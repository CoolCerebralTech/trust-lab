"""One wrapper around all LLM providers: cache, retries, rate limiting, logging.

Every call is logged to logs/calls.jsonl (request, response, timestamp, ok/error)
and cached to cache/<hash>.json so re-runs are cheap and resumable.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from trustlab.config import ROOT, make_client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(provider: str, model_id: str, prompt: str, **kw) -> str:
    h = hashlib.sha256()
    payload = {"provider": provider, "model": model_id, "prompt": prompt, "kw": kw}
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


class LLMClient:
    def __init__(self, settings: dict):
        self.settings = settings
        self.cache_enabled = bool(settings.get("cache_enabled", True))
        self.cache_dir = ROOT / settings.get("cache_dir", "cache")
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = ROOT / settings.get("log_dir", "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "calls.jsonl"

        self.rpm = float(settings.get("requests_per_minute", 20))
        self._min_interval = 60.0 / max(self.rpm, 1)
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clients: dict[str, object] = {}

    # ------------------------------------------------------------------ client
    def _client_for(self, provider_cfg: dict):
        key = __import__("os").environ.get(provider_cfg["key_env"], "").strip()
        if not key:
            return None
        pname = provider_cfg.get("name") or provider_cfg["key_env"]
        if pname not in self._clients:
            self._clients[pname] = make_client(provider_cfg)
        return self._clients[pname]

    # ------------------------------------------------------------- rate limit
    def _rate_limit(self, provider_name: str) -> None:
        with self._lock:
            now = time.time()
            last = self._last_call.get(provider_name, 0.0)
            wait = self._min_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._last_call[provider_name] = time.time()

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        if not self.cache_enabled:
            return None
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, record: dict) -> None:
        if not self.cache_enabled:
            return
        try:
            self._cache_path(key).write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception:
            pass  # cache is a performance aid, never fatal

    # -------------------------------------------------------------------- log
    def _log(self, record: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------- call
    def call(
        self,
        provider_name: str,
        provider_cfg: dict,
        model_id: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        dry_run: bool = False,
    ) -> tuple[str, dict]:
        max_tokens = max_tokens if max_tokens is not None else self.settings.get("max_tokens", 400)
        temperature = temperature if temperature is not None else self.settings.get("temperature", 0.0)
        retries = int(self.settings.get("retries", 3))
        backoff = float(self.settings.get("retry_backoff_seconds", 2))

        key = _cache_key(provider_name, model_id, prompt, max_tokens=max_tokens, temperature=temperature)

        if dry_run:
            answer = f"[DRY-RUN] {provider_name}::{model_id}: {prompt[:60]}..."
            record = {
                "time_utc": _now(), "dry_run": True,
                "provider": provider_name, "model": model_id,
                "prompt": prompt, "ok": True, "answer": answer,
                "seconds": 0.0, "cached": False,
            }
            self._log(record)
            return answer, record

        # Cache hit -> resume without spending an API call.
        cached = self._read_cache(key)
        if cached:
            record = dict(cached)
            record["cached"] = True
            self._log(record)
            return cached["answer"], record

        client = self._client_for(provider_cfg)
        if client is None:
            raise RuntimeError(f"No API key found for {provider_cfg['key_env']}")

        last_err = None
        for attempt in range(1, retries + 1):
            self._rate_limit(provider_name)
            start = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = (resp.choices[0].message.content or "").strip()
                seconds = round(time.time() - start, 2)
                record = {
                    "time_utc": _now(), "dry_run": False,
                    "provider": provider_name, "model": model_id,
                    "prompt": prompt, "ok": True, "answer": answer,
                    "seconds": seconds, "attempt": attempt, "cached": False,
                }
                self._write_cache(key, record)
                self._log(record)
                return answer, record
            except Exception as e:  # network, bad key, wrong URL: show it plainly
                last_err = e
                seconds = round(time.time() - start, 2)
                record = {
                    "time_utc": _now(), "dry_run": False,
                    "provider": provider_name, "model": model_id,
                    "prompt": prompt, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "seconds": seconds, "attempt": attempt, "cached": False,
                }
                self._log(record)
                if attempt < retries:
                    time.sleep(backoff * attempt)
        raise last_err
