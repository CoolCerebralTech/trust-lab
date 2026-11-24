"""Loads settings and builds API clients.

Design note: all three providers expose an OpenAI-compatible API, so one client
class works for all of them. Adding a fourth provider later is a config change,
not a code change.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# Repo root = two folders up from this file (src/trustlab/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Reads .env and puts the keys into environment variables for this process only.
load_dotenv(ROOT / ".env")


def load_models_config() -> dict:
    with open(ROOT / "config" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def provider_by_name(cfg: dict, name: str) -> dict:
    return cfg["providers"][name]


def model_provider_cfg(cfg: dict, model: dict) -> dict:
    return cfg["providers"][model["provider"]]


def make_client(provider_cfg: dict):
    """Return an OpenAI-compatible client, or None if the key is missing."""
    key = os.environ.get(provider_cfg["key_env"], "").strip()
    if not key:
        return None
    # max_retries=0 on purpose: in Stage 1 we write our own retry logic so that
    # EVERY attempt is logged. Some providers count failed attempts toward daily
    # limits, so hidden automatic retries can silently eat your quota.
    return OpenAI(api_key=key, base_url=provider_cfg["base_url"], timeout=60, max_retries=0)
