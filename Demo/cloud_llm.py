"""Thin wrapper for calling a configurable CLOUD LLM HTTP API."""
from __future__ import annotations

import os
from typing import Any, Dict

import requests


def call_cloud_llm(prompt: str, system: str, timeout: float = 20.0) -> str:
    """
    Call a CLOUD LLM via HTTP.

    Configuration (env):
        CLOUD_LLM_URL: base URL for the API (required)
        CLOUD_LLM_API_KEY: optional bearer/API key

    Expected JSON payload (generic):
        {
            "prompt": "<user text>",
            "system": "<system prompt>"
        }

    The exact schema can be adapted later; this function is a thin seam.
    """
    url = (os.environ.get("CLOUD_LLM_URL") or "").strip()
    if not url:
        return "(Cloud LLM not configured: set CLOUD_LLM_URL)"

    api_key = (os.environ.get("CLOUD_LLM_API_KEY") or "").strip()

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "prompt": (prompt or "").strip(),
        "system": (system or "").strip(),
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return f"(Cloud LLM HTTP error: {exc})"

    if resp.status_code != 200:
        return f"(Cloud LLM HTTP {resp.status_code})"

    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"(Cloud LLM JSON error: {exc})"

    # Be flexible about response schema; try common keys.
    for key in ("response", "text", "answer", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return "(Cloud LLM: empty response)"

