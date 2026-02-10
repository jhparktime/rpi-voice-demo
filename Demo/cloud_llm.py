"""Thin wrapper for calling a configurable CLOUD LLM (Gemini or custom HTTP API)."""
from __future__ import annotations

import os
from typing import Any, Dict

import requests

# Gemini (Google AI) free tier: https://ai.google.dev/gemini-api/docs
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _call_gemini(prompt: str, system: str, timeout: float) -> str:
    """Call Google Gemini API (e.g. Gemini 2.5 Flash free tier)."""
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("CLOUD_LLM_API_KEY") or "").strip()
    if not api_key:
        return "(Gemini not configured: set GEMINI_API_KEY)"

    model = (os.environ.get("CLOUD_LLM_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    url = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": (prompt or "").strip()}]}],
        "systemInstruction": {"parts": [{"text": (system or "").strip()}]},
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature": 0.7,
        },
    }

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return f"(Gemini HTTP error: {exc})"

    if resp.status_code != 200:
        return f"(Gemini HTTP {resp.status_code})"

    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"(Gemini JSON error: {exc})"

    try:
        candidates = data.get("candidates") or []
        if not candidates:
            return "(Gemini: empty response)"
        parts = (candidates[0].get("content") or {}).get("parts") or []
        if not parts:
            return "(Gemini: empty response)"
        text = (parts[0].get("text") or "").strip()
        return text or "(Gemini: empty response)"
    except (KeyError, IndexError, TypeError) as exc:  # noqa: BLE001
        return f"(Gemini parse error: {exc})"


def call_cloud_llm(prompt: str, system: str, timeout: float = 20.0) -> str:
    """
    Call a CLOUD LLM.

    If GEMINI_API_KEY is set: use Google Gemini (e.g. Gemini 2.5 Flash free tier).
    Else if CLOUD_LLM_URL is set: use custom HTTP API (generic payload).

    Env (Gemini):
        GEMINI_API_KEY: Google AI Studio API key (do not commit; use .env or shell export)
        CLOUD_LLM_MODEL: optional, default gemini-2.5-flash

    Env (custom HTTP):
        CLOUD_LLM_URL: base URL for the API
        CLOUD_LLM_API_KEY: optional bearer/API key
    """
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    url = (os.environ.get("CLOUD_LLM_URL") or "").strip()

    if api_key:
        return _call_gemini(prompt, system, timeout)
    if url:
        return _call_custom_http(prompt, system, timeout)
    return "(Cloud LLM not configured: set GEMINI_API_KEY or CLOUD_LLM_URL)"


def _call_custom_http(prompt: str, system: str, timeout: float) -> str:
    """Call custom CLOUD LLM via HTTP (generic payload)."""
    url = (os.environ.get("CLOUD_LLM_URL") or "").strip()
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

    for key in ("response", "text", "answer", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "(Cloud LLM: empty response)"

