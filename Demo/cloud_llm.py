"""Thin wrapper for calling a configurable CLOUD LLM (Gemini-first by default)."""
from __future__ import annotations

import os
from typing import Any, Dict

import requests

# OpenAI: https://platform.openai.com/docs/api-reference/chat
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Gemini (Google AI) free tier: https://ai.google.dev/gemini-api/docs
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _call_openai(prompt: str, system: str, timeout: float, max_output_tokens: int = 512, temperature: float = 0.7) -> str:
    """Call OpenAI Chat Completions API (GPT-4o, gpt-4o-mini, etc.)."""
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return "(OpenAI not configured: set OPENAI_API_KEY)"

    model = (os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
    print(f"[Cloud] Calling OpenAI API ({model})...", flush=True)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": (system or "").strip()},
            {"role": "user", "content": (prompt or "").strip()},
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = requests.post(OPENAI_CHAT_URL, json=payload, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return f"(OpenAI HTTP error: {exc})"

    if resp.status_code != 200:
        return f"(OpenAI HTTP {resp.status_code})"

    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"(OpenAI JSON error: {exc})"

    try:
        choices = data.get("choices") or []
        if not choices:
            return "(OpenAI: empty response)"
        content = (choices[0].get("message") or {}).get("content") or ""
        return (content or "(OpenAI: empty response)").strip()
    except (KeyError, IndexError, TypeError) as exc:  # noqa: BLE001
        return f"(OpenAI parse error: {exc})"


def _call_gemini(
    prompt: str,
    system: str,
    timeout: float,
    max_output_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Call Google Gemini API (e.g. Gemini 2.5 Flash free tier)."""
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("CLOUD_LLM_API_KEY") or "").strip()
    if not api_key:
        return "(Gemini not configured: set GEMINI_API_KEY)"

    model = (os.environ.get("CLOUD_LLM_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    print(f"[Cloud] Calling Gemini API ({model})...", flush=True)
    url = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": (prompt or "").strip()}]}],
        "systemInstruction": {"parts": [{"text": (system or "").strip()}]},
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
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


def call_cloud_llm(
    prompt: str,
    system: str,
    timeout: float = 20.0,
    max_output_tokens: int = 512,
    temperature: float = 0.7,
    preferred_provider: str | None = None,
) -> str:
    """
    Call a CLOUD LLM.

    Priority: MAIN_LLM_PROVIDER (default: Gemini) → OPENAI_API_KEY → GEMINI_API_KEY → CLOUD_LLM_URL.

    Env (OpenAI):
        OPENAI_API_KEY: OpenAI API key
        OPENAI_MODEL: optional, default gpt-4o-mini

    Env (Gemini):
        GEMINI_API_KEY: Google AI Studio API key
        CLOUD_LLM_MODEL: optional, default gemini-2.5-flash

    Env (custom HTTP):
        CLOUD_LLM_URL: base URL for the API
        CLOUD_LLM_API_KEY: optional bearer/API key
    """
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    url = (os.environ.get("CLOUD_LLM_URL") or "").strip()
    provider = (preferred_provider or os.environ.get("MAIN_LLM_PROVIDER") or "gemini").strip().lower()

    if provider == "openai":
        if openai_key:
            return _call_openai(prompt, system, timeout, max_output_tokens=max_output_tokens, temperature=temperature)
        return "(OpenAI not configured: set OPENAI_API_KEY)"
    if provider == "gemini":
        if gemini_key:
            return _call_gemini(prompt, system, timeout, max_output_tokens=max_output_tokens, temperature=temperature)
        return "(Gemini not configured: set GEMINI_API_KEY)"
    if provider == "custom":
        if url:
            return _call_custom_http(prompt, system, timeout)
        return "(Cloud custom endpoint not configured: set CLOUD_LLM_URL)"

    if gemini_key:
        return _call_gemini(
            prompt,
            system,
            timeout,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if openai_key:
        return _call_openai(prompt, system, timeout, max_output_tokens=max_output_tokens, temperature=temperature)
    if url:
        return _call_custom_http(prompt, system, timeout)
    return "(Cloud LLM not configured: set MAIN_LLM_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY, or CLOUD_LLM_URL)"


def _call_custom_http(prompt: str, system: str, timeout: float) -> str:
    """Call custom CLOUD LLM via HTTP (generic payload)."""
    url = (os.environ.get("CLOUD_LLM_URL") or "").strip()
    print(f"[Cloud] Calling custom HTTP LLM ({url})...", flush=True)
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
