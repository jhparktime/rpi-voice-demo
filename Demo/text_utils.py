"""Text postprocessing and LLM prompt constants."""
from __future__ import annotations

import re
from typing import Tuple


def _limit_words(s: str, max_words: int) -> Tuple[str, bool]:
    if max_words <= 0:
        return (s or "").strip(), False
    parts = (s or "").strip().split()
    if len(parts) <= max_words:
        return " ".join(parts).strip(), False
    return " ".join(parts[:max_words]).strip(), True


def _limit_sentences(s: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return (s or "").strip()
    text = (s or "").strip()
    if not text:
        return ""
    text = text.replace("...", "…")
    text = re.sub(r"\.\s*\.\s*\.", "…", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return text
    return " ".join(parts[:max_sentences]).strip()


def postprocess_output(text: str, max_sentences: int, max_words: int) -> str:
    s = (text or "").strip()
    if not s:
        return s
    s = " ".join([ln.strip() for ln in s.splitlines() if ln.strip()]).strip()
    s = _limit_sentences(s, max_sentences=max_sentences)
    s, truncated = _limit_words(s, max_words=max_words)
    if truncated:
        s = s.rstrip(" ,:;") + "…"
    else:
        if s and s[-1] not in ".!?":
            if s[-1] not in "\"'\"\u201d')]}":
                s = s.rstrip(",:;") + "."
    return s.strip()


OLLAMA_DEFAULT_SYSTEM = (
    "You are a warm, supportive friend. Reply in English in 1-2 short sentences. "
    "No emojis, no lists. Sound natural and spoken."
)

# ONNX LLM: discourage echoing; small models tend to repeat the user
ONNX_DEFAULT_SYSTEM = (
    "You are a warm, supportive friend. Reply in English in 1-2 short sentences. "
    "Do NOT repeat or echo the user's words. Give your own brief reply (e.g. answer a question, react to what they said). "
    "No emojis, no lists. Sound natural and spoken."
)
