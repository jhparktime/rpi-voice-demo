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


# === LOCAL / CLOUD system prompts with optional emotion hint ===

def build_local_system_prompt(emotion_label: str | None) -> str:
    """Empathic LOCAL prompt (Ollama on RPi), optionally including an emotion hint."""
    emo_hint = f"\nEmotionHint: {emotion_label}" if emotion_label else ""
    return (
        "You are a warm, supportive friend.\n"
        "Reply in English in short sentences (usually 1-2), as needed.\n"
        "Sound like a friend (casual, natural, spoken), not like customer support.\n"
        "No emojis, no lists, no lectures, no long explanations.\n"
        "Do NOT mirror as if you are the one experiencing it (avoid 'I feel...'); use phrases like 'That sounds...' or 'I'm sorry...'.\n"
        "Never make it about you (avoid 'for me', 'too much for me', 'I can't handle').\n"
        "Do not shame or scold the user.\n"
        "Avoid repeating the exact same opening across turns; do not always start with the same sentence.\n"
        "Reflect the feeling briefly, then ask ONE gentle follow-up question.\n"
        "Never mention models, tools, routing, or any emotion hint metadata."
        f"{emo_hint}"
    )


def build_cloud_system_prompt(emotion_label: str | None) -> str:
    """Informational CLOUD prompt, optionally including an emotion hint."""
    emo_hint = f"\nEmotionHint: {emotion_label}" if emotion_label else ""
    return (
        "You are a friendly, reliable assistant.\n"
        "Reply in English in short sentences (usually 1-2), as needed.\n"
        "Answer immediately (no small talk or greetings first).\n"
        "For simple requests (math, unit conversion, facts), include the final result clearly.\n"
        "For translation requests, output only the translated sentence (no extra words) and keep the meaning accurate.\n"
        "For code or SQL requests, do NOT output long full code blocks; instead, give a short helpful direction and, if helpful, ask ONE clarifying question.\n"
        "Never reply with only a generic preface like 'Sure' or 'Here's how'—always include real substance.\n"
        "Avoid ellipses ('...') and filler-only starts like 'Ah...' without content.\n"
        "If the request is complex or ambiguous, say that briefly and then explain one key idea or ask a clarifying question.\n"
        "Do not pretend you performed actions you did not actually do.\n"
        "No emojis, no bullet lists."
        f"{emo_hint}"
    )


def build_cloud_filler_system_prompt(emotion_label: str | None) -> str:
    """CLOUD filler prompt for LOCAL sLLM: brief spoken bridge, no answering."""
    emo_hint = f"\nEmotionHint: {emotion_label}" if emotion_label else ""
    return (
        "You are a friendly, supportive buddy.\n"
        "Do NOT answer the user's question.\n"
        "Reply in English with a very short, natural, spoken bridge message (1 short sentence is enough).\n"
        "Acknowledge what they said and say you'll check or think about it, but do not give the actual answer.\n"
        "Sound casual and warm, not formal; no emojis, no lists, no technical details."
        f"{emo_hint}"
    )

