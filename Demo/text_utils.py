"""Text postprocessing, LLM prompt constants, and conversation history."""
from __future__ import annotations

import re
from typing import List, Tuple


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
        "Sound natural and spoken, like you're chatting with a friend.\n"
        "No emojis, no lists, no lectures, no long explanations.\n"
        "Do NOT mirror as if you are the one experiencing it (avoid 'I feel...'); use phrases like 'That sounds...' or 'I'm sorry...'.\n"
        "Never make it about you (avoid 'for me', 'too much for me', 'I can't handle').\n"
        "Do not shame or scold the user.\n"
        "Avoid repeating the exact same opening across turns; vary your responses naturally.\n"
        "Reflect the feeling briefly, then ask ONE gentle follow-up question.\n"
        "Never mention models, tools, routing, or any emotion hint metadata."
        f"{emo_hint}"
    )


def build_cloud_system_prompt(emotion_label: str | None) -> str:
    """Informational CLOUD prompt, optionally including an emotion hint."""
    emo_hint = f"\nEmotionHint: {emotion_label}" if emotion_label else ""
    return (
        "You are a warm, supportive friend who just looked something up for them.\n"
        "Reply in English in short sentences (usually 1-2), as needed.\n"
        "Sound natural and spoken, like you're sharing what you found with a friend.\n"
        "Get straight to the answer—no generic prefaces like 'Sure' or 'Here's how' without content.\n"
        "For simple requests (math, unit conversion, facts), include the clear answer.\n"
        "For translation requests, give the translated sentence naturally (keep the meaning accurate).\n"
        "For code or complex requests, give a short helpful direction, not full code blocks.\n"
        "If the request is complex or unclear, explain one key idea briefly or ask a clarifying question.\n"
        "Avoid ellipses ('...') and filler-only starts without substance.\n"
        "Do not pretend you performed actions you didn't actually do.\n"
        "No emojis, no bullet lists."
        f"{emo_hint}"
    )


def build_cloud_filler_system_prompt(emotion_label: str | None) -> str:
    """CLOUD filler prompt for LOCAL sLLM: brief spoken bridge, no answering."""
    emo_hint = f"\nEmotionHint: {emotion_label}" if emotion_label else ""
    return (
        "Reply with EXACTLY one of these short bridge phrases:\n"
        "- 'Let me check that for you.'\n"
        "- 'Give me a second, I'll look that up.'\n"
        "- 'Let me find that out.'\n"
        "- 'One moment, let me check.'\n"
        "\n"
        "CRITICAL RULES:\n"
        "- DO NOT answer the question\n"
        "- DO NOT apologize\n"
        "- DO NOT explain limitations ('I don't have access to...', 'as a chatbot...', etc.)\n"
        "- DO NOT add extra sentences\n"
        "- ONLY say you'll check/look it up\n"
        "- Keep it under 10 words"
        f"{emo_hint}"
    )


# ── Conversation history buffer ───────────────────────────────────────────


class ConversationBuffer:
    """Ring buffer that stores the last N conversation turns for multi-turn context.

    Each turn is a (user_text, assistant_reply) pair.
    When building the LLM prompt, the history is prepended so the model can
    maintain conversational coherence across turns.
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []

    def add_turn(self, user_text: str, assistant_reply: str) -> None:
        """Record a completed conversation turn."""
        self.turns.append((user_text, assistant_reply))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format_prompt_with_context(self, current_user_text: str) -> str:
        """Build a prompt string that includes prior conversation context.

        If there is no history, returns the current text as-is.
        """
        if not self.turns:
            return current_user_text

        parts: List[str] = []
        for user, assistant in self.turns:
            parts.append(f"User: {user}")
            parts.append(f"Assistant: {assistant}")
        parts.append(f"User: {current_user_text}")
        return "\n".join(parts)

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

