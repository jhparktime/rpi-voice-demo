"""Text postprocessing, LLM prompt constants, and conversation history."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


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


def split_into_chunks(text: str, max_words_per_chunk: int | None = None) -> List[str]:
    """Split text into sentence/word chunks for streaming TTS.

    1) First split on sentence boundaries (. ! ?) while preserving punctuation.
    2) Optionally re-split long sentences into smaller chunks by word count
       (max_words_per_chunk), so that each chunk is TTS-friendly on low-power
       devices (e.g., Raspberry Pi).

    Returns:
        List of sentence strings (each including its punctuation).
        Empty strings are filtered out.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Split on sentence boundaries while capturing the punctuation
    # Use lookahead to avoid splitting on common abbreviations
    # Pattern: split after .!? followed by space (or end), but not after common abbreviations
    parts = re.split(r'(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bProf)([.!?]+)\s+', text)

    # Combine text with its punctuation
    sentence_chunks: List[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i+1] in ['.', '!', '?', '..', '...', '!?', '?!']:
            # Combine text with punctuation
            chunk = parts[i] + parts[i+1]
            sentence_chunks.append(chunk.strip())
            i += 2
        else:
            # Text without captured punctuation (last chunk or already has punctuation)
            if parts[i].strip():
                sentence_chunks.append(parts[i].strip())
            i += 1

    # If no word limit requested, return sentence chunks as-is
    if not max_words_per_chunk or max_words_per_chunk <= 0:
        return [c for c in sentence_chunks if c]

    # Re-split each sentence chunk by word count
    final_chunks: List[str] = []
    for sent in sentence_chunks:
        words = sent.split()
        if len(words) <= max_words_per_chunk:
            final_chunks.append(sent.strip())
            continue

        current: List[str] = []
        for w in words:
            current.append(w)
            if len(current) >= max_words_per_chunk:
                final_chunks.append(" ".join(current).strip())
                current = []
        if current:
            final_chunks.append(" ".join(current).strip())

    return [c for c in final_chunks if c]


_FILLER_FORBIDDEN_PATTERNS = [
    re.compile(r"\bhere(?:'s| is)\b", re.IGNORECASE),
    re.compile(r"\bthe answer\b", re.IGNORECASE),
    re.compile(r"\bsql query\b", re.IGNORECASE),
    re.compile(r"\bi(?: am|'m)?\s*sorry\b", re.IGNORECASE),
    re.compile(r"\bjust kidding\b", re.IGNORECASE),
    re.compile(r"\bbecause\b", re.IGNORECASE),
]

_FILLER_NAME_ENTITY_HINT_PATTERNS = [
    re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"),
    re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+\b"),
]

_FILLER_FALLBACKS = [
    "One moment.",
    "Just a sec.",
    "Checking that now.",
    "Working on it.",
    "Let me check.",
]


def validate_cloud_filler_output(text: str, min_words: int = 3, max_words: int = 12) -> Optional[str]:
    """Validate generated filler with soft constraints; return normalized text or None."""
    s = (text or "").strip()
    if not s:
        return None
    s = " ".join([ln.strip() for ln in s.splitlines() if ln.strip()]).strip().strip("\"'")
    if not s:
        return None
    if "?" in s or re.search(r"\d", s):
        return None
    if any(p.search(s) for p in _FILLER_NAME_ENTITY_HINT_PATTERNS):
        return None
    if len(re.findall(r"[.!?]+", s)) > 1:
        return None
    if any(p.search(s) for p in _FILLER_FORBIDDEN_PATTERNS):
        return None
    words = s.split()
    if len(words) < min_words or len(words) > max_words:
        return None
    if s[-1] not in ".!?":
        s = s.rstrip(",:;") + "."
    return s


def should_emit_long_filler(route_mode: str, elapsed_s: float, delay_ms: int) -> bool:
    """Gate helper used by the speech-delay filler rule."""
    try:
        if (route_mode or "").strip().upper() != "LONG":
            return False
        delay_seconds = float(delay_ms) / 1000.0
        return delay_seconds > 0.0 and float(elapsed_s) >= delay_seconds
    except (TypeError, ValueError):
        return False


def fallback_cloud_filler(user_text: str = "") -> str:
    """Return a short natural fallback filler; deterministic by user text hash."""
    idx = abs(hash(user_text or "")) % len(_FILLER_FALLBACKS)
    return _FILLER_FALLBACKS[idx]


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
        "You are a knowledgeable assistant providing accurate, concise information.\n"
        "Reply in English in 2–3 sentences, focusing on the single most important idea for the user.\n"
        "Start by directly answering the question in 1 clear sentence, then add 1–2 short supporting details or examples.\n"
        "Sound professional yet conversational—like an expert explaining things clearly to an interested friend.\n"
        "\n"
        "Guidelines:\n"
        "- Get straight to the main answer with specific facts or explanations\n"
        "- For technical topics: state the core idea, then give one intuitive example\n"
        "- For 'how-to' questions: outline only the key steps at a high level\n"
        "- Use simple, spoken-style sentences (no bullet lists or code blocks)\n"
        "\n"
        "Avoid:\n"
        "- Long, essay-style explanations\n"
        "- Generic prefaces without substance ('Sure, here's...', 'Let me tell you...')\n"
        "- Emojis, roleplay, or overly casual language\n"
        "- Pretending you did actions you can't do\n"
        "\n"
        "If the user seems emotional or personally affected, you may briefly acknowledge their feeling in one short sentence before or after the main explanation."
        f"{emo_hint}"
    )


def build_cloud_filler_system_prompt(emotion_label: str | None) -> str:
    """CLOUD filler prompt for LOCAL sLLM: brief spoken bridge, no answering."""
    emo_hint = f"\nEmotionHint: {emotion_label}" if emotion_label else ""
    return (
        "You are generating a SHORT spoken bridge phrase while a heavier Cloud model is thinking.\n"
        "You may briefly acknowledge the TOPIC of the user's question, but you must NOT answer it.\n"
        "Keep the phrase neutral and supportive.\n"
        "\n"
        "Generate a SHORT bridge phrase (ideally 6-10 words) while I process the request.\n"
        "Examples (feel free to vary naturally):\n"
        "- 'Let me think about that AI question with you.'\n"
        "- 'Give me a second to organize that CPU idea.'\n"
        "- 'One moment, I am lining up the key points.'\n"
        "- 'Let me gather the most important details for you.'\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Keep it between 6 and 12 words\n"
        "- You may mention the topic, but DO NOT answer the question\n"
        "- DO NOT apologize or explain limitations\n"
        "- DO NOT use random poetic imagery (flowers, wind, weather, etc.) unrelated to the topic\n"
        "- Just a quick bridge phrase, not an answer\n"
        "- Sound natural and conversational"
        f"{emo_hint}"
    )


# ── Conversation history buffer ───────────────────────────────────────────


def _extract_pinned_facts(statement: str) -> Dict[str, str]:
    text = (statement or "").strip()
    if not text:
        return {}
    facts: Dict[str, str] = {}
    patterns: Dict[str, re.Pattern[str]] = {
        "name": re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z0-9_\\- ]{1,64})", re.IGNORECASE),
        "location": re.compile(r"\b(?:i live in|i am in|i'm in)\s+([A-Za-z][A-Za-z0-9_\\- ]{1,64})", re.IGNORECASE),
        "preference": re.compile(r"\b(?:i prefer|i like|i need)\s+([A-Za-z][A-Za-z0-9_\\- ]{1,64})", re.IGNORECASE),
    }
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            value = (match.group(1) or "").strip().strip(",.")
            if value:
                facts[key] = value
    return facts


def _summarize_turns_fallback(turns: List[Tuple[str, str]], max_words: int) -> str:
    if not turns:
        return ""
    compact: List[str] = []
    for user, assistant in turns:
        user = (user or "").strip()
        assistant = (assistant or "").strip()
        if not user or not assistant:
            continue
        compact.append(f"User asked: {user[:72].strip()} | Assistant said: {assistant[:72].strip()}")
    if not compact:
        return ""
    return postprocess_output(" ".join(compact), max_sentences=4, max_words=max_words)


def _summarize_turns_bert(turns: List[Tuple[str, str]], max_words: int) -> str:
    model_name = os.environ.get("MEMORY_BERT_SUMMARIZER")
    if not model_name:
        return _summarize_turns_fallback(turns, max_words=max_words)
    try:
        from transformers import pipeline  # type: ignore

        summarizer = pipeline("summarization", model=model_name)
        text = " ".join([f"{u} {a}" for u, a in turns]).strip()
        if not text:
            return ""
        result = summarizer(text, max_length=max(20, max_words), min_length=max(12, max_words // 3), do_sample=False)
        if isinstance(result, list) and result:
            return postprocess_output(
                result[0].get("summary_text", ""),
                max_sentences=4,
                max_words=max_words,
            )
    except Exception:
        pass
    return _summarize_turns_fallback(turns, max_words=max_words)


def update_memory(
    history: List[Tuple[str, str]],
    n_recent_turns: int = 3,
    max_summary_turns: int = 12,
    summary_word_budget: int = 120,
) -> Dict[str, Any]:
    """Build a deterministic memory state snapshot from a raw history list."""
    buffer = ConversationBuffer(
        max_turns=max(1, n_recent_turns),
        max_summary_turns=max_summary_turns,
        summary_word_budget=max(summary_word_budget, 20),
    )
    for user_text, assistant_text in history:
        buffer.add_turn(user_text, assistant_text)
    return buffer.as_dict()


class ConversationBuffer:
    """Maintain rolling summary + recent turns + pinned facts for LLM context."""

    def __init__(
        self,
        max_turns: int = 3,
        max_summary_turns: int = 12,
        summary_word_budget: int = 120,
    ) -> None:
        self.max_turns = max(1, max_turns)
        self.max_summary_turns = max(0, max_summary_turns)
        self.summary_word_budget = max(20, summary_word_budget)
        self.turns: List[Tuple[str, str]] = []
        self._archived_turns: List[Tuple[str, str]] = []
        self.rolling_summary = ""
        self.pinned_facts: Dict[str, str] = {}

    def add_turn(self, user_text: str, assistant_reply: str) -> None:
        """Record a completed conversation turn and update rolling memory."""
        user = (user_text or "").strip()
        assistant = (assistant_reply or "").strip()
        if not user or not assistant:
            return

        if len(self.turns) >= self.max_turns:
            self._archived_turns.append(self.turns.pop(0))
            if len(self._archived_turns) > self.max_summary_turns > 0:
                self._archived_turns = self._archived_turns[-self.max_summary_turns :]

        self.turns.append((user, assistant))
        self.pinned_facts.update(_extract_pinned_facts(user))
        self.rolling_summary = _summarize_turns_bert(self._archived_turns, max_words=self.summary_word_budget)

    def format_prompt_with_context(self, current_user_text: str) -> str:
        """Build context prompt with summary + pinned facts + recent raw turns."""
        user_text = (current_user_text or "").strip()
        parts: List[str] = []
        if self.rolling_summary:
            parts.append(f"RollingSummary: {self.rolling_summary}")
        if self.pinned_facts:
            facts = ", ".join(f"{k}={v}" for k, v in self.pinned_facts.items())
            parts.append(f"PinnedFacts: {facts}")
        if self.turns:
            parts.append("RecentTurns:")
            for user, assistant in self.turns:
                parts.append(f"User: {user}")
                parts.append(f"Assistant: {assistant}")
        parts.append(f"User: {user_text}")
        return "\n".join(parts) if parts else user_text

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rolling_summary": self.rolling_summary,
            "pinned_facts": dict(self.pinned_facts),
            "recent_raw_turns": list(self.turns),
        }

    def clear(self) -> None:
        self.turns.clear()
        self._archived_turns.clear()
        self.pinned_facts.clear()
        self.rolling_summary = ""

    def __len__(self) -> int:
        return len(self.turns)
