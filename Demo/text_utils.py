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
        "Generate a SHORT bridge phrase (ideally 3-8 words) while I process the request.\n"
        "Examples (feel free to vary naturally):\n"
        "- 'Let me think about that AI question with you.'\n"
        "- 'Give me a second to organize that CPU idea.'\n"
        "- 'One moment, I am lining up the key points.'\n"
        "- 'Let me gather the most important details for you.'\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Keep it under 10 words\n"
        "- You may mention the topic, but DO NOT answer the question\n"
        "- DO NOT apologize or explain limitations\n"
        "- DO NOT use random poetic imagery (flowers, wind, weather, etc.) unrelated to the topic\n"
        "- Just a quick bridge phrase, not an answer\n"
        "- Sound natural and conversational"
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

