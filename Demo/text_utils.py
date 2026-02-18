"""Text postprocessing, LLM prompt constants, and conversation history."""
from __future__ import annotations

import os
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


_MEMORY_MINILM_ENCODER: Optional[Any] = None


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
        "name": re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z0-9_\- ]{1,64})", re.IGNORECASE),
        "location": re.compile(r"\b(?:i live in|i am in|i'm in)\s+([A-Za-z][A-Za-z0-9_\- ]{1,64})", re.IGNORECASE),
        "preference": re.compile(r"\b(?:i prefer|i like|i need)\s+([A-Za-z][A-Za-z0-9_\- ]{1,64})", re.IGNORECASE),
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


def _resolve_minilm_model_name() -> str:
    return (
        os.environ.get("MEMORY_MINILM_ONNX_DIR", "").strip()
        or os.environ.get("MEMORY_BERT_SUMMARIZER", "").strip()
        or "Xenova/all-MiniLM-L6-v2"
    )


def _resolve_minilm_snapshot_dir(model_name: str) -> str:
    """Resolve local directory for MiniLM ONNX bundle (path or HF repo id)."""
    from pathlib import Path

    value = (model_name or "").strip()
    if not value:
        return ""

    local_dir = Path(value).expanduser()
    if local_dir.exists():
        return str(local_dir)

    cache_dir = (os.environ.get("MEMORY_MINILM_CACHE_DIR") or "").strip() or None
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snap = snapshot_download(
            repo_id=value,
            cache_dir=cache_dir,
            allow_patterns=[
                "onnx/*.onnx",
                "*.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
                "config.json",
            ],
        )
        return str(Path(snap))
    except Exception:
        return ""


def _resolve_minilm_onnx_path(model_dir: str) -> str:
    """Pick ONNX graph path from a MiniLM model directory."""
    from pathlib import Path

    base = Path(model_dir).expanduser()
    candidates = [
        base / "model.int8.onnx",
        base / "model.onnx",
        base / "onnx" / "model.int8.onnx",
        base / "onnx" / "model.onnx",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


class _MiniLMOnnxEncoder:
    """Minimal ONNX sentence encoder compatible with extractive memory scoring."""

    def __init__(self, model_dir: str) -> None:
        from pathlib import Path
        from transformers import AutoTokenizer  # type: ignore
        import onnxruntime as ort  # type: ignore

        base = Path(model_dir).expanduser()
        onnx_path = _resolve_minilm_onnx_path(str(base))
        if not onnx_path:
            raise FileNotFoundError(f"MiniLM ONNX graph not found under: {base}")

        tokenizer_dir = base
        if not (tokenizer_dir / "tokenizer.json").exists():
            onnx_tok_dir = base / "onnx"
            if (onnx_tok_dir / "tokenizer.json").exists():
                tokenizer_dir = onnx_tok_dir

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.max_length = int(os.environ.get("MEMORY_MINILM_MAX_LENGTH", "256"))
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def encode(
        self,
        sentences: Any,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            texts = [sentences]
        else:
            texts = [str(s) for s in (sentences or [])]

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max(8, self.max_length),
            return_tensors="np",
        )
        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(
            tokenized.get("attention_mask", np.ones_like(input_ids)),
            dtype=np.int64,
        )

        feeds: Dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name in tokenized:
                feeds[name] = np.asarray(tokenized[name], dtype=np.int64)
            elif name == "input_ids":
                feeds[name] = input_ids
            elif name == "attention_mask":
                feeds[name] = attention_mask
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(input_ids, dtype=np.int64)
            else:
                feeds[name] = input_ids

        raw_outputs = self.session.run(None, feeds)
        token_embeddings: Optional[np.ndarray] = None
        sentence_embeddings: Optional[np.ndarray] = None

        for out in raw_outputs:
            arr = np.asarray(out)
            if arr.ndim == 3 and token_embeddings is None:
                token_embeddings = arr
            elif arr.ndim == 2 and sentence_embeddings is None:
                sentence_embeddings = arr

        if token_embeddings is not None:
            mask = attention_mask.astype(np.float32)[..., None]
            summed = (token_embeddings * mask).sum(axis=1)
            denom = np.clip(mask.sum(axis=1), 1e-6, None)
            sent = summed / denom
        elif sentence_embeddings is not None:
            sent = sentence_embeddings
        else:
            raise RuntimeError("MiniLM ONNX output shape is not supported.")

        embeddings = np.asarray(sent, dtype=np.float32)
        if normalize_embeddings and embeddings.size > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)
        if convert_to_numpy:
            return embeddings
        return embeddings


def _get_memory_embedding_model() -> Optional[Any]:
    global _MEMORY_MINILM_ENCODER
    if _MEMORY_MINILM_ENCODER is not None:
        return _MEMORY_MINILM_ENCODER

    model_name = _resolve_minilm_model_name()
    if not model_name:
        return None

    try:
        model_dir = _resolve_minilm_snapshot_dir(model_name)
        if not model_dir:
            return None
        _MEMORY_MINILM_ENCODER = _MiniLMOnnxEncoder(model_dir)
        return _MEMORY_MINILM_ENCODER
    except Exception:
        return None


def _split_turn_sentences(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    normalized = re.sub(r"\s+", " ", raw)
    chunks = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not sentences:
        return [normalized]
    return sentences


def _extractive_turn_summary(turn: Tuple[str, str], max_words: int) -> str:
    user, assistant = turn
    combined = f"{(user or '').strip()} {(assistant or '').strip()}".strip()
    if not combined:
        return ""

    candidates = _split_turn_sentences(combined)
    if len(candidates) == 1:
        return postprocess_output(candidates[0], max_sentences=1, max_words=max_words)

    try:
        encoder = _get_memory_embedding_model()
        if encoder is None:
            raise RuntimeError("memory_encoder_unavailable")

        embeddings = encoder.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
        if embeddings is None or len(embeddings) == 0:
            raise RuntimeError("memory_embedding_empty")

        arr = np.asarray(embeddings, dtype=np.float32)
        centroid = np.mean(arr, axis=0)
        scores = arr @ centroid
        top_idx = int(np.argmax(scores))
        return postprocess_output(candidates[top_idx], max_sentences=1, max_words=max_words)
    except Exception:
        return _summarize_turns_fallback([turn], max_words=max_words)

def _summarize_text(text: str, max_words: int) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    return postprocess_output(raw, max_sentences=4, max_words=max_words)


def _summarize_turns_bert(turns: List[Tuple[str, str]], max_words: int) -> str:
    if not turns:
        return ""
    if len(turns) == 1:
        return _extractive_turn_summary(turns[0], max_words)

    fragments: List[str] = []
    per_turn_budget = max(12, max_words // max(1, len(turns)))
    for turn in turns:
        fragment = _extractive_turn_summary(turn, max_words=per_turn_budget)
        if fragment:
            fragments.append(fragment)

    if not fragments:
        return ""
    merged = " ".join(fragments).strip()
    if not merged:
        return ""
    if len(merged.split()) <= max_words:
        return _summarize_text(merged, max_words=max_words)
    return _summarize_text(merged, max_words=max_words)


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
        self._summary_fragments: List[str] = []
        self.pinned_facts: Dict[str, str] = {}

    @property
    def _summary_fragment_budget(self) -> int:
        if self.max_summary_turns <= 0:
            return self.summary_word_budget
        return max(12, self.summary_word_budget // self.max_summary_turns)

    def _append_summary_fragment(self, archived_turn: Tuple[str, str]) -> None:
        if self.max_summary_turns <= 0:
            return
        user, assistant = archived_turn
        # Prefer MiniLM-based extractive summarization per archived turn.
        # _extractive_turn_summary internally falls back to deterministic text
        # compaction when the embedding model is unavailable.
        fragment = _extractive_turn_summary(
            (user, assistant),
            max_words=self._summary_fragment_budget,
        )
        if fragment:
            self._summary_fragments.append(fragment)
            if len(self._summary_fragments) > self.max_summary_turns:
                self._summary_fragments = self._summary_fragments[-self.max_summary_turns :]

    def _refresh_rolling_summary(self) -> str:
        if self.max_summary_turns <= 0 or not self._summary_fragments:
            return ""
        text = " ".join(self._summary_fragments).strip()
        if not text:
            return ""
        if len(text.split()) <= self.summary_word_budget:
            return postprocess_output(text, max_sentences=4, max_words=self.summary_word_budget)
        return _summarize_text(text, max_words=self.summary_word_budget)

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
            self._append_summary_fragment(self._archived_turns[-1])

        self.turns.append((user, assistant))
        self.pinned_facts.update(_extract_pinned_facts(user))
        self.rolling_summary = self._refresh_rolling_summary()

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
        self._summary_fragments.clear()
        self.pinned_facts.clear()
        self.rolling_summary = ""

    def __len__(self) -> int:
        return len(self.turns)
