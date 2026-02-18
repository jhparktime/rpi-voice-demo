#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests


SYSTEM_PROMPT = (
    "You are a realtime voice filler generator for smooth turn-taking. "
    "Output exactly one short bridge sentence (6-14 words, max 16 tokens). "
    "Keep the response natural and spoken. Do not answer the user. "
    "Do not add facts. Say what you are about to do next so it feels like a semantic bridge."
)

SAFE_FILLERS = [
    "I'm taking a breath to organize the key points clearly.",
    "Give me one second, I'll structure this cleanly before I continue.",
    "One moment, I'm preparing a clearer next sentence.",
    "Got it, I'm checking the details and will continue in a second.",
    "Sure, I'm framing the key ideas so it sounds natural.",
    "Let me align the response structure, then I'll continue.",
    "I'll quickly verify the context and keep the flow smooth.",
    "One second, I'll take care of this and continue cleanly.",
    "Got it, I'm organizing the explanation for a cleaner follow-up.",
    "I'm pausing briefly to keep the next step crisp.",
]

EMPATHY_BRIDGE_TEMPLATES = [
    "I hear you. I'll hold that and keep this response calm and natural.",
    "Let me take a breath and respond to this feeling with a smoother tone.",
    "I'll pause briefly, then continue in a steady, natural way.",
    "I got you. I'll keep this reply warm and easy to follow.",
    "I'm taking a moment to keep the flow grounded and gentle.",
]

UNKNOWN_BRIDGE_TEMPLATES = [
    "Let me collect the core thought so this response flows naturally.",
    "I'm just setting up the next clear step, so this stays smooth.",
    "I'll prepare the flow so the follow-up feels natural.",
    "I'm organizing the reply direction so this stays consistent.",
    "I'm preparing a clean transition before I continue.",
]

UNKNOWN_TAIL_PREFIXES = ("for", "of", "about", "with", "in", "under", "through", "across", "around")

FILLER_OPENERS = [
    "Give me a moment",
    "One moment please",
    "Just a second",
    "Let me check",
    "I'm on it",
    "Thanks, one moment",
    "Got it, one moment",
    "Okay, one second",
]

FILLER_ACTIONS = [
    "to check",
    "to review",
    "to verify",
    "to organize",
    "to frame",
    "to structure",
]

FILLER_ENDINGS = [
    "the key points and continue.",
    "the context and continue.",
    "the details and continue.",
    "the core thread and continue.",
    "the flow and continue.",
]

USER_UTTERANCES = [
    "Explain recursion simply.",
    "What is the capital of Germany?",
    "Can you summarize this paragraph?",
    "Translate this sentence to Korean.",
    "Help me debug this SQL query.",
    "How does photosynthesis work?",
    "Write a short email draft for me.",
    "I feel anxious about tomorrow.",
]

SYNTH_ACTIONS = [
    "explain",
    "summarize",
    "translate",
    "compare",
    "debug",
    "calculate",
    "plan",
    "outline",
    "review",
    "clarify",
    "write",
    "solve",
]

SYNTH_TOPICS = [
    "binary search",
    "SQL joins",
    "photosynthesis",
    "quantum tunneling",
    "IPv6",
    "transformer models",
    "cache coherence",
    "loan interest",
    "thermodynamics",
    "vaccines",
    "Bayes theorem",
    "Git rebase",
    "network latency",
    "distributed locks",
    "Docker networking",
    "Kubernetes pods",
    "time complexity",
    "prompt engineering",
    "A/B testing",
    "REST authentication",
]

SYNTH_CONSTRAINTS = [
    "in simple terms",
    "for a beginner",
    "with one example",
    "step by step",
    "in under two minutes",
    "for an interview context",
    "for practical use",
    "for production systems",
    "for a school assignment",
    "for a quick review",
]

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "smollm2:360m"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

FORBIDDEN_PATTERNS = [
    re.compile(r"\bhere(?:'s| is)\b", re.IGNORECASE),
    re.compile(r"\bthe answer\b", re.IGNORECASE),
    re.compile(r"\bsql query\b", re.IGNORECASE),
    re.compile(r"\btranslate(?:d|s|)\b", re.IGNORECASE),
    re.compile(r"\bsummar(?:y|ize|ized)\b", re.IGNORECASE),
    re.compile(r"\bi'm not sure\b", re.IGNORECASE),
    re.compile(r"\bjust kidding\b", re.IGNORECASE),
    re.compile(r"\bsorry\b", re.IGNORECASE),
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
    re.compile(r"\bit is\b", re.IGNORECASE),
    re.compile(r"\bit's\b", re.IGNORECASE),
    re.compile(r"\bbecause\b", re.IGNORECASE),
    re.compile(r"\bfor example\b", re.IGNORECASE),
    re.compile(r"\bin summary\b", re.IGNORECASE),
    re.compile(r"\btherefore\b", re.IGNORECASE),
]

STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "can", "could", "would",
    "should", "please", "you", "this", "that", "into", "from", "about", "with", "just", "very",
    "really", "quickly", "help", "need", "project", "before", "beginner", "simple", "terms",
    "example", "step", "assignment", "interview", "context", "practical", "production", "school",
    "review", "under", "minutes", "quick", "for", "on", "in", "of", "or", "and", "a", "an", "the",
    "to", "it", "i", "i'm", "im", "you're", "my", "me", "we", "we're", "let", "us", "their",
    "them", "their", "his", "her", "its", "our", "be", "have", "has", "had", "do", "does", "did",
    "are", "is", "was", "were", "am", "been", "being", "can", "could", "will", "would", "should",
    "might", "may", "must", "mustn't", "if", "as", "at", "by", "up", "out", "off", "so", "we'll",
    "translate", "summarize", "summarised", "summaries", "explain", "compare", "debug", "calculate",
    "plan", "outline", "review", "clarify", "look", "show", "give", "make", "build", "check", "pull",
    "organize", "verify", "find", "show", "run", "tell", "ask", "feel", "feeling", "feels", "today",
    "tomorrow", "grateful", "thank", "thanks", "sorry", "anxious", "sad", "nervous", "helpful", "your",
    "before", "after", "while", "because", "while", "before", "proceed", "matter", "matters", "goes", "go",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero",
    "use", "need", "provide", "around", "towards", "toward", "topic", "topics", "request",
    "statement", "example", "examples", "draft", "drafting", "response",
    "quickly", "quick", "detail", "details", "first", "second", "step", "steps", "next", "nexts",
    "go", "goes", "go on", "hmm", "well", "actually", "here", "there", "here's", "there's",
    "unpack", "restate",
}

TOPIC_ACTION_VERBS = {
    "explain", "summarize", "compare", "debug", "calculate", "plan", "outline", "review", "clarify",
    "translate", "check", "analyze", "analyse", "evaluate", "inspect", "help", "helping", "provide", "give",
    "write", "solve", "draft", "fix", "analyser", "find", "define", "mean", "means", "caused", "cause", "compute", "start",
    "listen", "unpack", "restate",
}

TOPIC_SUFFIX_STOP = {
    "for", "with", "in", "under", "by", "as", "like", "on", "at", "of", "through",
    "throughout", "while", "when", "unless", "to", "step", "steps", "after", "before", "until", "if", "so", "as",
    "and", "or", "important",
}

WEAK_TOPIC_TOKENS = {
    "it", "this", "that", "these", "those", "you", "your", "i", "me", "we",
    "day", "moment", "thing", "things", "someone", "thanks", "thank", "made", "beautiful",
    "hilarious", "ugly", "ugh", "annoying", "given", "ok", "okay", "fine", "sure",
    "alright", "wow", "nice", "great", "good", "bad", "hard", "trying", "best", "listen",
    "hi", "hello", "hey", "how", "are", "doing", "today", "need", "want", "don't", "dont",
    "encouragement", "encourage", "hurt", "feels", "feeling", "feel", "appears", "little",
    "everything", "reflect", "back",
}

EMOTIONAL_KEYWORDS = {
    "anxious", "anxiety", "lonely", "overwhelmed", "sad", "stressed", "frustrated",
    "nervous", "worried", "tired", "angry", "upset", "down", "low", "blue", "confused",
    "miss", "missing", "happy", "grateful", "alone", "scared", "excited", "exciting", "rough", "bad", "hate", "love", "disappointed",
}

TOPIC_TRAIL_STOP = {
    "important", "especially", "mainly", "basically", "currently", "basically", "better",
    "clearly", "actually", "exactly", "mostly", "quickly", "briefly", "directly", "simply",
    "especially", "generally",
}

CLASS_PROFILE_SMOLLM2_10K = {
    "semantic": 0.58,
    "generic": 0.17,
    "empathy": 0.15,
    "clarify": 0.07,
    "barge_recovery": 0.03,
}

CLASS_ORDER = ("semantic", "generic", "empathy", "clarify", "barge_recovery")

GENERIC_USER_UTTERANCES = [
    "Hey, are you there?",
    "Can we keep going?",
    "What are you up to?",
    "Let's keep this moving.",
    "I have a quick follow-up.",
    "Hold on a second.",
]

EMPATHY_USER_UTTERANCES = [
    "I feel anxious about tomorrow.",
    "I'm overwhelmed right now.",
    "I feel lonely lately.",
    "I'm having a rough day.",
    "I just need someone to listen.",
]

CLARIFY_USER_UTTERANCES = [
    "Can you clarify that point first?",
    "Wait, what do you mean exactly?",
    "Please restate this more clearly.",
    "I need a clearer explanation.",
]

BARGE_USER_UTTERANCES = [
    "Hold on, let me jump in.",
    "Sorry to interrupt, I want to redirect.",
    "Wait, before you continue, pause there.",
    "Can I cut in for a second?",
]

GENERIC_FLOW_OBJECTS = [
    "next step",
    "response flow",
    "reply direction",
    "context thread",
    "transition",
    "follow-up",
]

GENERIC_FLOW_STYLES = [
    "smooth",
    "natural",
    "steady",
    "clear",
    "coherent",
    "consistent",
]

EMPATHY_TONES = [
    "calm",
    "steady",
    "gentle",
    "grounded",
    "supportive",
]

CLARIFY_FOCUS_WORDS = [
    "scope",
    "intent",
    "meaning",
    "direction",
    "point",
]


@dataclass
class CandidateResult:
    accepted: bool
    text: str
    reason: str


def make_record(user_text: str, filler_text: str) -> Dict[str, object]:
    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
        "assistant": filler_text,
        "tags": ["auto_generated", "filler_bridge"],
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _is_weak_topic(topic: str) -> bool:
    toks = _tokens(topic)
    if not toks:
        return True
    if any(tok in WEAK_TOPIC_TOKENS for tok in toks):
        return True
    if len(toks) == 1 and len(toks[0]) <= 3:
        return True
    if len(toks) == 2 and toks[0] in {"this", "that", "made", "beautiful", "hilarious", "thanks", "thank", "given"}:
        return True
    return False


def _strip_polite_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(can you|could you|would you|please|i need you to|i'd like you to|help me|for my project,?|before we proceed,?)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _trim_topic_tokens(tokens: List[str], *, max_tokens: int = 2) -> List[str]:
    if not tokens:
        return []
    while tokens and tokens[0] in {
        "to", "a", "an", "the", "this", "that", "these", "those", "it", "its", "don't", "dont", "need",
        "want", "can", "could", "would", "should", "may", "might", "i", "you", "we", "they", "them", "my",
        "his", "her", "our", "their", "for", "with", "from", "about", "if",
    }:
        tokens = tokens[1:]
    if not tokens:
        return []
    if tokens and tokens[0] in TOPIC_ACTION_VERBS:
        tokens = tokens[1:]
    if not tokens:
        return []
    for i, tok in enumerate(tokens):
        if tok in TOPIC_SUFFIX_STOP:
            tokens = tokens[:i]
            break
    if not tokens:
        return []
    trimmed = [tok for tok in tokens if tok and tok not in STOP_WORDS and not tok.isdigit()]
    if len(trimmed) <= max_tokens:
        return trimmed
    return trimmed[:max_tokens]


_ACTION_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(TOPIC_ACTION_VERBS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _load_sources_from_json(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    if isinstance(data, dict):
        for key in ("local_anchors", "cloud_anchors", "utterances"):
            val = data.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        out.append(item.strip())
        return out

    return out


def _load_sources_from_jsonl(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, str) and row.strip():
                out.append(row.strip())
            elif isinstance(row, dict):
                text = row.get("user") or row.get("text") or row.get("utterance")
                if isinstance(text, str) and text.strip():
                    out.append(text.strip())
    return out


def load_source_utterances(source_file: str | None, include_anchors: bool) -> List[str]:
    out: List[str] = list(USER_UTTERANCES)

    if include_anchors:
        anchors_path = Path("dataset/anchors.json")
        if anchors_path.exists():
            out.extend(_load_sources_from_json(anchors_path))

    if source_file:
        path = Path(source_file)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        if path.suffix == ".txt":
            with path.open("r", encoding="utf-8") as f:
                out.extend([ln.strip() for ln in f if ln.strip()])
        elif path.suffix == ".json":
            out.extend(_load_sources_from_json(path))
        elif path.suffix == ".jsonl":
            out.extend(_load_sources_from_jsonl(path))
        else:
            raise ValueError("source file must be .txt, .json, or .jsonl")

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for text in out:
        key = _normalize_text(text).lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(text.strip())
    return uniq


def build_synthetic_utterances(count: int, seed: int) -> List[str]:
    """Build diverse synthetic user utterances for filler training scale-up."""
    rng = random.Random(seed)
    templates = [
        "Can you {action} {topic} {constraint}?",
        "Please {action} {topic} {constraint}.",
        "I need you to {action} {topic} {constraint}.",
        "Could you quickly {action} {topic} {constraint}?",
        "Help me {action} {topic} {constraint}.",
        "For my project, {action} {topic} {constraint}.",
        "Before we proceed, {action} {topic} {constraint}.",
    ]
    out: List[str] = []
    for _ in range(max(0, count)):
        t = rng.choice(templates)
        out.append(
            t.format(
                action=rng.choice(SYNTH_ACTIONS),
                topic=rng.choice(SYNTH_TOPICS),
                constraint=rng.choice(SYNTH_CONSTRAINTS),
            ).strip()
        )
    return out


def build_generic_utterances(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    qualifiers = [
        "for a moment",
        "from here",
        "without changing topic",
        "in this flow",
        "before we switch",
        "right now",
        "for this part",
        "while we continue",
    ]
    templates = [
        "Can we keep going {q}?",
        "I have a quick follow-up {q}.",
        "Hold on, let's keep this moving {q}.",
        "Give me a second, I want to continue {q}.",
        "Let's continue naturally {q}.",
    ]
    out: List[str] = []
    for _ in range(max(0, count)):
        out.append(rng.choice(templates).format(q=rng.choice(qualifiers)).strip())
    return out


def build_empathy_utterances(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    emotions = [
        "anxious", "overwhelmed", "stressed", "tired", "sad", "frustrated",
        "nervous", "confused", "lonely", "upset", "worried", "drained",
    ]
    contexts = [
        "work", "my project", "tomorrow", "today", "this week", "my interview",
        "the situation", "everything lately", "my family", "my task",
    ]
    templates = [
        "I'm feeling {e} about {c}.",
        "I feel {e} right now.",
        "{c} is making me feel {e}.",
        "I'm {e} and need a calm response.",
        "Honestly, I feel {e} today.",
    ]
    out: List[str] = []
    for _ in range(max(0, count)):
        out.append(rng.choice(templates).format(e=rng.choice(emotions), c=rng.choice(contexts)).strip())
    return out


def build_clarify_utterances(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    topics = list(SYNTH_TOPICS) + [
        "this point",
        "your last sentence",
        "the scope",
        "the conclusion",
        "the reasoning",
        "the approach",
        "the previous step",
        "the intent",
        "the key assumption",
        "the tradeoff",
        "the exact boundary",
        "the core definition",
        "the final goal",
    ]
    qualifiers = [
        "before we continue",
        "in plain terms",
        "for this context",
        "for this step",
        "without jargon",
        "for this case",
        "right now",
        "before the next part",
    ]
    lead_ins = [
        "Can you",
        "Could you",
        "Please",
        "I need you to",
    ]
    templates = [
        "{lead} clarify {t} {q}?",
        "Wait, what do you mean by {t} {q}?",
        "{lead} restate {t} more clearly {q}.",
        "I need a clearer explanation of {t} {q}.",
        "{lead} unpack {t} step by step {q}.",
    ]
    out: List[str] = []
    for _ in range(max(0, count)):
        out.append(
            rng.choice(templates).format(
                t=rng.choice(topics),
                q=rng.choice(qualifiers),
                lead=rng.choice(lead_ins),
            ).strip()
        )
    return out


def build_barge_utterances(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    topics = list(SYNTH_TOPICS) + [
        "this direction",
        "that point",
        "the latest point",
        "your last line",
        "the main thread",
        "the original question",
        "the previous detail",
    ]
    pivots = [
        "for a second",
        "for this part",
        "before the next step",
        "right now",
        "before we continue",
        "for this direction",
    ]
    templates = [
        "Hold on {p}, let me jump in.",
        "Sorry to interrupt {p}, I want to redirect this.",
        "Wait {p}, before you continue, switch to {t}.",
        "Can I cut in {p} and move to {t}?",
        "Pause there {p}, let's pivot to {t}.",
        "Stop {p}, I need to jump in.",
    ]
    out: List[str] = []
    for _ in range(max(0, count)):
        out.append(rng.choice(templates).format(t=rng.choice(topics), p=rng.choice(pivots)).strip())
    return out


def _dedupe_texts(texts: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for text in texts:
        key = _normalize_text(text).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(text.strip())
    return out


def _allocate_counts(total: int, weights: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    if not weights:
        return []
    w_sum = sum(max(0.0, w) for w in weights)
    if w_sum <= 0:
        base = [0 for _ in weights]
        base[0] = total
        return base
    scaled = [(max(0.0, w) / w_sum) * total for w in weights]
    floors = [int(v) for v in scaled]
    rem = total - sum(floors)
    fracs = sorted(((scaled[i] - floors[i], i) for i in range(len(weights))), reverse=True)
    for _, idx in fracs[:rem]:
        floors[idx] += 1
    return floors


def _class_from_tags(tags: Sequence[object]) -> str:
    for t in tags:
        if isinstance(t, str) and t.startswith("class:"):
            return t.split(":", 1)[1]
    return "unlabeled"


def _p95(values: Sequence[int]) -> int:
    if not values:
        return 0
    arr = sorted(values)
    idx = int(0.95 * (len(arr) - 1))
    return arr[idx]


def _extract_topic_hint(user_text: str) -> str:
    base_text = _strip_polite_prefix(_normalize_text(user_text).split(":")[0])
    low_text = base_text.lower()

    for topic in sorted(SYNTH_TOPICS, key=len, reverse=True):
        if topic.lower() in low_text:
            return topic

    if re.match(r"^\s*convert\s+\d", low_text):
        return "unit conversion"
    if re.match(r"^\s*convert\s", low_text):
        return "unit conversion"

    question_match = re.match(
        r"^(?:could you|can you|can|could|please|help me|i need you to)?\s*(?:what|which|how|why|where|when|who|whom)\s+(?:is|are|was|were|do|does|did|can|could|would|should)?\s*(.+)$",
        low_text,
    )
    if question_match:
        q_tokens = _trim_topic_tokens(_tokens(question_match.group(1)), max_tokens=3)
        if q_tokens and not _is_weak_topic(" ".join(q_tokens)):
            return " ".join(q_tokens)

    action_match = _ACTION_PATTERN.search(low_text)
    if action_match:
        tail = low_text[action_match.end():].strip()
        body_tokens = _trim_topic_tokens(_tokens(tail))
        if body_tokens and body_tokens[0] in {"sentence", "sentences"}:
            return "that sentence"
        if body_tokens and len(body_tokens) > 1 and body_tokens[1] in TOPIC_TRAIL_STOP:
            return body_tokens[0]
        if body_tokens == ["sentence"]:
            return "that sentence"
        if body_tokens and _is_weak_topic(" ".join(body_tokens)):
            return "your request"
        if body_tokens:
            return " ".join(body_tokens)

    for prefix in UNKNOWN_TAIL_PREFIXES:
        m = re.search(rf"\b{re.escape(prefix)}\s+(.+)$", low_text)
        if m:
            tail_tokens = _trim_topic_tokens(_tokens(m.group(1)), max_tokens=2)
            if tail_tokens:
                return " ".join(tail_tokens)

    words = _trim_topic_tokens(_tokens(low_text), max_tokens=4)
    if not words:
        return "your request"
    if any(tok in EMOTIONAL_KEYWORDS for tok in words):
        return "your feelings"

    if len(words) >= 3 and words[0] in {"difference", "meaning", "meanings"} and words[1] == "between":
        if len(words) >= 3:
            return words[2]


    if words and words[0] in ("what", "why", "how", "which", "where", "when", "who", "whom", "that"):
        words = words[1:]
    if words and words[0] in TOPIC_ACTION_VERBS and len(words) > 1:
        words = words[1:]
    if words and words[0] == "that":
        words = words[1:]
    words = _trim_topic_tokens(words)
    if not words:
        return "your request"

    if len(words) > 1:
        # Avoid attaching trailing conjunction fragments like "x and y" as one odd topic.
        for stop_word in ("and", "or", "but", "while", "because", "if", "then"):
            if stop_word in words:
                idx = words.index(stop_word)
                words = words[:idx] if idx > 0 else words[:1]
                break

    if not words:
        return "your request"
    if len(words) == 1:
        if _is_weak_topic(words[0]):
            return "your request"
        return words[0]
    result = f"{words[0]} {words[1]}"
    if _is_weak_topic(result):
        return "your request"
    return result


def _topic_display(topic: str) -> str:
    topic = _normalize_text(topic)
    if not topic:
        return "your request"
    low_topic = topic.lower()
    if low_topic in {"that", "this", "it", "they", "them", "your request", "request", "request topic"}:
        return "your request"
    if low_topic == "your feelings":
        return "your feelings"
    if low_topic == "that sentence":
        return "that sentence"
    if low_topic in ("your request", "this request", "that request"):
        return "your request"
    if low_topic == "your request":
        return "your request"
    parts = [part for part in low_topic.split() if part and part not in STOP_WORDS and not part.isdigit()]
    if not parts:
        return "your request"
    if _is_weak_topic(" ".join(parts)):
        return "your request"
    if len(parts) == 1:
        if len(parts[0]) >= 3:
            return parts[0]
        return "your request"
    topic_text = f"{parts[0]} {parts[1]}"
    if _is_weak_topic(topic_text):
        return "your request"
    return topic_text


BRIDGE_TEMPLATES = [
    "Got it, I'm building the main idea around {topic}, then continuing in order.",
    "I'll frame {topic} cleanly, then move into the details smoothly.",
    "I'll anchor the {topic} context, then expand with the next detail.",
    "I'm organizing a short {topic} thread so the follow-up is natural.",
    "I'll align the core of {topic}, then continue with the next part.",
    "I'm pulling together {topic} so this explanation stays coherent.",
]

SEMANTIC_BRIDGE_TEMPLATES = [
    "I'll set a clear outline for {topic}, then move into the detailed explanation.",
    "I'm grounding {topic} first, so the next details land clearly.",
    "I'll take {topic} through a quick structure, then continue in detail.",
    "I'll bridge from the overview of {topic} into the specific details.",
    "I'll map the {topic} flow first, then step through the details.",
]


GENERIC_BRIDGE_TEMPLATES = [
    "I'll take a second to structure this so it flows cleanly.",
    "Give me one moment, I'll keep the response chain smooth.",
    "I'm just setting up the next clear step for continuity.",
    "I'm briefly organizing the flow so this stays natural.",
    "Thanks, I'm preparing the next sentence to sound smoother.",
]


CLARIFY_BRIDGE_TEMPLATES = [
    "Let me clarify {topic} first, then continue with cleaner detail.",
    "I'll pin down {topic} first, then continue step by step.",
    "One moment, I'll disambiguate {topic} and continue clearly.",
    "I'll resolve the ambiguity around {topic}, then continue smoothly.",
]

BARGE_RECOVERY_TEMPLATES = [
    "I caught the interruption, I'll restart from {topic} and continue smoothly.",
    "Let me switch to {topic}, then continue from there naturally.",
    "I'll pause and realign to {topic}, then continue in sequence.",
    "I heard you cut in, I'll continue from {topic} now.",
]


def _fallback_topic_for_mode(topic: str, mode: str) -> str:
    if mode == "clarify":
        return random.choice(CLARIFY_FOCUS_WORDS)
    if mode == "barge_recovery":
        return "your latest point"
    if mode == "semantic":
        return "the main point"
    return "your request"


def _generate_generic_bridge() -> str:
    obj = random.choice(GENERIC_FLOW_OBJECTS)
    style = random.choice(GENERIC_FLOW_STYLES)
    templates = [
        "Give me a second, I'll organize the {obj} so this stays {style}.",
        "One moment, I'll align the {obj}, then continue in a {style} way.",
        "I'm taking a short pause to shape the {obj} before continuing.",
        "I'll briefly set up the {obj} so the next line feels {style}.",
        "Let me prepare the {obj}, then I'll continue in a {style} way.",
    ]
    return _normalize_text(random.choice(templates).format(obj=obj, style=style))


def _generate_empathy_bridge() -> str:
    tone = random.choice(EMPATHY_TONES)
    templates = [
        "I hear you, I'll keep this {tone} and continue gently.",
        "I got you, I'll respond with a {tone} tone and keep it steady.",
        "I'll take a breath, keep this {tone}, and continue naturally.",
        "Thanks for sharing that, I'll continue in a {tone} tone.",
        "I'll hold this carefully and continue in a {tone} flow.",
    ]
    return _normalize_text(random.choice(templates).format(tone=tone))


def _generate_clarify_bridge(topic: str) -> str:
    low_topic = topic.lower().strip()
    if low_topic in {"your request", "your feelings", "previous", "unpack", "restate", "clarify"} or _is_weak_topic(low_topic):
        safe_topic = _fallback_topic_for_mode(topic, "clarify")
    else:
        safe_topic = topic
    return _normalize_text(random.choice(CLARIFY_BRIDGE_TEMPLATES).format(topic=safe_topic))


def _generate_barge_recovery_bridge(topic: str) -> str:
    safe_topic = topic if topic not in {"your request", "your feelings"} else _fallback_topic_for_mode(topic, "barge_recovery")
    return _normalize_text(random.choice(BARGE_RECOVERY_TEMPLATES).format(topic=safe_topic))


def generate_template_candidate(user_text: str, topic_aware: bool, forced_mode: str | None = None) -> str:
    if not topic_aware:
        return random.choice(SAFE_FILLERS)

    topic = _topic_display(_extract_topic_hint(user_text))
    mode = forced_mode

    if mode == "generic":
        return _generate_generic_bridge()
    if mode == "empathy":
        return _generate_empathy_bridge()
    if mode == "clarify":
        return _generate_clarify_bridge(topic)
    if mode == "barge_recovery":
        return _generate_barge_recovery_bridge(topic)
    if mode == "semantic":
        if topic in {"your request", "your feelings"}:
            topic = _fallback_topic_for_mode(topic, "semantic")
        r = random.random()
        if r < 0.65:
            return _normalize_text(random.choice(BRIDGE_TEMPLATES).format(topic=topic))
        return _normalize_text(random.choice(SEMANTIC_BRIDGE_TEMPLATES).format(topic=topic))

    if topic == "your feelings":
        return _normalize_text(random.choice(EMPATHY_BRIDGE_TEMPLATES))
    if topic == "your request":
        return _normalize_text(random.choice(UNKNOWN_BRIDGE_TEMPLATES))
    r = random.random()
    if r < 0.6:
        return _normalize_text(random.choice(BRIDGE_TEMPLATES).format(topic=topic))
    if r < 0.8:
        return _normalize_text(random.choice(SEMANTIC_BRIDGE_TEMPLATES).format(topic=topic))
    return _normalize_text(random.choice(GENERIC_BRIDGE_TEMPLATES))


def _call_ollama(prompt: str, system: str, model: str, url: str, timeout: float) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 16,
            "num_ctx": 128,
        },
    }
    try:
        res = requests.post(url, json=payload, timeout=timeout)
    except Exception:
        return ""
    if res.status_code != 200:
        return ""
    try:
        data = res.json()
    except Exception:
        return ""
    out = data.get("response")
    return _normalize_text(out if isinstance(out, str) else "")


def _call_openai(prompt: str, system: str, timeout: float, model: str) -> str:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return ""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 20,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception:
        return ""
    if res.status_code != 200:
        return ""
    try:
        data = res.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        return _normalize_text(content)
    except Exception:
        return ""


def _call_gemini(prompt: str, system: str, timeout: float, model: str) -> str:
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("CLOUD_LLM_API_KEY") or "").strip()
    if not api_key:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 20,
        },
    }
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
    except Exception:
        return ""
    if res.status_code != 200:
        return ""
    try:
        data = res.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        if not parts:
            return ""
        text = parts[0].get("text") or ""
        return _normalize_text(text)
    except Exception:
        return ""


def _call_cloud(prompt: str, system: str, timeout: float, model: str | None) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        m = model or DEFAULT_OPENAI_MODEL
        return _call_openai(prompt, system, timeout=timeout, model=m)
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("CLOUD_LLM_API_KEY"):
        m = model or DEFAULT_GEMINI_MODEL
        return _call_gemini(prompt, system, timeout=timeout, model=m)
    return ""


def _teacher_prompt(user_text: str) -> str:
    return (
        "User said: "
        + repr(user_text)
        + "\nReturn ONLY one short bridge sentence (6-14 words). "
        + "Do not answer. No facts. No names. No numbers."
    )


def generate_teacher_candidate(
    backend: str,
    user_text: str,
    teacher_model: str | None,
    ollama_url: str,
    timeout: float,
) -> str:
    prompt = _teacher_prompt(user_text)
    system = (
        "You are a filler generator for realtime voice latency masking. "
        "Output exactly one short spoken sentence. Do not answer the question."
    )
    if backend == "ollama":
        return _call_ollama(
            prompt=prompt,
            system=system,
            model=teacher_model or DEFAULT_OLLAMA_MODEL,
            url=ollama_url,
            timeout=timeout,
        )
    if backend == "cloud":
        return _call_cloud(prompt=prompt, system=system, timeout=timeout, model=teacher_model)
    return ""


def validate_candidate(text: str, min_words: int, max_words: int, strict_filter: bool) -> CandidateResult:
    t = _normalize_text(text)
    if not t:
        return CandidateResult(False, t, "empty")
    if "\n" in t:
        return CandidateResult(False, t, "multiline")
    if "?" in t:
        return CandidateResult(False, t, "contains_question_mark")

    words = t.split()
    if len(words) < min_words:
        return CandidateResult(False, t, "too_few_words")
    if len(words) > max_words:
        return CandidateResult(False, t, "too_many_words")

    sentence_breaks = re.findall(r"[.!?]+", t)
    if len(sentence_breaks) > 1:
        return CandidateResult(False, t, "multi_sentence")

    if strict_filter:
        if re.search(r"\d", t):
            return CandidateResult(False, t, "contains_number")
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(t):
                return CandidateResult(False, t, f"forbidden_pattern:{pat.pattern}")
        if re.search(r"\b(that now|it carefully now|the key points now)\b", t, re.IGNORECASE):
            return CandidateResult(False, t, "awkward_phrase_pattern")
        # heuristic: avoid obvious entities in very short fillers
        caps = re.findall(r"\b[A-Z][a-z]{2,}\b", t)
        if len(caps) >= 2:
            return CandidateResult(False, t, "possible_named_entity")

    return CandidateResult(True, t, "ok")


def _dedupe_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for r in records:
        key = (str(r["user"]).strip().lower(), str(r["assistant"]).strip().lower())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _cap_phrase_frequency(records: List[Dict[str, object]], max_per_phrase: int) -> List[Dict[str, object]]:
    if max_per_phrase <= 0:
        return records
    counts: Dict[str, int] = {}
    out: List[Dict[str, object]] = []
    for r in records:
        key = str(r["assistant"]).strip().lower()
        c = counts.get(key, 0)
        if c >= max_per_phrase:
            continue
        counts[key] = c + 1
        out.append(r)
    return out


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_profile_source_pools(base_sources: Sequence[str], total_target: int, synthetic_sources: int, seed: int) -> Dict[str, List[str]]:
    semantic_count = max(synthetic_sources, total_target * 2)
    generic_count = max(3000, int(total_target * 1.2))
    empathy_count = max(2500, int(total_target * 1.0))
    clarify_count = max(2200, int(total_target * 0.9))
    barge_count = max(1500, int(total_target * 0.7))

    pools = {
        "semantic": _dedupe_texts(list(base_sources) + build_synthetic_utterances(semantic_count, seed=seed + 11)),
        "generic": _dedupe_texts(list(GENERIC_USER_UTTERANCES) + build_generic_utterances(generic_count, seed=seed + 21)),
        "empathy": _dedupe_texts(list(EMPATHY_USER_UTTERANCES) + build_empathy_utterances(empathy_count, seed=seed + 31)),
        "clarify": _dedupe_texts(list(CLARIFY_USER_UTTERANCES) + build_clarify_utterances(clarify_count, seed=seed + 41)),
        "barge_recovery": _dedupe_texts(list(BARGE_USER_UTTERANCES) + build_barge_utterances(barge_count, seed=seed + 51)),
    }
    return pools


def _summarize_dataset(
    train_rows: Sequence[Dict[str, object]],
    val_rows: Sequence[Dict[str, object]],
    test_rows: Sequence[Dict[str, object]],
    rejects: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    all_rows = list(train_rows) + list(val_rows) + list(test_rows)
    class_counter = Counter()
    split_class_counter: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        for row in rows:
            cls = _class_from_tags(row.get("tags", [])) if isinstance(row, dict) else "unlabeled"
            class_counter[cls] += 1
            split_class_counter[split_name][cls] += 1

    words = [len(str(row.get("assistant", "")).split()) for row in all_rows]
    assistant_texts = [str(row.get("assistant", "")).strip().lower() for row in all_rows]
    pair_keys = {(str(row.get("user", "")).strip().lower(), str(row.get("assistant", "")).strip().lower()) for row in all_rows}

    reject_reason_counts = Counter()
    for r in rejects:
        if isinstance(r, dict):
            reject_reason_counts[str(r.get("reason", "unknown"))] += 1

    return {
        "total_rows": len(all_rows),
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "class_distribution": dict(class_counter),
        "class_distribution_by_split": {
            "train": dict(split_class_counter["train"]),
            "val": dict(split_class_counter["val"]),
            "test": dict(split_class_counter["test"]),
        },
        "assistant_word_stats": {
            "avg": round(sum(words) / max(1, len(words)), 3),
            "min": min(words) if words else 0,
            "max": max(words) if words else 0,
            "p95": _p95(words),
        },
        "unique_assistant_ratio": round(len(set(assistant_texts)) / max(1, len(assistant_texts)), 4),
        "unique_pair_ratio": round(len(pair_keys) / max(1, len(all_rows)), 4),
        "top_assistant_phrases": [
            {"text": text, "count": cnt}
            for text, cnt in Counter(assistant_texts).most_common(15)
        ],
        "reject_count": len(rejects),
        "top_reject_reasons": dict(reject_reason_counts.most_common(10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filler SFT dataset with template/teacher + strict filters")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-file", type=str, default=None, help="Optional .txt/.json/.jsonl source utterances")
    parser.add_argument("--include-anchors", action="store_true", default=True, help="Include dataset/anchors.json utterances")
    parser.add_argument("--synthetic-sources", type=int, default=1200, help="Number of synthetic source utterances to add")
    parser.add_argument("--teacher-backend", choices=["template", "ollama", "cloud"], default="template")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model name (backend dependent)")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--n-candidates", type=int, default=3, help="Candidates per user utterance")
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=16)
    parser.add_argument("--max-per-phrase", type=int, default=6, help="Cap repeated identical assistant phrases")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--strict-filter", action="store_true", default=True)
    parser.add_argument("--topic-aware-templates", action="store_true", default=True)
    parser.add_argument("--profile", choices=["default", "smollm2_10k"], default="default")
    parser.add_argument("--out-train", default="training/data/filler_train.jsonl")
    parser.add_argument("--out-val", default="training/data/filler_val.jsonl")
    parser.add_argument("--out-test", default="training/data/filler_test.jsonl")
    parser.add_argument("--out-rejects", default="training/data/filler_rejects.jsonl")
    parser.add_argument("--out-report", default="training/data/filler_report.json")
    args = parser.parse_args()

    random.seed(args.seed)
    total_target = args.train_size + args.val_size + args.test_size

    sources = load_source_utterances(args.source_file, include_anchors=args.include_anchors)
    if args.synthetic_sources > 0:
        sources.extend(build_synthetic_utterances(args.synthetic_sources, seed=args.seed))
        # Re-dedupe after synthetic extension
        seen = set()
        deduped: List[str] = []
        for s in sources:
            k = _normalize_text(s).lower()
            if k and k not in seen:
                seen.add(k)
                deduped.append(s)
        sources = deduped
    if not sources:
        raise ValueError("no source utterances available")

    rejects: List[Dict[str, object]] = []
    train_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    candidate_count = 0

    if args.profile == "smollm2_10k":
        if args.teacher_backend != "template":
            raise ValueError("profile=smollm2_10k currently supports --teacher-backend template only")

        split_plan = [("train", args.train_size), ("val", args.val_size), ("test", args.test_size)]
        split_weights = [float(sz) for _, sz in split_plan]
        class_weights = [CLASS_PROFILE_SMOLLM2_10K[c] for c in CLASS_ORDER]
        class_totals = _allocate_counts(total_target, class_weights)
        class_targets = {cls: class_totals[i] for i, cls in enumerate(CLASS_ORDER)}

        split_class_targets: Dict[str, Dict[str, int]] = {name: {} for name, _ in split_plan}
        for cls in CLASS_ORDER:
            alloc = _allocate_counts(class_targets[cls], split_weights)
            for idx, (split_name, _) in enumerate(split_plan):
                split_class_targets[split_name][cls] = alloc[idx]

        pools = _build_profile_source_pools(
            base_sources=sources,
            total_target=total_target,
            synthetic_sources=args.synthetic_sources,
            seed=args.seed,
        )

        split_rows: Dict[str, List[Dict[str, object]]] = {name: [] for name, _ in split_plan}
        seen_pairs = set()
        assistant_phrase_counts: Dict[str, int] = {}
        effective_max_per_phrase = 0 if args.max_per_phrase == 6 else args.max_per_phrase

        for split_name, _ in split_plan:
            for cls in CLASS_ORDER:
                target = split_class_targets[split_name].get(cls, 0)
                if target <= 0:
                    continue
                attempts = 0
                max_attempts = max(target * 160, 5000)
                filled = 0
                pool = pools.get(cls) or sources
                while filled < target and attempts < max_attempts:
                    attempts += 1
                    user_text = random.choice(pool)
                    candidate = generate_template_candidate(user_text, topic_aware=True, forced_mode=cls)
                    v = validate_candidate(
                        candidate,
                        min_words=args.min_words,
                        max_words=args.max_words,
                        strict_filter=args.strict_filter,
                    )
                    if not v.accepted:
                        rejects.append({"user": user_text, "candidate": candidate, "reason": v.reason, "class": cls})
                        continue

                    pair_key = (user_text.strip().lower(), v.text.strip().lower())
                    if pair_key in seen_pairs:
                        rejects.append({"user": user_text, "candidate": v.text, "reason": "duplicate_pair", "class": cls})
                        continue

                    phrase_key = v.text.strip().lower()
                    if effective_max_per_phrase > 0 and assistant_phrase_counts.get(phrase_key, 0) >= effective_max_per_phrase:
                        rejects.append({"user": user_text, "candidate": v.text, "reason": "phrase_cap", "class": cls})
                        continue

                    row = make_record(user_text, v.text)
                    tags = list(row.get("tags", []))
                    tags.append(f"backend:{args.teacher_backend}")
                    tags.append(f"class:{cls}")
                    row["tags"] = tags
                    split_rows[split_name].append(row)
                    seen_pairs.add(pair_key)
                    assistant_phrase_counts[phrase_key] = assistant_phrase_counts.get(phrase_key, 0) + 1
                    filled += 1

                if filled < target:
                    raise RuntimeError(
                        f"profile generation failed for split={split_name}, class={cls}: "
                        f"have={filled}, need={target}. Try relaxing strict filters or increasing source diversity."
                    )

        train_rows = split_rows["train"]
        val_rows = split_rows["val"]
        test_rows = split_rows["test"]
        random.shuffle(train_rows)
        random.shuffle(val_rows)
        random.shuffle(test_rows)
        candidate_count = len(train_rows) + len(val_rows) + len(test_rows)
    else:
        candidates: List[Dict[str, object]] = []
        # Build candidate pool
        for user_text in sources:
            for _ in range(args.n_candidates):
                if args.teacher_backend == "template":
                    c = generate_template_candidate(user_text, topic_aware=args.topic_aware_templates)
                else:
                    c = generate_teacher_candidate(
                        backend=args.teacher_backend,
                        user_text=user_text,
                        teacher_model=args.teacher_model,
                        ollama_url=args.ollama_url,
                        timeout=args.timeout,
                    )
                    if not c:
                        c = generate_template_candidate(user_text, topic_aware=False)

                v = validate_candidate(
                    c,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    strict_filter=args.strict_filter,
                )
                if v.accepted:
                    r = make_record(user_text, v.text)
                    tags = list(r.get("tags", []))
                    tags.append(f"backend:{args.teacher_backend}")
                    r["tags"] = tags
                    candidates.append(r)
                else:
                    rejects.append({"user": user_text, "candidate": c, "reason": v.reason})

        candidates = _dedupe_records(candidates)
        candidates = _cap_phrase_frequency(candidates, max_per_phrase=args.max_per_phrase)
        random.shuffle(candidates)

        # Top up with safe templates if pool is too small
        max_topup_attempts = max(total_target * 50, 1000)
        attempts = 0
        while len(candidates) < total_target and attempts < max_topup_attempts:
            attempts += 1
            u = random.choice(sources)
            c = generate_template_candidate(u, topic_aware=False)
            v = validate_candidate(
                c,
                min_words=args.min_words,
                max_words=args.max_words,
                strict_filter=args.strict_filter,
            )
            if not v.accepted:
                continue
            r = make_record(u, v.text)
            tags = list(r.get("tags", []))
            tags.append("backend:template_fallback")
            r["tags"] = tags
            candidates.append(r)
            candidates = _dedupe_records(candidates)
            candidates = _cap_phrase_frequency(candidates, max_per_phrase=args.max_per_phrase)

        if len(candidates) < total_target:
            raise RuntimeError(
                f"unable to reach requested dataset size: have={len(candidates)}, need={total_target}. "
                "Try reducing train/val/test size, adding more sources, or relaxing strict filters."
            )

        train_rows = candidates[: args.train_size]
        val_rows = candidates[args.train_size : args.train_size + args.val_size]
        test_rows = candidates[args.train_size + args.val_size : args.train_size + args.val_size + args.test_size]
        candidate_count = len(candidates)

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_test = Path(args.out_test)
    out_rejects = Path(args.out_rejects)
    out_report = Path(args.out_report)

    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_val, val_rows)
    if args.test_size > 0:
        _write_jsonl(out_test, test_rows)
    _write_jsonl(out_rejects, rejects)

    report = _summarize_dataset(train_rows=train_rows, val_rows=val_rows, test_rows=test_rows, rejects=rejects)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print(
        json.dumps(
            {
                "status": "ok",
                "profile": args.profile,
                "teacher_backend": args.teacher_backend,
                "source_count": len(sources),
                "candidate_count": candidate_count,
                "reject_count": len(rejects),
                "train_file": str(out_train),
                "val_file": str(out_val),
                "test_file": str(out_test) if args.test_size > 0 else "",
                "reject_file": str(out_rejects),
                "report_file": str(out_report),
                "train_size": len(train_rows),
                "val_size": len(val_rows),
                "test_size": len(test_rows),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
