#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests


SYSTEM_PROMPT = (
    "You are a realtime voice filler generator. Output exactly one short bridge sentence "
    "(6-14 words, max 16 tokens). Do not answer the user. Do not add facts."
)

SAFE_FILLERS = [
    "Give me a moment while I check that for you.",
    "One second, I am pulling that together now.",
    "Let me take a moment to review that clearly.",
    "Okay, give me a second to verify the details.",
    "Got it, I will check that and get right back.",
    "One moment, I am organizing the key points now.",
    "Let me quickly review that before I respond.",
    "I am on it, give me a brief second.",
    "Give me a second while I validate that.",
    "One moment while I gather that for you.",
]

FILLER_OPENERS = [
    "Give me a moment",
    "One moment please",
    "Just a second",
    "Let me check",
    "I am on it",
    "Thanks, one moment",
    "Got it, one moment",
    "Okay, one second",
]

FILLER_ACTIONS = [
    "while I check",
    "while I review",
    "while I look into",
    "while I verify",
    "while I organize",
    "while I gather",
]

FILLER_ENDINGS = [
    "this for you.",
    "that now.",
    "the details for you.",
    "it carefully.",
    "the key points clearly.",
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


def _extract_topic_hint(user_text: str) -> str:
    low_text = user_text.lower()
    for topic in sorted(SYNTH_TOPICS, key=len, reverse=True):
        if topic in low_text:
            return topic

    words = re.findall(r"[A-Za-z']+", user_text)
    if not words:
        return "that"
    stop = {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "how",
        "please",
        "could",
        "would",
        "should",
        "can",
        "you",
        "this",
        "that",
        "into",
        "from",
        "about",
        "with",
        "just",
        "very",
        "really",
        "please",
        "quickly",
        "project",
        "before",
        "need",
        "beginner",
        "simple",
        "terms",
        "example",
        "step",
        "assignment",
        "interview",
        "context",
        "practical",
        "production",
        "school",
        "review",
        "under",
        "minutes",
        "quick",
        "help",
        "proceed",
        "explain",
        "summarize",
        "translate",
        "compare",
        "debug",
        "calculate",
        "plan",
        "outline",
        "clarify",
    }
    keywords = [w.lower() for w in words if len(w) >= 4 and w.lower() not in stop]
    if not keywords:
        return "that"
    if len(keywords) >= 2:
        return f"{keywords[0]} {keywords[1]}"
    return keywords[0]


def generate_template_candidate(user_text: str, topic_aware: bool) -> str:
    if not topic_aware:
        return random.choice(SAFE_FILLERS)

    topic = _extract_topic_hint(user_text)

    # Blend curated templates with compositional patterns for diversity.
    curated = [
        f"One moment while I check the details on {topic}.",
        f"Give me a second to review {topic} clearly.",
        f"I am checking {topic} and will be right back.",
        "One moment, I am checking that now.",
        "Give me a moment while I check that for you.",
        "Let me quickly check that for you.",
    ]
    if random.random() < 0.4:
        return random.choice(curated)

    opener = random.choice(FILLER_OPENERS)
    action = random.choice(FILLER_ACTIONS)
    ending = random.choice(FILLER_ENDINGS)
    if random.random() < 0.7 and topic != "that":
        candidate = f"{opener} {action} {topic} {ending}"
    else:
        candidate = f"{opener} {action} {ending}"
    return _normalize_text(candidate)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filler SFT dataset with template/teacher + strict filters")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=60)
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
    parser.add_argument("--out-train", default="training/data/filler_train.jsonl")
    parser.add_argument("--out-val", default="training/data/filler_val.jsonl")
    parser.add_argument("--out-rejects", default="training/data/filler_rejects.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    total_target = args.train_size + args.val_size

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

    candidates: List[Dict[str, object]] = []
    rejects: List[Dict[str, object]] = []

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
            "Try reducing train/val size, adding more sources, or relaxing strict filters."
        )

    train_rows = candidates[: args.train_size]
    val_rows = candidates[args.train_size : args.train_size + args.val_size]

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_rejects = Path(args.out_rejects)

    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_val, val_rows)
    _write_jsonl(out_rejects, rejects)

    print(
        json.dumps(
            {
                "status": "ok",
                "teacher_backend": args.teacher_backend,
                "source_count": len(sources),
                "candidate_count": len(candidates),
                "reject_count": len(rejects),
                "train_file": str(out_train),
                "val_file": str(out_val),
                "reject_file": str(out_rejects),
                "train_size": len(train_rows),
                "val_size": len(val_rows),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
