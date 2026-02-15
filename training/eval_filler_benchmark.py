#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You are a realtime voice filler generator. "
    "Output exactly one short bridge sentence (6-14 words, max 16 tokens). "
    "Do not answer the user. Do not add facts."
)

BASE_EVAL_PROMPTS = [
    "What is the difference between SRAM and DRAM?",
    "Can you summarize this article in plain language?",
    "Translate this sentence to Japanese.",
    "How does DNS caching work?",
    "Write a SQL query to find duplicates.",
    "I feel nervous before tomorrow's interview.",
    "How many continents are there?",
    "Explain quantum tunneling for a beginner.",
    "Can you plan a two day trip to Busan?",
    "What causes inflation in economics?",
    "Debug this Python traceback for me.",
    "Compare REST and GraphQL quickly.",
    "Help me outline a machine learning project.",
    "What is Bayes theorem in simple terms?",
    "How can I improve API latency?",
    "Can you review this resume summary?",
    "Is Kubernetes hard to learn?",
    "Explain cache invalidation strategies.",
    "How do vaccines train the immune system?",
    "Give me interview tips for backend engineering.",
]

ANSWER_RISK_PATTERNS = [
    r"\bthe answer is\b",
    r"\bbecause\b",
    r"\bfor example\b",
    r"\bmeans\b",
    r"\bfirst\b",
    r"\bsecond\b",
]

STOP = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "can", "could", "would", "please",
    "you", "this", "that", "into", "from", "about", "with", "just", "very", "really", "quickly", "help",
}


@dataclass
class Row:
    user: str
    output: str
    words: int
    pass_rules: bool
    answer_risk: bool
    multi_sentence: bool
    has_question_mark: bool
    topic_copy: bool
    latency_ms: float


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.split("\n")[0].strip()
    return text


def _extract_topic(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    for w in words:
        if len(w) >= 4 and w not in STOP:
            return w
    return ""


def _load_model(base_model_id: str, adapter_dir: str | None, device: str, local_files_only: bool):
    tok_src = adapter_dir if adapter_dir and (Path(adapter_dir) / "tokenizer.json").exists() else base_model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, local_files_only=local_files_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=local_files_only)
    else:
        model = base
    model = model.to(device)
    model.eval()
    return tokenizer, model


def _generate_one(model, tokenizer, user_text: str, max_new_tokens: int, do_sample: bool, temperature: float):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    end = time.perf_counter()

    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return _clean_text(text), (end - start) * 1000.0


def _evaluate_output(user_text: str, output: str, latency_ms: float, min_words: int, max_words: int) -> Row:
    words = output.split()
    w = len(words)
    has_q = "?" in output
    multi = len(re.findall(r"[.!?]+", output)) > 1
    low = output.lower()
    risk = bool(re.search(r"\d", low)) or any(re.search(p, low) for p in ANSWER_RISK_PATTERNS)
    topic = _extract_topic(user_text)
    topic_copy = bool(topic) and (topic in low)
    passed = bool(output) and (min_words <= w <= max_words) and (not has_q) and (not multi) and (not risk)
    return Row(
        user=user_text,
        output=output,
        words=w,
        pass_rules=passed,
        answer_risk=risk,
        multi_sentence=multi,
        has_question_mark=has_q,
        topic_copy=topic_copy,
        latency_ms=round(latency_ms, 2),
    )


def _distinct_ngrams(outputs: List[str], n: int) -> float:
    grams: List[str] = []
    for out in outputs:
        toks = out.lower().split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            grams.append(" ".join(toks[i : i + n]))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def build_prompt_set(multiplier: int) -> List[str]:
    prompts: List[str] = []
    for i in range(max(1, multiplier)):
        for p in BASE_EVAL_PROMPTS:
            if i == 0:
                prompts.append(p)
            else:
                prompts.append(f"{p} Please keep it concise for a voice reply.")
    return prompts


def run_eval(name: str, base_model_id: str, adapter_dir: str | None, device: str, prompts: List[str], max_new_tokens: int, do_sample: bool, temperature: float, local_files_only: bool, min_words: int, max_words: int) -> Dict[str, Any]:
    tokenizer, model = _load_model(base_model_id, adapter_dir, device, local_files_only=local_files_only)
    rows: List[Row] = []
    for user in prompts:
        out, latency_ms = _generate_one(model, tokenizer, user, max_new_tokens, do_sample, temperature)
        rows.append(_evaluate_output(user, out, latency_ms, min_words=min_words, max_words=max_words))

    outputs = [r.output for r in rows]
    pass_count = sum(1 for r in rows if r.pass_rules)
    risk_count = sum(1 for r in rows if r.answer_risk)
    summary = {
        "name": name,
        "total": len(rows),
        "pass_rate": round(pass_count / max(1, len(rows)), 4),
        "answer_risk_rate": round(risk_count / max(1, len(rows)), 4),
        "avg_words": round(statistics.mean(r.words for r in rows), 3),
        "avg_latency_ms": round(statistics.mean(r.latency_ms for r in rows), 2),
        "p95_latency_ms": round(sorted(r.latency_ms for r in rows)[int(0.95 * (len(rows)-1))], 2),
        "topic_copy_rate": round(sum(1 for r in rows if r.topic_copy) / max(1, len(rows)), 4),
        "unique_output_ratio": round(len(set(outputs)) / max(1, len(outputs)), 4),
        "distinct_1": round(_distinct_ngrams(outputs, 1), 4),
        "distinct_2": round(_distinct_ngrams(outputs, 2), 4),
        "rows": [asdict(r) for r in rows],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark base vs LoRA for filler quality/latency/diversity")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter-dir", default="outputs/smollm2-135m-filler-lora")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--prompt-multiplier", type=int, default=3, help="20 * multiplier eval prompts")
    ap.add_argument("--out", default="outputs/smollm2-135m-filler-lora/benchmark.json")
    ap.add_argument("--allow-remote", action="store_true", default=False, help="Allow downloading model/tokenizer from Hugging Face")
    ap.add_argument("--min-words", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=16)
    args = ap.parse_args()

    prompts = build_prompt_set(args.prompt_multiplier)

    base_summary = run_eval(
        name="base_135m",
        base_model_id=args.base_model,
        adapter_dir=None,
        device=args.device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        local_files_only=(not args.allow_remote),
        min_words=args.min_words,
        max_words=args.max_words,
    )
    tuned_summary = run_eval(
        name="lora_135m",
        base_model_id=args.base_model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        local_files_only=(not args.allow_remote),
        min_words=args.min_words,
        max_words=args.max_words,
    )

    report = {
        "config": {
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "prompt_count": len(prompts),
        },
        "base": base_summary,
        "tuned": tuned_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "out": str(out_path),
        "base_pass_rate": base_summary["pass_rate"],
        "tuned_pass_rate": tuned_summary["pass_rate"],
        "base_avg_latency_ms": base_summary["avg_latency_ms"],
        "tuned_avg_latency_ms": tuned_summary["avg_latency_ms"],
        "base_distinct_2": base_summary["distinct_2"],
        "tuned_distinct_2": tuned_summary["distinct_2"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
