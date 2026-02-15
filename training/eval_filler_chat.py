#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = (
    "You are a realtime voice filler generator. "
    "Output exactly one short bridge sentence (6-14 words, max 16 tokens). "
    "Do not answer the user. Do not add facts."
)

TEST_UTTERANCES = [
    "What is quantum tunneling in simple terms?",
    "Can you summarize this article for me?",
    "Write a SQL query to find duplicate emails.",
    "How many continents are there on Earth?",
    "I feel very anxious about my interview tomorrow.",
    "Translate this sentence into Korean.",
    "What is 127 multiplied by 43?",
    "Can you explain transformers in machine learning?",
]

ANSWER_RISK_PATTERNS = [
    r"\bthe answer is\b",
    r"\bis\s+\d+\b",
    r"\bbecause\b",
    r"\bfor example\b",
    r"\bmeans\b",
]


@dataclass
class EvalRow:
    user: str
    output: str
    words: int
    has_question_mark: bool
    multi_sentence: bool
    answer_risk: bool
    pass_rules: bool


def _clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _is_multi_sentence(t: str) -> bool:
    parts = re.findall(r"[.!?]+", t)
    return len(parts) > 1


def _answer_risk(t: str) -> bool:
    low = t.lower()
    if re.search(r"\d", low):
        return True
    return any(re.search(pat, low) for pat in ANSWER_RISK_PATTERNS)


def generate_one(model, tokenizer, user_text: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _clean_text(text.split("\n")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned filler model by simulated chat turns")
    parser.add_argument("--adapter-dir", default="outputs/smollm2-135m-filler-lora")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=16)
    parser.add_argument("--allow-remote", action="store_true", default=False, help="Allow downloading model/tokenizer from Hugging Face")
    args = parser.parse_args()
    local_files_only = not args.allow_remote

    adapter_dir = Path(args.adapter_dir)
    with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg["base_model_name_or_path"]

    tokenizer_dir = adapter_dir if (adapter_dir / "tokenizer.json").exists() else adapter_dir.parent
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception:
        # Some checkpoints may not reconstruct a fast tokenizer cleanly.
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            use_fast=False,
            local_files_only=local_files_only,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "cpu":
        torch.set_num_threads(1)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=local_files_only)
    model = model.to(args.device)
    model.eval()

    rows: List[EvalRow] = []
    for user_text in TEST_UTTERANCES[: max(1, args.limit)]:
        out = generate_one(model, tokenizer, user_text, max_new_tokens=args.max_new_tokens)
        words = len(out.split())
        has_q = "?" in out
        multi = _is_multi_sentence(out)
        risk = _answer_risk(out)
        passed = (args.min_words <= words <= args.max_words) and (not has_q) and (not multi) and (not risk) and bool(out)
        rows.append(
            EvalRow(
                user=user_text,
                output=out,
                words=words,
                has_question_mark=has_q,
                multi_sentence=multi,
                answer_risk=risk,
                pass_rules=passed,
            )
        )

    passed = sum(1 for r in rows if r.pass_rules)
    summary = {
        "total": len(rows),
        "pass_count": passed,
        "pass_rate": round(passed / max(len(rows), 1), 4),
        "rows": [asdict(r) for r in rows],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
