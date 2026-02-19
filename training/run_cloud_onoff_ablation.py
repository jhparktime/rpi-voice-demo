#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from Demo import cloud_llm
from Demo import text_utils
from Demo.stt_tts_test import _build_gemini_prompt, _generate_filler_ollama


DEFAULT_QUERIES = [
    "Explain quantum superposition with a simple analogy.",
    "Compare LoRA and full fine-tuning for edge deployment.",
    "Explain quantization for on-device inference in detail.",
    "How does DNS caching work step by step?",
    "Explain transformer architecture with pros and cons.",
]

ANSWER_LIKE_PATTERNS = [
    r"\bthe answer is\b",
    r"\bbecause\b",
    r"\bfor example\b",
    r"\bmeans\b",
    r"\bfirst\b",
    r"\bsecond\b",
]

FORBIDDEN_PATTERNS = [
    r"\bhere(?:'s| is)\b",
    r"\bthe answer\b",
    r"\bsql query\b",
    r"\bi(?: am|'m)?\s*sorry\b",
    r"\bjust kidding\b",
    r"\bbecause\b",
]


def _load_queries(path: str | None) -> List[str]:
    if not path:
        return list(DEFAULT_QUERIES)
    p = Path(path)
    rows: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            rows.append(s)
    return rows if rows else list(DEFAULT_QUERIES)


def _token_count_proxy(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _has_named_entity_like(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text or "")
    return any(re.fullmatch(r"[A-Z][a-z]{2,}", w) for w in words[1:])


def _p95(values: List[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[int(0.95 * (len(s) - 1))]


def _build_filler_args(filler_provider: str) -> Any:
    return SimpleNamespace(
        cloud_filler=True,
        filler_provider=filler_provider,
        onnx_model="HuggingFaceTB/SmolLM2-135M-Instruct",
    )


def _evaluate_filler_policy(text: str) -> Dict[str, bool]:
    s = (text or "").strip()
    low = s.lower()
    return {
        "has_number": bool(re.search(r"\d", s)),
        "has_question": "?" in s,
        "has_named_entity": _has_named_entity_like(s),
        "answer_like": any(re.search(p, low) for p in ANSWER_LIKE_PATTERNS),
        "forbidden_pattern": any(re.search(p, low) for p in FORBIDDEN_PATTERNS),
    }


def run_ablation(
    queries: List[str],
    repeats: int,
    filler_provider: str,
    delay_ms: int,
    timeout_s: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    delay_s = max(0.0, float(delay_ms) / 1000.0)
    filler_args = _build_filler_args(filler_provider=filler_provider)

    for _ in range(max(1, repeats)):
        for q in queries:
            t0 = time.perf_counter()
            gemini_prompt = _build_gemini_prompt("LONG", q)
            system_prompt = text_utils.build_cloud_system_prompt(None)

            filler_text = ""
            filler_triggered = False
            first_signal_s: float | None = None

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                fut_cloud = ex.submit(
                    cloud_llm.call_cloud_llm,
                    gemini_prompt,
                    system_prompt,
                    timeout_s,
                    None,
                    0.35,
                    "gemini",
                )
                try:
                    cloud_reply = fut_cloud.result(timeout=delay_s)
                    t_cloud = time.perf_counter()
                    first_signal_s = t_cloud - t0
                except concurrent.futures.TimeoutError:
                    filler_triggered = True
                    if filler_provider.lower() != "off":
                        filler_text = _generate_filler_ollama(
                            filler_args,
                            None,
                            q,
                            timeout=min(timeout_s, 3.0),
                        )
                        t_fill = time.perf_counter()
                        if filler_text:
                            first_signal_s = t_fill - t0
                    cloud_reply = fut_cloud.result()
                    t_cloud = time.perf_counter()

            if first_signal_s is None:
                first_signal_s = t_cloud - t0

            row = {
                "query": q,
                "filler_provider": filler_provider,
                "filler_triggered": filler_triggered,
                "filler_text": filler_text,
                "cloud_reply": cloud_reply,
                "first_signal_s": round(first_signal_s, 4),
                "cloud_ready_s": round(t_cloud - t0, 4),
                "input_tokens_proxy": _token_count_proxy(gemini_prompt),
                "output_tokens_proxy": _token_count_proxy(cloud_reply),
            }
            row["filler_policy"] = _evaluate_filler_policy(filler_text) if filler_text else {}
            rows.append(row)

    first_signal = [r["first_signal_s"] for r in rows]
    cloud_ready = [r["cloud_ready_s"] for r in rows]
    input_toks = [r["input_tokens_proxy"] for r in rows]
    output_toks = [r["output_tokens_proxy"] for r in rows]
    cloud_calls = len(rows)

    fillers = [r for r in rows if r.get("filler_text")]
    if fillers:
        policy_viols = 0
        forbidden = 0
        for r in fillers:
            fp = r["filler_policy"]
            if fp.get("forbidden_pattern"):
                forbidden += 1
            if (
                fp.get("has_number")
                or fp.get("has_question")
                or fp.get("has_named_entity")
                or fp.get("answer_like")
            ):
                policy_viols += 1
        policy_violation_rate = policy_viols / len(fillers)
        hallucination_proxy_rate = forbidden / len(fillers)
    else:
        policy_violation_rate = 0.0
        hallucination_proxy_rate = 0.0

    summary = {
        "filler_provider": filler_provider,
        "turns": len(rows),
        "cloud_calls": cloud_calls,
        "first_signal_avg_s": round(statistics.mean(first_signal), 4) if first_signal else None,
        "first_signal_p95_s": round(_p95(first_signal), 4) if first_signal else None,
        "cloud_ready_avg_s": round(statistics.mean(cloud_ready), 4) if cloud_ready else None,
        "cloud_ready_p95_s": round(_p95(cloud_ready), 4) if cloud_ready else None,
        "avg_input_tokens_proxy": round(statistics.mean(input_toks), 2) if input_toks else None,
        "avg_output_tokens_proxy": round(statistics.mean(output_toks), 2) if output_toks else None,
        "filler_count": len(fillers),
        "policy_violation_rate": round(policy_violation_rate, 4),
        "hallucination_proxy_forbidden_rate": round(hallucination_proxy_rate, 4),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cloud LONG filler on/off ablation without STT/TTS")
    ap.add_argument("--queries-file", type=str, default="", help="Optional text file with one query per line")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--filler-provider", choices=["smollm2", "off"], default="smollm2")
    ap.add_argument("--delay-ms", type=int, default=800)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    queries = _load_queries(args.queries_file or None)
    result = run_ablation(
        queries=queries,
        repeats=args.repeats,
        filler_provider=args.filler_provider,
        delay_ms=args.delay_ms,
        timeout_s=args.timeout_s,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
