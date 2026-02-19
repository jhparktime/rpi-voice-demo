#!/usr/bin/env bash
set -euo pipefail

# RPi ablation runner (no STT/TTS required)
# 1) LONG filler ON vs OFF (cloud text-only)
# 2) base 135M vs fine-tuned 135M (filler benchmark)
# 3) fixed phrase bank baseline

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-outputs/ablation_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-5}"
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-10}"
BASE_MODEL="${BASE_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/smollm2-135m-filler-lora}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$OUT_DIR"

echo "[ablation] output_dir=$OUT_DIR"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[warn] GEMINI_API_KEY is not set. Cloud filler ON/OFF ablation will fail."
else
  echo "[ablation] running cloud LONG filler ON..."
  python training/run_cloud_onoff_ablation.py \
    --filler-provider smollm2 \
    --repeats "$REPEATS" \
    --out "$OUT_DIR/cloud_long_filler_on.json"

  echo "[ablation] running cloud LONG filler OFF..."
  python training/run_cloud_onoff_ablation.py \
    --filler-provider off \
    --repeats "$REPEATS" \
    --out "$OUT_DIR/cloud_long_filler_off.json"
fi

echo "[ablation] running base vs fine-tuned filler benchmark..."
python training/eval_filler_benchmark.py \
  --base-model "$BASE_MODEL" \
  --adapter-dir "$ADAPTER_DIR" \
  --device "$DEVICE" \
  --prompt-multiplier "$PROMPT_MULTIPLIER" \
  --max-new-tokens 16 \
  --allow-remote \
  --out "$OUT_DIR/base_vs_tuned.json"

echo "[ablation] building fixed phrase bank baseline..."
python - <<'PY' > "$OUT_DIR/fixed_phrase_bank.json"
import json
import re
import statistics

prompts = [
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
] * 10

bank = [
    "One moment.",
    "Just a sec.",
    "Checking that now.",
    "Working on it.",
    "Let me check.",
]

answer_like = [
    r"\bthe answer is\b",
    r"\bbecause\b",
    r"\bfor example\b",
    r"\bmeans\b",
    r"\bfirst\b",
    r"\bsecond\b",
]

rows = []
for i, _ in enumerate(prompts):
    out = bank[i % len(bank)]
    words = len(out.split())
    has_q = "?" in out
    multi = len(re.findall(r"[.!?]+", out)) > 1
    risk = bool(re.search(r"\d", out)) or any(re.search(p, out, re.I) for p in answer_like)
    passed = (6 <= words <= 16) and (not has_q) and (not multi) and (not risk)
    rows.append({"words": words, "pass_rules": passed})

print(
    json.dumps(
        {
            "name": "fixed_phrase_bank",
            "total": len(rows),
            "pass_rate": round(sum(r["pass_rules"] for r in rows) / max(1, len(rows)), 4),
            "avg_words": round(statistics.mean(r["words"] for r in rows), 3),
        },
        indent=2,
    )
)
PY

echo "[ablation] done."
echo "[ablation] files:"
echo "  - $OUT_DIR/base_vs_tuned.json"
echo "  - $OUT_DIR/fixed_phrase_bank.json"
echo "  - $OUT_DIR/cloud_long_filler_on.json (if GEMINI_API_KEY set)"
echo "  - $OUT_DIR/cloud_long_filler_off.json (if GEMINI_API_KEY set)"
