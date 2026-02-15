#!/usr/bin/env bash
set -euo pipefail

python training/run_filler_sft.py \
  --config training/configs/smollm2_135m_filler_lora_l40s.yaml

python training/eval_filler_benchmark.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --prompt-multiplier 3 \
  --out outputs/smollm2-135m-filler-lora-l40s/benchmark_det.json

python training/eval_filler_benchmark.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --prompt-multiplier 3 \
  --do-sample --temperature 0.6 \
  --out outputs/smollm2-135m-filler-lora-l40s/benchmark_sampled.json

# Optional: export trained adapter as ONNX bundle for RPi runtime.
# python training/export_lora_to_onnx.py \
#   --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
#   --out-dir outputs/smollm2-135m-filler-l40s-onnx-bundle \
#   --quantize-int8
