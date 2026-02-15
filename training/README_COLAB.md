# Colab Setup (GPU)

Use this when local CPU is too slow for LoRA fine-tuning.

## 1) Runtime

- Colab menu: `Runtime` -> `Change runtime type` -> `L4 / A100 / L40S` (recommended)

## 2) Clone repo

```bash
%cd /content
!git clone <YOUR_REPO_URL> rpi-voice-demo
%cd /content/rpi-voice-demo
```

## 3) Install dependencies

```bash
!pip install -q -U pip
!pip install -q -r training/requirements-colab.txt
```

Notes:
- Do not reinstall `torch` in Colab unless you explicitly need a different CUDA build.

## 4) Build larger filler dataset (L40S)

```bash
!python training/build_filler_dataset.py \
  --train-size 8000 --val-size 1200 \
  --seed 42 \
  --teacher-backend template \
  --n-candidates 6 \
  --synthetic-sources 10000 \
  --min-words 6 --max-words 16 \
  --max-per-phrase 4 \
  --strict-filter \
  --topic-aware-templates \
  --out-train training/data/filler_train_l40s.jsonl \
  --out-val training/data/filler_val_l40s.jsonl \
  --out-rejects training/data/filler_rejects_l40s.jsonl
```

## 5) Train LoRA on GPU (L40S config)

```bash
!python training/run_filler_sft.py \
  --config training/configs/smollm2_135m_filler_lora_l40s.yaml
```

Output adapter:
- `outputs/smollm2-135m-filler-lora-l40s`

Optional ablation (attention-only LoRA):

```bash
!python training/run_filler_sft.py \
  --config training/configs/smollm2_135m_filler_lora_l40s_attn_only.yaml
```

## 6) Evaluate (base vs tuned)

Deterministic:

```bash
!python training/eval_filler_benchmark.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --prompt-multiplier 3 \
  --out outputs/smollm2-135m-filler-lora-l40s/benchmark_det.json
```

Sampled:

```bash
!python training/eval_filler_benchmark.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --prompt-multiplier 3 \
  --do-sample --temperature 0.6 \
  --out outputs/smollm2-135m-filler-lora-l40s/benchmark_sampled.json
```

Quick chat-style sanity check:

```bash
!python training/eval_filler_chat.py \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --max-new-tokens 14 --limit 8
```

Or run all steps after dataset build:

```bash
!bash training/run_colab_l40s.sh
```

## 7) Export to ONNX bundle (for RPi)

```bash
!python training/export_lora_to_onnx.py \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --out-dir outputs/smollm2-135m-filler-l40s-onnx-bundle \
  --quantize-int8
```

This produces:
- tokenizer files at bundle root
- ONNX graphs under `onnx/`
- optional `*.int8.onnx` files for CPU-friendly deployment
## 8) Save artifacts to Drive (optional)

```bash
from google.colab import drive
import shutil

drive.mount('/content/drive')
shutil.make_archive('/content/smollm2-135m-filler-lora-l40s', 'zip', '/content/rpi-voice-demo/outputs/smollm2-135m-filler-lora-l40s')
shutil.copy('/content/smollm2-135m-filler-lora-l40s.zip', '/content/drive/MyDrive/')
```
