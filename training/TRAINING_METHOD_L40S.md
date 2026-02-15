# L40S Training Method (This Branch)

This guide describes the end-to-end training flow for the filler/bridge LoRA model in this branch.

## 0) Environment

- Recommended: Colab or server with NVIDIA L40S GPU
- Python 3.10+

Install dependencies:

```bash
pip install -U pip
pip install -r training/requirements-colab.txt
```

## 1) Build Dataset (8K / 1.2K)

```bash
python training/build_filler_dataset.py \
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

## 2) Train LoRA (Main Config)

```bash
python training/run_filler_sft.py \
  --config training/configs/smollm2_135m_filler_lora_l40s.yaml
```

Output:

- `outputs/smollm2-135m-filler-lora-l40s`

## 3) Ablation: Attention-only LoRA

```bash
python training/run_filler_sft.py \
  --config training/configs/smollm2_135m_filler_lora_l40s_attn_only.yaml
```

Output:

- `outputs/smollm2-135m-filler-lora-l40s-attn`

## 4) Evaluate (Base vs Tuned)

Deterministic:

```bash
python training/eval_filler_benchmark.py \
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
python training/eval_filler_benchmark.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --prompt-multiplier 3 \
  --do-sample --temperature 0.6 \
  --out outputs/smollm2-135m-filler-lora-l40s/benchmark_sampled.json
```

Quick chat sanity check:

```bash
python training/eval_filler_chat.py \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --device cuda \
  --allow-remote \
  --min-words 6 --max-words 16 \
  --max-new-tokens 14 --limit 8
```

## 5) Export to ONNX Bundle (RPi Deployment)

```bash
python training/export_lora_to_onnx.py \
  --adapter-dir outputs/smollm2-135m-filler-lora-l40s \
  --out-dir outputs/smollm2-135m-filler-l40s-onnx-bundle \
  --quantize-int8
```

Output bundle:

- tokenizer files at bundle root
- `onnx/*.onnx`
- optional `onnx/*.int8.onnx`

## 6) One-command Run (after dataset build)

```bash
bash training/run_colab_l40s.sh
```

