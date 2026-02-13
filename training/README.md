# Filler/Bridge Fine-tuning for SmolLM2-135M

This folder contains a practical starter pipeline to fine-tune `HuggingFaceTB/SmolLM2-135M-Instruct` for strict filler/bridge behavior.

## Goal

Train a small local model that:

- outputs only one short bridge sentence
- does not answer the question
- avoids factual claims and hallucination
- stays within low-latency output length bounds

## Files

- `configs/smollm2_135m_filler_lora.yaml`: experiment config
- `data/filler_train.sample.jsonl`: sample training data
- `data/filler_val.sample.jsonl`: sample validation data
- `run_filler_sft.py`: LoRA SFT training entrypoint
- `requirements-finetune.txt`: dependencies for fine-tuning

## Dataset JSONL format

Each line is one object:

```json
{
  "system": "System instruction for filler behavior",
  "user": "Original user utterance",
  "assistant": "Target filler response",
  "tags": ["optional", "metadata"]
}
```

Constraints for `assistant` target:

- one sentence
- max 12 tokens (whitespace tokenization as a cheap bound)
- no factual answer
- no named entities if avoidable

## Dataset Generation

### A. Template bootstrap (fast, no API dependency)

```bash
python training/build_filler_dataset.py \
  --teacher-backend template \
  --train-size 300 --val-size 60 \
  --n-candidates 3
```

### B. Distillation with local teacher (Ollama)

```bash
python training/build_filler_dataset.py \
  --teacher-backend ollama \
  --teacher-model smollm2:360m \
  --ollama-url http://localhost:11434/api/generate \
  --train-size 1000 --val-size 200 \
  --n-candidates 4
```

### C. Distillation with cloud teacher (OpenAI/Gemini env key required)

```bash
python training/build_filler_dataset.py \
  --teacher-backend cloud \
  --train-size 1000 --val-size 200 \
  --n-candidates 4
```

Generated files:

- train: `training/data/filler_train.jsonl`
- val: `training/data/filler_val.jsonl`
- rejects: `training/data/filler_rejects.jsonl`

## Quick Start (Fine-tuning)

```bash
python -m venv .venv-ft
source .venv-ft/bin/activate
pip install -r training/requirements-finetune.txt

python training/run_filler_sft.py --config training/configs/smollm2_135m_filler_lora.yaml
```

## Suggested first experiment

1. Start with template-heavy data (safe and deterministic).
2. Add teacher-distilled data (ollama/cloud) with strict filtering.
3. Evaluate:
- role-break rate (answers the question)
- hallucination rate
- output length violations
- latency per token / total generation time
