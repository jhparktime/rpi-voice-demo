# L40S Filler Dataset Generation Record (for Research Notes)

## 1) Purpose
Build a constrained utterance-level dataset for SFT training of short bridge/filler responses for low-latency voice interaction.
- Goal: one concise utterance per turn
- Constraint: 6~16 words
- Safety: no answering style, no facts, no question marks, single sentence

## 2) Command (reproducible)
```bash
cd /Users/parkjaehyun/Desktop/rpi-voice-demo
python training/build_filler_dataset.py \
  --train-size 8000 --val-size 1200 --seed 42 \
  --teacher-backend template --n-candidates 6 \
  --synthetic-sources 10000 --min-words 6 --max-words 16 \
  --max-per-phrase 4 --strict-filter --topic-aware-templates \
  --out-train training/data/filler_train_l40s.jsonl \
  --out-val training/data/filler_val_l40s.jsonl \
  --out-rejects training/data/filler_rejects_l40s.jsonl
```

## 3) Generation result
- Backend: `template`
- Seed: `42`
- Generated at: `2026-02-18 10:21:30 UTC`
- Train rows: `8000`
- Validation rows: `1200`
- Rejected candidates saved: `8089`

## 4) Rejection breakdown (Top-10)
- `awkward_phrase_pattern`: `8089`

## 5) Output paths
- `training/data/filler_train_l40s.jsonl`
- `training/data/filler_val_l40s.jsonl`
- `training/data/filler_rejects_l40s.jsonl`

## 6) Research-use logging artifact
- Meta file: `training/data/.filler_l40s_generation_meta.json`
- Contains: timestamp, seed, backend, train/val/reject counts, top reject reasons, file paths.

## 7) Notes for paper/report
- This generation is deterministic for given `--seed` and source corpus settings.
- The generated dataset is intentionally synthetic/template-heavy to reduce upstream API dependency and improve reproducibility.
- Recommended next step for ablation: compare with `--teacher-backend cloud` and/or additional synthetic coverage, while keeping quality filters and target length constraints fixed.
