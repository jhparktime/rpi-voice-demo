# Research Checklist (Implemented)

## A. Training Correctness

- [x] `assistant-only loss` masking implemented in `training/run_filler_sft.py`
- [x] configurable max target length (`data.max_target_words`) implemented
- [x] best checkpoint loading enabled (`load_best_model_at_end`, `metric_for_best_model`)

## B. Evaluation Consistency

- [x] evaluation prompt constraints aligned to current policy (6-14 words, max 16 tokens)
- [x] eval scripts expose `--min-words/--max-words`

## C. LoRA Ablation

- [x] full target config exists (`training/configs/smollm2_135m_filler_lora_l40s.yaml`)
- [x] attention-only target config added (`training/configs/smollm2_135m_filler_lora_l40s_attn_only.yaml`)

## D. Deployment Path (RPi ONNX)

- [x] LoRA merge + ONNX export script added (`training/export_lora_to_onnx.py`)
- [x] optional dynamic int8 quantization hook added (`--quantize-int8`)

## E. Data Policy

- [x] dataset size expanded to 8K/1.2K (`training/data/filler_train_l40s.jsonl`, `training/data/filler_val_l40s.jsonl`)
- [x] awkward pattern suppression applied in dataset builder
