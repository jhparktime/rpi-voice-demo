#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_record(rec: Dict[str, Any], idx: int, max_target_words: int) -> None:
    required = ["system", "user", "assistant"]
    missing = [k for k in required if k not in rec or not str(rec[k]).strip()]
    if missing:
        raise ValueError(f"Record {idx} missing required fields: {missing}")

    out = str(rec["assistant"]).strip()
    words = out.split()
    if len(words) > max_target_words:
        raise ValueError(
            f"Record {idx} assistant too long ({len(words)} > {max_target_words} words): {out!r}"
        )


def _make_text(tokenizer: Any, system: str, user: str, assistant: str) -> str:
    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _make_prompt_prefix(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _read_jsonl(path: str, max_target_words: int, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            _validate_record(rec, i, max_target_words=max_target_words)
            rows.append(rec)
    return rows


class TextLMDataset:
    def __init__(self, input_ids: List[List[int]], attention_mask: List[List[int]], labels: List[List[int]]) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import torch

        ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


try:
    import torch as _torch
    _Tensor = _torch.Tensor
except Exception:  # pragma: no cover
    _Tensor = Any

TextLMDataset.__annotations__["__getitem__"] = Dict[str, _Tensor]  # type: ignore[index]


def _tokenize_rows(rows: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> TextLMDataset:
    all_input_ids: List[List[int]] = []
    all_attention_mask: List[List[int]] = []
    all_labels: List[List[int]] = []
    dropped = 0

    for r in rows:
        full_text = _make_text(tokenizer, r["system"], r["user"], r["assistant"])
        prefix_text = _make_prompt_prefix(tokenizer, r["system"], r["user"])

        enc_full = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        enc_prefix = tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        input_ids = list(enc_full["input_ids"])
        attention_mask = list(enc_full["attention_mask"])
        prefix_len = len(enc_prefix["input_ids"])
        non_pad_len = int(sum(attention_mask))

        # If assistant segment is fully truncated, skip this record.
        if prefix_len >= non_pad_len:
            dropped += 1
            continue

        labels = list(input_ids)
        for i in range(len(labels)):
            if i < prefix_len or attention_mask[i] == 0:
                labels[i] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    if dropped > 0:
        print(f"[ft] dropped rows due to truncation(no assistant tokens): {dropped}", flush=True)
    return TextLMDataset(all_input_ids, all_attention_mask, all_labels)


def main() -> None:
    print("[ft] startup", flush=True)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    print("[ft] heavy imports ready", flush=True)

    parser = argparse.ArgumentParser(description="LoRA SFT for SmolLM2 filler/bridge behavior (lightweight)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    model_id = cfg["model_name_or_path"]
    out_cfg = cfg["output"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    runtime_cfg = cfg.get("runtime", {})
    local_files_only = bool(runtime_cfg.get("local_files_only", True))
    use_fast_tokenizer = bool(runtime_cfg.get("use_fast_tokenizer", True))

    output_dir = out_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"[ft] runtime local_files_only={local_files_only} use_fast_tokenizer={use_fast_tokenizer}",
        flush=True,
    )
    print("[ft] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast_tokenizer,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[ft] loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)

    print("[ft] applying LoRA...", flush=True)
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    mts = data_cfg.get("max_train_samples")
    mes = data_cfg.get("max_eval_samples")
    max_target_words = int(data_cfg.get("max_target_words", 12))
    train_rows = _read_jsonl(
        data_cfg["train_file"],
        max_target_words=max_target_words,
        limit=int(mts) if mts else None,
    )
    val_rows = _read_jsonl(
        data_cfg["val_file"],
        max_target_words=max_target_words,
        limit=int(mes) if mes else None,
    )
    print(f"[ft] loaded rows: train={len(train_rows)} val={len(val_rows)}", flush=True)

    max_length = int(data_cfg.get("max_seq_length", 192))
    train_ds = _tokenize_rows(train_rows, tokenizer, max_length=max_length)
    val_ds = _tokenize_rows(val_rows, tokenizer, max_length=max_length)
    print("[ft] tokenization done", flush=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        eval_strategy=str(train_cfg["evaluation_strategy"]),
        save_strategy=str(train_cfg["save_strategy"]),
        bf16=bool(train_cfg["bf16"]),
        fp16=bool(train_cfg["fp16"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        seed=int(train_cfg["seed"]),
        report_to="none",
        remove_unused_columns=False,
        logging_first_step=True,
        disable_tqdm=True,
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(train_cfg.get("greater_is_better", False)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("[ft] training start", flush=True)
    trainer.train()
    print("[ft] training done", flush=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"status": "ok", "output_dir": output_dir}, ensure_ascii=True))


if __name__ == "__main__":
    main()
