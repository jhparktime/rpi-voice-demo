#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_record(rec: Dict[str, Any], idx: int) -> None:
    required = ["system", "user", "assistant"]
    missing = [k for k in required if k not in rec or not str(rec[k]).strip()]
    if missing:
        raise ValueError(f"Record {idx} missing required fields: {missing}")

    out = str(rec["assistant"]).strip()
    words = out.split()
    if len(words) > 12:
        raise ValueError(f"Record {idx} assistant too long ({len(words)} > 12 words): {out!r}")


def _make_text(tokenizer: AutoTokenizer, system: str, user: str, assistant: str) -> str:
    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _prepare_split(dataset_split: Any, tokenizer: AutoTokenizer) -> Any:
    for i, rec in enumerate(dataset_split):
        _validate_record(rec, i)

    def _map_fn(rec: Dict[str, Any]) -> Dict[str, str]:
        txt = _make_text(tokenizer, rec["system"], rec["user"], rec["assistant"])
        return {"text": txt}

    return dataset_split.map(_map_fn, remove_columns=dataset_split.column_names)


@dataclass
class BuildResult:
    train_dataset: Any
    eval_dataset: Any


def build_datasets(cfg: Dict[str, Any], tokenizer: AutoTokenizer) -> BuildResult:
    data_cfg = cfg["data"]
    train_file = data_cfg["train_file"]
    val_file = data_cfg["val_file"]

    ds = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
    )
    train_ds = ds["train"]
    val_ds = ds["validation"]

    mts = data_cfg.get("max_train_samples")
    mes = data_cfg.get("max_eval_samples")
    if mts:
        train_ds = train_ds.select(range(min(int(mts), len(train_ds))))
    if mes:
        val_ds = val_ds.select(range(min(int(mes), len(val_ds))))

    train_ds = _prepare_split(train_ds, tokenizer)
    val_ds = _prepare_split(val_ds, tokenizer)
    return BuildResult(train_dataset=train_ds, eval_dataset=val_ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for SmolLM2 filler/bridge behavior")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    model_id = cfg["model_name_or_path"]
    out_cfg = cfg["output"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]

    output_dir = out_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    datasets = build_datasets(cfg, tokenizer)

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        logging_dir=out_cfg.get("logging_dir"),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
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
        dataset_text_field="text",
        max_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(json.dumps({"status": "ok", "output_dir": output_dir}, ensure_ascii=True))


if __name__ == "__main__":
    main()
