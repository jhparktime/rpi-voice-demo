#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print('[onnx] run:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    names = [
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'chat_template.jinja',
        'vocab.json',
        'merges.txt',
        'sentencepiece.bpe.model',
    ]
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _quantize_onnx_files(onnx_dir: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as e:
        raise RuntimeError('onnxruntime quantization is unavailable. Install onnxruntime.') from e

    for p in sorted(onnx_dir.rglob('*.onnx')):
        out = p.with_name(p.stem + '.int8.onnx')
        print(f'[onnx] quantize: {p} -> {out}', flush=True)
        quantize_dynamic(
            model_input=str(p),
            model_output=str(out),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge LoRA adapter and export ONNX bundle for RPi inference')
    ap.add_argument('--adapter-dir', required=True, help='Path to trained LoRA adapter dir')
    ap.add_argument('--base-model', default=None, help='Override base model id/path')
    ap.add_argument('--out-dir', required=True, help='Output bundle dir (contains tokenizer + onnx/)')
    ap.add_argument('--task', default='text-generation-with-past', help='Optimum export task')
    ap.add_argument('--quantize-int8', action='store_true', default=False, help='Also produce *.int8.onnx files')
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    merged_dir = out_dir / '_merged'
    onnx_dir = out_dir / 'onnx'

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(adapter_dir / 'adapter_config.json', 'r', encoding='utf-8') as f:
        adapter_cfg = json.load(f)
    base_model = args.base_model or adapter_cfg['base_model_name_or_path']

    print(f'[onnx] base_model={base_model}', flush=True)
    print(f'[onnx] adapter_dir={adapter_dir}', flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=False)
    base = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=False)
    model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=False)
    model = model.merge_and_unload()

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    _run([
        'optimum-cli',
        'export',
        'onnx',
        '--task',
        args.task,
        '--model',
        str(merged_dir),
        str(onnx_dir),
    ])

    _copy_tokenizer_files(merged_dir, out_dir)

    if args.quantize_int8:
        _quantize_onnx_files(onnx_dir)

    print(json.dumps({
        'status': 'ok',
        'bundle_dir': str(out_dir),
        'onnx_dir': str(onnx_dir),
        'merged_dir': str(merged_dir),
    }, ensure_ascii=True))


if __name__ == '__main__':
    main()
