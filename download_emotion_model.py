"""Download emotion_onnx_int8 model assets from HuggingFace.

Usage (on Raspberry Pi, inside venv):

    python download_emotion_model.py

This will create the following layout if it does not already exist:

emotion_onnx_int8/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
  onnx/
    model_quantized.onnx

Demo/emotion.py then uses this directory as the model_dir.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable


TOKENIZER_REPO = "joeddav/distilbert-base-uncased-go-emotions-student"
ONNX_REPO = "Cohee/distilbert-base-uncased-go-emotions-onnx"

TOKENIZER_FILES: Iterable[str] = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def main() -> int:
    root = Path(__file__).resolve().parent
    emotion_dir = root / "emotion_onnx_int8"
    onnx_dir = emotion_dir / "onnx"
    target_onnx = onnx_dir / "model_quantized.onnx"

    # Idempotent: if everything is already there, exit early.
    if target_onnx.exists():
        print(f"[info] Emotion ONNX already present at {target_onnx}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print(
            "[error] huggingface_hub is not installed. "
            "Run 'pip install -r requirements.txt' first.",
            file=sys.stderr,
        )
        return 1

    emotion_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download tokenizer/config/vocab
    print(f"[info] Downloading tokenizer/config from {TOKENIZER_REPO}...")
    try:
        tok_cache = snapshot_download(
            repo_id=TOKENIZER_REPO,
            allow_patterns=list(TOKENIZER_FILES),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to download tokenizer/config: {exc}", file=sys.stderr)
        return 1

    tok_cache_path = Path(tok_cache)
    for name in TOKENIZER_FILES:
        src = tok_cache_path / name
        dst = emotion_dir / name
        if not src.exists():
            print(f"[error] Missing expected file in tokenizer repo: {src}", file=sys.stderr)
            return 1
        print(f"[info] Copying {src} -> {dst}")
        shutil.copy2(src, dst)

    # 2) Download ONNX model
    print(f"[info] Downloading ONNX model from {ONNX_REPO}...")
    try:
        onnx_cache = snapshot_download(
            repo_id=ONNX_REPO,
            allow_patterns=["onnx/model_quantized.onnx"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to download ONNX model: {exc}", file=sys.stderr)
        return 1

    onnx_cache_path = Path(onnx_cache) / "onnx" / "model_quantized.onnx"
    if not onnx_cache_path.exists():
        print(f"[error] Missing expected ONNX file: {onnx_cache_path}", file=sys.stderr)
        return 1

    print(f"[info] Copying {onnx_cache_path} -> {target_onnx}")
    shutil.copy2(onnx_cache_path, target_onnx)

    print(f"[done] Emotion model ready under {emotion_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

