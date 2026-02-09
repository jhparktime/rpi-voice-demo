"""Download emotion_onnx_int8, Kokoro TTS, and (optionally) sherpa-onnx STT assets.

Usage (on Raspberry Pi, inside venv):

    python download_model.py

This will create the following layouts if they do not already exist:

emotion_onnx_int8/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
  onnx/
    model_quantized.onnx

Demo/emotion.py then uses this directory as the model_dir.

models/kokoro/
  voices-v1.0.bin
  model_quantized.onnx

Demo/stt_tts_cli.py + Demo/tts_kokoro.py use these Kokoro files.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

import requests


TOKENIZER_REPO = "joeddav/distilbert-base-uncased-go-emotions-student"
ONNX_REPO = "Cohee/distilbert-base-uncased-go-emotions-onnx"

TOKENIZER_FILES: Iterable[str] = (
    "config.json",
    # Many DistilBERT checkpoints do not provide tokenizer.json; we rely on
    # tokenizer_config.json + vocab.txt + special_tokens_map.json instead.
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)

KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"

# sherpa-onnx STT: we assume you manually download a small streaming English model
# and place the core files under sherpa_stt/:
#   tokens.txt
#   encoder.onnx
#   decoder.onnx
#   joiner.onnx
# The download URLs change over time, so we don't hard-code them here.


def _download_file(url: str, dst: Path, desc: str) -> None:
    """Stream download a file to dst."""
    print(f"[info] Downloading {desc} from {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
    size = dst.stat().st_size if dst.exists() else 0
    print(f"[info] Saved {desc} to {dst} ({size} bytes)")


def main() -> int:
    root = Path(__file__).resolve().parent

    # === Emotion model (HuggingFace) ===
    emotion_dir = root / "emotion_onnx_int8"
    onnx_dir = emotion_dir / "onnx"
    target_onnx = onnx_dir / "model_quantized.onnx"

    need_emotion = not target_onnx.exists()
    if need_emotion:
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
                # Some files (like tokenizer.json) may be omitted in certain repos.
                print(f"[warn] Optional tokenizer file missing in repo: {src}")
                continue
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
    else:
        print(f"[info] Emotion ONNX already present at {target_onnx}")

    # === Kokoro TTS model + voices (GitHub release) ===
    kokoro_root = root / "models" / "kokoro"
    kokoro_onnx = kokoro_root / "model_quantized.onnx"
    kokoro_voices = kokoro_root / "voices-v1.0.bin"

    have_kokoro = kokoro_onnx.exists() and kokoro_voices.exists()
    if have_kokoro:
        print(f"[info] Kokoro TTS already present under {kokoro_root}")
    else:
        try:
            _download_file(KOKORO_VOICES_URL, kokoro_voices, "Kokoro voices-v1.0.bin")
            _download_file(KOKORO_ONNX_URL, kokoro_onnx, "Kokoro ONNX model_quantized.onnx")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] Failed to download Kokoro TTS assets: {exc}", file=sys.stderr)
            return 1

    # sherpa-onnx STT directory check (manual download for now)
    sherpa_dir = root / "sherpa_stt"
    tokens = sherpa_dir / "tokens.txt"
    encoder = sherpa_dir / "encoder.onnx"
    decoder = sherpa_dir / "decoder.onnx"
    joiner = sherpa_dir / "joiner.onnx"
    sherpa_ready = all(p.exists() for p in (tokens, encoder, decoder, joiner))
    if sherpa_ready:
        print(f"[info] sherpa-onnx STT assets present under {sherpa_dir}")
    else:
        sherpa_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[info] sherpa-onnx STT assets not found. "
            f"Expected tokens/encoder/decoder/joiner under {sherpa_dir}. "
            "Please download a small English streaming model from the sherpa-onnx docs "
            "and place the files there.",
            file=sys.stderr,
        )

    # Final status summary (best-effort)
    try:
        emo_size = target_onnx.stat().st_size if target_onnx.exists() else 0
        kokoro_size = kokoro_onnx.stat().st_size if kokoro_onnx.exists() else 0
        voices_size = kokoro_voices.stat().st_size if kokoro_voices.exists() else 0
        print(f"[summary] Emotion ONNX: {target_onnx} ({emo_size} bytes)")
        print(f"[summary] Kokoro ONNX:  {kokoro_onnx} ({kokoro_size} bytes)")
        print(f"[summary] Kokoro voices:{kokoro_voices} ({voices_size} bytes)")
        if sherpa_ready:
            print(f"[summary] sherpa-onnx STT: {sherpa_dir}")
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

