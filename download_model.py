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

# sherpa-onnx STT: small English streaming Zipformer model (~20M params).
# We hard-code a stable release archive and extract/copy the needed files.
SHERPA_ASR_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
)
SHERPA_ASR_ARCHIVE_NAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"

# sherpa-onnx TTS: English VITS Piper model (GLaDOS voice, ~61 MB).
SHERPA_TTS_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "vits-piper-en_US-glados.tar.bz2"
)
SHERPA_TTS_ARCHIVE_NAME = "vits-piper-en_US-glados.tar.bz2"


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


def _ensure_sherpa_stt(root: Path) -> bool:
    """Download and prepare sherpa-onnx English streaming STT into sherpa_stt/.

    Returns True if ready, False on failure (but does not raise).
    """
    sherpa_dir = root / "sherpa_stt"
    tokens = sherpa_dir / "tokens.txt"
    encoder = sherpa_dir / "encoder.onnx"
    decoder = sherpa_dir / "decoder.onnx"
    joiner = sherpa_dir / "joiner.onnx"

    # If already present, nothing to do.
    if all(p.exists() for p in (tokens, encoder, decoder, joiner)):
        print(f"[info] sherpa-onnx STT assets already present under {sherpa_dir}")
        return True

    # Download archive into a temp location under root.
    tmp_dir = root / ".tmp_sherpa_asr"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / SHERPA_ASR_ARCHIVE_NAME

    try:
        if not archive_path.exists():
            _download_file(SHERPA_ASR_ARCHIVE_URL, archive_path, "sherpa-onnx English streaming ASR")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to download sherpa-onnx ASR archive: {exc}", file=sys.stderr)
        return False

    # Extract archive
    try:
        import tarfile

        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=tmp_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to extract sherpa-onnx ASR archive: {exc}", file=sys.stderr)
        return False

    # Find extracted model directory (assume single top-level folder)
    candidates = [p for p in tmp_dir.iterdir() if p.is_dir() and "zipformer-en-20M" in p.name]
    if not candidates:
        # Fallback: any directory with tokens.txt inside.
        for p in tmp_dir.iterdir():
            if p.is_dir() and (p / "tokens.txt").exists():
                candidates.append(p)
                break
    if not candidates:
        print(f"[error] Could not locate extracted sherpa-onnx model in {tmp_dir}", file=sys.stderr)
        return False
    model_root = candidates[0]

    # Locate encoder/decoder/joiner ONNX files (prefer int8 if multiple)
    def _pick(pattern: str) -> Path | None:
        matches = sorted(model_root.glob(pattern))
        return matches[0] if matches else None

    src_tokens = model_root / "tokens.txt"
    src_encoder = _pick("encoder*.onnx")
    src_decoder = _pick("decoder*.onnx")
    src_joiner = _pick("joiner*.onnx")

    if not (src_tokens.exists() and src_encoder and src_decoder and src_joiner):
        print(f"[error] Missing expected sherpa-onnx files under {model_root}", file=sys.stderr)
        return False

    sherpa_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Preparing sherpa-onnx STT assets under {sherpa_dir}")
    shutil.copy2(src_tokens, tokens)
    shutil.copy2(src_encoder, encoder)
    shutil.copy2(src_decoder, decoder)
    shutil.copy2(src_joiner, joiner)

    return True


def _ensure_sherpa_tts(root: Path) -> bool:
    """Download and prepare sherpa-onnx English TTS into sherpa_tts/.

    Uses the vits-piper-en_US-glados model (~61 MB).
    Returns True if ready, False on failure (but does not raise).
    """
    tts_dir = root / "sherpa_tts"
    model = tts_dir / "model.onnx"
    tokens = tts_dir / "tokens.txt"
    data_dir = tts_dir / "espeak-ng-data"

    if model.exists() and tokens.exists() and data_dir.exists():
        print(f"[info] sherpa-onnx TTS assets already present under {tts_dir}")
        return True

    tmp_dir = root / ".tmp_sherpa_tts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / SHERPA_TTS_ARCHIVE_NAME

    try:
        if not archive_path.exists():
            _download_file(SHERPA_TTS_ARCHIVE_URL, archive_path, "sherpa-onnx English TTS (vits-piper-en_US-glados)")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to download sherpa-onnx TTS archive: {exc}", file=sys.stderr)
        return False

    # Extract archive
    try:
        import tarfile

        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=tmp_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to extract sherpa-onnx TTS archive: {exc}", file=sys.stderr)
        return False

    # Expect directory vits-piper-en_US-glados/
    model_root = tmp_dir / "vits-piper-en_US-glados"
    if not model_root.exists():
        # Fallback: first dir with expected files
        for p in tmp_dir.iterdir():
            if p.is_dir() and (p / "tokens.txt").exists():
                model_root = p
                break

    src_model = model_root / "en_US-glados.onnx"
    src_tokens = model_root / "tokens.txt"
    src_data = model_root / "espeak-ng-data"

    if not (src_model.exists() and src_tokens.exists() and src_data.exists()):
        print(f"[error] Missing expected sherpa-onnx TTS files under {model_root}", file=sys.stderr)
        return False

    tts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Preparing sherpa-onnx TTS assets under {tts_dir}")
    shutil.copy2(src_model, model)
    shutil.copy2(src_tokens, tokens)

    # Copy espeak-ng-data directory (merge if already exists)
    dst_data = tts_dir / "espeak-ng-data"
    if dst_data.exists():
        shutil.rmtree(dst_data)
    shutil.copytree(src_data, dst_data)

    return True


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

    # sherpa-onnx STT assets (English, streaming Zipformer, ~20M params)
    sherpa_stt_ready = _ensure_sherpa_stt(root)

    # sherpa-onnx TTS assets (English VITS Piper, GLaDOS voice)
    sherpa_tts_ready = _ensure_sherpa_tts(root)

    # Final status summary (best-effort)
    try:
        emo_size = target_onnx.stat().st_size if target_onnx.exists() else 0
        kokoro_size = kokoro_onnx.stat().st_size if kokoro_onnx.exists() else 0
        voices_size = kokoro_voices.stat().st_size if kokoro_voices.exists() else 0
        print(f"[summary] Emotion ONNX: {target_onnx} ({emo_size} bytes)")
        print(f"[summary] Kokoro ONNX:  {kokoro_onnx} ({kokoro_size} bytes)")
        print(f"[summary] Kokoro voices:{kokoro_voices} ({voices_size} bytes)")
        if sherpa_stt_ready:
            print(f"[summary] sherpa-onnx STT: {root / 'sherpa_stt'}")
        if sherpa_tts_ready:
            print(f"[summary] sherpa-onnx TTS: {root / 'sherpa_tts'}")
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

