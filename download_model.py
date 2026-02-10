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
# We hard-code the same archive that was used in the previous Raspberry Pi
# experiments and extract/copy the needed files.
SHERPA_ASR_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
)
SHERPA_ASR_ARCHIVE_NAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"

SHERPA_TTS_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "vits-piper-en_US-amy-low.tar.bz2"
)
SHERPA_TTS_ARCHIVE_NAME = "vits-piper-en_US-amy-low.tar.bz2"

# (참고) Piper 모델은 espeak-ng-data가 모델 압축파일 안에 포함된 경우가 많지만,
# 기존 로직과의 호환성을 위해 아래 data 파일은 그대로 둬도 무방합니다.
ESPEAK_DATA_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "espeak-ng-data.tar.bz2"
)
ESPEAK_DATA_ARCHIVE_NAME = "espeak-ng-data.tar.bz2"

# VAD: silero-vad ONNX model for speech activity detection.
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)


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

    Uses the vits-piper-en_US-amy-low model (Ultra-fast, Optimized for RPi 5)
    plus a shared espeak-ng-data directory.
    Returns True if ready, False on failure (but does not raise).
    """
    tts_dir = root / "sherpa_tts"
    model = tts_dir / "model.onnx"
    tokens = tts_dir / "tokens.txt"
    data_dir = tts_dir / "espeak-ng-data"

    # 이미 설치되어 있으면 패스
    if model.exists() and tokens.exists() and data_dir.exists():
        print(f"[info] sherpa-onnx TTS assets already present under {tts_dir}")
        return True

    tmp_dir = root / ".tmp_sherpa_tts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tts_archive_path = tmp_dir / SHERPA_TTS_ARCHIVE_NAME
    espeak_archive_path = tmp_dir / ESPEAK_DATA_ARCHIVE_NAME

    try:
        # 모델 다운로드
        if not tts_archive_path.exists():
            _download_file(
                SHERPA_TTS_ARCHIVE_URL,
                tts_archive_path,
                "sherpa-onnx TTS (vits-piper-en_US-amy-low)",
            )
        # espeak-ng-data 다운로드
        if not espeak_archive_path.exists():
            _download_file(
                ESPEAK_DATA_ARCHIVE_URL,
                espeak_archive_path,
                "sherpa-onnx TTS espeak-ng-data",
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to download sherpa-onnx TTS archives: {exc}", file=sys.stderr)
        return False

    # 압축 해제
    try:
        import tarfile

        with tarfile.open(tts_archive_path, "r:bz2") as tar:
            tar.extractall(path=tmp_dir)
        with tarfile.open(espeak_archive_path, "r:bz2") as tar:
            tar.extractall(path=tmp_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to extract sherpa-onnx TTS archives: {exc}", file=sys.stderr)
        return False

    # [수정된 부분] 폴더 이름을 정확히 'vits-piper-en_US-amy-low'로 지정
    model_root = tmp_dir / "vits-piper-en_US-amy-low"
    
    # 혹시 폴더명이 다를 경우를 대비한 자동 찾기 로직 (안전장치)
    if not model_root.exists():
        for p in tmp_dir.iterdir():
            if p.is_dir() and (p / "tokens.txt").exists() and list(p.glob("*.onnx")):
                model_root = p
                break

    if not model_root.exists():
        print(f"[error] Could not locate sherpa-onnx TTS model directory in {tmp_dir}", file=sys.stderr)
        return False

    # ONNX 파일 찾기 (보통 en_US-amy-low.onnx 이름임)
    onnx_candidates = sorted(model_root.glob("*.onnx"))
    if not onnx_candidates:
        print(f"[error] No ONNX model found under {model_root}", file=sys.stderr)
        return False
    src_model = onnx_candidates[0]
    src_tokens = model_root / "tokens.txt"

    # espeak-ng-data 찾기
    espeak_candidates = []
    for p in tmp_dir.rglob("espeak-ng-data"):
        if p.is_dir():
            espeak_candidates.append(p)
            break
    
    # Piper 모델의 경우 모델 폴더 안에 espeak-ng-data가 있을 수도 있음
    if not espeak_candidates:
         if (model_root / "espeak-ng-data").exists():
             espeak_candidates.append(model_root / "espeak-ng-data")

    if not espeak_candidates:
        print(f"[error] espeak-ng-data directory not found under {tmp_dir}", file=sys.stderr)
        return False
    src_data = espeak_candidates[0]

    # 최종 확인 및 복사
    if not (src_model.exists() and src_tokens.exists() and src_data.exists()):
        print(f"[error] Missing expected sherpa-onnx TTS files under {model_root}", file=sys.stderr)
        return False

    tts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Preparing sherpa-onnx TTS assets under {tts_dir}")
    
    # [중요] 여기서 이름을 'model.onnx'로 통일해서 복사해줍니다.
    # 따라서 tts_sherpa.py에서는 경로를 항상 './sherpa_tts/model.onnx'로 쓰면 됩니다.
    shutil.copy2(src_model, model)
    shutil.copy2(src_tokens, tokens)

    dst_data = tts_dir / "espeak-ng-data"
    if dst_data.exists():
        shutil.rmtree(dst_data)
    shutil.copytree(src_data, dst_data)

    return True


def _ensure_sherpa_vad(root: Path) -> bool:
    """Download silero_vad.onnx for sherpa-onnx VAD into sherpa_vad/.

    Returns True if ready, False on failure (but does not raise).
    """
    vad_dir = root / "sherpa_vad"
    vad_model = vad_dir / "silero_vad.onnx"

    if vad_model.exists():
        print(f"[info] silero_vad.onnx already present at {vad_model}")
        return True

    try:
        _download_file(SILERO_VAD_URL, vad_model, "silero_vad.onnx (VAD)")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to download silero_vad.onnx: {exc}", file=sys.stderr)
        return False

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

    # sherpa-onnx TTS assets (English VITS LJSpeech, via vits-coqui-en-ljspeech)
    sherpa_tts_ready = _ensure_sherpa_tts(root)

    # sherpa-onnx VAD (silero-vad ONNX) for always-listening mode
    sherpa_vad_ready = _ensure_sherpa_vad(root)

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
        if sherpa_vad_ready:
            print(f"[summary] sherpa-onnx VAD: {root / 'sherpa_vad' / 'silero_vad.onnx'}")
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

