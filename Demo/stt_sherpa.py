"""sherpa-onnx based STT backend for baseline mode."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sherpa_onnx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _RECOGNIZER = None
else:
    _RECOGNIZER = None


def _init_recognizer() -> Optional[object]:
    """Lazy-initialize sherpa-onnx OfflineRecognizer for chunked STT.

    Expects model files under:

        sherpa_stt/
          tokens.txt
          encoder.onnx
          decoder.onnx
          joiner.onnx

    You can override the directory with SHERPA_STT_DIR env var.
    """
    global _RECOGNIZER
    if _RECOGNIZER is not None:
        return _RECOGNIZER

    root = Path(__file__).resolve().parent.parent
    model_dir = Path(os.environ.get("SHERPA_STT_DIR", root / "sherpa_stt"))
    tokens = model_dir / "tokens.txt"
    encoder = model_dir / "encoder.onnx"
    decoder = model_dir / "decoder.onnx"
    joiner = model_dir / "joiner.onnx"
    if not (tokens.exists() and encoder.exists() and decoder.exists() and joiner.exists()):
        print(f"[sherpa] STT models not found under {model_dir}; falling back.", flush=True)
        return None

    try:
        # Use OfflineRecognizer.from_transducer for non-streaming, chunked STT.
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(  # type: ignore[attr-defined]
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            lm="",
            lm_scale=0.0,
            lodr_fst="",
            lodr_scale=0.0,
            decoding_method="greedy_search",
            hotwords_file="",
            hotwords_score=1.5,
            modeling_unit="",
            bpe_vocab="",
            blank_penalty=0.0,
            debug=False,
        )
        print(f"[sherpa] OfflineRecognizer initialized from {model_dir}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] Failed to init recognizer: {exc}", flush=True)
        return None

    _RECOGNIZER = recognizer
    return recognizer


def transcribe_sherpa(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a single utterance using sherpa-onnx OnlineRecognizer.

    Falls back to empty string if recognizer is not available.
    """
    recognizer = _init_recognizer()
    if recognizer is None:
        return ""

    # sherpa-onnx expects float32 mono waveform
    wav = audio.astype(np.float32)
    # Light-weight debugging to help diagnose empty-text issues on devices.
    # 이 로그는 문제가 없으면 크게 부담을 주지 않습니다.
    if wav.size:
        max_abs = float(np.max(np.abs(wav)))
        print(f"[sherpa] input len={wav.size} max_abs={max_abs:.4f}", flush=True)
    else:
        print("[sherpa] empty waveform passed to transcribe_sherpa()", flush=True)

    try:
        # OfflineRecognizer API: create a stream, accept the full waveform once,
        # then decode all streams in a batch.
        stream = recognizer.create_stream()  # type: ignore[attr-defined]
        stream.accept_waveform(sample_rate, wav)  # type: ignore[attr-defined]

        recognizer.decode_streams([stream])  # type: ignore[attr-defined]

        # Offline stream exposes a .result field with .text
        result = getattr(stream, "result", None)
        text = getattr(result, "text", "") if result is not None else ""
        print(f"[sherpa] raw result: {repr(text)}", flush=True)
        return text.strip()
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] transcribe error: {exc}", flush=True)
        return ""

