"""sherpa-onnx based TTS backend for baseline mode."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import sherpa_onnx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _TTS = None
else:
    _TTS: Optional["sherpa_onnx.OfflineTts"] = None  # type: ignore[name-defined]


def _init_tts() -> Optional["sherpa_onnx.OfflineTts"]:  # type: ignore[name-defined]
    """Lazy-initialize sherpa-onnx OfflineTts using a VITS English model.

    Expects files under:

        sherpa_tts/
          model.onnx
          tokens.txt
          espeak-ng-data/

    You can override the directory with SHERPA_TTS_DIR env var.
    """
    global _TTS
    if _TTS is not None:
        return _TTS

    try:
        sherpa_onnx  # type: ignore[name-defined]
    except Exception:
        return None

    root = Path(__file__).resolve().parent.parent
    base_dir = Path(os.environ.get("SHERPA_TTS_DIR", root / "sherpa_tts"))
    model_path = base_dir / "model.onnx"
    tokens_path = base_dir / "tokens.txt"
    data_dir = base_dir / "espeak-ng-data"

    if not (model_path.exists() and tokens_path.exists() and data_dir.exists()):
        print(f"[sherpa-tts] TTS assets not found under {base_dir}; falling back.", flush=True)
        return None

    try:
        tts_config = sherpa_onnx.OfflineTtsConfig(  # type: ignore[attr-defined]
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    data_dir=str(data_dir),
                ),
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(),
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(),
                kitten=sherpa_onnx.OfflineTtsKittenModelConfig(),
                provider="cpu",
                debug=False,
                num_threads=1,
            ),
            rule_fsts="",
            max_num_sentences=1,
        )
        if not tts_config.validate():
            print("[sherpa-tts] Invalid TTS config; falling back.", flush=True)
            return None

        tts = sherpa_onnx.OfflineTts(tts_config)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-tts] Failed to init OfflineTts: {exc}", flush=True)
        return None

    _TTS = tts
    return tts


def synthesize_sherpa_tts(text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
    """Synthesize speech for the given text.

    Returns (audio, sample_rate). On failure, returns (empty_array, 0).
    """
    tts = _init_tts()
    if tts is None:
        return np.zeros(0, dtype=np.float32), 0

    try:
        audio = tts.generate(text, sid=0, speed=speed)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-tts] generate() error: {exc}", flush=True)
        return np.zeros(0, dtype=np.float32), 0

    samples = np.asarray(audio.samples, dtype=np.float32)
    return samples, int(audio.sample_rate)

