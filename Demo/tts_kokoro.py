"""Kokoro ONNX TTS wrapper."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro


def synthesize_kokoro(tts: "Kokoro", text: str, voice: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
    wav, sr = tts.create(text=text, voice=voice, speed=speed, lang="en-us")
    return wav.astype(np.float32), sr
