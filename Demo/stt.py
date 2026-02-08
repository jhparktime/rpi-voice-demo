"""Faster-Whisper STT wrapper."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# Warn when transcribe exceeds this (seconds); suggests CPU/thermal/memory check on RPi
TRANSCRIBE_SLOW_WARN_THRESHOLD = 10.0


def transcribe_faster_whisper(model: "WhisperModel", audio: np.ndarray, beam_size: int = 1) -> str:
    segments, _info = model.transcribe(
        audio,
        beam_size=beam_size,
        language="en",
        vad_filter=False,
    )
    texts = [seg.text.strip() for seg in segments if seg.text]
    return " ".join(texts).strip()
