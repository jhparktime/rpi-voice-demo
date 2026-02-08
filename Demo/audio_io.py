"""Audio I/O: record, resample, trim, play (sounddevice + scipy)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy import signal

SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 16000  # ReSpeaker 4 Mic Array default output rate


def record_audio(seconds: float, sample_rate: int, device: Optional[int]) -> np.ndarray:
    frames = int(seconds * sample_rate)
    if frames <= 0:
        raise ValueError("record duration must be positive")
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio[:, 0].copy()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Resample to target_sr so playback device accepts it (e.g. 44100 Hz)."""
    if orig_sr == target_sr:
        return audio, target_sr
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples).astype(np.float32)
    return resampled, target_sr


def trim_start_seconds(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    """Trim the first `seconds` from the audio (reduces TTS lead-in)."""
    if seconds <= 0 or not len(audio):
        return audio
    drop = int(seconds * sample_rate)
    if drop >= len(audio):
        return audio[:1].copy()  # keep at least 1 sample
    return audio[drop:].astype(np.float32)


def play_audio(audio: np.ndarray, sample_rate: int, device: Optional[int], volume: float = 1.0) -> None:
    audio, sr = resample_audio(audio, sample_rate, PLAYBACK_SAMPLE_RATE)
    if volume != 1.0:
        audio = (audio * volume).astype(np.float32)
    sd.play(audio, samplerate=sr, device=device, blocking=True)
