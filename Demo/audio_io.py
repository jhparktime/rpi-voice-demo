"""Audio I/O: record, resample, trim, play (sounddevice + scipy)."""
from __future__ import annotations

import threading
import time
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


# ---------------------------------------------------------------------------
# Interruptible playback for sentence-streaming mode
# ---------------------------------------------------------------------------


class AudioPlayer:
    """Non-blocking audio player with interrupt capability.
    
    Plays audio in a background thread and allows stopping mid-playback.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        device: Optional[int] = None,
        volume: float = 1.0,
    ) -> None:
        self._audio = audio
        self._sample_rate = sample_rate
        self._device = device
        self._volume = volume
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(target=self._play, daemon=False)
        self._started = False

    def start(self) -> None:
        """Start playback in background thread."""
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        """Stop playback immediately."""
        self._stop_flag.set()
        sd.stop()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for playback to finish (or timeout)."""
        if self._started:
            self._thread.join(timeout=timeout)

    def is_playing(self) -> bool:
        """Check if playback thread is still active."""
        return self._started and self._thread.is_alive()

    def _play(self) -> None:
        """Internal playback worker."""
        try:
            audio, sr = resample_audio(self._audio, self._sample_rate, PLAYBACK_SAMPLE_RATE)
            if self._volume != 1.0:
                audio = (audio * self._volume).astype(np.float32)
            
            # Play non-blocking so we can check stop_flag
            sd.play(audio, samplerate=sr, device=self._device, blocking=False)
            
            # Poll until done or stopped
            while sd.get_stream().active:
                if self._stop_flag.is_set():
                    sd.stop()
                    break
                time.sleep(0.01)  # 10ms poll interval
        except Exception as exc:
            print(f"[audio] playback error: {exc}")


def play_audio_interruptible(
    audio: np.ndarray,
    sample_rate: int,
    device: Optional[int] = None,
    volume: float = 1.0,
) -> AudioPlayer:
    """Play audio with interrupt capability (non-blocking).
    
    Returns an AudioPlayer handle. Call .start() to begin, .stop() to interrupt.
    """
    player = AudioPlayer(audio, sample_rate, device, volume)
    player.start()
    return player
