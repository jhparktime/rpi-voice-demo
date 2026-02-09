"""sherpa-onnx streaming STT backend using OnlineRecognizer + optional VAD.

Mirrors the official sherpa-onnx microphone example:
  - OnlineRecognizer.from_transducer with enable_endpoint_detection
  - Audio fed in small chunks with decode_stream per chunk
  - Endpoint detection for automatic turn boundary

Provides three recognition modes:
  1. transcribe_sherpa(audio, sr)             – feed a pre-recorded buffer in chunks
  2. stream_recognize_until_endpoint(...)      – live mic with auto endpoint (Phase 2)
  3. vad_stream_recognize_one(...)             – VAD + streaming for one utterance (Phase 3)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import sherpa_onnx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sherpa_onnx = None  # type: ignore[assignment]

_RECOGNIZER = None
_VAD = None

# Size of each chunk fed to accept_waveform (in seconds).
CHUNK_SECONDS = 0.1
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# OnlineRecognizer (streaming transducer)
# ---------------------------------------------------------------------------


def _init_recognizer() -> Optional[object]:
    """Lazy-initialize sherpa-onnx OnlineRecognizer (streaming transducer).

    Expects model files under sherpa_stt/ (or SHERPA_STT_DIR env var).
    """
    global _RECOGNIZER
    if _RECOGNIZER is not None:
        return _RECOGNIZER

    if sherpa_onnx is None:
        print("[sherpa] sherpa_onnx not installed; STT unavailable.", flush=True)
        return None

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
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=1,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
            decoding_method="greedy_search",
            provider="cpu",
            hotwords_file="",
            hotwords_score=1.5,
            blank_penalty=0.0,
        )
        print(f"[sherpa] OnlineRecognizer initialized from {model_dir}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] Failed to init recognizer: {exc}", flush=True)
        return None

    _RECOGNIZER = recognizer
    return recognizer


def get_recognizer() -> Optional[object]:
    """Return the lazily-initialized OnlineRecognizer (or None)."""
    return _init_recognizer()


# ---------------------------------------------------------------------------
# VAD (silero-vad via sherpa-onnx)
# ---------------------------------------------------------------------------


def _init_vad() -> Optional[object]:
    """Lazy-initialize sherpa-onnx VoiceActivityDetector (silero-vad).

    Expects silero_vad.onnx under sherpa_vad/ (or SHERPA_VAD_DIR env var).
    """
    global _VAD
    if _VAD is not None:
        return _VAD

    if sherpa_onnx is None:
        print("[sherpa-vad] sherpa_onnx not installed; VAD unavailable.", flush=True)
        return None

    root = Path(__file__).resolve().parent.parent
    vad_dir = Path(os.environ.get("SHERPA_VAD_DIR", root / "sherpa_vad"))
    vad_model = vad_dir / "silero_vad.onnx"
    if not vad_model.exists():
        print(f"[sherpa-vad] silero_vad.onnx not found at {vad_model}", flush=True)
        return None

    try:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_model)
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.5
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.window_size = 512  # 32ms at 16kHz
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        config.provider = "cpu"
        config.debug = False

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        print(f"[sherpa-vad] VoiceActivityDetector initialized from {vad_model}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-vad] Failed to init VAD: {exc}", flush=True)
        return None

    _VAD = vad
    return vad


def get_vad() -> Optional[object]:
    """Return the lazily-initialized VAD (or None)."""
    return _init_vad()


# ---------------------------------------------------------------------------
# Mode 1: transcribe a pre-recorded buffer (Phase 1 fallback)
# ---------------------------------------------------------------------------


def transcribe_sherpa(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a pre-recorded utterance by feeding it in small chunks.

    The audio is fed in CHUNK_SECONDS increments to mimic the streaming
    behaviour of the official microphone example.
    """
    recognizer = _init_recognizer()
    if recognizer is None:
        return ""

    wav = audio.astype(np.float32)
    if wav.size:
        max_abs = float(np.max(np.abs(wav)))
        print(f"[sherpa] input len={wav.size} max_abs={max_abs:.4f}", flush=True)
    else:
        print("[sherpa] empty waveform passed to transcribe_sherpa()", flush=True)
        return ""

    try:
        stream = recognizer.create_stream()

        chunk_size = int(CHUNK_SECONDS * sample_rate)
        offset = 0
        while offset < len(wav):
            end = min(offset + chunk_size, len(wav))
            chunk = wav[offset:end]
            stream.accept_waveform(sample_rate, chunk)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            offset = end

        # Append a short silence tail and signal end-of-stream.
        tail = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        text = getattr(result, "text", "") or ""
        print(f"[sherpa] raw result: {repr(text)}", flush=True)
        return text.strip()
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] transcribe error: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Mode 2: streaming mic + endpoint detection (Phase 2)
# ---------------------------------------------------------------------------


def stream_recognize_until_endpoint(
    input_device: Optional[int] = None,
    max_seconds: float = 15.0,
) -> Tuple[str, float]:
    """Open the microphone and stream-recognize until endpoint or max_seconds.

    Returns (recognized_text, elapsed_seconds).
    """
    import sounddevice as sd

    recognizer = _init_recognizer()
    if recognizer is None:
        return "", 0.0

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    stream = recognizer.create_stream()
    text = ""
    t0 = time.perf_counter()

    print("[sherpa] Listening... (speak now, endpoint will auto-detect)", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        ) as mic:
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= max_seconds:
                    print(f"[sherpa] max_seconds ({max_seconds}s) reached.", flush=True)
                    break

                samples, _ = mic.read(chunk_samples)
                samples = samples.reshape(-1)
                stream.accept_waveform(SAMPLE_RATE, samples)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                is_endpoint = recognizer.is_endpoint(stream)
                result = recognizer.get_result(stream)
                current_text = getattr(result, "text", "") or ""

                if is_endpoint and current_text.strip():
                    text = current_text.strip()
                    print(f"[sherpa] endpoint detected: {repr(text)}", flush=True)
                    recognizer.reset(stream)
                    break
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa] stream_recognize error: {exc}", flush=True)

    elapsed = time.perf_counter() - t0

    if not text:
        # Finalize: feed silence tail + input_finished
        try:
            tail = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, tail)
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            text = (getattr(result, "text", "") or "").strip()
        except Exception:
            pass

    print(f"[sherpa] final text: {repr(text)} ({elapsed:.2f}s)", flush=True)
    return text, elapsed


# ---------------------------------------------------------------------------
# Mode 3: VAD + streaming OnlineRecognizer — one utterance (Phase 3)
# ---------------------------------------------------------------------------


def vad_stream_recognize_one(
    input_device: Optional[int] = None,
    max_seconds: float = 30.0,
) -> Tuple[str, float]:
    """Listen with VAD; when a speech segment is detected and finishes,
    return the streaming-recognized text for that segment.

    The mic is opened at the start and closed after one utterance.
    Returns (recognized_text, elapsed_seconds).
    """
    import sounddevice as sd

    recognizer = _init_recognizer()
    vad = _init_vad()
    if recognizer is None:
        return "", 0.0
    if vad is None:
        print("[sherpa-vad] VAD not available, falling back to endpoint-only mode.", flush=True)
        return stream_recognize_until_endpoint(input_device, max_seconds)

    # silero-vad window size (must match config)
    window_size = 512
    speech_active = False
    stream = None
    text = ""
    t0 = time.perf_counter()

    print("[sherpa-vad] Listening... (VAD will detect speech automatically)", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        ) as mic:
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= max_seconds:
                    print(f"[sherpa-vad] max_seconds ({max_seconds}s) reached.", flush=True)
                    break

                data, _ = mic.read(window_size)
                samples = data.reshape(-1).astype(np.float32)

                vad.accept_waveform(samples)
                is_speech = vad.is_speech_detected()

                # Speech just started → create a fresh OnlineRecognizer stream
                if is_speech and not speech_active:
                    speech_active = True
                    stream = recognizer.create_stream()
                    print("[sherpa-vad] speech detected", flush=True)

                # Feed audio to recognizer while speech is active
                if speech_active and stream is not None:
                    stream.accept_waveform(SAMPLE_RATE, samples)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                # Check for complete speech segments from VAD
                while not vad.empty():
                    vad.pop()
                    if speech_active and stream is not None:
                        # Finalize this utterance
                        tail = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
                        stream.accept_waveform(SAMPLE_RATE, tail)
                        stream.input_finished()
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)

                        result = recognizer.get_result(stream)
                        text = (getattr(result, "text", "") or "").strip()
                        print(f"[sherpa-vad] utterance: {repr(text)}", flush=True)

                        stream = None
                        speech_active = False
                        # One utterance done → return
                        if text:
                            elapsed = time.perf_counter() - t0
                            return text, elapsed
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-vad] error: {exc}", flush=True)

    # If we timed out but had an active stream, try to get partial result
    if speech_active and stream is not None:
        try:
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            text = (getattr(result, "text", "") or "").strip()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    print(f"[sherpa-vad] final text: {repr(text)} ({elapsed:.2f}s)", flush=True)
    return text, elapsed
