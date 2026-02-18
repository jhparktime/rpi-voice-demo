"""sherpa-onnx streaming STT backend using OnlineRecognizer + optional VAD.

Mirrors the official sherpa-onnx microphone example:
  - OnlineRecognizer.from_transducer with enable_endpoint_detection
  - Audio fed in small chunks with decode_stream per chunk
  - Endpoint detection for automatic turn boundary

Provides four recognition modes:
  1. transcribe_sherpa(audio, sr)             – feed a pre-recorded buffer in chunks
  2. stream_recognize_until_endpoint(...)      – live mic with auto endpoint (Phase 2)
  3. vad_stream_recognize_one(...)             – VAD + streaming for one utterance (Phase 3)
  4. stream_recognize_sentences(...)           – sentence-by-sentence streaming with 1.5s silence
"""
from __future__ import annotations

import os
import time
from pathlib import Path
import sys
from typing import Callable, Generator, Optional, Tuple

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

    def _env_float(name: str, default: float) -> float:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _env_int(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    threshold = _env_float("SHERPA_VAD_THRESHOLD", 0.5)
    min_silence = _env_float("SHERPA_VAD_MIN_SILENCE", 0.5)
    min_speech = _env_float("SHERPA_VAD_MIN_SPEECH", 0.25)
    window_size = _env_int("SHERPA_VAD_WINDOW_SIZE", 512)
    buffer_seconds = _env_int("SHERPA_VAD_BUFFER_SECONDS", 30)

    try:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_model)
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = min_silence
        config.silero_vad.min_speech_duration = min_speech
        config.silero_vad.window_size = window_size  # 32ms at 16kHz when 512
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        config.provider = "cpu"
        config.debug = False

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=buffer_seconds)
        print(
            f"[sherpa-vad] VoiceActivityDetector initialized from {vad_model} "
            f"(thr={threshold}, min_speech={min_speech}, min_silence={min_silence}, win={window_size})",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-vad] Failed to init VAD: {exc}", flush=True)
        return None

    _VAD = vad
    return vad


def get_vad() -> Optional[object]:
    """Return the lazily-initialized VAD (or None)."""
    return _init_vad()


def prime_microphone_input(
    input_device: Optional[int] = None,
    seconds: float = 0.8,
) -> bool:
    """Open mic once and discard a short buffer to stabilize first-turn capture."""
    if seconds <= 0:
        return False

    try:
        import sounddevice as sd
    except Exception:
        return False

    frames = max(1, int(seconds * SAMPLE_RATE))
    window = 512
    consumed = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        ) as mic:
            while consumed < frames:
                n = min(window, frames - consumed)
                mic.read(n)
                consumed += n
        print(f"[sherpa] mic primed ({seconds:.2f}s)", flush=True)
        return True
    except Exception as exc:
        print(f"[sherpa] mic prime skipped: {exc}", flush=True)
        return False


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
        chunk_count = 0
        while offset < len(wav):
            end = min(offset + chunk_size, len(wav))
            chunk = wav[offset:end]
            stream.accept_waveform(sample_rate, chunk)
            
            decode_count = 0
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
                decode_count += 1
            
            chunk_count += 1
            # Log partial result every 10 chunks (~1 second)
            if chunk_count % 10 == 0:
                partial_result = recognizer.get_result(stream)
                partial_text = str(partial_result).strip()
                if partial_text:
                    print(f"[sherpa-debug] partial @{chunk_count}chunks: {repr(partial_text)}", flush=True)
            
            offset = end

        print(f"[sherpa] processed {chunk_count} chunks, finalizing...", flush=True)

        # Append a short silence tail and signal end-of-stream.
        tail = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        text = str(result).strip()
        print(f"[sherpa] raw result: {repr(text)}", flush=True)
        return text
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
    warmup_chunks = 4  # ~0.4s at CHUNK_SECONDS=0.1
    text = ""
    t0 = time.perf_counter()

    print("[sherpa] Listening... (speak now, endpoint will auto-detect)", flush=True)

    # Prime decoder path with short silence to reduce first-token clipping.
    try:
        prime = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
        stream.accept_waveform(SAMPLE_RATE, prime)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
    except Exception:
        pass

    last_logged_text = ""
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        ) as mic:
            try:
                print("[sherpa] warming up mic...", flush=True)
                for _ in range(warmup_chunks):
                    mic.read(chunk_samples)
                print("[sherpa] ready, detecting speech...", flush=True)
            except Exception:
                print("[sherpa] warm-up skipped due to error.", flush=True)

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
                # sherpa-onnx OnlineRecognizer returns a result object that has __str__
                current_text = str(result).strip()

                # Debug: log partial results when they change
                if current_text and current_text != last_logged_text:
                    print(f"[sherpa-debug] partial: {repr(current_text)}", flush=True)
                    last_logged_text = current_text

                if is_endpoint:
                    text = current_text
                    if text:
                        print(f"[sherpa] endpoint detected: {repr(text)}", flush=True)
                    else:
                        print("[sherpa] endpoint detected but text is empty", flush=True)
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
            text = str(result).strip()
            if text:
                print(f"[sherpa] finalized text: {repr(text)}", flush=True)
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

    # Keep ASR stream always running. VAD is used only for utterance boundary.
    window_size = 512
    warmup_windows = 5  # ~0.16s at 16kHz
    speech_active = False
    text = ""
    stream = recognizer.create_stream()
    t0 = time.perf_counter()

    print("[sherpa-vad] Listening... (VAD will detect speech automatically)", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        ) as mic:
            try:
                print("[sherpa-vad] warming up mic...", flush=True)
                for _ in range(warmup_windows):
                    mic.read(window_size)
                print("[sherpa-vad] ready, detecting speech...", flush=True)
            except Exception:
                print("[sherpa-vad] warm-up skipped due to error.", flush=True)

            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= max_seconds:
                    print(f"[sherpa-vad] max_seconds ({max_seconds}s) reached.", flush=True)
                    break

                data, _ = mic.read(window_size)
                samples = data.reshape(-1).astype(np.float32)

                # ASR always-on path: collect/decode continuously.
                stream.accept_waveform(SAMPLE_RATE, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                # VAD only determines turn boundary.
                vad.accept_waveform(samples)
                is_speech = vad.is_speech_detected()
                if is_speech and not speech_active:
                    speech_active = True
                    print("[sherpa-vad] speech detected", flush=True)

                while not vad.empty():
                    vad.pop()
                    if not speech_active:
                        continue

                    # End of one VAD utterance: finalize current stream snapshot.
                    tail = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
                    stream.accept_waveform(SAMPLE_RATE, tail)
                    stream.input_finished()
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    result = recognizer.get_result(stream)
                    text = str(result).strip()
                    print(f"[sherpa-vad] utterance: {repr(text)}", flush=True)
                    if text:
                        elapsed = time.perf_counter() - t0
                        return text, elapsed

                    # Prepare next segment even if current segment was empty.
                    speech_active = False
                    stream = recognizer.create_stream()
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-vad] error: {exc}", flush=True)

    # Timed out: finalize whatever has been collected.
    try:
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream)
        text = str(result).strip()
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    print(f"[sherpa-vad] final text: {repr(text)} ({elapsed:.2f}s)", flush=True)
    return text, elapsed


# ---------------------------------------------------------------------------
# Mode 4: sentence-by-sentence streaming (Phase 4)
# ---------------------------------------------------------------------------


def stream_recognize_sentences(
    input_device: Optional[int] = None,
    sentence_silence_threshold: float = 1.5,
    max_total_seconds: float = 120.0,
    on_speech_start: Optional[Callable[[], None]] = None,
) -> Generator[Tuple[str, float], None, None]:
    """Stream mic continuously, yield each sentence when 1.5s silence detected.

    This enables natural multi-sentence conversations:
    - User speaks sentence 1 → pause 1.5s → yield sentence 1
    - User continues with sentence 2 → pause 1.5s → yield sentence 2
    - Continues until max_total_seconds or external stop

    Yields (sentence_text, elapsed_since_start) tuples.
    If on_speech_start is provided, it is called once when non-empty text first appears
    in an active utterance segment (useful to interrupt assistant playback).
    """
    import sounddevice as sd

    recognizer = _init_recognizer()
    if recognizer is None:
        print("[sherpa] Recognizer not available for sentence streaming.", flush=True)
        return

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    stream = recognizer.create_stream()
    
    t_session_start = time.perf_counter()
    t_last_text_change = t_session_start
    last_text = ""
    speech_started = False
    
    print(f"[sherpa-sentence] Listening... (sentence boundary: {sentence_silence_threshold}s silence)", flush=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        ) as mic:
            # Warm-up: discard first 3 chunks to avoid cutting off first words
            print("[sherpa-sentence] warming up mic...", flush=True)
            for _ in range(3):
                mic.read(chunk_samples)
            print("[sherpa-sentence] ready, listening...", flush=True)
            
            while True:
                elapsed_total = time.perf_counter() - t_session_start
                if elapsed_total >= max_total_seconds:
                    print(f"[sherpa-sentence] session timeout ({max_total_seconds}s)", flush=True)
                    break

                # Read and feed audio chunk
                samples, _ = mic.read(chunk_samples)
                samples = samples.reshape(-1)
                stream.accept_waveform(SAMPLE_RATE, samples)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                result = recognizer.get_result(stream)
                current_text = str(result).strip()

                # Check if text changed (new words detected)
                if current_text != last_text:
                    if current_text and not last_text:
                        # Speech started
                        print(f"[sherpa-sentence] speech started", flush=True)
                        if on_speech_start is not None and not speech_started:
                            try:
                                on_speech_start()
                            except Exception as exc:  # noqa: BLE001
                                print(f"[sherpa-sentence] speech-start callback error: {exc}", file=sys.stderr, flush=True)
                            speech_started = True
                    elif current_text:
                        # Text updated
                        print(f"[sherpa-sentence] partial: {repr(current_text)}", flush=True)
                    
                    last_text = current_text
                    t_last_text_change = time.perf_counter()
                    continue

                # Text hasn't changed — check silence duration
                silence_duration = time.perf_counter() - t_last_text_change
                
                if silence_duration >= sentence_silence_threshold and current_text:
                    # Sentence boundary detected!
                    sentence = current_text
                    elapsed = time.perf_counter() - t_session_start
                    print(f"[sherpa-sentence] sentence confirmed ({silence_duration:.2f}s silence): {repr(sentence)}", flush=True)
                    
                    # Yield this sentence
                    yield sentence, elapsed
                    
                    # Reset stream for next sentence
                    recognizer.reset(stream)
                    last_text = ""
                    speech_started = False
                    t_last_text_change = time.perf_counter()

    except GeneratorExit:
        print("[sherpa-sentence] generator stopped by caller", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[sherpa-sentence] error: {exc}", flush=True)
