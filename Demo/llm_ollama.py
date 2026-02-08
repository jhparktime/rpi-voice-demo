"""Ollama /api/generate: sync, stream, and async stream with TTS chunking."""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import requests

from . import audio_io
from . import text_utils
from . import tts_kokoro

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro


def generate_ollama(
    prompt: str,
    model: str,
    system: str,
    url: str,
    num_predict: int = 24,
    temperature: float = 0.3,
    stop: Optional[List[str]] = None,
    keep_alive: str = "10m",
    num_thread: int = 4,
    num_ctx: int = 256,
    num_batch: int = 16,
    max_sentences: int = 2,
    max_words: int = 36,
    timeout: int = 20,
) -> str:
    """Call Ollama /api/generate and return postprocessed reply (options aligned with run_brain)."""
    if not (prompt or "").strip():
        return ""
    stop = stop if stop is not None else ["\n"]
    options: dict = {
        "num_predict": num_predict,
        "temperature": temperature,
        "num_thread": num_thread,
        "num_ctx": num_ctx,
        "stop": stop,
    }
    if num_batch > 0:
        options["num_batch"] = num_batch
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "system": system or text_utils.OLLAMA_DEFAULT_SYSTEM,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        if res.status_code != 200:
            return f"(Ollama error: HTTP {res.status_code})"
        data = res.json()
        out = (data.get("response") or "").strip()
        return text_utils.postprocess_output(out, max_sentences=max_sentences, max_words=max_words)
    except requests.exceptions.Timeout:
        return "(Ollama error: timeout)"
    except Exception as e:
        return f"(Ollama error: {e})"


def _extract_flush_chunk(buffer: str, max_words: int) -> Tuple[str, str]:
    """
    Extract the first flushable chunk from buffer (sentence end or max_words).
    Returns (chunk_to_speak, remainder).
    """
    buffer = (buffer or "").strip()
    if not buffer:
        return "", ""

    # Sentence end: first . ! ? followed by space
    match = re.search(r"(?<=[.!?])\s+", buffer)
    if match:
        end = match.end()
        chunk = buffer[:end].strip()
        remainder = buffer[end:].lstrip()
        if chunk:
            return chunk, remainder

    # Buffer ends with sentence-ending punctuation (no trailing space)
    if buffer.rstrip().endswith((".", "!", "?")):
        return buffer.strip(), ""

    # No sentence end: flush up to max_words
    parts = buffer.split()
    if len(parts) >= max_words:
        chunk = " ".join(parts[:max_words])
        remainder = " ".join(parts[max_words:])
        return chunk, remainder

    return "", buffer


_AUDIO_SENTINEL: Tuple[Optional[np.ndarray], int] = (None, 0)


def stream_ollama_tts_chunks(
    prompt: str,
    model: str,
    system: str,
    url: str,
    keep_alive: str,
    num_thread: int,
    num_ctx: int,
    num_batch: int,
    num_predict: int,
    temperature: float,
    stop: List[str],
    max_words_per_chunk: int,
    tts: "Kokoro",
    voice: str,
    output_device: Optional[int],
    volume: float,
    trim_start: float,
    timeout: int = 20,
    first_play_timestamp: Optional[List[float]] = None,
) -> str:
    """
    Stream Ollama response (NDJSON), flush on sentence end or max_words, synthesize and play each chunk.
    Returns full response text or error string.
    """
    if not (prompt or "").strip():
        return ""
    stop = stop if stop else ["\n"]
    options: dict = {
        "num_predict": num_predict,
        "temperature": temperature,
        "num_thread": num_thread,
        "num_ctx": num_ctx,
        "stop": stop,
    }
    if num_batch > 0:
        options["num_batch"] = num_batch
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "system": system or text_utils.OLLAMA_DEFAULT_SYSTEM,
        "stream": True,
        "keep_alive": keep_alive,
        "options": options,
    }
    buffer = ""
    full_response: List[str] = []
    try:
        res = requests.post(url, json=payload, stream=True, timeout=timeout)
        if res.status_code != 200:
            return f"(Ollama error: HTTP {res.status_code})"
        for line in res.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_text = data.get("response") or ""
            if isinstance(chunk_text, str) and chunk_text:
                full_response.append(chunk_text)
                buffer += chunk_text
                while True:
                    to_speak, remainder = _extract_flush_chunk(buffer, max_words_per_chunk)
                    if not to_speak:
                        break
                    buffer = remainder
                    try:
                        tts_audio, tts_sr = tts_kokoro.synthesize_kokoro(tts, to_speak, voice)
                        if trim_start > 0.0:
                            tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, trim_start)
                        if first_play_timestamp is not None and len(first_play_timestamp) == 0:
                            first_play_timestamp.append(time.perf_counter())
                        audio_io.play_audio(tts_audio, tts_sr, output_device, volume=volume)
                    except Exception as e:  # noqa: BLE001
                        print(f"[error] TTS chunk failed: {e}", file=sys.stderr)
            if data.get("done"):
                break
        # Flush remainder
        if buffer.strip():
            try:
                tts_audio, tts_sr = tts_kokoro.synthesize_kokoro(tts, buffer.strip(), voice)
                if trim_start > 0.0:
                    tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, trim_start)
                if first_play_timestamp is not None and len(first_play_timestamp) == 0:
                    first_play_timestamp.append(time.perf_counter())
                audio_io.play_audio(tts_audio, tts_sr, output_device, volume=volume)
            except Exception as e:  # noqa: BLE001
                print(f"[error] TTS chunk failed: {e}", file=sys.stderr)
        return "".join(full_response).strip()
    except requests.exceptions.Timeout:
        return "(Ollama error: timeout)"
    except Exception as e:
        return f"(Ollama error: {e})"


def stream_ollama_tts_chunks_async(
    prompt: str,
    model: str,
    system: str,
    url: str,
    keep_alive: str,
    num_thread: int,
    num_ctx: int,
    num_batch: int,
    num_predict: int,
    temperature: float,
    stop: List[str],
    max_words_per_chunk: int,
    tts: "Kokoro",
    voice: str,
    output_device: Optional[int],
    volume: float,
    trim_start: float,
    timeout: int = 20,
    first_play_timestamp: Optional[List[float]] = None,
) -> str:
    """
    Stream Ollama, push text chunks to a queue; Synth thread synthesizes, Play thread plays.
    Play of chunk N overlaps with synthesis of chunk N+1 for seamless playback.
    Returns full response text or error string.
    """
    if not (prompt or "").strip():
        return ""
    stop = stop if stop else ["\n"]
    options: dict = {
        "num_predict": num_predict,
        "temperature": temperature,
        "num_thread": num_thread,
        "num_ctx": num_ctx,
        "stop": stop,
    }
    if num_batch > 0:
        options["num_batch"] = num_batch
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "system": system or text_utils.OLLAMA_DEFAULT_SYSTEM,
        "stream": True,
        "keep_alive": keep_alive,
        "options": options,
    }
    text_queue: queue.Queue[Optional[str]] = queue.Queue()
    audio_queue: queue.Queue[Tuple[Optional[np.ndarray], int]] = queue.Queue()
    synth_error: List[Optional[Exception]] = []
    play_error: List[Optional[Exception]] = []

    def synth_worker() -> None:
        try:
            while True:
                chunk_text = text_queue.get()
                if chunk_text is None:
                    audio_queue.put(_AUDIO_SENTINEL)
                    break
                try:
                    tts_audio, tts_sr = tts_kokoro.synthesize_kokoro(tts, chunk_text, voice)
                    if trim_start > 0.0:
                        tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, trim_start)
                    audio_queue.put((tts_audio, tts_sr))
                except Exception as e:  # noqa: BLE001
                    print(f"[error] TTS chunk failed: {e}", file=sys.stderr)
                    synth_error.append(e)
                    audio_queue.put(_AUDIO_SENTINEL)
                    break
        except Exception as e:  # noqa: BLE001
            synth_error.append(e)
            try:
                audio_queue.put(_AUDIO_SENTINEL)
            except Exception:  # noqa: S110
                pass

    def play_worker() -> None:
        try:
            while True:
                audio, sr = audio_queue.get()
                if audio is None:
                    break
                if first_play_timestamp is not None and len(first_play_timestamp) == 0:
                    first_play_timestamp.append(time.perf_counter())
                try:
                    audio_io.play_audio(audio, sr, output_device, volume=volume)
                except Exception as e:  # noqa: BLE001
                    print(f"[error] playback failed: {e}", file=sys.stderr)
                    play_error.append(e)
                    break
        except Exception as e:  # noqa: BLE001
            play_error.append(e)

    synth_thread = threading.Thread(target=synth_worker, daemon=False)
    play_thread = threading.Thread(target=play_worker, daemon=False)
    synth_thread.start()
    play_thread.start()

    buffer = ""
    full_response: List[str] = []
    try:
        res = requests.post(url, json=payload, stream=True, timeout=timeout)
        if res.status_code != 200:
            text_queue.put(None)
            synth_thread.join(timeout=5.0)
            play_thread.join(timeout=5.0)
            return f"(Ollama error: HTTP {res.status_code})"
        for line in res.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_text = data.get("response") or ""
            if isinstance(chunk_text, str) and chunk_text:
                full_response.append(chunk_text)
                buffer += chunk_text
                while True:
                    to_speak, remainder = _extract_flush_chunk(buffer, max_words_per_chunk)
                    if not to_speak:
                        break
                    buffer = remainder
                    text_queue.put(to_speak)
            if data.get("done"):
                break
        if buffer.strip():
            text_queue.put(buffer.strip())
        text_queue.put(None)
    except requests.exceptions.Timeout:
        text_queue.put(None)
        synth_thread.join(timeout=5.0)
        play_thread.join(timeout=5.0)
        return "(Ollama error: timeout)"
    except Exception as e:
        try:
            text_queue.put(None)
        except Exception:  # noqa: S110
            pass
        synth_thread.join(timeout=5.0)
        play_thread.join(timeout=5.0)
        return f"(Ollama error: {e})"

    synth_thread.join()
    play_thread.join()
    if synth_error:
        return f"(Ollama/TTS error: {synth_error[0]})"
    if play_error:
        return f"(playback error: {play_error[0]})"
    return "".join(full_response).strip()
