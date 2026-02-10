"""
Sherpa-onnx STT -> Emotion + Intent routing -> LOCAL/CLOUD LLM -> sherpa-onnx VITS TTS demo (RPi).

STT modes (controlled by CLI flags):
  --streaming       (default) Stream mic to OnlineRecognizer with endpoint detection; Enter to start.
  --no-streaming    Fixed-duration recording + chunked transcription (Phase 1 fallback).
  --vad             Always-listening: VAD detects speech automatically, no Enter needed.

LOCAL: Ollama (e.g., smollm2:360m) with empathic prompt.
CLOUD: external HTTP LLM (if configured) with informational prompt.

The sLLM provides semantic fillers during CLOUD LLM latency so the conversation
feels natural even when the main answer takes a few seconds.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from . import audio_io
from . import cloud_llm
from . import llm_ollama
from . import llm_onnx
from . import stt_sherpa
from . import stt_tts_cli
from . import text_utils
from . import tts_sherpa
from .emotion import EmotionClassifierONNX, EmotionResult
from .intent_router import classify_intent_easy_or_complex
from . import router_anchors_runtime


ENABLE_EMOTION = os.environ.get("ENABLE_EMOTION", "1").strip() not in {"0", "false", "False", "no", "NO"}
ENABLE_INTENT_ROUTER = os.environ.get("ENABLE_INTENT_ROUTER", "1").strip() not in {"0", "false", "False", "no", "NO"}
ENABLE_CLOUD_FILLER = os.environ.get("ENABLE_CLOUD_FILLER", "0").strip() not in {"0", "false", "False", "no", "NO"}
FORCE_MODE = (os.environ.get("FORCE_MODE", "") or "").strip().upper()

SLOW_ASR_WARN_THRESHOLD = 10.0


# ── TTS helper ─────────────────────────────────────────────────────────────

def _synthesize_tts(tts: Any, voice: str, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
    """TTS helper: always use sherpa-onnx OfflineTts backend (tts/voice are unused)."""
    audio, sr = tts_sherpa.synthesize_sherpa_tts(text, speed=speed)
    if sr <= 0 or audio.size == 0:
        raise RuntimeError("sherpa-onnx TTS synthesis failed")
    return audio, sr


def _play_chunks_pipelined(
    chunks: List[str],
    tts: Any,
    voice: str,
    args: Any,
    filler_player: Optional[audio_io.AudioPlayer] = None,
) -> Tuple[List[np.ndarray], Optional[int]]:
    """Play text chunks with pipelined TTS to eliminate gaps.
    
    Producer thread generates TTS for all chunks in background.
    Main thread plays each chunk as soon as it's ready.
    First chunk waits for filler to finish.
    
    Returns (list of audio arrays, sample_rate).
    """
    if not chunks:
        return [], None
    
    tts_queue: Queue = Queue(maxsize=2)  # Buffer up to 2 chunks ahead
    audio_chunks = []
    sample_rate = None
    
    def _tts_producer():
        """Background thread: TTS all chunks and push to queue."""
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                t_start = time.perf_counter()
                chunk_audio, chunk_sr = _synthesize_tts(tts, voice, chunk)
                t_end = time.perf_counter()
                print(f"  [LATENCY] Chunk-{i+1} TTS: {t_end - t_start:.2f}s", flush=True)
                tts_queue.put((i, chunk_audio, chunk_sr))
            except Exception as exc:  # noqa: BLE001
                print(f"[error] TTS chunk {i+1} failed: {exc}", file=sys.stderr)
        tts_queue.put(None)  # Sentinel: all chunks done
    
    # Start producer thread
    producer = Thread(target=_tts_producer, daemon=True)
    producer.start()
    
    # Consumer (main thread): play chunks as they become ready
    first_chunk = True
    while True:
        item = tts_queue.get()
        if item is None:
            break
        
        chunk_idx, chunk_audio, chunk_sr = item
        
        # Track sample rate from first chunk
        if sample_rate is None:
            sample_rate = chunk_sr
        
        # First chunk: wait for filler to finish
        if first_chunk and filler_player and filler_player.is_playing():
            print("[Cloud] Waiting for filler to finish...", flush=True)
            filler_player.wait()
            first_chunk = False
        
        # Trim start only on first chunk
        if chunk_idx == 0 and args.trim_start > 0.0:
            chunk_audio = audio_io.trim_start_seconds(chunk_audio, chunk_sr, args.trim_start)
        
        # Play this chunk (blocking)
        print(f"  Playing chunk {chunk_idx+1}/{len(chunks)}...", flush=True)
        t_play_start = time.perf_counter()
        try:
            audio_io.play_audio(chunk_audio, chunk_sr, args.output_device, volume=args.volume)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] playback chunk {chunk_idx+1} failed: {exc}", file=sys.stderr)
        t_play_end = time.perf_counter()
        print(f"  [LATENCY] Chunk-{chunk_idx+1} Play: {t_play_end - t_play_start:.2f}s", flush=True)
        
        audio_chunks.append(chunk_audio)
    
    # Wait for producer thread to finish
    producer.join(timeout=5.0)
    
    return audio_chunks, sample_rate


# ── Turn handlers (ONNX LLM, Ollama stream, Ollama single, Brain router) ──

def _run_turn_onnx_llm(
    t0: float,
    args: Any,
    text: str,
    tts: Any,
    voice: str,
    onnx_model: Any,
    onnx_tokenizer: Any,
) -> None:
    """ONNX LLM -> synthesize -> play."""
    print("ONNX LLM...", flush=True)
    t_llm_start = time.perf_counter()
    reply = llm_onnx.generate_onnx_llm(
        prompt=text,
        system=text_utils.ONNX_DEFAULT_SYSTEM,
        model=onnx_model,
        tokenizer=onnx_tokenizer,
        max_new_tokens=args.onnx_max_new_tokens,
        temperature=args.onnx_temperature,
        max_sentences=2,
        max_words=36,
    )
    t_llm_end = time.perf_counter()
    print(f"[time] onnx-llm: {t_llm_end - t_llm_start:.2f}s", flush=True)
    tts_text = reply.strip() if reply and not reply.startswith("(ONNX LLM") else text
    if reply and not reply.startswith("(ONNX LLM"):
        print(f"[LLM] {reply}", flush=True)
    elif reply:
        print(reply, flush=True)
    t2b = time.perf_counter()
    print("Synthesizing...", flush=True)
    try:
        tts_audio, tts_sr = _synthesize_tts(tts, voice, tts_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] TTS failed: {exc}", file=sys.stderr)
        return
    t3 = time.perf_counter()
    print(f"[time] synthesize: {t3 - t2b:.2f}s", flush=True)
    if args.trim_start > 0.0:
        tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
    print("Playing...", flush=True)
    try:
        audio_io.play_audio(tts_audio, tts_sr, args.output_device, volume=args.volume)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] playback failed: {exc}", file=sys.stderr)
    t4 = time.perf_counter()
    print(f"[time] play: {t4 - t3:.2f}s (total: {t4 - t0:.2f}s)", flush=True)


def _run_turn_ollama_stream(t0: float, args: Any, text: str, tts: Any, voice: str) -> None:
    """Ollama stream (async or sync) -> TTS per chunk."""
    print("Ollama (stream)...", flush=True)
    t_ollama_start = time.perf_counter()
    first_play_ts: List[float] = []
    if args.ollama_stream_async:
        reply = llm_ollama.stream_ollama_tts_chunks_async(
            prompt=text,
            model=args.ollama_model,
            system=args.ollama_system or text_utils.OLLAMA_DEFAULT_SYSTEM,
            url=args.ollama_url,
            keep_alive=args.ollama_keep_alive,
            num_thread=args.ollama_num_thread,
            num_ctx=args.ollama_num_ctx,
            num_batch=args.ollama_num_batch,
            num_predict=args.ollama_num_predict,
            temperature=args.ollama_temperature,
            stop=["\n"],
            max_words_per_chunk=args.ollama_stream_max_words,
            tts=tts,
            voice=voice,
            output_device=args.output_device,
            volume=args.volume,
            trim_start=args.trim_start,
            timeout=20,
            first_play_timestamp=first_play_ts,
        )
    else:
        reply = llm_ollama.stream_ollama_tts_chunks(
            prompt=text,
            model=args.ollama_model,
            system=args.ollama_system or text_utils.OLLAMA_DEFAULT_SYSTEM,
            url=args.ollama_url,
            keep_alive=args.ollama_keep_alive,
            num_thread=args.ollama_num_thread,
            num_ctx=args.ollama_num_ctx,
            num_batch=args.ollama_num_batch,
            num_predict=args.ollama_num_predict,
            temperature=args.ollama_temperature,
            stop=["\n"],
            max_words_per_chunk=args.ollama_stream_max_words,
            tts=tts,
            voice=voice,
            output_device=args.output_device,
            volume=args.volume,
            trim_start=args.trim_start,
            timeout=20,
            first_play_timestamp=first_play_ts,
        )
    t_ollama_end = time.perf_counter()
    if first_play_ts:
        print(f"[time] first TTS: {first_play_ts[0] - t_ollama_start:.2f}s", flush=True)
    print(f"[time] ollama+tts stream: {t_ollama_end - t_ollama_start:.2f}s", flush=True)
    if reply and not reply.startswith("(Ollama error"):
        print(f"[LLM] {reply}", flush=True)
    elif reply:
        print(reply, flush=True)
    print(f"[time] total: {t_ollama_end - t0:.2f}s", flush=True)


def _run_turn_ollama_or_direct(t0: float, args: Any, text: str, tts: Any, voice: str) -> None:
    """Ollama single reply -> synthesize -> play, or direct TTS."""
    if args.ollama:
        print("Ollama...", flush=True)
        t_ollama_start = time.perf_counter()
        reply = llm_ollama.generate_ollama(
            prompt=text,
            model=args.ollama_model,
            system=args.ollama_system or text_utils.OLLAMA_DEFAULT_SYSTEM,
            url=args.ollama_url,
            num_predict=args.ollama_num_predict,
            temperature=args.ollama_temperature,
            stop=["\n"],
            keep_alive=args.ollama_keep_alive,
            num_thread=args.ollama_num_thread,
            num_ctx=args.ollama_num_ctx,
            num_batch=args.ollama_num_batch,
            max_sentences=2,
            max_words=36,
            timeout=20,
        )
        t_ollama_end = time.perf_counter()
        print(f"[time] ollama: {t_ollama_end - t_ollama_start:.2f}s", flush=True)
        tts_text = reply.strip() if reply and not reply.startswith("(Ollama error") else text
        if reply:
            print(f"[LLM] {reply}", flush=True)
    else:
        tts_text = text

    t2b = time.perf_counter()
    print("Synthesizing...", flush=True)
    try:
        tts_audio, tts_sr = _synthesize_tts(tts, voice, tts_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] TTS failed: {exc}", file=sys.stderr)
        return
    t3 = time.perf_counter()
    print(f"[time] synthesize: {t3 - t2b:.2f}s", flush=True)
    if args.trim_start > 0.0:
        tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
    print("Playing...", flush=True)
    try:
        audio_io.play_audio(tts_audio, tts_sr, args.output_device, volume=args.volume)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] playback failed: {exc}", file=sys.stderr)
    t4 = time.perf_counter()
    print(f"[time] play: {t4 - t3:.2f}s (total: {t4 - t0:.2f}s)", flush=True)


# ── Filler generation ──────────────────────────────────────────────────────

def _generate_filler_ollama(
    args: Any,
    emotion_label: Optional[str],
    timeout: float = 3.0,
) -> str:
    """Generate short Ollama filler phrase for Cloud waiting period.
    
    Returns empty string if Ollama unavailable, filler disabled, or generation fails.
    """
    # Check if filler is enabled (CLI flag or env var)
    filler_enabled = getattr(args, 'cloud_filler', ENABLE_CLOUD_FILLER)
    if not filler_enabled:
        print("[Filler] Skipped (disabled by --no-cloud-filler)", flush=True)
        return ""
    
    if not args.ollama:
        print("[Filler] Skipped (Ollama not enabled; use --ollama flag)", flush=True)
        return ""
    
    print("[Filler] Generating Ollama filler...", flush=True)
    system = text_utils.build_cloud_filler_system_prompt(emotion_label)
    try:
        filler = llm_ollama.generate_ollama(
            prompt="Generate a short bridge phrase.",  # Explicit prompt for filler
            model=args.ollama_model,
            system=system,
            url=args.ollama_url,
            num_predict=15,  # Short response
            temperature=0.7,
            stop=["\n"],
            keep_alive=args.ollama_keep_alive,
            num_thread=args.ollama_num_thread,
            num_ctx=128,  # Minimal context
            num_batch=args.ollama_num_batch,
            max_sentences=1,
            max_words=10,
            timeout=timeout,
        )
        result = filler.strip() if filler and not filler.startswith("(") else ""
        if not result:
            print(f"[Filler] Generation returned empty/error: {filler[:100] if filler else 'empty'}", flush=True)
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[Filler] Generation failed: {exc}", flush=True, file=sys.stderr)
        return ""


# ── Brain: Emotion + Intent routing → LOCAL / CLOUD LLM → TTS ─────────────

def _run_turn_brain(
    t0: float,
    args: Any,
    text: str,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> Optional[str]:
    """Route using emotion + intent, call LOCAL or CLOUD LLM, then TTS.

    For CLOUD requests the sLLM first generates a quick semantic filler
    ("Let me look that up…") so the user hears something immediately while
    the heavier CLOUD model is processing.

    Returns the assistant reply text (or None if no LLM reply was generated).
    """
    route_mode = "LOCAL"
    emotion_result: Optional[EmotionResult] = None

    # Build context-enriched prompt for LLM (multi-turn history)
    llm_prompt = text
    if conversation is not None and len(conversation) > 0:
        llm_prompt = conversation.format_prompt_with_context(text)

    if FORCE_MODE in {"LOCAL", "CLOUD"}:
        route_mode = FORCE_MODE
        if ENABLE_EMOTION and emotion_classifier and emotion_classifier.available:
            try:
                emotion_result = emotion_classifier.predict(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[Emotion] error: {exc}", file=sys.stderr)
                emotion_result = None
    else:
        def _run_emotion() -> Optional[EmotionResult]:
            if not (ENABLE_EMOTION and emotion_classifier and emotion_classifier.available):
                return None
            try:
                return emotion_classifier.predict(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[Emotion] error: {exc}", file=sys.stderr)
                return None

        def _run_intent() -> str:
            if not ENABLE_INTENT_ROUTER:
                return "LOCAL"
            try:
                rr = router_anchors_runtime.route_local_or_cloud(text)
                print(
                    f"[Route] mode={rr.mode} conf={rr.confidence:.2f} "
                    f"(local={rr.best_local:.2f}, cloud={rr.best_cloud:.2f}, Δ={rr.delta:.2f}) | "
                    f"anchor={rr.matched_anchor!r}",
                    flush=True,
                )
                return rr.mode
            except Exception as exc:  # noqa: BLE001
                print(f"[Intent] anchors-router error: {exc}", file=sys.stderr)
                try:
                    return classify_intent_easy_or_complex(text)
                except Exception as exc2:  # noqa: BLE001
                    print(f"[Intent] simple classifier error: {exc2}", file=sys.stderr)
                    return "LOCAL"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_emotion = ex.submit(_run_emotion)
            fut_intent = ex.submit(_run_intent)
            emotion_result = fut_emotion.result()
            route_mode = fut_intent.result()

    emotion_label = emotion_result.label if emotion_result is not None else None
    assistant_reply: Optional[str] = None

    # ── CLOUD path ──────────────────────────────────────────────────────
    if route_mode == "CLOUD":
        print("\n" + "="*60)
        print("[CLOUD MODE] Starting (with Ollama filler)...")
        print("="*60 + "\n", flush=True)

        system = text_utils.build_cloud_system_prompt(emotion_label)

        # Parallel: Ollama filler + Cloud LLM
        def _call_filler():
            return _generate_filler_ollama(args, emotion_label, timeout=3.0)

        def _call_cloud():
            return cloud_llm.call_cloud_llm(
                prompt=llm_prompt,
                system=system,
                timeout=20.0,
            )

        t_parallel_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_filler = executor.submit(_call_filler)
            fut_cloud = executor.submit(_call_cloud)

            filler_text = fut_filler.result()
            t_filler_ready = time.perf_counter()

            # Filler TTS+재생을 별도 스레드에서 실행
            filler_player: Optional[audio_io.AudioPlayer] = None
            fut_filler_tts: Optional[concurrent.futures.Future] = None
            if filler_text:
                print(f"[FILLER] {filler_text}")
                print(
                    f"[LATENCY] Filler generation: {t_filler_ready - t_parallel_start:.2f}s\n",
                    flush=True,
                )

                def _run_filler_tts() -> Optional[audio_io.AudioPlayer]:
                    t_tts_start = time.perf_counter()
                    try:
                        filler_audio, filler_sr = _synthesize_tts(tts, voice, filler_text)
                        t_tts_end = time.perf_counter()
                        print(
                            f"[LATENCY] Filler TTS: {t_tts_end - t_tts_start:.2f}s",
                            flush=True,
                        )
                        player = audio_io.AudioPlayer(
                            filler_audio,
                            filler_sr,
                            device=args.output_device,
                            volume=args.volume,
                        )
                        player.start()
                        print("[FILLER] Playing...", flush=True)
                        return player
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[FILLER] TTS/playback failed: {exc}",
                            flush=True,
                            file=sys.stderr,
                        )
                        return None

                fut_filler_tts = executor.submit(_run_filler_tts)

            # Wait for Cloud result (병렬로 Filler TTS 수행 중)
            cloud_reply = fut_cloud.result()
            t_cloud_ready = time.perf_counter()

            # Filler TTS 완료 후 player 핸들을 받아온다 (재생은 이미 시작됨)
            if fut_filler_tts is not None:
                try:
                    filler_player = fut_filler_tts.result()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[FILLER] TTS future failed: {exc}",
                        flush=True,
                        file=sys.stderr,
                    )
                    filler_player = None

        print(f"[LATENCY] Cloud LLM call: {t_cloud_ready - t_parallel_start:.2f}s")

        cloud_text = cloud_reply.strip() if cloud_reply and not cloud_reply.startswith("(") else text
        assistant_reply = cloud_text

        if cloud_reply:
            print(f"[LLM-CLOUD] {cloud_reply}\n", flush=True)

        # Split and play with pipelined TTS
        chunks = text_utils.split_into_chunks(cloud_text)
        if not chunks:
            chunks = [cloud_text]

        print(f"[CLOUD] Streaming {len(chunks)} chunk(s)...\n", flush=True)

        t_tts_start = time.perf_counter()
        _, _ = _play_chunks_pipelined(chunks, tts, voice, args, filler_player=filler_player)
        t_tts_end = time.perf_counter()

        print(f"\n[LATENCY] Total TTS+Play: {t_tts_end - t_tts_start:.2f}s")
        print(f"[LATENCY] End-to-end: {t_tts_end - t0:.2f}s")
        print("="*60 + "\n", flush=True)

        return assistant_reply

    # ── LOCAL path ──────────────────────────────────────────────────────
    system = text_utils.build_local_system_prompt(emotion_label)

    if args.ollama and args.ollama_stream:
        args.ollama_system = system
        _run_turn_ollama_stream(t0, args, llm_prompt, tts, voice)
        # Streaming reply is logged but not easily captured; return None for now.
        return None

    reply = ""
    if args.ollama:
        print("Ollama (LOCAL)...", flush=True)
        t_ollama_start = time.perf_counter()
        reply = llm_ollama.generate_ollama(
            prompt=llm_prompt,
            model=args.ollama_model,
            system=system,
            url=args.ollama_url,
            num_predict=args.ollama_num_predict,
            temperature=args.ollama_temperature,
            stop=["\n"],
            keep_alive=args.ollama_keep_alive,
            num_thread=args.ollama_num_thread,
            num_ctx=args.ollama_num_ctx,
            num_batch=args.ollama_num_batch,
            max_sentences=2,
            max_words=36,
            timeout=20,
        )
        t_ollama_end = time.perf_counter()
        print(f"[time] ollama(local): {t_ollama_end - t_ollama_start:.2f}s", flush=True)

    tts_text = reply.strip() if reply and not reply.startswith("(") else text
    if reply:
        print(f"[LLM-LOCAL] {reply}", flush=True)
    assistant_reply = tts_text

    t2b = time.perf_counter()
    print("Synthesizing...", flush=True)
    try:
        tts_audio, tts_sr = _synthesize_tts(tts, voice, tts_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] TTS failed: {exc}", file=sys.stderr)
        return assistant_reply
    t3 = time.perf_counter()
    print(f"[time] synthesize: {t3 - t2b:.2f}s", flush=True)
    if args.trim_start > 0.0:
        tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
    print("Playing...", flush=True)
    try:
        audio_io.play_audio(tts_audio, tts_sr, args.output_device, volume=args.volume)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] playback failed: {exc}", file=sys.stderr)
    t4 = time.perf_counter()
    print(f"[time] play: {t4 - t3:.2f}s (total: {t4 - t0:.2f}s)", flush=True)
    return assistant_reply


def _run_turn_brain_sentence(
    t0: float,
    args: Any,
    text: str,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[int]]:
    """Sentence-streaming version: returns (reply_text, tts_audio, tts_sr) without playing.
    
    Simplified brain path for sentence-by-sentence processing:
    - No streaming modes (those don't fit sentence boundaries)
    - Returns audio for caller to play interruptibly
    """
    route_mode = "LOCAL"
    emotion_result: Optional[EmotionResult] = None

    # Build context-enriched prompt for LLM
    llm_prompt = text
    if conversation is not None and len(conversation) > 0:
        llm_prompt = conversation.format_prompt_with_context(text)

    if FORCE_MODE in {"LOCAL", "CLOUD"}:
        route_mode = FORCE_MODE
        if ENABLE_EMOTION and emotion_classifier and emotion_classifier.available:
            try:
                emotion_result = emotion_classifier.predict(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[Emotion] error: {exc}", file=sys.stderr)
    else:
        def _run_emotion() -> Optional[EmotionResult]:
            if not (ENABLE_EMOTION and emotion_classifier and emotion_classifier.available):
                return None
            try:
                return emotion_classifier.predict(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[Emotion] error: {exc}", file=sys.stderr)
                return None

        def _run_intent() -> str:
            if not ENABLE_INTENT_ROUTER:
                return "LOCAL"
            try:
                rr = router_anchors_runtime.route_local_or_cloud(text)
                print(
                    f"[Route] mode={rr.mode} conf={rr.confidence:.2f} "
                    f"(local={rr.best_local:.2f}, cloud={rr.best_cloud:.2f}, Δ={rr.delta:.2f}) | "
                    f"anchor={rr.matched_anchor!r}",
                    flush=True,
                )
                return rr.mode
            except Exception as exc:  # noqa: BLE001
                print(f"[Route] error: {exc}", file=sys.stderr)
                return "LOCAL"

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_emotion = pool.submit(_run_emotion)
            fut_intent = pool.submit(_run_intent)
            emotion_result = fut_emotion.result()
            route_mode = fut_intent.result()

    emotion_label = emotion_result.label if emotion_result else "neutral"

    # ── CLOUD path ──────────────────────────────────────────────────────────
    if route_mode == "CLOUD":
        print("\n" + "="*60)
        print("[CLOUD MODE] Starting (with Ollama filler)...")
        print("="*60 + "\n", flush=True)

        system = text_utils.build_cloud_system_prompt(emotion_label)

        # Parallel: Ollama filler + Cloud LLM
        def _call_filler():
            return _generate_filler_ollama(args, emotion_label, timeout=3.0)

        def _call_cloud():
            return cloud_llm.call_cloud_llm(
                prompt=llm_prompt,
                system=system,
                timeout=10.0,
            )

        t_parallel_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_filler = executor.submit(_call_filler)
            fut_cloud = executor.submit(_call_cloud)

            filler_text = fut_filler.result()
            t_filler_ready = time.perf_counter()

            # Filler TTS+재생을 별도 스레드에서 실행
            filler_player: Optional[audio_io.AudioPlayer] = None
            fut_filler_tts: Optional[concurrent.futures.Future] = None
            if filler_text:
                print(f"[FILLER] {filler_text}")
                print(
                    f"[LATENCY] Filler generation: {t_filler_ready - t_parallel_start:.2f}s\n",
                    flush=True,
                )

                def _run_filler_tts() -> Optional[audio_io.AudioPlayer]:
                    t_tts_start = time.perf_counter()
                    try:
                        filler_audio, filler_sr = _synthesize_tts(tts, voice, filler_text)
                        t_tts_end = time.perf_counter()
                        print(
                            f"[LATENCY] Filler TTS: {t_tts_end - t_tts_start:.2f}s",
                            flush=True,
                        )
                        player = audio_io.AudioPlayer(
                            filler_audio,
                            filler_sr,
                            device=args.output_device,
                            volume=args.volume,
                        )
                        player.start()
                        print("[FILLER] Playing...", flush=True)
                        return player
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[FILLER] TTS/playback failed: {exc}",
                            flush=True,
                            file=sys.stderr,
                        )
                        return None

                fut_filler_tts = executor.submit(_run_filler_tts)

            # Wait for Cloud result (병렬로 Filler TTS 수행 중)
            cloud_reply = fut_cloud.result()
            t_cloud_ready = time.perf_counter()

            # Filler TTS 완료 후 player 핸들을 받아온다 (재생은 이미 시작됨)
            if fut_filler_tts is not None:
                try:
                    filler_player = fut_filler_tts.result()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[FILLER] TTS future failed: {exc}",
                        flush=True,
                        file=sys.stderr,
                    )
                    filler_player = None

        print(f"[LATENCY] Cloud LLM call: {t_cloud_ready - t_parallel_start:.2f}s")

        cloud_text = cloud_reply.strip() if cloud_reply and not cloud_reply.startswith("(") else text

        if cloud_reply:
            print(f"[LLM-CLOUD] {cloud_reply}\n", flush=True)

        # Split and play with pipelined TTS
        chunks = text_utils.split_into_chunks(cloud_text)
        if not chunks:
            chunks = [cloud_text]

        print(f"[CLOUD] Streaming {len(chunks)} chunk(s)...\n", flush=True)

        t_tts_start = time.perf_counter()
        combined_audio_chunks, sample_rate = _play_chunks_pipelined(chunks, tts, voice, args, filler_player=filler_player)
        t_tts_end = time.perf_counter()

        print(f"\n[LATENCY] Total TTS+Play: {t_tts_end - t_tts_start:.2f}s")
        print("="*60 + "\n", flush=True)

        # Combine all chunks into single audio array for return
        if combined_audio_chunks and sample_rate:
            combined_audio = np.concatenate(combined_audio_chunks)
            return cloud_text, combined_audio, sample_rate
        else:
            return cloud_text, None, None

    # ── LOCAL path ──────────────────────────────────────────────────────────
    system = text_utils.build_local_system_prompt(emotion_label)

    reply = ""
    if args.ollama:
        print("Ollama (LOCAL)...", flush=True)
        t_ollama_start = time.perf_counter()
        reply = llm_ollama.generate_ollama(
            prompt=llm_prompt,
            model=args.ollama_model,
            system=system,
            url=args.ollama_url,
            num_predict=args.ollama_num_predict,
            temperature=args.ollama_temperature,
            stop=["\n"],
            keep_alive=args.ollama_keep_alive,
            num_thread=args.ollama_num_thread,
            num_ctx=args.ollama_num_ctx,
            num_batch=args.ollama_num_batch,
            max_sentences=2,
            max_words=36,
            timeout=20,
        )
        t_ollama_end = time.perf_counter()
        print(f"[time] ollama(local): {t_ollama_end - t_ollama_start:.2f}s", flush=True)

    tts_text = reply.strip() if reply and not reply.startswith("(") else text
    if reply:
        print(f"[LLM-LOCAL] {reply}", flush=True)

    t2b = time.perf_counter()
    print("Synthesizing...", flush=True)
    try:
        tts_audio, tts_sr = _synthesize_tts(tts, voice, tts_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] TTS failed: {exc}", file=sys.stderr)
        return tts_text, None, None
    t3 = time.perf_counter()
    print(f"[time] synthesize: {t3 - t2b:.2f}s", flush=True)
    
    if args.trim_start > 0.0:
        tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
    
    return tts_text, tts_audio, tts_sr


# ── Warmup helpers ─────────────────────────────────────────────────────────

def _warmup(args: Any, emotion_classifier: Optional[EmotionClassifierONNX]) -> None:
    """Warm-up all major components: STT, TTS, Emotion, Intent Router, LLMs."""
    print("[Warmup] Initializing all modules...", flush=True)
    
    # 1. STT Recognizer (sherpa-onnx OnlineRecognizer)
    print("[Warmup] Loading STT recognizer...", flush=True)
    stt_sherpa.get_recognizer()
    if args.vad:
        print("[Warmup] Loading VAD...", flush=True)
        stt_sherpa.get_vad()
    
    # 2. TTS (sherpa-onnx OfflineTts)
    print("[Warmup] Loading TTS...", flush=True)
    try:
        _, _ = tts_sherpa.synthesize_sherpa_tts("Warming up text to speech.", speed=1.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[Warmup] TTS warning: {exc}", file=sys.stderr)
    
    # 3. Intent Router (SentenceTransformer + embeddings)
    if ENABLE_INTENT_ROUTER:
        print("[Warmup] Loading intent router...", flush=True)
        try:
            _ = router_anchors_runtime.route_local_or_cloud("Warmup for routing.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Intent router warning: {exc}", file=sys.stderr)
    
    # 4. Emotion Classifier (ONNX BERT)
    if ENABLE_EMOTION and emotion_classifier is not None and emotion_classifier.available:
        print("[Warmup] Loading emotion classifier...", flush=True)
        try:
            _ = emotion_classifier.predict("Hello, just warming up.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Emotion warning: {exc}", file=sys.stderr)
    
    # 5. Ollama LLM (if enabled)
    if args.ollama:
        print(f"[Warmup] Warming up Ollama ({args.ollama_model})...", flush=True)
        try:
            _ = llm_ollama.generate_ollama(
                prompt="Hi",
                model=args.ollama_model,
                system=text_utils.OLLAMA_DEFAULT_SYSTEM,
                url=args.ollama_url,
                num_predict=1,
                temperature=0.0,
                keep_alive=args.ollama_keep_alive,
                num_thread=args.ollama_num_thread,
                num_ctx=args.ollama_num_ctx,
                num_batch=args.ollama_num_batch,
                max_sentences=1,
                max_words=2,
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Ollama warning: {exc}", file=sys.stderr)
    
    # 6. Cloud LLM (if configured: OpenAI, Gemini, or custom URL)
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    cloud_url = (os.environ.get("CLOUD_LLM_URL") or "").strip()
    if openai_key or gemini_key or cloud_url:
        print("[Warmup] Testing cloud LLM connection...", flush=True)
        try:
            _ = cloud_llm.call_cloud_llm(
                prompt="Warmup request.",
                system="You are a friendly, reliable assistant. This is a warmup call; a short reply is fine.",
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Cloud LLM warning: {exc}", file=sys.stderr)
    
    print("[Warmup] All modules ready!", flush=True)


# ── Main entry point ───────────────────────────────────────────────────────

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = stt_tts_cli.parse_args(argv)
    if not 0.0 <= args.volume <= 1.0:
        print("[error] --volume must be between 0.0 and 1.0.", file=sys.stderr)
        return 1
    if args.trim_start < 0.0:
        print("[error] --trim-start must be >= 0.", file=sys.stderr)
        return 1

    tts = None
    voice = "sherpa_default"

    # ── Determine STT mode ──
    if args.sentence_streaming:
        stt_mode = "sentence_streaming"
    elif args.vad:
        stt_mode = "vad"
    elif args.streaming:
        stt_mode = "streaming"
    else:
        stt_mode = "fixed"

    stt_mode_label = {
        "sentence_streaming": f"sentence-by-sentence streaming ({args.sentence_silence}s silence = sentence boundary)",
        "vad": "VAD always-listening + streaming OnlineRecognizer",
        "streaming": "streaming OnlineRecognizer with endpoint detection (Enter to start)",
        "fixed": f"fixed {args.record_seconds}s recording + chunked transcription",
    }

    print("Voice demo (sherpa-onnx)")
    print(f"- STT: {stt_mode_label[stt_mode]}")
    print("- TTS: sherpa-onnx VITS (vits-coqui-en-ljspeech)")
    print(f"- volume: {args.volume}")
    print(f"- trim-start: {args.trim_start}s")
    if stt_mode != "fixed":
        print(f"- max-listen-seconds: {args.max_listen_seconds}")
    else:
        print(f"- record seconds: {args.record_seconds}")

    # LLM config display
    if args.ollama:
        print(f"- ollama: {args.ollama_model} @ {args.ollama_url}")
        print(f"- ollama-stream: {args.ollama_stream} (async: {args.ollama_stream_async}, max-words/chunk: {args.ollama_stream_max_words})")
        filler_enabled = getattr(args, 'cloud_filler', ENABLE_CLOUD_FILLER)
        print(f"- cloud-filler: {'enabled' if filler_enabled else 'disabled'} (Ollama bridge during Cloud LLM latency)")

    stt_tts_cli.print_config(args, voice)

    # Emotion classifier
    emotion_classifier: Optional[EmotionClassifierONNX] = None
    if ENABLE_EMOTION:
        model_dir = Path(__file__).resolve().parent.parent / "emotion_onnx_int8"
        emotion_classifier = EmotionClassifierONNX(str(model_dir))
        if not getattr(emotion_classifier, "available", False):
            emotion_classifier = None

    # Multi-turn conversation buffer
    conversation: Optional[text_utils.ConversationBuffer] = None
    if args.max_turns > 0:
        conversation = text_utils.ConversationBuffer(max_turns=args.max_turns)
        print(f"- multi-turn context: last {args.max_turns} turns")

    # Warm-up
    _warmup(args, emotion_classifier)

    if stt_mode == "sentence_streaming":
        print("Sentence-streaming mode: Press Enter to start session, speak naturally with pauses. Ctrl+C to quit.")
    elif stt_mode == "vad":
        print("Always-listening mode (VAD). Say something — no Enter needed. Ctrl+C to quit.")
    else:
        print("Press Enter to record, or Ctrl+C to quit.")

    # ── Main loop ──────────────────────────────────────────────────────
    if stt_mode == "sentence_streaming":
        _main_loop_sentence_streaming(args, tts, voice, emotion_classifier, conversation)
    elif stt_mode == "vad":
        _main_loop_vad(args, tts, voice, emotion_classifier, conversation)
    elif stt_mode == "streaming":
        _main_loop_streaming(args, tts, voice, emotion_classifier, conversation)
    else:
        _main_loop_fixed(args, tts, voice, emotion_classifier, conversation)

    return 0


# ── Loop: VAD always-listening (Phase 3) ──────────────────────────────────

def _main_loop_vad(
    args: Any,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> None:
    """Always-listening loop using VAD + streaming OnlineRecognizer."""
    while True:
        try:
            t0 = time.perf_counter()
            text, elapsed = stt_sherpa.vad_stream_recognize_one(
                input_device=args.input_device,
                max_seconds=args.max_listen_seconds,
            )
        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[error] VAD recognition failed: {exc}", file=sys.stderr)
            continue

        print(f"[time] stt(vad): {elapsed:.2f}s", flush=True)

        if not text:
            print("[info] no speech detected.")
            continue

        print(f"[ASR] {text}")
        reply = _run_turn_brain(t0, args, text, tts, voice, emotion_classifier, conversation)
        if conversation is not None and reply:
            conversation.add_turn(text, reply)
        print("Done.", flush=True)
        gc.collect()


# ── Loop: streaming mic + endpoint detection (Phase 2) ────────────────────

def _main_loop_streaming(
    args: Any,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> None:
    """Enter-triggered loop with streaming OnlineRecognizer + endpoint detection."""
    while True:
        try:
            input(">>> ")
        except KeyboardInterrupt:
            print("\nBye.")
            break

        t0 = time.perf_counter()

        try:
            text, elapsed = stt_sherpa.stream_recognize_until_endpoint(
                input_device=args.input_device,
                max_seconds=args.max_listen_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] streaming recognition failed: {exc}", file=sys.stderr)
            continue

        print(f"[time] stt(streaming): {elapsed:.2f}s", flush=True)

        if not text:
            print("[info] no speech detected / empty text.")
            continue

        print(f"[ASR] {text}")
        reply = _run_turn_brain(t0, args, text, tts, voice, emotion_classifier, conversation)
        if conversation is not None and reply:
            conversation.add_turn(text, reply)
        print("Done. Press Enter to record again.", flush=True)
        gc.collect()


# ── Loop: fixed-duration recording (Phase 1 fallback) ─────────────────────

def _main_loop_fixed(
    args: Any,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> None:
    """Enter-triggered loop with fixed-duration recording + chunked transcription."""
    while True:
        try:
            input(">>> ")
        except KeyboardInterrupt:
            print("\nBye.")
            break

        print(f"Recording... ({args.record_seconds}s)", flush=True)
        t0 = time.perf_counter()
        try:
            audio = audio_io.record_audio(args.record_seconds, audio_io.SAMPLE_RATE, args.input_device)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] recording failed: {exc}", file=sys.stderr)
            continue
        t1 = time.perf_counter()
        print(f"[time] record: {t1 - t0:.2f}s", flush=True)

        if not len(audio):
            print("[warn] empty audio, retry.")
            continue

        print("Transcribing...", flush=True)
        try:
            text = stt_sherpa.transcribe_sherpa(audio, audio_io.SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] sherpa-onnx ASR failed: {exc}", file=sys.stderr)
            continue
        t2 = time.perf_counter()
        transcribe_sec = t2 - t1
        print(f"[time] transcribe: {transcribe_sec:.2f}s", flush=True)
        if transcribe_sec >= SLOW_ASR_WARN_THRESHOLD:
            print(
                f"[warn] transcribe very slow ({transcribe_sec:.1f}s); check CPU/thermal/memory",
                file=sys.stderr,
                flush=True,
            )

        if not text:
            print("[info] no speech detected / empty text.")
            continue

        print(f"[ASR] {text}")
        reply = _run_turn_brain(t0, args, text, tts, voice, emotion_classifier, conversation)
        if conversation is not None and reply:
            conversation.add_turn(text, reply)
        print("Done. Press Enter to record again.", flush=True)
        gc.collect()


def _main_loop_sentence_streaming(
    args: Any,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> None:
    """Sentence-by-sentence streaming loop with interruptible TTS.
    
    User presses Enter once → continuous listening:
    - Each sentence (0.8s silence) → brain → TTS (interruptible)
    - New sentence stops previous TTS immediately
    - Ctrl+C to exit
    """
    while True:
        try:
            input(">>> ")
        except KeyboardInterrupt:
            print("\nBye.")
            break

        current_player: Optional[audio_io.AudioPlayer] = None
        sentence_count = 0
        session_start = time.perf_counter()

        print(f"[sentence-streaming] Session started (speak naturally, {args.sentence_silence:.1f}s pauses = sentence boundaries)", flush=True)

        try:
            for sentence_text, elapsed in stt_sherpa.stream_recognize_sentences(
                input_device=args.input_device,
                sentence_silence_threshold=args.sentence_silence,
                max_total_seconds=args.max_listen_seconds,
            ):
                sentence_count += 1
                print(f"[ASR-sentence-{sentence_count}] {sentence_text}", flush=True)

                # Stop any active TTS from previous sentence
                if current_player is not None and current_player.is_playing():
                    print("[sentence-streaming] Interrupting previous TTS", flush=True)
                    current_player.stop()
                    current_player = None

                # Process this sentence through brain (non-blocking TTS synthesis)
                t0 = time.perf_counter()
                reply_text, tts_audio, tts_sr = _run_turn_brain_sentence(
                    t0, args, sentence_text, tts, voice, emotion_classifier, conversation
                )

                # Add to conversation context
                if conversation is not None and reply_text:
                    conversation.add_turn(sentence_text, reply_text)

                # Start interruptible TTS playback
                if tts_audio is not None and tts_sr is not None:
                    print("Playing (interruptible)...", flush=True)
                    try:
                        current_player = audio_io.play_audio_interruptible(
                            tts_audio, tts_sr, args.output_device, volume=args.volume
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[error] playback failed: {exc}", file=sys.stderr)
                        current_player = None

                t_total = time.perf_counter() - t0
                print(f"[time] sentence-{sentence_count} total: {t_total:.2f}s", flush=True)
                gc.collect()

        except KeyboardInterrupt:
            print("\n[sentence-streaming] Session interrupted", flush=True)
            if current_player is not None:
                current_player.stop()

        session_elapsed = time.perf_counter() - session_start
        print(f"[sentence-streaming] Session ended. Processed {sentence_count} sentence(s) in {session_elapsed:.1f}s", flush=True)
        print("Press Enter to start new session.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
