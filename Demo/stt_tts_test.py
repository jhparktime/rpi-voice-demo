"""Sherpa-onnx STT -> router -> Gemini LLM -> sherpa-onnx VITS TTS demo (RPi)."""
from __future__ import annotations

import gc
import os
import sys
import time
import re
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
from . import short_long_router


ENABLE_EMOTION = os.environ.get("ENABLE_EMOTION", "1").strip() not in {"0", "false", "False", "no", "NO"}
ENABLE_INTENT_ROUTER = os.environ.get("ENABLE_INTENT_ROUTER", "1").strip() not in {"0", "false", "False", "no", "NO"}
ENABLE_CLOUD_FILLER = os.environ.get("ENABLE_CLOUD_FILLER", "0").strip() not in {"0", "false", "False", "no", "NO"}
FORCE_MODE = (os.environ.get("FORCE_MODE", "") or "").strip().upper()

SLOW_ASR_WARN_THRESHOLD = 10.0
_GEMINI_TEMPLATE_SHORT = "gemini_short.txt"
_GEMINI_TEMPLATE_LONG = "gemini_long.txt"
_GEMINI_PROMPT_CACHE: dict[str, str] = {}
_FILLER_ONNX_MODEL: Any = None
_FILLER_ONNX_TOKENIZER: Any = None
_FILLER_ONNX_MODEL_ID = ""


def _load_prompt_template(filename: str, fallback: str) -> str:
    cache_key = filename
    if cache_key in _GEMINI_PROMPT_CACHE:
        return _GEMINI_PROMPT_CACHE[cache_key]
    prompt_path = Path(__file__).resolve().parent / "prompts" / filename
    if prompt_path.exists():
        try:
            text = prompt_path.read_text(encoding="utf-8")
        except Exception:
            text = fallback
        if not text.strip():
            text = fallback
    else:
        text = fallback
    _GEMINI_PROMPT_CACHE[cache_key] = text
    return text


def _build_gemini_prompt(mode: str, context: str) -> str:
    template = _load_prompt_template(
        _GEMINI_TEMPLATE_SHORT if mode == "SHORT" else _GEMINI_TEMPLATE_LONG,
        fallback=(
            "You are a concise voice assistant.\n"
            "Context:\n{{CONTEXT}}\n\n"
            "Reply using the current user request from the context."
            if mode == "SHORT"
            else
            "You are a clear and concise voice assistant.\n"
            "Context:\n{{CONTEXT}}\n\n"
            "Start with a 1-2 sentence short answer, then add concise details."
        ),
    )
    if "{{CONTEXT}}" in template:
        return template.replace("{{CONTEXT}}", (context or "").strip())
    return f"{template}\n\nContext:\n{(context or '').strip()}\n"


def _normalize_asr_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"\s+", " ", raw)
    cleaned = re.sub(r"^[`'\".,!?;:()\[\]{}\-_/\\]+", "", cleaned)
    cleaned = re.sub(r"^(?:'S|S)\b\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _repair_asr_leading_fragment(text: str) -> str:
    cleaned = _normalize_asr_text(text)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    if len(tokens) >= 3 and len(tokens[0]) <= 3 and len(tokens[1]) >= 5:
        cleaned = " ".join(tokens[1:]).strip()
    return cleaned


def _is_low_quality_asr_input(text: str) -> bool:
    cleaned = _normalize_asr_text(text)
    if not cleaned:
        return True
    tokens = re.findall(r"[A-Za-z0-9']+", cleaned)
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return True
    if len(tokens) == 2 and all(len(t) <= 2 for t in tokens):
        return True
    return False


def _build_repeat_prompt() -> str:
    return "I may have missed the beginning. Please say that once more."


# ── TTS helper ─────────────────────────────────────────────────────────────

def _synthesize_tts(tts: Any, voice: str, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
    """TTS helper: always use sherpa-onnx OfflineTts backend (tts/voice are unused)."""
    audio, sr = tts_sherpa.synthesize_sherpa_tts(text, speed=speed)
    if sr <= 0 or audio.size == 0:
        raise RuntimeError("sherpa-onnx TTS synthesis failed")
    return audio, sr


def _play_audio_with_barge_in(audio: np.ndarray, sample_rate: int, args: Any) -> bool:
    """Play one TTS chunk (barge-in removed; kept for call compatibility)."""
    if not audio.size or sample_rate <= 0:
        return False

    audio_io.play_audio(audio, sample_rate, args.output_device, volume=args.volume)
    return False


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

    t0 = time.perf_counter()
    tts_queue: Queue = Queue(maxsize=2)  # Buffer up to 2 chunks ahead
    audio_chunks = []
    sample_rate = None
    
    def _tts_producer():
        """Background thread: TTS all chunks and push to queue."""
        first_done_logged = False
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                t_start = time.perf_counter()
                chunk_audio, chunk_sr = _synthesize_tts(tts, voice, chunk)
                t_end = time.perf_counter()
                print(f"  [LATENCY] Chunk-{i+1} TTS: {t_end - t_start:.2f}s", flush=True)
                if i == 0 and not first_done_logged:
                    print(
                        f"  [LATENCY] First cloud chunk TTS done at: {t_end - t0:.2f}s",
                        flush=True,
                    )
                    first_done_logged = True
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
        
        # First chunk: wait for filler to finish, then log TTFS
        if first_chunk and filler_player and filler_player.is_playing():
            print("[Cloud] Waiting for filler to finish...", flush=True)
            filler_player.wait()
        if first_chunk:
            print(
                f"  [LATENCY] TTFS: {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
            first_chunk = False
        
        # Trim start only on first chunk
        if chunk_idx == 0 and args.trim_start > 0.0:
            chunk_audio = audio_io.trim_start_seconds(chunk_audio, chunk_sr, args.trim_start)
        
        # Play this chunk (blocking)
        if chunk_idx == 0:
            print(
                f"  [LATENCY] First cloud chunk play start: {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        print(f"  Playing chunk {chunk_idx+1}/{len(chunks)}...", flush=True)
        t_play_start = time.perf_counter()
        try:
            _play_audio_with_barge_in(chunk_audio, chunk_sr, args)
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
        _play_audio_with_barge_in(tts_audio, tts_sr, args)
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
            stop_event=None,
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
            stop_event=None,
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
        _play_audio_with_barge_in(tts_audio, tts_sr, args)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] playback failed: {exc}", file=sys.stderr)
    t4 = time.perf_counter()
    print(f"[time] play: {t4 - t3:.2f}s (total: {t4 - t0:.2f}s)", flush=True)


# ── Filler generation ──────────────────────────────────────────────────────

def _resolve_filler_onnx_model_id(args: Any) -> str:
    env_model = (os.environ.get("FILLER_ONNX_MODEL") or "").strip()
    if env_model:
        return env_model

    local_bundle = Path(__file__).resolve().parent.parent / "models" / "smollm2-135m-filler-colab-onnx-bundle"
    if local_bundle.exists():
        return str(local_bundle)

    arg_model = (getattr(args, "onnx_model", "") or "").strip()
    if arg_model:
        return arg_model

    return "HuggingFaceTB/SmolLM2-135M-Instruct"


def _get_filler_onnx_runtime(args: Any) -> Tuple[Any, Any, str]:
    global _FILLER_ONNX_MODEL, _FILLER_ONNX_TOKENIZER, _FILLER_ONNX_MODEL_ID

    model_id = _resolve_filler_onnx_model_id(args)
    if _FILLER_ONNX_MODEL is not None and _FILLER_ONNX_TOKENIZER is not None and _FILLER_ONNX_MODEL_ID == model_id:
        return _FILLER_ONNX_MODEL, _FILLER_ONNX_TOKENIZER, model_id

    model, tokenizer = llm_onnx._load_onnx_llm(model_id)
    if model is None or tokenizer is None:
        return None, None, model_id

    _FILLER_ONNX_MODEL = model
    _FILLER_ONNX_TOKENIZER = tokenizer
    _FILLER_ONNX_MODEL_ID = model_id
    return model, tokenizer, model_id


def _generate_filler_ollama(
    args: Any,
    emotion_label: Optional[str],
    user_text: str,
    timeout: float = 3.0,
) -> str:
    """Generate short filler phrase using fine-tuned ONNX model for Gemini delay gate."""
    filler_enabled = getattr(args, "cloud_filler", ENABLE_CLOUD_FILLER)
    if not filler_enabled:
        print("[Filler] Skipped (disabled by --no-cloud-filler)", flush=True)
        return ""

    filler_provider = (getattr(args, "filler_provider", "off") or "off").strip().lower()
    if filler_provider not in {"smollm2", "onnx", "smollm2_onnx"}:
        print(f"[Filler] Skipped (provider={filler_provider})", flush=True)
        return ""

    print("[Filler] Generating ONNX filler...", flush=True)
    system = text_utils.build_cloud_filler_system_prompt(emotion_label)
    try:
        model, tokenizer, model_id = _get_filler_onnx_runtime(args)
        if model is None or tokenizer is None:
            print(f"[Filler] ONNX model unavailable: {model_id}", flush=True)
            return text_utils.fallback_cloud_filler(user_text)

        filler_prompt = (
            "User just said: "
            f"{user_text!r}\n"
            "Generate ONE short spoken bridge phrase (3-12 words) while you think. "
            "Do NOT answer the question or give details."
        )
        filler = llm_onnx.generate_onnx_llm(
            prompt=filler_prompt,
            system=system,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=0.5,
            max_sentences=1,
            max_words=12,
        )
        raw = filler.strip() if filler and not filler.startswith("(ONNX LLM") else ""
        result = text_utils.validate_cloud_filler_output(raw)
        if not result:
            if raw:
                print(f"[Filler] Guardrail fallback (raw={raw[:100]!r})", flush=True)
            else:
                print(f"[Filler] Generation returned empty/error: {filler[:100] if filler else 'empty'}", flush=True)
            return text_utils.fallback_cloud_filler(user_text)
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[Filler] Generation failed: {exc}", flush=True, file=sys.stderr)
        return text_utils.fallback_cloud_filler(user_text)


# ── Routing + Gemini orchestration helpers ─────────────────────────────────

def _build_gemini_routes(
    args: Any,
    text: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
) -> Tuple[str, Optional[EmotionResult], Optional[short_long_router.RouteDecision]]:
    route_mode = "SHORT" if args.router_mode == "short_long" else "LOCAL"
    route_decision: Optional[short_long_router.RouteDecision] = None
    emotion_result: Optional[EmotionResult] = None

    def _run_emotion() -> Optional[EmotionResult]:
        if not (ENABLE_EMOTION and emotion_classifier and emotion_classifier.available):
            return None
        try:
            return emotion_classifier.predict(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[Emotion] error: {exc}", file=sys.stderr)
            return None

    def _run_legacy_router() -> str:
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

    # legacy force mode: LOCAL/CLOUD
    if FORCE_MODE in {"LOCAL", "CLOUD", "SHORT", "LONG"}:
        force_mode = FORCE_MODE
        if force_mode == "LOCAL":
            route_mode = "SHORT" if args.router_mode == "short_long" else "LOCAL"
        elif force_mode == "CLOUD":
            route_mode = "LONG" if args.router_mode == "short_long" else "CLOUD"
        elif force_mode in {"SHORT", "LONG"}:
            route_mode = force_mode
        if route_mode in {"LONG", "SHORT"}:
            print(f"[Route] forced mode={route_mode} (FORCE_MODE)", flush=True)
        else:
            print(f"[Route] forced mode={route_mode} (FORCE_MODE)", flush=True)

        emotion_result = _run_emotion()
        return route_mode, emotion_result, route_decision

    # short-long router
    if args.router_mode == "short_long":
        try:
            route_decision = short_long_router.route_query(
                text,
                min_score=args.router_min_score,
                margin=args.router_margin,
            )
            route_mode = route_decision.mode
            print(
                f"[Route] mode={route_decision.mode} conf={route_decision.confidence:.2f} "
                f"reason={route_decision.reason}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Route] short_long router error: {exc}", file=sys.stderr)
            route_mode = "SHORT"
        return route_mode, _run_emotion(), route_decision

    # legacy router
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_emotion = ex.submit(_run_emotion)
        fut_intent = ex.submit(_run_legacy_router)
        emotion_result = fut_emotion.result()
        route_mode = fut_intent.result()

    return route_mode, emotion_result, route_decision


def _play_filler_tts(tts: Any, voice: str, filler_text: str, args: Any) -> Optional[audio_io.AudioPlayer]:
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
        print(f"[FILLER] TTS/playback failed: {exc}", file=sys.stderr, flush=True)
        return None


def _run_gemini_turn(
    t0: float,
    args: Any,
    route_mode: str,
    prompt_context: str,
    user_text: str,
    tts: Any,
    voice: str,
    emotion_label: Optional[str],
    return_audio: bool = False,
    allow_filler: bool = True,
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[int]]:
    """Run Gemini (SHORT/LONG), with optional long-mode filler delay gate."""
    gemini_prompt = _build_gemini_prompt(route_mode, prompt_context)
    gemini_tokens = (
        args.gemini_short_max_tokens if route_mode == "SHORT" else args.gemini_long_max_tokens
    )

    def _call_gemini() -> str:
        return cloud_llm.call_cloud_llm(
            prompt=gemini_prompt,
            system=text_utils.build_cloud_system_prompt(emotion_label),
            timeout=20.0,
            max_output_tokens=gemini_tokens,
            temperature=0.35,
            preferred_provider="gemini",
        )

    t_parallel_start = time.perf_counter()
    filler_triggered = False
    filler_player: Optional[audio_io.AudioPlayer] = None

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_gemini = executor.submit(_call_gemini)

        delay_ms = int(getattr(args, "filler_delay_ms", 750))
        delay_s = max(0.0, float(delay_ms) / 1000.0)

        if route_mode == "LONG" and allow_filler and delay_s > 0.0:
            try:
                cloud_reply = fut_gemini.result(timeout=delay_s)
                t_cloud_ready = time.perf_counter()
            except concurrent.futures.TimeoutError:
                filler_triggered = True
                t_gate = time.perf_counter()
                print(
                    f"[Filler] Delay gate reached: elapsed={t_gate - t_parallel_start:.2f}s "
                    f"delay_ms={delay_ms}",
                    flush=True,
                )
                filler_text = _generate_filler_ollama(args, emotion_label, user_text, timeout=3.0)
                if filler_text:
                    print(f"[FILLER] {filler_text}", flush=True)
                    fut_filler_tts = executor.submit(_play_filler_tts, tts, voice, filler_text, args)
                    # Wait for full Gemini answer while filler is being spoken.
                    cloud_reply = fut_gemini.result()
                    t_cloud_ready = time.perf_counter()
                    try:
                        filler_player = fut_filler_tts.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[FILLER] TTS future failed: {exc}", file=sys.stderr, flush=True)
                        filler_player = None
                else:
                    cloud_reply = fut_gemini.result()
                    t_cloud_ready = time.perf_counter()
        else:
            cloud_reply = fut_gemini.result()
            t_cloud_ready = time.perf_counter()

    print(f"[Route] filler_triggered={filler_triggered} for {route_mode} mode")
    print(f"[LATENCY] Gemini call: {t_cloud_ready - t_parallel_start:.2f}s")

    if cloud_reply and not cloud_reply.startswith("("):
        if route_mode == "SHORT":
            assistant_reply = text_utils.postprocess_output(
                cloud_reply,
                max_sentences=2,
                max_words=60,
            )
        else:
            assistant_reply = text_utils.postprocess_output(
                cloud_reply,
                max_sentences=20,
                max_words=220,
            )
    else:
        assistant_reply = user_text

    if not assistant_reply:
        assistant_reply = user_text

    if cloud_reply:
        print(f"[LLM-GEMINI-{route_mode}] {cloud_reply}", flush=True)

    if return_audio:
        t2b = time.perf_counter()
        try:
            tts_audio, tts_sr = _synthesize_tts(tts, voice, assistant_reply)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] TTS failed: {exc}", file=sys.stderr)
            return assistant_reply, None, None
        t3 = time.perf_counter()
        print(f"[time] synthesize: {t3 - t2b:.2f}s", flush=True)
        if args.trim_start > 0.0:
            tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
        return assistant_reply, tts_audio, tts_sr

    max_words_per_chunk = getattr(args, "cloud_tts_max_words_per_chunk", 20)
    chunks = text_utils.split_into_chunks(assistant_reply, max_words_per_chunk=max_words_per_chunk)
    if not chunks:
        chunks = [assistant_reply]

    print(f"[GEMINI] Streaming {len(chunks)} chunk(s)...", flush=True)
    t_tts_start = time.perf_counter()
    _, _ = _play_chunks_pipelined(chunks, tts, voice, args, filler_player=filler_player)
    t_tts_end = time.perf_counter()
    print(f"[LATENCY] Total TTS+Play: {t_tts_end - t_tts_start:.2f}s")
    print(f"[LATENCY] End-to-end: {t_tts_end - t0:.2f}s")
    print("=" * 60 + "\n", flush=True)
    return assistant_reply, None, None


# ── Brain: Emotion + Intent routing → Gemini / legacy LOCAL/CLOUD → TTS ────

def _run_turn_brain(
    t0: float,
    args: Any,
    text: str,
    tts: Any,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
    conversation: Optional[text_utils.ConversationBuffer] = None,
) -> Optional[str]:
    """Route using router, call Gemini or legacy path, then TTS."""
    repaired_text = _repair_asr_leading_fragment(text)
    if repaired_text != (text or "").strip():
        print(f"[ASR] normalized -> {repaired_text!r}", flush=True)
    if _is_low_quality_asr_input(repaired_text):
        retry_text = _build_repeat_prompt()
        print("[ASR] low-quality turn; requesting repeat.", flush=True)
        try:
            tts_audio, tts_sr = _synthesize_tts(tts, voice, retry_text)
            if args.trim_start > 0.0:
                tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
            audio_io.play_audio(tts_audio, tts_sr, args.output_device, volume=args.volume)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] TTS failed on repeat prompt: {exc}", file=sys.stderr)
        return retry_text
    text = repaired_text

    llm_prompt = text
    if conversation is not None and len(conversation) > 0:
        llm_prompt = conversation.format_prompt_with_context(text)

    route_mode, emotion_result, _ = _build_gemini_routes(args, text, emotion_classifier)
    if args.router_mode == "short_long":
        route_mode = route_mode if route_mode in {"SHORT", "LONG"} else "SHORT"
        print("\n" + "=" * 60, flush=True)
        print(f"[GEMINI ROUTE] mode={route_mode}")
        print("=" * 60 + "\n", flush=True)
        assistant_reply, _, _ = _run_gemini_turn(
            t0=t0,
            args=args,
            route_mode=route_mode,
            prompt_context=llm_prompt,
            user_text=text,
            tts=tts,
            voice=voice,
            emotion_label=emotion_result.label if emotion_result else None,
            return_audio=False,
            allow_filler=True,
        )
        return assistant_reply

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
            return _generate_filler_ollama(args, emotion_label, text, timeout=3.0)

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

        # Postprocess Cloud reply length (sentences/words) for RPi-friendly TTS
        if cloud_reply and not cloud_reply.startswith("("):
            max_sents = getattr(args, "cloud_max_sentences", 2)
            max_words = getattr(args, "cloud_max_words", 60)
            cloud_text = text_utils.postprocess_output(
                cloud_reply,
                max_sentences=max_sents,
                max_words=max_words,
            )
        else:
            cloud_text = text
        assistant_reply = cloud_text

        if cloud_reply:
            print(f"[LLM-CLOUD] {cloud_reply}\n", flush=True)

        # Split and play with pipelined TTS
        max_words_per_chunk = getattr(args, "cloud_tts_max_words_per_chunk", 20)
        chunks = text_utils.split_into_chunks(cloud_text, max_words_per_chunk=max_words_per_chunk)
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
    """Sentence-streaming version: returns (reply_text, tts_audio, tts_sr) without playing."""
    repaired_text = _repair_asr_leading_fragment(text)
    if repaired_text != (text or "").strip():
        print(f"[ASR] normalized -> {repaired_text!r}", flush=True)
    if _is_low_quality_asr_input(repaired_text):
        retry_text = _build_repeat_prompt()
        try:
            tts_audio, tts_sr = _synthesize_tts(tts, voice, retry_text)
            if args.trim_start > 0.0:
                tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
            return retry_text, tts_audio, tts_sr
        except Exception as exc:  # noqa: BLE001
            print(f"[error] TTS failed on repeat prompt: {exc}", file=sys.stderr)
            return retry_text, None, None
    text = repaired_text

    llm_prompt = text
    if conversation is not None and len(conversation) > 0:
        llm_prompt = conversation.format_prompt_with_context(text)

    route_mode, emotion_result, _ = _build_gemini_routes(args, text, emotion_classifier)
    if args.router_mode == "short_long":
        route_mode = route_mode if route_mode in {"SHORT", "LONG"} else "SHORT"
        assistant_reply, tts_audio, tts_sr = _run_gemini_turn(
            t0=t0,
            args=args,
            route_mode=route_mode,
            prompt_context=llm_prompt,
            user_text=text,
            tts=tts,
            voice=voice,
            emotion_label=emotion_result.label if emotion_result else None,
            return_audio=True,
            allow_filler=False,
        )
        if tts_audio is not None and tts_sr is not None and args.trim_start > 0.0:
            tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
        return assistant_reply, tts_audio, tts_sr

    emotion_label = emotion_result.label if emotion_result else "neutral"

    # ── CLOUD path ──────────────────────────────────────────────────────────
    if route_mode == "CLOUD":
        print("\n" + "="*60)
        print("[CLOUD MODE] Starting (with Ollama filler)...")
        print("="*60 + "\n", flush=True)

        system = text_utils.build_cloud_system_prompt(emotion_label)

        # Parallel: Ollama filler + Cloud LLM
        def _call_filler():
            return _generate_filler_ollama(args, emotion_label, text, timeout=3.0)

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

        # Postprocess Cloud reply length (sentences/words) for RPi-friendly TTS
        if cloud_reply and not cloud_reply.startswith("("):
            max_sents = getattr(args, "cloud_max_sentences", 2)
            max_words = getattr(args, "cloud_max_words", 60)
            cloud_text = text_utils.postprocess_output(
                cloud_reply,
                max_sentences=max_sents,
                max_words=max_words,
            )
        else:
            cloud_text = text

        if cloud_reply:
            print(f"[LLM-CLOUD] {cloud_reply}\n", flush=True)

        # Split and play with pipelined TTS
        max_words_per_chunk = getattr(args, "cloud_tts_max_words_per_chunk", 20)
        chunks = text_utils.split_into_chunks(cloud_text, max_words_per_chunk=max_words_per_chunk)
        if not chunks:
            chunks = [cloud_text]

        print(f"[CLOUD] Streaming {len(chunks)} chunk(s)...\n", flush=True)

        t_tts_start = time.perf_counter()
        combined_audio_chunks, sample_rate = _play_chunks_pipelined(chunks, tts, voice, args, filler_player=filler_player)
        t_tts_end = time.perf_counter()

        print(f"\n[LATENCY] Total TTS+Play: {t_tts_end - t_tts_start:.2f}s")
        print("="*60 + "\n", flush=True)

        # Audio has already been played via _play_chunks_pipelined for CLOUD path.
        # We return no audio here so that the caller does NOT play it a second time.
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
    warmup_all_onnx = os.environ.get("WARMUP_ALL_ONNX", "1").strip() not in {
        "0",
        "false",
        "False",
        "no",
        "NO",
    }
    
    # 1. STT Recognizer (sherpa-onnx OnlineRecognizer)
    print("[Warmup] Loading STT recognizer...", flush=True)
    try:
        recognizer = stt_sherpa.get_recognizer()
        if recognizer is not None:
            # Prime first decode path with short silence.
            stream = recognizer.create_stream()
            stream.accept_waveform(stt_sherpa.SAMPLE_RATE, np.zeros(1600, dtype=np.float32))
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
    except Exception as exc:  # noqa: BLE001
        print(f"[Warmup] STT warning: {exc}", file=sys.stderr)

    if args.vad or warmup_all_onnx:
        print("[Warmup] Loading VAD...", flush=True)
        try:
            _ = stt_sherpa.get_vad()
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] VAD warning: {exc}", file=sys.stderr)
        try:
            # Prime mic input so the first real user turn is not clipped.
            _ = stt_sherpa.prime_microphone_input(
                input_device=args.input_device,
                seconds=float(os.environ.get("SHERPA_MIC_PRIME_SECONDS", "0.8")),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Mic prime warning: {exc}", file=sys.stderr)
    
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
    if args.router_mode == "short_long":
        print("[Warmup] Loading short/long router...", flush=True)
        try:
            _ = short_long_router.route_query("Warmup for short long routing.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Short/long router warning: {exc}", file=sys.stderr)
    
    # 4. Emotion Classifier (ONNX BERT)
    if ENABLE_EMOTION and emotion_classifier is not None and emotion_classifier.available:
        print("[Warmup] Loading emotion classifier...", flush=True)
        try:
            _ = emotion_classifier.predict("Hello, just warming up.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Emotion warning: {exc}", file=sys.stderr)

    # 5. Memory encoder (MiniLM ONNX) for rolling summary extraction
    if warmup_all_onnx and getattr(args, "max_turns", 0) > 0:
        print("[Warmup] Loading memory MiniLM ONNX encoder...", flush=True)
        try:
            memory_encoder = text_utils._get_memory_embedding_model()
            if memory_encoder is not None:
                _ = memory_encoder.encode(
                    [
                        "Warmup memory context one.",
                        "Warmup memory context two.",
                    ],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            else:
                print("[Warmup] Memory encoder unavailable (fallback summarizer will be used).", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] Memory ONNX warning: {exc}", file=sys.stderr)

    # 6. ONNX filler model warmup
    filler_provider = (getattr(args, "filler_provider", "off") or "off").strip().lower()
    if warmup_all_onnx and filler_provider in {"smollm2", "onnx", "smollm2_onnx"}:
        print("[Warmup] Loading ONNX filler model...", flush=True)
        try:
            filler_model, filler_tokenizer, _ = _get_filler_onnx_runtime(args)
            if filler_model is not None and filler_tokenizer is not None:
                _ = llm_onnx.generate_onnx_llm(
                    prompt="Warmup filler",
                    system=text_utils.build_cloud_filler_system_prompt(None),
                    model=filler_model,
                    tokenizer=filler_tokenizer,
                    max_new_tokens=4,
                    temperature=0.0,
                    max_sentences=1,
                    max_words=8,
                )
            else:
                print("[Warmup] ONNX filler model unavailable.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] ONNX filler warning: {exc}", file=sys.stderr)

    # 7. Optional ONNX LLM warmup
    warmup_onnx_llm = getattr(args, "onnx_llm", False) or (
        os.environ.get("WARMUP_ONNX_LLM", "0").strip() not in {"0", "false", "False", "no", "NO"}
    )
    if warmup_onnx_llm:
        print("[Warmup] Loading ONNX LLM...", flush=True)
        try:
            onnx_model, onnx_tokenizer = llm_onnx._load_onnx_llm(args.onnx_model)
            if onnx_model is not None and onnx_tokenizer is not None:
                _ = llm_onnx.generate_onnx_llm(
                    prompt="Warmup",
                    system=text_utils.ONNX_DEFAULT_SYSTEM,
                    model=onnx_model,
                    tokenizer=onnx_tokenizer,
                    max_new_tokens=4,
                    temperature=0.0,
                    max_sentences=1,
                    max_words=8,
                )
            else:
                print("[Warmup] ONNX LLM unavailable.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warmup] ONNX LLM warning: {exc}", file=sys.stderr)
    
    # 8. Ollama LLM (if enabled)
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
        conversation = text_utils.ConversationBuffer(
            max_turns=args.max_turns,
            max_summary_turns=max(getattr(args, "memory_max_summary_turns", 12), 0),
            summary_word_budget=max(getattr(args, "memory_summary_word_budget", 120), 20),
        )
        print(f"- multi-turn context: last {args.max_turns} turns")
        print(
            f"- memory summary: {getattr(args, 'memory_max_summary_turns', 12)} fragments, "
            f"{getattr(args, 'memory_summary_word_budget', 120)} word budget"
        )

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
    - Each sentence (1.5s silence) → brain → TTS (interruptible)
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
                on_speech_start=None,
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
