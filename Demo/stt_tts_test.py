"""
Sherpa-onnx STT -> Emotion + Intent routing -> LOCAL/CLOUD LLM -> sherpa-onnx VITS TTS demo (RPi).

LOCAL: Ollama (e.g., smollm2:360m) with empathic prompt.
CLOUD: external HTTP LLM (if configured) with informational prompt.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import concurrent.futures
from pathlib import Path
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
ENABLE_CLOUD_FILLER = os.environ.get("ENABLE_CLOUD_FILLER", "1").strip() not in {"0", "false", "False", "no", "NO"}
FORCE_MODE = (os.environ.get("FORCE_MODE", "") or "").strip().upper()

# Warn when STT transcribe time exceeds this (seconds); suggests CPU/thermal/memory check on RPi.
SLOW_ASR_WARN_THRESHOLD = 10.0


def _synthesize_tts(tts: Any, voice: str, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
    """TTS helper: always use sherpa-onnx OfflineTts backend (tts/voice are unused)."""
    audio, sr = tts_sherpa.synthesize_sherpa_tts(text, speed=speed)
    if sr <= 0 or audio.size == 0:
        raise RuntimeError("sherpa-onnx TTS synthesis failed")
    return audio, sr


def _run_turn_onnx_llm(
    t0: float,
    args: Any,
    text: str,
    tts: Kokoro,
    voice: str,
    onnx_model: Any,
    onnx_tokenizer: Any,
) -> None:
    """ONNX LLM -> synthesize -> play. Logs timing."""
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


def _run_turn_ollama_stream(t0: float, args: Any, text: str, tts: Kokoro, voice: str) -> None:
    """Ollama stream (async or sync) -> TTS per chunk. Logs timing."""
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


def _run_turn_ollama_or_direct(t0: float, args: Any, text: str, tts: Kokoro, voice: str) -> None:
    """Ollama single reply -> synthesize -> play, or direct TTS. Logs timing."""
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


def _run_turn_brain(
    t0: float,
    args: Any,
    text: str,
    tts: Kokoro,
    voice: str,
    emotion_classifier: Optional[EmotionClassifierONNX],
) -> None:
    """Route using emotion + intent, call LOCAL or CLOUD LLM, then TTS."""
    route_mode = "LOCAL"
    emotion_result: Optional[EmotionResult] = None

    # Respect FORCE_MODE override first
    if FORCE_MODE in {"LOCAL", "CLOUD"}:
        route_mode = FORCE_MODE
        if ENABLE_EMOTION and emotion_classifier and emotion_classifier.available:
            try:
                emotion_result = emotion_classifier.predict(text)
            except Exception as exc:  # noqa: BLE001
                print(f"[Emotion] error: {exc}", file=sys.stderr)
                emotion_result = None
    else:
        # Optionally run emotion + intent in parallel
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
            # Prefer anchors-based router; fall back to simple classifier on error.
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

    # Build system prompt and behavior based on route
    if route_mode == "CLOUD":
        # Phase 1: optional LOCAL filler via Ollama (short bridge, no answering)
        if ENABLE_CLOUD_FILLER and args.ollama:
            filler_system = text_utils.build_cloud_filler_system_prompt(emotion_label)
            print("Ollama (CLOUD filler)...", flush=True)
            t_fill_start = time.perf_counter()
            filler_reply = llm_ollama.generate_ollama(
                prompt=text,
                model=args.ollama_model,
                system=filler_system,
                url=args.ollama_url,
                num_predict=min(args.ollama_num_predict, 24),
                temperature=args.ollama_temperature,
                stop=["\n"],
                keep_alive=args.ollama_keep_alive,
                num_thread=args.ollama_num_thread,
                num_ctx=args.ollama_num_ctx,
                num_batch=args.ollama_num_batch,
                max_sentences=1,
                max_words=24,
                timeout=10,
            )
            t_fill_end = time.perf_counter()
            print(f"[time] ollama(cloud-filler): {t_fill_end - t_fill_start:.2f}s", flush=True)
            if filler_reply and not filler_reply.startswith("(Ollama error"):
                print(f"[LLM-CLOUD-FILLER] {filler_reply}", flush=True)
                t2b = time.perf_counter()
                print("Synthesizing (filler)...", flush=True)
                try:
                    tts_audio, tts_sr = _synthesize_tts(tts, voice, filler_reply)
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] TTS filler failed: {exc}", file=sys.stderr)
                else:
                    t3 = time.perf_counter()
                    print(f"[time] synthesize(filler): {t3 - t2b:.2f}s", flush=True)
                    if args.trim_start > 0.0:
                        tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
                    print("Playing (filler)...", flush=True)
                    try:
                        audio_io.play_audio(tts_audio, tts_sr, args.output_device, volume=args.volume)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[error] playback filler failed: {exc}", file=sys.stderr)

        # Phase 2: main CLOUD answer via HTTP LLM
        system = text_utils.build_cloud_system_prompt(emotion_label)
        reply = cloud_llm.call_cloud_llm(prompt=text, system=system, timeout=20.0)

        tts_text = reply.strip() if reply and not reply.startswith("(Cloud LLM") else text
        if reply:
            print(f"[LLM-CLOUD] {reply}", flush=True)

        t2b = time.perf_counter()
        print("Synthesizing (cloud)...", flush=True)
        try:
            tts_audio, tts_sr = _synthesize_tts(tts, voice, tts_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] TTS failed: {exc}", file=sys.stderr)
            return
        t3 = time.perf_counter()
        print(f"[time] synthesize(cloud): {t3 - t2b:.2f}s", flush=True)
        if args.trim_start > 0.0:
            tts_audio = audio_io.trim_start_seconds(tts_audio, tts_sr, args.trim_start)
        print("Playing (cloud)...", flush=True)
        try:
            audio_io.play_audio(tts_audio, tts_sr, args.output_device, volume=args.volume)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] playback failed: {exc}", file=sys.stderr)
        t4 = time.perf_counter()
        print(f"[time] play(cloud): {t4 - t3:.2f}s (total: {t4 - t0:.2f}s)", flush=True)
        return

    # LOCAL (default) – use Ollama if enabled, otherwise just echo ASR text
    system = text_utils.build_local_system_prompt(emotion_label)

    # If streaming is enabled, reuse the existing streaming path so that
    # Ollama + Kokoro TTS run as a pipeline and the user hears output sooner.
    if args.ollama and args.ollama_stream:
        # Pass emotion-aware system prompt via args.ollama_system
        args.ollama_system = system
        _run_turn_ollama_stream(t0, args, text, tts, voice)
        return

    reply = ""
    if args.ollama:
        print("Ollama (LOCAL)...", flush=True)
        t_ollama_start = time.perf_counter()
        reply = llm_ollama.generate_ollama(
            prompt=text,
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = stt_tts_cli.parse_args(argv)
    if not 0.0 <= args.volume <= 1.0:
        print("[error] --volume must be between 0.0 and 1.0.", file=sys.stderr)
        return 1
    if args.trim_start < 0.0:
        print("[error] --trim-start must be >= 0.", file=sys.stderr)
        return 1

    # sherpa-onnx TTS backend (OfflineTts) is configured in Demo/tts_sherpa.py
    # and download_model.py, so we don't need to load a separate TTS model here.
    tts = None
    voice = "sherpa_default"

    print("Voice demo (sherpa-onnx single mode)")
    print("- ASR: sherpa-onnx (streaming Zipformer)")
    print("- TTS: sherpa-onnx VITS (vits-coqui-en-ljspeech)")
    print(f"- volume: {args.volume}")
    print(f"- trim-start: {args.trim_start}s")
    print(f"- record seconds: {args.record_seconds}")

    # LOCAL LLM (Ollama) warmup if enabled
    if args.ollama:
        print(f"- ollama: {args.ollama_model} @ {args.ollama_url}")
        print(f"- ollama-stream: {args.ollama_stream} (async: {args.ollama_stream_async}, max-words/chunk: {args.ollama_stream_max_words})")
        print("Preloading Ollama model...", flush=True)
        try:
            llm_ollama.generate_ollama(
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
                timeout=60,
            )
            print("Ollama model ready.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Ollama preload failed: {e}", file=sys.stderr)

    stt_tts_cli.print_config(args, voice)
    print("Press Enter to record, or Ctrl+C to quit.")

    # Emotion classifier (ONNX BERT) – optional, controlled by ENABLE_EMOTION
    emotion_classifier: Optional[EmotionClassifierONNX] = None
    if ENABLE_EMOTION:
        model_dir = Path(__file__).resolve().parent.parent / "emotion_onnx_int8"
        emotion_classifier = EmotionClassifierONNX(str(model_dir))
        if not getattr(emotion_classifier, "available", False):
            emotion_classifier = None

    # --- Warm-up phase: embedder / emotion / cloud LLM ---
    # This increases startup time a bit but avoids first-turn spikes.
    try:
        if ENABLE_INTENT_ROUTER:
            # Warm up anchors-based router (loads SentenceTransformer + anchor embeddings).
            _ = router_anchors_runtime.route_local_or_cloud("Warmup for routing.")
        if ENABLE_EMOTION and emotion_classifier is not None and emotion_classifier.available:
            # Warm up emotion classifier ONNX + tokenizer.
            _ = emotion_classifier.predict("Hello, just warming up.")
        cloud_url = (os.environ.get("CLOUD_LLM_URL") or "").strip()
        if cloud_url:
            # Best-effort CLOUD LLM warm-up; ignore errors so LOCAL path still works.
            _ = cloud_llm.call_cloud_llm(
                prompt="Warmup request.",
                system="You are a friendly, reliable assistant. This is a warmup call; a short reply is fine.",
                timeout=5.0,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[Warmup] warning: {exc}", file=sys.stderr)

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

        _run_turn_brain(t0, args, text, tts, voice, emotion_classifier)

        print("Done. Press Enter to record again.", flush=True)
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
