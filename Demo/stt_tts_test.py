"""
Faster-Whisper (distil-small.en) STT -> Kokoro v0.19 ONNX TTS demo (RPi).

Optional: --ollama (Ollama/smolLM2) or --onnx-llm (ONNX SmolLM2-135M) between STT and TTS for voice chatbot.
"""
from __future__ import annotations

import gc
import sys
import time
from typing import Any, Iterable, List, Optional

from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

from . import audio_io
from . import llm_ollama
from . import llm_onnx
from . import stt
from . import stt_tts_cli
from . import text_utils
from . import tts_kokoro


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
        tts_audio, tts_sr = tts_kokoro.synthesize_kokoro(tts, tts_text, voice)
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
        tts_audio, tts_sr = tts_kokoro.synthesize_kokoro(tts, tts_text, voice)
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

    try:
        asr_model = WhisperModel(
            "distil-small.en",
            device="cpu",
            compute_type=args.asr_compute_type,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[fatal] ASR model load failed: {exc}", file=sys.stderr)
        return 1

    try:
        tts = Kokoro(
            model_path=str(args.kokoro_model),
            voices_path=str(args.kokoro_voices),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[fatal] TTS model load failed: {exc}", file=sys.stderr)
        return 1

    available_voices = tts.get_voices()
    voice = args.voice
    if voice not in available_voices:
        voice = available_voices[0] if available_voices else "af_alloy"
        print(f"[info] voice '{args.voice}' not found, using '{voice}'. Available: {available_voices}", file=sys.stderr)

    print("Faster-Whisper + Kokoro demo")
    print(f"- ASR: distil-small.en (compute_type={args.asr_compute_type})")
    print(f"- TTS model: {args.kokoro_model}")
    print(f"- TTS voices: {args.kokoro_voices}")
    print(f"- voice: {voice} (available: {available_voices})")
    print(f"- volume: {args.volume}")
    print(f"- trim-start: {args.trim_start}s")
    print(f"- record seconds: {args.record_seconds}")

    onnx_model, onnx_tokenizer = None, None
    if args.onnx_llm:
        print(f"- onnx-llm: {args.onnx_model}")
        print("Loading ONNX LLM...", flush=True)
        try:
            onnx_model, onnx_tokenizer = llm_onnx._load_onnx_llm(args.onnx_model)
            if onnx_model is not None and onnx_tokenizer is not None:
                try:
                    _ = llm_onnx.generate_onnx_llm(
                        "Hi", text_utils.ONNX_DEFAULT_SYSTEM, onnx_model, onnx_tokenizer,
                        max_new_tokens=1, temperature=0.0,
                    )
                except Exception:  # noqa: BLE001
                    pass
                print("ONNX LLM ready.", flush=True)
            else:
                print("[warn] ONNX LLM load failed; install optimum[onnxruntime] and ensure model has onnx/ subfolder.", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ONNX LLM preload failed: {e}", file=sys.stderr)
    elif args.ollama:
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
            text = stt.transcribe_faster_whisper(asr_model, audio, beam_size=args.beam_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] ASR failed: {exc}", file=sys.stderr)
            continue
        t2 = time.perf_counter()
        transcribe_sec = t2 - t1
        print(f"[time] transcribe: {transcribe_sec:.2f}s", flush=True)
        if transcribe_sec >= stt.TRANSCRIBE_SLOW_WARN_THRESHOLD:
            print(
                f"[warn] transcribe very slow ({transcribe_sec:.1f}s); check CPU/thermal/memory",
                file=sys.stderr,
                flush=True,
            )

        if not text:
            print("[info] no speech detected / empty text.")
            continue

        print(f"[ASR] {text}")

        if args.onnx_llm and onnx_model is not None and onnx_tokenizer is not None:
            _run_turn_onnx_llm(t0, args, text, tts, voice, onnx_model, onnx_tokenizer)
        elif args.ollama and args.ollama_stream:
            _run_turn_ollama_stream(t0, args, text, tts, voice)
        else:
            _run_turn_ollama_or_direct(t0, args, text, tts, voice)

        print("Done. Press Enter to record again.", flush=True)
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
