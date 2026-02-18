"""CLI: parse_args and print_config for STT/TTS demo."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable, Optional


_DIALOG_PRESETS: dict[str, dict[str, int | float]] = {
    "natural_balanced": {
        "max_turns": 3,
        "cloud_max_sentences": 3,
        "cloud_max_words": 68,
        "cloud_tts_max_words_per_chunk": 22,
        "memory_max_summary_turns": 8,
        "memory_summary_word_budget": 100,
        "router_min_score": 0.20,
        "router_margin": 0.05,
        "gemini_short_max_tokens": 170,
        "gemini_long_max_tokens": 420,
        "filler_delay_ms": 800,
        "ollama_num_ctx": 640,
        "ollama_temperature": 0.32,
    },
    "accurate": {
        "max_turns": 4,
        "cloud_max_sentences": 4,
        "cloud_max_words": 90,
        "cloud_tts_max_words_per_chunk": 24,
        "memory_max_summary_turns": 12,
        "memory_summary_word_budget": 140,
        "router_min_score": 0.18,
        "router_margin": 0.04,
        "gemini_short_max_tokens": 190,
        "gemini_long_max_tokens": 560,
        "filler_delay_ms": 950,
        "ollama_num_ctx": 768,
        "ollama_temperature": 0.35,
    },
}


def _resolve_dialog_profile() -> str:
    profile = (os.environ.get("DIALOG_PROFILE") or "").strip().lower()
    if profile in _DIALOG_PRESETS:
        return profile
    return "natural_balanced"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sherpa-onnx STT/TTS voice demo (Raspberry Pi)")
    argv_list = list(argv) if argv is not None else None
    profile = _resolve_dialog_profile()
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument(
        "--dialog-profile",
        choices=sorted(_DIALOG_PRESETS.keys()),
        default=profile,
    )
    profile_from_argv = profile_parser.parse_known_args(argv_list)[0].dialog_profile
    if profile_from_argv in _DIALOG_PRESETS:
        profile = profile_from_argv

    preset_values = _DIALOG_PRESETS[profile]
    env_int = lambda env_var, key: int(os.environ.get(env_var, str(preset_values[key])))
    env_float = lambda env_var, key: float(os.environ.get(env_var, str(preset_values[key])))

    # --- Recording & I/O ---
    parser.add_argument("--record-seconds", type=float, default=3.0, help="Record duration per turn when --no-streaming (seconds).")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index (None=default).")
    parser.add_argument("--output-device", type=int, default=None, help="sounddevice output device index (None=default).")
    parser.add_argument("--volume", type=float, default=1.0, help="Playback volume 0.0–1.0 (default 1.0).")
    parser.add_argument("--trim-start", type=float, default=0.0, help="Trim this many seconds from start of TTS audio (default 0).")

    # --- STT mode selection ---
    parser.add_argument(
        "--streaming", action="store_true", default=True,
        help="(Default) Stream mic to OnlineRecognizer with endpoint detection.",
    )
    parser.add_argument(
        "--no-streaming", action="store_false", dest="streaming",
        help="Fixed-duration recording + chunked transcription (Phase 1 fallback).",
    )
    parser.add_argument(
        "--max-listen-seconds", type=float, default=15.0,
        help="Max seconds to listen in streaming/VAD mode before timeout (default 15).",
    )
    parser.add_argument(
        "--vad", action="store_true", default=False,
        help="Always-listening mode: VAD detects speech automatically (no Enter needed).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=env_int("DIALOG_MAX_TURNS", "max_turns"),
        help="Number of conversation turns to remember for multi-turn context (0 to disable).",
    )
    parser.add_argument(
        "--sentence-streaming", action="store_true", default=False,
        help="Sentence-by-sentence streaming mode: each sentence (1.5s silence) triggers immediate processing.",
    )
    parser.add_argument(
        "--sentence-silence", type=float, default=1.5,
        help="Silence duration (seconds) to detect sentence boundaries in sentence-streaming mode (default 1.5).",
    )

    # --- Ollama (LOCAL LLM) ---
    parser.add_argument("--ollama", action="store_true", help="Send STT text to Ollama and TTS the reply (voice chatbot).")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama API URL.")
    parser.add_argument("--ollama-model", type=str, default="smollm2:360m", help="Ollama model name.")
    parser.add_argument("--ollama-system", type=str, default="", help="System prompt for Ollama (default: short friendly buddy).")
    parser.add_argument("--ollama-num-predict", type=int, default=24, help="Ollama max tokens.")
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=env_float("OLLAMA_TEMPERATURE", "ollama_temperature"),
        help="Ollama temperature.",
    )
    parser.add_argument("--ollama-keep-alive", type=str, default="10m", help="Keep model in memory (e.g. 10m, 60m).")
    parser.add_argument("--ollama-num-thread", type=int, default=4, help="Ollama num_thread (match run_brain).")
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=env_int("OLLAMA_NUM_CTX", "ollama_num_ctx"),
        help="Ollama num_ctx (match run_brain).",
    )
    parser.add_argument("--ollama-num-batch", type=int, default=16, help="Ollama num_batch; 0=omit (match run_brain).")
    parser.add_argument("--ollama-stream", action="store_true", default=True, help="Stream Ollama and TTS per chunk (default).")
    parser.add_argument("--no-ollama-stream", action="store_false", dest="ollama_stream", help="Disable streaming; one TTS after full response.")
    parser.add_argument("--ollama-stream-max-words", type=int, default=12, help="Max words per chunk when no sentence end (stream mode).")
    parser.add_argument("--ollama-stream-async", action="store_true", default=True, help="Overlap synth and play for seamless TTS (default).")
    parser.add_argument("--no-ollama-stream-async", action="store_false", dest="ollama_stream_async", help="Stream TTS sequential (synth then play per chunk).")

    # --- Cloud Filler ---
    parser.add_argument("--cloud-filler", action="store_true", default=True, help="Use Ollama filler during Cloud LLM latency (default).")
    parser.add_argument("--no-cloud-filler", action="store_false", dest="cloud_filler", help="Disable filler; wait silently for Cloud LLM.")

    # --- Cloud LLM output / TTS length control ---
    parser.add_argument(
        "--cloud-max-sentences",
        type=int,
        default=env_int("CLOUD_MAX_SENTENCES", "cloud_max_sentences"),
        help="Max sentences for Cloud LLM reply.",
    )
    parser.add_argument(
        "--cloud-max-words",
        type=int,
        default=env_int("CLOUD_MAX_WORDS", "cloud_max_words"),
        help="Approximate max words for Cloud LLM reply after postprocessing.",
    )
    parser.add_argument(
        "--cloud-tts-max-words-per-chunk",
        type=int,
        default=env_int("CLOUD_TTS_MAX_WORDS_PER_CHUNK", "cloud_tts_max_words_per_chunk"),
        help="Max words per Cloud TTS chunk after sentence splitting.",
    )
    parser.add_argument(
        "--memory-max-summary-turns",
        type=int,
        default=env_int("MEMORY_MAX_SUMMARY_TURNS", "memory_max_summary_turns"),
        help="Max archived turns to keep for rolling summary fragment maintenance.",
    )
    parser.add_argument(
        "--memory-summary-word-budget",
        type=int,
        default=env_int("MEMORY_SUMMARY_WORD_BUDGET", "memory_summary_word_budget"),
        help="Word budget for rolling summary.",
    )
    parser.add_argument(
        "--dialog-profile",
        type=str,
        choices=sorted(_DIALOG_PRESETS.keys()),
        default=profile,
        help="Preset optimized for speech-dialogue quality/latency trade-off.",
    )

    # --- ONNX LLM ---
    parser.add_argument("--onnx-llm", action="store_true", help="Use ONNX LLM instead of Ollama (e.g. SmolLM2-135M-Instruct); no streaming, single TTS.")
    parser.add_argument("--onnx-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Hugging Face model id for ONNX LLM (must have onnx/ subfolder).")
    parser.add_argument("--onnx-max-new-tokens", type=int, default=24, help="ONNX LLM max new tokens.")
    parser.add_argument("--onnx-temperature", type=float, default=0.3, help="ONNX LLM temperature.")

    parser.add_argument(
        "--router-mode",
        type=str,
        choices=["legacy", "short_long"],
        default=os.environ.get("ROUTER_MODE", "short_long"),
        help="Routing strategy: legacy (LOCAL/CLOUD) or short_long (SHORT/LONG).",
    )
    parser.add_argument(
        "--router-min-score",
        type=float,
        default=env_float("ROUTER_MIN_SCORE", "router_min_score"),
        help="Minimum score threshold for confident LONG routing.",
    )
    parser.add_argument(
        "--router-margin",
        type=float,
        default=env_float("ROUTER_MARGIN", "router_margin"),
        help="Uncertainty threshold on (best score - second best).",
    )
    parser.add_argument(
        "--gemini-short-max-tokens",
        type=int,
        default=env_int("GEMINI_SHORT_MAX_TOKENS", "gemini_short_max_tokens"),
        help="Max output tokens for Gemini SHORT mode.",
    )
    parser.add_argument(
        "--gemini-long-max-tokens",
        type=int,
        default=env_int("GEMINI_LONG_MAX_TOKENS", "gemini_long_max_tokens"),
        help="Max output tokens for Gemini LONG mode.",
    )
    parser.add_argument(
        "--filler-provider",
        type=str,
        default=os.environ.get("FILLER_PROVIDER", "smollm2"),
        help="Filler provider: smollm2 (default) or off.",
    )
    parser.add_argument(
        "--filler-delay-ms",
        type=int,
        default=env_int("FILLER_DELAY_MS", "filler_delay_ms"),
        help="Wait this long before playing LONG filler (ms).",
    )
    parser.add_argument("--barge-in", action="store_true", default=True, help="Enable barge-in: stop playback when user speaks.")
    parser.add_argument("--no-barge-in", action="store_false", dest="barge_in", help="Disable barge-in interruption.")
    parser.add_argument(
        "--barge-in-energy-threshold",
        type=float,
        default=0.02,
        help="VAD-free barge-in energy threshold (higher = fewer false positives).",
    )
    parser.add_argument(
        "--barge-in-window-ms",
        type=int,
        default=80,
        help="Window length (ms) for each barge-in microphone check.",
    )
    parser.add_argument(
        "--barge-in-hit-count",
        type=int,
        default=2,
        help="Consecutive windows required before treating as barge-in speech.",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def _arg_to_json_value(value: Any) -> Any:
    if hasattr(value, "__fspath__"):
        return str(value)
    return value


def print_config(args: argparse.Namespace, voice: str) -> None:
    """Print current config (args + resolved voice) as JSON to stdout."""
    d: dict[str, Any] = {}
    for key in [
        "record_seconds",
        "streaming",
        "vad",
        "max_listen_seconds",
        "max_turns",
        "sentence_streaming",
        "sentence_silence",
        "input_device",
        "output_device",
        "volume",
        "trim_start",
        "ollama",
        "ollama_url",
        "ollama_model",
        "ollama_stream",
        "ollama_stream_async",
        "ollama_stream_max_words",
        "cloud_filler",
        "cloud_max_sentences",
        "cloud_max_words",
        "cloud_tts_max_words_per_chunk",
        "onnx_llm",
        "onnx_model",
        "onnx_max_new_tokens",
        "onnx_temperature",
        "router_mode",
        "router_min_score",
        "router_margin",
        "gemini_short_max_tokens",
        "gemini_long_max_tokens",
        "filler_provider",
        "filler_delay_ms",
        "memory_max_summary_turns",
        "memory_summary_word_budget",
        "dialog_profile",
        "barge_in",
        "barge_in_energy_threshold",
        "barge_in_window_ms",
        "barge_in_hit_count",
    ]:
        if hasattr(args, key):
            d[key] = _arg_to_json_value(getattr(args, key))
    d["voice"] = voice
    print("--- config ---")
    print(json.dumps(d, indent=2))
