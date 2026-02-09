"""CLI: parse_args and print_config for STT/TTS demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sherpa-onnx STT/TTS voice demo (Raspberry Pi)")
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--record-seconds", type=float, default=3.0, help="Record duration per turn (seconds).")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index (None=default).")
    parser.add_argument("--output-device", type=int, default=None, help="sounddevice output device index (None=default).")
    # sherpa-onnx STT/TTS does not require explicit model paths here; they are
    # controlled by download_model.py and environment variables.
    parser.add_argument("--volume", type=float, default=1.0, help="Playback volume 0.0–1.0 (default 1.0).")
    parser.add_argument("--trim-start", type=float, default=0.0, help="Trim this many seconds from start of TTS audio (default 0).")
    parser.add_argument("--ollama", action="store_true", help="Send STT text to Ollama and TTS the reply (voice chatbot).")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama API URL.")
    parser.add_argument("--ollama-model", type=str, default="smollm2:360m", help="Ollama model name.")
    parser.add_argument("--ollama-system", type=str, default="", help="System prompt for Ollama (default: short friendly buddy).")
    parser.add_argument("--ollama-num-predict", type=int, default=24, help="Ollama max tokens.")
    parser.add_argument("--ollama-temperature", type=float, default=0.3, help="Ollama temperature.")
    parser.add_argument("--ollama-keep-alive", type=str, default="10m", help="Keep model in memory (e.g. 10m, 60m).")
    parser.add_argument("--ollama-num-thread", type=int, default=4, help="Ollama num_thread (match run_brain).")
    parser.add_argument("--ollama-num-ctx", type=int, default=256, help="Ollama num_ctx (match run_brain).")
    parser.add_argument("--ollama-num-batch", type=int, default=16, help="Ollama num_batch; 0=omit (match run_brain).")
    parser.add_argument("--ollama-stream", action="store_true", default=True, help="Stream Ollama and TTS per chunk (default).")
    parser.add_argument("--no-ollama-stream", action="store_false", dest="ollama_stream", help="Disable streaming; one TTS after full response.")
    parser.add_argument("--ollama-stream-max-words", type=int, default=12, help="Max words per chunk when no sentence end (stream mode).")
    parser.add_argument("--ollama-stream-async", action="store_true", default=True, help="Overlap synth and play for seamless TTS (default).")
    parser.add_argument("--no-ollama-stream-async", action="store_false", dest="ollama_stream_async", help="Stream TTS sequential (synth then play per chunk).")
    parser.add_argument("--onnx-llm", action="store_true", help="Use ONNX LLM instead of Ollama (e.g. SmolLM2-135M-Instruct); no streaming, single TTS.")
    parser.add_argument("--onnx-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Hugging Face model id for ONNX LLM (must have onnx/ subfolder).")
    parser.add_argument("--onnx-max-new-tokens", type=int, default=24, help="ONNX LLM max new tokens.")
    parser.add_argument("--onnx-temperature", type=float, default=0.3, help="ONNX LLM temperature.")
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
        "onnx_llm",
        "onnx_model",
        "onnx_max_new_tokens",
        "onnx_temperature",
    ]:
        if hasattr(args, key):
            d[key] = _arg_to_json_value(getattr(args, key))
    d["voice"] = voice  # resolved voice (kept for backward compatibility)
    print("--- config ---")
    print(json.dumps(d, indent=2))
