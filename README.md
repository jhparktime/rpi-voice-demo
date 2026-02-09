rpi-voice-demo
=============

## Overview

Voice demo for Raspberry Pi:

- STT (baseline): sherpa-onnx (low-latency)
- STT (custom): Faster-Whisper (`distil-small.en`)
- Emotion: ONNX BERT classifier (GoEmotions-style, optional)
- Intent: simple LOCAL vs CLOUD routing based on sentence embeddings
- LOCAL LLM: Ollama (e.g. `smollm2:360m`) on the Pi
- CLOUD LLM: external HTTP API hook (optional)
- TTS (baseline): sherpa-onnx VITS (LJ Speech, single-speaker)
- TTS (custom): Kokoro ONNX

Run from the directory that contains the `Demo/` package:

```bash
python -m Demo
```

The behavior is controlled by `DEMO_MODE`:

- `DEMO_MODE=baseline` (default): sherpa-onnx STT + sherpa-onnx TTS front-end + existing emotion/intent/LLM pipeline.
- `DEMO_MODE=custom`: original Faster-Whisper STT + Kokoro TTS front-end.

Examples:

```bash
# Baseline (sherpa-onnx STT)
python -m Demo

# Custom pipeline (Faster-Whisper + Ollama)
DEMO_MODE=custom python -m Demo --ollama --ollama-model smollm2:360m
```

## Environment flags (RPi)

These flags let you gradually enable features on the Pi:

- `ENABLE_EMOTION` (default: `1`)
  - `1` / `true` / `yes`: run emotion classifier and inject an `EmotionHint` into system prompts
  - `0` / `false` / `no`: skip emotion (no extra latency, no hint)

- `ENABLE_INTENT_ROUTER` (default: `1`)
  - `1`: use a simple intent classifier to route between LOCAL and CLOUD
  - `0`: always treat as `LOCAL` (unless `FORCE_MODE` overrides)

- `FORCE_MODE`
  - Set to `LOCAL` or `CLOUD` to override routing for all turns.
  - Example:
    - `FORCE_MODE=LOCAL` → always use Ollama/local prompt
    - `FORCE_MODE=CLOUD` → always use CLOUD HTTP LLM (if configured)

## LOCAL LLM (Ollama) setup

On Raspberry Pi:

1. Install and pull model:

```bash
ollama pull smollm2:360m
```

2. Run the demo with Ollama enabled:

```bash
ENABLE_EMOTION=1 ENABLE_INTENT_ROUTER=1 \
ollama serve &

source venv/bin/activate
python -m Demo --ollama --ollama-model smollm2:360m
```

(`--ollama-url`, `--ollama-num-thread` 등은 `Demo/stt_tts_cli.py` 옵션으로 조정 가능)

## CLOUD LLM HTTP hook

To enable CLOUD mode, configure these env vars:

- `CLOUD_LLM_URL` (required)
  - Base URL for your HTTP LLM endpoint.
  - The demo POSTs JSON:
    - `{ "prompt": "<user text>", "system": "<system prompt>" }`

- `CLOUD_LLM_API_KEY` (optional)
  - If set, sent as `Authorization: Bearer <API_KEY>`.

If `CLOUD_LLM_URL` is not set, CLOUD routing will return a short error string and effectively fall back to LOCAL behavior from the user's perspective (since TTS will read the ASR text instead).

## Emotion model files

The ONNX BERT emotion classifier expects files under:

- `emotion_onnx_int8/`

This directory is **not stored in git** to keep the repo lightweight.  
On the Raspberry Pi, you can create it with a one-time download script:

```bash
source venv/bin/activate
python download_model.py
```

This will download:

- `config.json`, tokenizer files, vocab from `joeddav/distilbert-base-uncased-go-emotions-student`
- `onnx/model_quantized.onnx` from `Cohee/distilbert-base-uncased-go-emotions-onnx`

and place them into:

```text
emotion_onnx_int8/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
  onnx/
    model_quantized.onnx
```

The first run requires network access to HuggingFace; after that, the directory can be reused offline.

## Kokoro TTS model files

The Kokoro ONNX TTS model and voices are also **not stored in git**.  
On the Raspberry Pi, they are downloaded by the same helper script:

```bash
source venv/bin/activate
python download_model.py
```

This will additionally create:

```text
models/
  kokoro/
    voices-v1.0.bin
    model_quantized.onnx
```

using the official `kokoro-onnx` GitHub release URLs. After this, `python -m Demo` can use Kokoro for TTS.

## sherpa-onnx STT/TTS model files

For the baseline mode, we use sherpa-onnx for both STT and TTS.

- `Demo/stt_sherpa.py` expects an English streaming STT model under:

  ```text
  sherpa_stt/
    tokens.txt
    encoder.onnx
    decoder.onnx
    joiner.onnx
  ```

- `Demo/tts_sherpa.py` expects an English VITS TTS model under:

  ```text
  sherpa_tts/
    model.onnx
    tokens.txt
    espeak-ng-data/
  ```

These files are **not stored in git**.  
`download_model.py` will automatically download and prepare:

- A small English streaming STT model (Zipformer, ~20M params) into `sherpa_stt/`
- An English VITS TTS model (vits-coqui-en-ljspeech + espeak-ng-data) into `sherpa_tts/`
