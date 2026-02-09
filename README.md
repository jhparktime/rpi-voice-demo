rpi-voice-demo
=============

## Overview

Voice demo for Raspberry Pi:

- STT: Faster-Whisper (`distil-small.en`)
- Emotion: ONNX BERT classifier (GoEmotions-style, optional)
- Intent: simple LOCAL vs CLOUD routing based on sentence embeddings
- LOCAL LLM: Ollama (e.g. `smollm2:360m`) on the Pi
- CLOUD LLM: external HTTP API hook (optional)
- TTS: Kokoro ONNX

Run from the directory that contains the `Demo/` package:

```bash
python -m Demo
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
python download_emotion_model.py
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
