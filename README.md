rpi-voice-demo
=============

## Overview

Voice demo for Raspberry Pi — sherpa-onnx based STT/TTS with emotion-aware
intent routing and natural conversation flow:

- **STT**: sherpa-onnx streaming Zipformer (OnlineRecognizer, 4 modes)
  - `--sentence-streaming`: sentence-by-sentence mode — each 0.8s pause triggers processing
  - `--streaming` (default): Enter → mic stream → endpoint auto-detection → text
  - `--no-streaming`: Enter → fixed N-second recording → chunked transcription
  - `--vad`: always-listening — silero-vad detects speech automatically, no Enter needed
- **Emotion**: ONNX BERT classifier (GoEmotions-style, optional)
- **Intent**: LOCAL vs CLOUD routing based on sentence embeddings
- **LOCAL LLM**: Ollama (e.g. `smollm2:360m`) on the Pi — empathic filler + response
- **CLOUD LLM**: external HTTP API hook (optional) — sLLM provides semantic filler during CLOUD latency
- **TTS**: sherpa-onnx VITS (LJ Speech, single-speaker)

The sLLM (local Ollama) provides **semantic fillers** ("Let me look that up…")
during CLOUD LLM latency, so the conversation feels natural even when the main
answer takes a few seconds.

## Quick start

```bash
# 1. Set up venv and install deps (on RPi)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download all models (emotion, STT, TTS, VAD)
python download_model.py

# 3. Run the demo (default: streaming mode with endpoint detection)
python -m Demo

# 4. With Ollama LLM enabled
ollama serve &
python -m Demo --ollama --ollama-model smollm2:360m

# 5. Sentence-streaming mode (most natural — speak multiple sentences with pauses)
python -m Demo --sentence-streaming

# 6. Streaming mode (default — press Enter, speak, endpoint auto-detects end)
python -m Demo --streaming

# 7. VAD always-listening mode (no Enter needed, just speak)
python -m Demo --vad

# 8. Fixed-duration fallback (3 seconds)
python -m Demo --no-streaming --record-seconds 3
```

## STT modes

| Flag | Trigger | Description |
|------|---------|-------------|
| `--sentence-streaming` | Enter key | Continuous listening: each sentence (0.8s silence) triggers immediate processing. Most natural for multi-sentence conversations. |
| `--streaming` (default) | Enter key | Opens mic, streams to OnlineRecognizer, endpoint detection auto-stops |
| `--no-streaming` | Enter key | Records for `--record-seconds`, then feeds entire buffer in chunks |
| `--vad` | Automatic | Always-listening: silero-vad detects speech start/end, no Enter needed |

Common options:
- `--sentence-silence 0.8` — silence duration (seconds) for sentence boundaries in `--sentence-streaming` mode (default 0.8s)
- `--max-listen-seconds 15` — timeout for streaming/VAD modes (default 15s)
- `--record-seconds 3` — duration for `--no-streaming` mode (default 3s)

## Environment flags (RPi)

- `ENABLE_EMOTION` (default: `1`)
  - `1` / `true` / `yes`: run emotion classifier and inject an `EmotionHint` into system prompts
  - `0` / `false` / `no`: skip emotion (no extra latency, no hint)

- `ENABLE_INTENT_ROUTER` (default: `1`)
  - `1`: use intent classifier to route between LOCAL and CLOUD
  - `0`: always treat as `LOCAL` (unless `FORCE_MODE` overrides)

- `ENABLE_CLOUD_FILLER` (default: `1`)
  - `1`: for CLOUD requests, sLLM generates a quick spoken filler first
  - `0`: skip filler, wait silently for CLOUD response

- `FORCE_MODE`
  - Set to `LOCAL` or `CLOUD` to override routing for all turns.

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

## CLOUD LLM (Gemini or custom HTTP)

### Option A: Google Gemini (free tier, e.g. Gemini 2.5 Flash)

1. Get an API key: [Google AI Studio](https://aistudio.google.com/apikey) → Create API key.
2. Set the key via **environment variable only** (do not commit it):

   **Shell (current session):**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   python -m Demo --sentence-streaming
   ```

   **Or use a `.env` file (recommended, not committed):**
   ```bash
   # In project root, create .env (already in .gitignore)
   echo 'GEMINI_API_KEY=your-api-key-here' > .env
   ```
   Then load it before running (e.g. `set -a && source .env && set +a` or use `python-dotenv` in your script).

3. Optional: `CLOUD_LLM_MODEL` (default: `gemini-2.5-flash`).

### Option B: Custom HTTP endpoint

- `CLOUD_LLM_URL` (required) — Base URL for your HTTP LLM endpoint.
  POSTs JSON: `{ "prompt": "<user text>", "system": "<system prompt>" }`
- `CLOUD_LLM_API_KEY` (optional) — sent as `Authorization: Bearer <API_KEY>`.

## Model files

All models are **not stored in git**. Run `python download_model.py` to download:

### Emotion classifier
```
emotion_onnx_int8/
  config.json, tokenizer_config.json, special_tokens_map.json, vocab.txt
  onnx/model_quantized.onnx
```

### sherpa-onnx STT (streaming Zipformer, ~20M params)
```
sherpa_stt/
  tokens.txt, encoder.onnx, decoder.onnx, joiner.onnx
```

### sherpa-onnx TTS (VITS LJ Speech)
```
sherpa_tts/
  model.onnx, tokens.txt, espeak-ng-data/
```

### sherpa-onnx VAD (silero-vad)
```
sherpa_vad/
  silero_vad.onnx
```

## Architecture

```
Mic → [VAD] → OnlineRecognizer (streaming STT) → text
                                                    ↓
                                         Emotion + Intent Router
                                           ↓               ↓
                                        LOCAL            CLOUD
                                     (Ollama sLLM)    (sLLM filler → HTTP LLM)
                                           ↓               ↓
                                    sherpa-onnx VITS TTS → Speaker
```
