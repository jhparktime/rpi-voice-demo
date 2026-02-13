# Research Direction: Practical Latency-Masked Voice Agents on Edge Devices

## 1. Scope and Repo Mapping

This plan is grounded in the current implementation of `rpi-voice-demo`.

- Routing core: `Demo/router_anchors_runtime.py`
- Brain execution (LOCAL/CLOUD + filler parallelism): `Demo/stt_tts_test.py`
- Filler system prompt builder: `Demo/text_utils.py`
- Cloud LLM wrapper (non-streaming request/response): `Demo/cloud_llm.py`
- CLI control for ablations: `Demo/stt_tts_cli.py`

Core research question:

How can we design a latency-masked, interruptible, hallucination-safe voice agent under non-streaming cloud constraints and edge compute limits?

## 2. Repo-Verified Design Assumptions

1. Cloud path is effectively non-streaming at orchestration level.
   - Cloud is called as a blocking future and used after completion.
2. Filler and cloud are executed in parallel.
   - Filler generation and cloud request run concurrently in the CLOUD path.
3. Router is conditional and confidence-aware (anchors + margin).
   - LOCAL/CLOUD is selected from anchor similarity with margin logic.
4. Filler is short but still partially free-form.
   - This leaves role-break/hallucination risk for small models.

## 3. Hypotheses

- H1: Conditional routing reduces cloud invocation with minimal quality drop.
- H2: Parallel filler + cloud significantly improves TTFS and perceived responsiveness.
- H3: Strong constrained filler policy lowers hallucination/role-break versus current free-form filler prompt.
- H4: Latency-aware adaptive filler scheduling outperforms fixed-length filler in masking efficiency.

## 4. Experimental Variants

1. `LOCAL_ONLY`
- `FORCE_MODE=LOCAL`
- Cloud fully disabled by routing override.

2. `CLOUD_NO_FILLER`
- `FORCE_MODE=CLOUD --no-cloud-filler`
- Measures raw cloud wait and UX penalty.

3. `CLOUD_FILLER_CURRENT`
- Current repo behavior (parallel filler + cloud).

4. `CLOUD_FILLER_CONSTRAINED` (proposed)
- Replace free-form filler with constrained whitelist/template policy.

5. `ROUTER_PLUS_SCHEDULER` (proposed)
- Current router + adaptive filler length/cutoff by predicted cloud latency bins.

## 5. Data and Scenario Design

Use fixed prompts grouped into:

- LOCAL-friendly
  - small talk, check-ins, lightweight emotional support
- CLOUD-required
  - factual QA, technical explanations, coding/math
- Boundary/ambiguous
  - requests near local/cloud decision threshold

For each prompt, run multiple trials and record all timing logs.

## 6. Metrics

- Cloud Invocation Rate (CIR) ↓
- Routing Error Rate (RER) ↓
- Time to First Sound (TTFS) ↓
- End-to-End turn latency (E2E) ↓
- Masking Efficiency Ratio (MER) ↑
- Filler Hallucination Rate (FHR) ↓
- Named Entity Leakage in filler (NEL) ↓
- User-rated responsiveness / trust (MOS or Likert) ↑

## 7. Evaluation Table Template

| Variant | CIR ↓ | RER ↓ | TTFS (ms) ↓ | E2E (s) ↓ | MER ↑ | FHR ↓ | NEL ↓ | User MOS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LOCAL_ONLY |  |  |  |  |  |  |  |  |
| CLOUD_NO_FILLER |  |  |  |  |  |  |  |  |
| CLOUD_FILLER_CURRENT |  |  |  |  |  |  |  |  |
| CLOUD_FILLER_CONSTRAINED |  |  |  |  |  |  |  |  |
| ROUTER_PLUS_SCHEDULER |  |  |  |  |  |  |  |  |

## 8. Latency Modeling

Let:

- `t_u`: user utterance end
- `t_f`: filler audio start
- `t_c`: cloud response ready
- `t_a`: first cloud-answer audio start
- `d_f`: filler playback duration
- `w_c = t_c - t_u`: cloud waiting time

1. Time to First Sound:

\[
TTFS = t_f - t_u
\]

2. Masking Efficiency Ratio:

\[
MER = \frac{\min(d_f,\; w_c)}{w_c}
\]

3. Residual unmasked wait:

\[
W_{residual} = \max(0,\; w_c - d_f)
\]

4. Multi-objective system cost:

\[
J = \alpha E[Latency] + \beta E[CloudCost] + \gamma E[RoutingError] + \delta E[FillerRisk]
\]

## 9. Practical Runbook (Current Repo)

### 9.1 Baseline A: CLOUD without filler

```bash
FORCE_MODE=CLOUD ENABLE_CLOUD_FILLER=0 \
python -m Demo --ollama --no-cloud-filler
```

### 9.2 Baseline B: CLOUD with current filler

```bash
FORCE_MODE=CLOUD ENABLE_CLOUD_FILLER=1 \
python -m Demo --ollama --cloud-filler
```

### 9.3 Baseline C: LOCAL only

```bash
FORCE_MODE=LOCAL ENABLE_INTENT_ROUTER=0 \
python -m Demo --ollama
```

### 9.4 Router-enabled production mode

```bash
ENABLE_INTENT_ROUTER=1 ENABLE_CLOUD_FILLER=1 \
python -m Demo --ollama
```

## 10. Immediate Next Implementation Tasks

1. Add structured per-turn log output (`jsonl`) for:
- route decision/confidence
- filler generation latency
- cloud latency
- TTFS
- E2E

2. Add constrained filler mode toggle:
- `--filler-policy current|template|whitelist`

3. Add cloud latency binning:
- short/medium/long bins to adjust filler length.

4. Add interrupt-first controls:
- cancel cloud future on barge-in
- preempt filler audio playback

## 11. Contribution Framing (Paper Draft)

1. Conditional cloud invocation for edge-cloud voice agents.
2. Latency-aware masking scheduler for non-streaming cloud responses.
3. Hallucination-safe semantic filler policy for ultra-small local LLMs.
4. Interrupt-first architecture for practical realtime voice interaction.
