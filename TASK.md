# TASK.md — exep: Short/Long Router + Long-only smolLM2 Filler + BERT Memory

## Objective
Implement and test an experimental pipeline for our voice agent:

1) Always use Gemini API for the main response (no Local/Cloud routing).
2) Replace the router with a mode selector: `SHORT` vs `LONG` response generation.
3) Generate filler/bridge speech ONLY for `LONG` mode using a fine-tuned smolLM2 (or a safe fallback).
4) Maintain long conversation context via a BERT-based summarization memory layer, inserting:
   - Recent raw turns (last N turns)
   - A rolling summary of earlier turns
   - Pinned facts / key slots (optional)

Deliver everything on a new experiment branch prefixed with `exp/`.

---

## Required Git Workflow
1) Create a new branch name automatically using:
   - prefix `exp/`
   - a short descriptive slug `short-long-router`
   - date + base SHA to avoid collisions

Example branch name:
`exep/short-long-router-YYYYMMDD-<shortsha>`

2) Commit changes with a clear message.
3) Try to push to origin (if credentials/network allow). If push fails, stop and report the exact failure and the branch name.

---

## Step 0 — Repo Discovery (Do First)
Before edits:
- Print repo root structure (top-level files/folders).
- Identify language/tooling:
  - Python? (pyproject.toml / requirements.txt)
  - Node? (package.json)
  - Go/Rust? etc.
- Identify current pipeline entrypoint(s) for the voice agent, especially:
  - Router module (Local/Cloud routing)
  - Gemini API client wrapper
  - smolLM2 (Ollama) invocation
  - Context/memory handling

If there is an existing CLI (e.g., `python -m Demo ...`), locate and document it.

---

## Step 1 — Create Branch (exep/…)
Run commands:
- `git status`
- `git fetch --all`
- checkout base branch (prefer main/master as appropriate)
- pull latest
- generate branch name:
  - `DATE=$(date +%Y%m%d)`
  - `BASE=$(git rev-parse --short HEAD)`
  - `BRANCH="exep/short-long-router-${DATE}-${BASE}"`
- `git checkout -b "$BRANCH"`

---

## Step 2 — Implement Short/Long Router (Mode Selector)
### Goal
Replace “Local vs Cloud” decision with:
- `SHORT`: return short answer fast (1–2 sentences, ~<= 60 tokens target)
- `LONG`: return longer, structured answer (but still avoid extreme verbosity)

### Requirements
- Keep changes behind a feature flag so we can compare old vs new:
  - e.g. `ROUTER_MODE=short_long` or a CLI flag `--router short_long`
- The router should output a `mode` + optional `reason`:
  - `{"mode": "SHORT" | "LONG", "reason": "..."}`
- The router may use existing embeddings/anchors:
  - Create new anchors for `SHORT` and `LONG`
  - Add an uncertainty rule:
    - if top1 score < threshold OR (top1 - top2) < margin -> default `SHORT` (safer & cheaper)
- Add lightweight heuristic overrides:
  - If user asks for “explain / details / why / compare / pros & cons / step-by-step” -> LONG
  - If user asks for “quick / short / just the answer” -> SHORT

### Acceptance criteria
- Router is deterministic enough and testable (unit tests).
- Default is `SHORT` unless clearly `LONG`.

---

## Step 3 — Gemini API: Always Called, Prompted by Mode
### Goal
Always call Gemini for the main response, but with two prompt styles:

#### SHORT prompt rules
- 1–2 sentences max
- No extra preamble
- End with a gentle offer for details (optional): “Want a detailed version?”

#### LONG prompt rules
- Start with a concise 1–2 sentence “short answer”
- Then provide details in 3–6 short bullet points (voice-friendly)
- Avoid repeating the short answer verbatim
- Keep total output bounded (set max output tokens)

### Requirements
- Preserve current Gemini client integration.
- Add prompt templates:
  - `prompts/gemini_short.txt`
  - `prompts/gemini_long.txt`
  (or whichever structure matches the repo)
- Add `max_output_tokens` or equivalent config per mode.

### Acceptance criteria
- Mode changes output length.
- SHORT is consistently short.

---

## Step 4 — Context Memory Layer (BERT Summarization)
### Goal
Reduce prompt growth while maintaining context quality.

### Policy
- Keep last `N_RECENT_TURNS` raw (default 2–3).
- Summarize older turns into `rolling_summary`.
- Optionally maintain `pinned_facts` (names, locations, preferences) as simple key/value.

### Implementation Notes
- Use existing BERT/Emotion/Intent components if available.
- If there is no summarizer yet:
  - Add a simple summarization module interface:
    - `update_memory(history) -> {rolling_summary, pinned_facts, recent_raw_turns}`
  - If BERT summarization is heavy/unavailable, add a temporary fallback summarizer:
    - extractive (take last user+assistant pairs) or rule-based until BERT is wired
- IMPORTANT: Avoid summary drift:
  - Do not summarize the most recent turns.
  - Store user-provided facts verbatim in pinned facts (if present).

### Acceptance criteria
- Gemini prompt includes:
  - rolling summary + recent turns
- Token usage does not grow unbounded across long conversations.

---

## Step 5 — smolLM2 Filler/Bridge (LONG-only)
### Goal
When mode is LONG and Gemini response is not ready yet, generate a short filler line.

### Constraints (must be enforced)
- Filler must NOT answer the user question.
- Must NOT introduce facts, names, numbers, or advice.
- Must be max 1 short sentence (3–8 words preferred).
- Must be safe even for small models.

### Implementation
- Trigger filler only if Gemini call exceeds a delay gate:
  - e.g., do not speak filler if Gemini returns within 600–900ms.
- Use fine-tuned smolLM2 if configured:
  - config key: `FILLER_PROVIDER=smollm2` or CLI flag
- Add a strict validator:
  - Reject outputs containing:
    - `?` (questions)
    - digits
    - named entity-like patterns (basic heuristic)
    - more than one sentence
    - > 12 tokens
  - If invalid, fall back to a safe fixed phrase list.

### Fallback fixed list (English)
- “One moment.”
- “Just a sec.”
- “Checking that now.”
- “Working on it.”
- “Let me check.”

### Acceptance criteria
- No filler on SHORT mode.
- LONG mode produces filler only when needed.
- Filler never contains “answer-like” content (validator + tests).

---

## Step 6 — Wire into the Voice Agent Pipeline
### Requirements
- Ensure the pipeline order is:
  1) ASR text
  2) Router decides SHORT/LONG
  3) Prepare memory (summary + recent raw)
  4) Start Gemini request
  5) If LONG and delay > gate -> speak filler
  6) Speak Gemini answer when ready
- Keep interrupt/barge-in behavior intact (do not regress).
- Add logs/metrics:
  - mode decision
  - TTFS (time to first sound)
  - filler triggered or not
  - Gemini latency

---

## Step 7 — Tests + Lint + Demo Command
### Tests
Add/extend tests for:
- Router mode selection
- Memory summarization policy (recent raw + rolling summary)
- Filler validator + fallback behavior
- “LONG-only filler” gate logic

### Lint / Formatting
- Run existing lint command(s).
- Fix lint errors and keep diffs minimal.

### Demo
- Update or add a demo script/command showing:
  - SHORT query -> no filler
  - LONG query -> filler after gate + final answer

---

## Step 8 — Run Commands (Auto-detect, then execute)
Auto-detect the project’s standard commands. Prefer in order:

### Python
- `python -m Demo` (or `pytest -q`)
- `ruff check .` / `black --check .` if present

If commands are not found:
- Identify the correct ones from config files and run them.

---

## Step 9 — Commit (and Push if Possible)
- `git status` should be clean except intended changes.
- Commit message:
  - `exp: short/long router + long-only filler + bert memory`
- Attempt:
  - `git push -u origin "$BRANCH"`
- If push fails:
  - Do NOT retry endlessly.
  - Print the error and instructions to push manually.

---

## Deliverables
- New router mode selector (short/long)
- Gemini prompt templates (short/long)
- Context memory module using BERT summarization policy (or stub + interface)
- smolLM2 long-only filler with strict validator + fallback
- Tests + updated demo command
- A single commit on a new `exp/...` branch

---

## Final Output Requirements (when finished)
Print:
- Branch name
- What changed (bullet list)
- How to run tests
- How to run demo
- If push succeeded: remote branch URL if available
- If push failed: exact error + next steps