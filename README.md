# ASTRA — Autonomous Space Task Recognition & Assistance

**SIH26174 — AI Human Activity Recognition for On-board BAS Experiments**
Organization: ISRO · Category: Software

> Prototype / proof-of-concept, built on the ground with a webcam and two
> colored boxes, as explicitly permitted by the problem statement. This is
> **not** a space-ready, flight-certified, or edge-deployed system — see
> [Known Limitations](#known-limitations--honesty-notes).

ASTRA watches a person perform a predefined multi-step procedure through a
webcam, recognizes each action, validates it against the expected
experiment sequence, and gives corrective guidance (on-screen and by voice)
when a step is wrong, skipped, or performed out of order — then verifies
the correction and lets the experiment continue. The goal is to demonstrate
**procedural intelligence**: not just "what object is this" or "what pose is
this," but *what is supposed to happen next, and was it?*

## Current status: Phase 0 + Phase 1 complete

This repository is being built incrementally (see [Development Phases](#development-phases)).
**What exists right now and is fully runnable:**

- Full repo scaffold, config system, and the 5-step sample experiment as data (not hardcoded).
- A threaded camera capture layer with **three interchangeable sources** — `webcam`, `video_file`, and `synthetic` (a procedurally generated test pattern, used here because this dev environment has no physical camera).
- The **real, deterministic sequence-validation engine** (`backend/experiment/validator.py`) — not a mock — checked against the actual experiment config. It already implements CORRECT / OUT_OF_SEQUENCE / REPEATED_STEP / RECOVERED / LOW_CONFIDENCE / COMPLETE.
- Mock perception + mock temporal action recognizer (per the spec's "build mock inference interfaces first" guidance) that play back the canonical demo scenario: two correct steps → a deliberate wrong action → detection → recovery → completion. This exercises the *entire* CAMERA → PERCEPTION → ACTION → SEQUENCE → OUTPUT loop end-to-end today, before any real computer vision is wired in.
- A FastAPI backend exposing the REST + WebSocket API, verified live (see [Verification](#verification-already-performed)).
- 15 passing unit tests for experiment loading and sequence validation (no camera/hardware required).

**Not yet built** (see the phase list): real MediaPipe/HSV perception, hand-object interaction reasoning, the rule-based temporal recognizer, the full state machine (with SKIPPED_STEP vs OUT_OF_SEQUENCE distinguished), voice guidance, persistent logging/reports, video recording, the React dashboard, dataset tooling, and training scripts. Each is scoped and will replace the current mocks/stubs one phase at a time.

## Why a virtual environment

This project uses an isolated Python virtual environment (`.venv/`) rather
than the system interpreter, so ASTRA's dependencies (FastAPI, OpenCV,
MediaPipe, etc.) never collide with anything else on your machine and the
exact versions in `requirements.txt` are what actually run.

## Architecture

```
CAMERA → VIDEO CAPTURE → FRAME PREPROCESSING
  → PERCEPTION (objects / pose / hands)
  → TEMPORAL ACTION RECOGNITION
  → EXPERIMENT STATE MACHINE + SEQUENCE VALIDATION
  → DECISION ENGINE (CORRECT / WRONG / SKIPPED / REPEATED / RECOVERY)
  → OUTPUT (GUI, voice, event log, recording, optional stream)
```

**Perception vs. reasoning is a hard separation in the code** (spec principle):
Perception (`backend/perception/`) only answers "what do I see?" — objects,
pose, hands. It knows nothing about experiments. The sequence layer
(`backend/experiment/`) only answers "what does this mean for the
procedure?" — it consumes a recognized action and the known procedure, and
is deliberately **not** an ML model: the procedure is known in advance, so
correctness is decided with explicit, inspectable rules.

**Why HSV color detection instead of a trained object detector for the
red/blue boxes:** a COCO-pretrained detector has no concept of "red box" vs
"blue box," and training a custom one requires labeled data before *any*
demo works — which would block the MVP entirely. HSV thresholding needs zero
training data, runs in milliseconds on CPU, and returns the exact same
`{class, confidence, bbox}` shape a trained detector would — so it's a
drop-in replacement once/if a trained model is added later (see
`docs/architecture.md`, added in a later phase).

Full repo layout:

```
astra/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── api/routes.py           # REST endpoints
│   ├── websocket/live.py       # /ws/live
│   ├── config/settings.py      # config.yaml + env overrides
│   ├── services/
│   │   ├── camera_service.py   # threaded capture: webcam | video_file | synthetic
│   │   ├── inference_service.py# orchestrates the per-frame pipeline + WS broadcast
│   │   ├── synthetic_scene.py  # procedural test-pattern generator
│   │   └── frame_utils.py
│   ├── perception/
│   │   ├── base.py             # shared schemas (DetectedObject, PerceptionFrame, ...)
│   │   └── interfaces.py       # PerceptionEngine / ActionRecognizer / SequenceEngine ABCs
│   ├── experiment/
│   │   ├── experiment_loader.py# loads + validates experiment JSON
│   │   └── validator.py        # real rule-based sequence validation
│   └── mocks/mock_engines.py   # scripted mock perception + action recognizer
├── experiments/bas_sample_001.json
├── config/config.yaml
├── tools/
│   ├── test_camera.py
│   ├── make_synthetic_clip.py
│   └── _ws_smoke_test.py       # manual end-to-end WebSocket check (not pytest)
├── tests/                      # pytest — no camera required
├── frontend/                   # scaffolded, not yet built (Phase 10)
└── data/{raw,processed,annotations,models,recordings,reports}
```

## The sample experiment

`experiments/bas_sample_001.json` defines a 5-step tabletop stand-in for an
on-board BAS experiment — nothing about it is hardcoded in Python:

1. PICK the RED container
2. PLACE the RED container in the experiment area
3. PICK the BLUE container
4. PLACE the BLUE container in the experiment area
5. COMPLETE the experiment

Swapping in a different procedure later is a matter of writing a new JSON
file in the same shape, not touching code.

## Installation

Requires Python 3.11+.

```bash
cd astra
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the backend

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

By default `config/config.yaml` sets `camera.source: synthetic`, so this
runs with **no webcam required** — useful for development, CI, or this
sandbox. Open `http://localhost:8000/docs` for interactive API docs.

To use a real webcam instead (on your own machine):

```yaml
# config/config.yaml
camera:
  source: webcam
  camera_index: 0
```

or override without editing the file: `CAMERA_SOURCE=webcam CAMERA_INDEX=0 uvicorn backend.main:app`.

### Try it

```bash
# Check status
curl http://localhost:8000/api/status

# Start the experiment (spins up the camera + inference loop)
curl -X POST http://localhost:8000/api/experiment/start

# Watch the sequence unfold live (requires: pip install websockets)
python tools/_ws_smoke_test.py
```

`tools/_ws_smoke_test.py` prints every `sequence_event` as it happens; with
the default synthetic source and mock engines you'll see the canonical demo
scenario play out automatically: two correct steps, a deliberate deviation,
detection, recovery, and completion — the same closed loop described in the
problem statement's target demo flow.

### Camera / clip utilities

```bash
# Sanity-check any camera source in isolation (saves a sample frame):
python tools/test_camera.py --source synthetic --seconds 3 --no-display

# On your own machine with a webcam attached:
python tools/test_camera.py --source webcam --index 0

# Render the synthetic test pattern to an .mp4 (useful as a video_file source):
python tools/make_synthetic_clip.py --seconds 25 --out data/recordings/demo_clip.mp4
```

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

15 tests currently cover experiment-config loading/validation and every
sequence-validation scenario named in the spec (correct sequence, wrong
action, skipped step, recovery, repeated step, completion, low confidence,
reset) — all without needing a camera or any ML model.

## Verification already performed

Everything above has actually been run, not just written:

- `pytest tests/` → **15/15 passed**.
- Backend started with `uvicorn` and exercised live: `/`, `/api/status`,
  `/api/experiment`, `POST /api/experiment/start` all returned correct
  responses; camera FPS measured at ~29.8 fps on the synthetic source.
- `/ws/live` streamed the full canonical demo scenario end-to-end —
  `PICK_RED_BOX` (CORRECT) → `PICK_BLUE_BOX` (OUT_OF_SEQUENCE, expected
  `PLACE_RED_BOX`) → `PLACE_RED_BOX` (RECOVERED) → `PICK_BLUE_BOX` (CORRECT)
  → `PLACE_BLUE_BOX` (CORRECT) → `COMPLETE_EXPERIMENT` (COMPLETE) — with
  live-camera JPEG frames also arriving over the same socket.

No performance numbers (FPS, latency, accuracy) beyond what's explicitly
stated above have been measured; none are claimed.

## Development Phases

Phases 0–1 are complete (this delivery). Remaining phases, in order:

2. Real perception — MediaPipe pose/hands + HSV red/blue box + experiment-area detection
3. Hand-object interaction reasoning (approach/touch/hold/move/release/place)
4. Rule-based temporal action recognizer + stability/hysteresis smoothing
5. Full experiment state machine (SKIPPED_STEP vs OUT_OF_SEQUENCE distinguished)
6. Decision engine refinement (all statuses, confidence-aware)
7. Offline voice guidance (async, non-blocking)
8. Persistent event logger + `experiment_report.json`
9. Wire the real pipeline into the backend (replacing mocks); video recording
10. Polished React + Vite mission-control dashboard
11. Dataset recording tool
12. Training/evaluation pipeline (baseline, then optional LSTM)
13. Demo-mode polish, packaging, full docs

## Known limitations / honesty notes

- No real computer vision runs yet — object/action recognition is a
  scripted mock. This is intentional (spec section 35: "mock inference
  interfaces... before the real ML pipeline is complete") and is being
  replaced incrementally, not skipped.
- The sequence validator currently merges "skipped step" and "true
  out-of-order" into a single `OUT_OF_SEQUENCE` status; splitting them
  precisely needs the full state machine (Phase 5).
- No GPU/CUDA is used or claimed; MediaPipe (added Phase 2) runs on CPU.
- No accuracy, latency, or FPS claims beyond the measured numbers stated
  above under Verification.
- Built and tested in a Linux cloud sandbox with no physical camera;
  webcam-source testing on Windows with real hardware is the next
  concrete step once a machine with a camera is available.

## Tech stack

Python 3.11, FastAPI, Uvicorn, WebSockets, Pydantic, OpenCV, MediaPipe
(installed, not yet used), pyttsx3 (installed, not yet used), NumPy,
Pandas, pytest. React + Vite + TypeScript + Tailwind planned for the
dashboard (Phase 10).
