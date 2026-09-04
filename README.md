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

## Current status: Phases 0–12 complete

The system is fully wired end-to-end: real webcam perception, rule-based
action recognition, full sequence validation, voice guidance, persistent
logging, video recording, and a live React dashboard. See
[Development Phases](#development-phases) for what each phase added, and
`docs/demo.md` for how to actually run it.

**What's real and running today:**

- Threaded camera capture with three interchangeable sources — `webcam`,
  `video_file`, `synthetic`.
- **Real perception**: HSV color detection (red/blue box + experiment
  area) and MediaPipe pose + hand tracking, composed into one
  `RealPerceptionEngine`.
- **Hand-object interaction reasoning**: a per-object state machine
  (approach → touch → hold → move → release/place) driving actual pick/
  place detection, not just raw landmark positions.
- **Rule-based temporal action recognizer**: turns interaction state
  transitions into stability-debounced `PICK_*`/`PLACE_*` actions.
- **Full sequence validation**: CORRECT / WRONG_OBJECT / SKIPPED_STEP /
  OUT_OF_SEQUENCE / REPEATED_STEP / LOW_CONFIDENCE / RECOVERED / COMPLETE,
  each genuinely distinguished by *why* a mismatch happened, not just that
  one did — plus automatic advancement through the final "close out the
  experiment" step (no gesture exists for that one).
- **Offline voice guidance** (pyttsx3, background thread) — proactively
  tells the astronaut what to do next on success, explains what went wrong
  on a deviation.
- **Persistent logging**: every event to a JSONL run log, plus a final
  `experiment_report.json` (per-step first-try-correct vs.
  needed-correction, total deviations, duration).
- **Annotated video recording** of every run.
- **A live React (Next.js) dashboard** — camera feed with detection
  overlay, step checklist, error/recovery banners, scrolling event log,
  Start/Stop/Reset — all driven by real REST + WebSocket data, no scripted
  replay.
- **Dataset recording + training-pipeline scaffolding** (`tools/
  record_dataset.py`, `training/`) — explicitly a placeholder; the real
  action recognition is rule-based and needs none of this to work.
- The original scripted mock perception/action engines are **still there**
  as an opt-in fallback (`features.use_mock_engines`) for headless/CI
  verification with no camera or props.
- 79 passing unit tests — every layer (color detection, interaction
  reasoning, action recognition, sequence validation, voice, logging,
  recording, feature extraction) is independently, deterministically
  tested with no camera or hardware required.

## Why a virtual environment

This project uses an isolated Python virtual environment (`.venv/`) rather
than the system interpreter, so ASTRA's dependencies (FastAPI, OpenCV,
MediaPipe, etc.) never collide with anything else on your machine and the
exact versions in `requirements.txt` are what actually run.

## Architecture

See **`docs/architecture.md`** for the full picture — layer separation,
why every layer is rule-based rather than a trained model, the mismatch
classification logic, the gotchas already found and fixed, and the
complete directory layout. Short version:

```
CAMERA → VIDEO CAPTURE → FRAME PREPROCESSING
  → PERCEPTION (objects / pose / hands)
  → TEMPORAL ACTION RECOGNITION
  → EXPERIMENT STATE MACHINE + SEQUENCE VALIDATION
  → DECISION ENGINE (CORRECT / WRONG_OBJECT / SKIPPED_STEP / OUT_OF_SEQUENCE
     / REPEATED_STEP / LOW_CONFIDENCE / RECOVERED / COMPLETE)
  → OUTPUT (dashboard, voice, event log, recording)
```

**Perception vs. reasoning is a hard separation in the code**: perception
(`backend/perception/`) only answers "what do I see?" and knows nothing
about experiments; temporal recognition (`backend/temporal/`) only
answers "what is happening, over time?"; the sequence layer
(`backend/experiment/`) only answers "what does this mean for the
procedure?" — and is deliberately **not** an ML model, since the
procedure is known in advance.

## The sample experiment

`experiments/bas_sample_001.json` defines a 5-step tabletop stand-in for an
on-board BAS experiment — nothing about it is hardcoded in Python:

1. PICK the RED container
2. PLACE the RED container in the experiment area
3. PICK the BLUE container
4. PLACE the BLUE container in the experiment area
5. COMPLETE the experiment (auto-advanced — no gesture for this one)

Swapping in a different procedure later is a matter of writing a new JSON
file in the same shape, not touching code.

## Installation

```bash
cd astra_phase0_1
python -m venv .venv
.venv\Scripts\activate           # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

cd frontend
npm install
```

## Running the full system

See **`docs/demo.md`** for the complete walkthrough (webcam setup, HSV
calibration, troubleshooting, the mock-engine fallback). Quick version:

```bash
# Terminal 1 — backend
CAMERA_SOURCE=webcam CAMERA_INDEX=0 uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open `http://localhost:3000`. By default `config/config.yaml` sets
`camera.source: synthetic`, so the backend also runs with **no webcam
required** — useful for development/CI. Interactive API docs are at
`http://localhost:8000/docs`.

### Camera / clip utilities

```bash
# Sanity-check any camera source in isolation:
python tools/test_camera.py --source webcam --index 0

# Exercise real perception standalone, without the backend running:
python tools/live_preview.py --source webcam --index 0

# Interactively tune HSV ranges if detection is unreliable under your lighting:
python tools/calibrate_hsv.py --source webcam --color red_box

# Record a labeled dataset (Phase 12 groundwork):
python tools/record_dataset.py --source webcam
```

## Testing

```bash
pytest tests/ -v
```

79 tests cover every layer deterministically — HSV color detection,
hand-object interaction reasoning, the temporal action recognizer,
full sequence validation (all 8 statuses), voice message construction,
persistent logging, video recording, and feature extraction — none of it
needs a camera, a display, or real hardware. Interactive tools
(`live_preview.py`, `calibrate_hsv.py`, `record_dataset.py`) and the full
wired backend were additionally verified live against real webcam hardware
during development — see the phase commit messages for specifics.

## Development Phases

All complete:

0. Repo scaffold, config system, sample experiment as data.
1. Threaded camera capture, FastAPI + WebSocket skeleton, scripted mocks
   exercising the full pipeline shape end-to-end.
2. Real perception — MediaPipe pose/hands + HSV red/blue box + experiment-
   area detection.
3. Hand-object interaction reasoning (approach/touch/hold/move/release/
   place).
4. Rule-based temporal action recognizer + stability/hysteresis smoothing.
5/6. Full sequence validation (WRONG_OBJECT vs SKIPPED_STEP vs
   OUT_OF_SEQUENCE distinguished) + COMPLETE_EXPERIMENT auto-advance.
7. Offline voice guidance (async, non-blocking).
8. Persistent event logger + `experiment_report.json`.
9. Real perception + action recognition wired into the backend
   (mocks kept as opt-in fallback); annotated video recording.
10. React (Next.js) mission-control dashboard, live REST/WebSocket data.
11. Dataset recording tool.
12. Training/evaluation pipeline stubs (explicitly placeholder).
13. Docs (this pass) + packaging.

## Known limitations / honesty notes

- Real inference throughput varies with machine load — measured anywhere
  from ~2 fps to ~21 fps on the same development hardware. Not a bug to
  chase for the MVP; see `docs/architecture.md`.
- `tools/record_dataset.py` captures single isolated frames, not temporal
  windows, so `training/`'s displacement/velocity features carry no
  signal on data recorded today — see `docs/dataset.md`.
- `training/evaluate.py` genuinely reports `N/A — insufficient evaluation
  data` — no dataset has been recorded in this repo, and no accuracy
  number is fabricated to fill the gap.
- HSV detector "confidence" and the rule-based action recognizer's
  confidence are both documented heuristics, not calibrated probabilities
  — never presented as more than that.
- No GPU/CUDA is used or claimed; MediaPipe runs on CPU.
- No performance claims beyond what's been explicitly measured and stated
  in the relevant module or doc.

## Tech stack

Python 3.10+, FastAPI, Uvicorn, WebSockets, Pydantic, OpenCV, MediaPipe,
pyttsx3, NumPy, Pandas, scikit-learn, pytest. Next.js (App Router) +
TypeScript + Tailwind v4 for the dashboard.
