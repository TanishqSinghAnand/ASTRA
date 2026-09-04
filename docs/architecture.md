# ASTRA — Architecture

SIH26174 — AI Human Activity Recognition for on-board BAS experiments.
This is the canonical reference for how the system is put together and
*why* each piece works the way it does — the decisions below were made
once, early, and every later phase was built to preserve them.

## Pipeline

```
CAMERA → VIDEO CAPTURE → FRAME PREPROCESSING
  → PERCEPTION (objects / pose / hands)
  → TEMPORAL ACTION RECOGNITION
  → EXPERIMENT STATE MACHINE + SEQUENCE VALIDATION
  → DECISION ENGINE (CORRECT / WRONG_OBJECT / SKIPPED_STEP / OUT_OF_SEQUENCE /
     REPEATED_STEP / LOW_CONFIDENCE / RECOVERED / COMPLETE)
  → OUTPUT (WebSocket dashboard, voice, event log, recording)
```

Each arrow is a real module boundary, not just a conceptual one — see
"Layer separation" below.

## Layer separation (the one rule every phase preserves)

- **`backend/perception/`** only answers *"what do I see?"* — objects,
  pose, hands. It has never heard of the word "experiment" and never will.
  `color_detector.py` (HSV), `pose.py` / `hands.py` (MediaPipe), and
  `interaction.py` (hand-object state machine — Phase 3) all live here.
  `real_engine.py`'s `RealPerceptionEngine` composes the three into one
  `PerceptionEngine`.
- **`backend/temporal/`** only answers *"what is happening, over time?"* —
  it consumes a rolling stream of `PerceptionFrame`s and turns interaction
  state transitions into a discrete `ActionPrediction` (e.g.
  `PICK_RED_BOX`). It doesn't know what step of the procedure that
  corresponds to.
- **`backend/experiment/`** only answers *"what does this mean for the
  procedure?"* — `validator.py`'s `RuleBasedSequenceEngine` consumes an
  `ActionPrediction` and the loaded experiment definition and decides
  CORRECT / WRONG_OBJECT / SKIPPED_STEP / etc. It's deliberately **not**
  ML: the procedure is known in advance (`experiments/*.json`), so
  correctness is decided by explicit, inspectable rules, not a model.

Everything upstream of the sequence layer is swappable without touching
anything downstream: `backend/perception/interfaces.py` defines
`PerceptionEngine` / `ActionRecognizer` / `SequenceEngine` as abstract
contracts, and `backend/mocks/mock_engines.py`'s scripted mocks satisfy
the exact same interfaces `RealPerceptionEngine` /
`RuleBasedActionRecognizer` do — `inference_service.py` picks one pair at
construction time (`features.use_mock_engines`) and nothing else in the
pipeline changes.

## Why rules instead of trained models, at every layer

| Layer | Approach | Why |
|---|---|---|
| Object detection (red/blue box) | HSV color thresholding | A COCO-pretrained detector has no concept of "red box" vs "blue box"; training a custom one needs labeled data before *any* demo works. HSV needs none, runs in low-single-digit ms on CPU, and returns the same `{class, confidence, bbox}` shape a trained detector would — a drop-in replacement later (see `training/`). |
| Action recognition (PICK/PLACE) | Rule-based geometric/temporal reasoning over MediaPipe landmarks + HSV boxes (`interaction.py` + `action_recognizer.py`) | Same reason — no labeled action dataset exists yet. The `ActionRecognizer` interface is intentionally kept swappable for a trained model later (`training/` is the placeholder for that, not the real thing yet). |
| Sequence validation | Deterministic rule-based state machine | The procedure is *known* (it's in `experiments/*.json`), so this is a correctness question with a definite right answer, not a pattern-recognition problem. |

**No LLM anywhere in the core recognition/validation/voice loop.** Everything
runs offline/local — no cloud calls, no network dependency for inference,
ever.

## Mismatch classification (`validator.py`)

A detected action that isn't the expected one is classified by *why* it
doesn't match, not just *that* it doesn't:

- **REPEATED_STEP** — matches an already-completed step.
- **WRONG_OBJECT** — matches a not-yet-completed step whose action *type*
  (PICK/PLACE) is the same as what's currently expected (e.g. expected
  `PICK_RED_BOX`, got `PICK_BLUE_BOX`) — same gesture, wrong item.
- **SKIPPED_STEP** — matches a not-yet-completed step with a *different*
  action type than expected (e.g. expected `PLACE_RED_BOX`, got
  `PICK_BLUE_BOX`) — a real future step, but not a same-type substitution:
  the current step was skipped over.
- **OUT_OF_SEQUENCE** — true fallback: the detected key matches no known
  step at all. Shouldn't normally happen (the recognizer only emits known
  action keys); kept defensively.

## COMPLETE_EXPERIMENT auto-advance

There is no physical gesture for "close out the experiment" — the last
step in the sample experiment has no PICK/PLACE behind it. Rather than
have the action recognizer invent a gesture that doesn't exist,
`RuleBasedActionRecognizer` never emits `COMPLETE_EXPERIMENT` at all, and
`RuleBasedSequenceEngine` auto-advances through a trailing COMPLETE-typed
step itself, the instant the prior real step is confirmed. Both the real
step's own event and the auto-generated COMPLETE event are recorded in
`history` (for logging), even though `submit_action()`'s one-event-per-call
interface means only the COMPLETE event is returned/broadcast live —
`inference_service.py`'s `_run_loop` diffs `history` length before/after
each call specifically to not lose the other one.

## Honesty rules (followed everywhere, not just where convenient)

- **HSV detector "confidence" is a heuristic** (blob fill ratio), not a
  calibrated probability — documented in `color_detector.py`, never
  implied otherwise anywhere else (UI, docs, logs).
- **The rule-based action recognizer's confidence** blends that same HSV
  heuristic with a fixed "interaction certainty" constant — also
  documented as a heuristic, not a real probability
  (`backend/temporal/action_recognizer.py`).
- **Don't fabricate metrics.** `training/evaluate.py` reports
  `"N/A — insufficient evaluation data"` until a real recorded dataset
  exists, rather than inventing a number. See `docs/dataset.md`.
- **No performance claims beyond what's actually been measured** — fps
  numbers, accuracy, latency: if it wasn't run and observed, it isn't
  claimed.

## Known operational limitations

- Real perception inference throughput varies by machine load — observed
  anywhere from ~2 fps to ~21 fps on the same development hardware
  depending on what else was running. `RecordingService`'s `.mp4` writer
  uses a nominal fps for the container regardless, so a recorded clip's
  playback speed reflects the achieved rate, not necessarily real time —
  documented in `recording_service.py`, not silently ignored.
- `tools/record_dataset.py` currently saves one isolated frame per
  labeled keypress, not a temporal window — so `training/features.py`'s
  displacement/velocity features are the `MISSING` sentinel for every
  example recorded that way today. See `docs/dataset.md`.

## Gotchas already found and fixed (reuse the existing fix, don't rediscover)

1. **MediaPipe Pose `model_complexity=0`** silently downloads a model from
   `storage.googleapis.com` on first use, breaking the offline guarantee
   (and failing outright on a restricted network). `model_complexity=1`
   ships inside the `mediapipe` pip package itself — already set correctly
   in `pose.py`.
2. **`cv2.imshow`/`cv2.namedWindow` on Linux with no display server**
   doesn't raise a catchable `cv2.error` — the underlying Qt plugin loader
   aborts the whole process (SIGABRT). `backend/services/display_utils.py`'s
   `gui_available()` checks for a display *before* any GUI call; every
   interactive tool (`test_camera.py`, `live_preview.py`, `calibrate_hsv.py`,
   `record_dataset.py`) uses it.
3. **`OBJECT_PLACED`/`OBJECT_RELEASED` must persist, not just fire for one
   frame** — discovered while building Phase 4's temporal smoother, which
   needs a state to hold stable across several consecutive frames to
   debounce it. `interaction.py`'s "resting" states now stay put until the
   object is physically touched again, matching reality (a placed object
   doesn't teleport back to "no interaction" just because the hand let go).
4. **A backgrounded webcam's driver may ignore the requested resolution**
   — `camera_service.py` sets `CAP_PROP_FRAME_WIDTH/HEIGHT` from config,
   but the actual captured frame can still come back at a different size.
   `RecordingService`'s `cv2.VideoWriter` is created lazily on the first
   real frame (matching its *actual* size) rather than trusting config.

## Directory layout

```
astra_phase0_1/
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── api/routes.py              # REST endpoints
│   ├── websocket/live.py          # /ws/live
│   ├── config/settings.py         # config.yaml + env overrides
│   ├── services/
│   │   ├── camera_service.py      # threaded capture: webcam | video_file | synthetic
│   │   ├── inference_service.py   # orchestrates the per-frame pipeline + WS broadcast
│   │   ├── synthetic_scene.py     # procedural test-pattern generator
│   │   ├── overlay.py             # draws detection/pose/hand overlay onto a frame
│   │   ├── voice_service.py       # offline pyttsx3 TTS, background thread + queue
│   │   ├── logging_service.py     # persistent run log (JSONL) + experiment_report.json
│   │   ├── recording_service.py   # annotated .mp4 writer
│   │   └── frame_utils.py
│   ├── perception/
│   │   ├── base.py                # shared schemas (DetectedObject, PerceptionFrame, ...)
│   │   ├── interfaces.py          # PerceptionEngine / ActionRecognizer / SequenceEngine ABCs
│   │   ├── color_detector.py      # HSV red/blue/experiment_area detection
│   │   ├── pose.py / hands.py     # MediaPipe wrappers
│   │   ├── interaction.py         # hand-object InteractionState reasoning (Phase 3)
│   │   └── real_engine.py         # composes the above into one PerceptionEngine
│   ├── temporal/
│   │   ├── action_recognizer.py   # RuleBasedActionRecognizer (Phase 4)
│   │   └── temporal_smoother.py   # stability/hysteresis debounce
│   ├── experiment/
│   │   ├── experiment_loader.py   # loads + validates experiment JSON
│   │   └── validator.py           # real rule-based sequence validation
│   └── mocks/mock_engines.py      # scripted mock perception + action recognizer
├── frontend/                      # Next.js + TypeScript + Tailwind dashboard (Phase 10)
├── experiments/bas_sample_001.json
├── config/config.yaml
├── tools/                         # test_camera.py, live_preview.py, calibrate_hsv.py,
│                                   # record_dataset.py, and a few one-off _*.py smoke scripts
├── training/                      # features.py / train_baseline.py / evaluate.py (Phase 12 stub)
├── tests/                         # pytest — no camera required
└── data/{raw,processed,annotations,models,recordings,reports}
```
