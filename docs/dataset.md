# ASTRA — Dataset & Training Pipeline

**Status: no real dataset has been recorded in this repository yet.**
Everything below describes the *tooling* that exists for building one, not
a claim that training has happened — `training/evaluate.py` will honestly
print `N/A — insufficient evaluation data` until someone actually runs
`tools/record_dataset.py`.

## Why this exists at all

The MVP demo does **not** need any of this — `backend/temporal/
action_recognizer.py`'s `RuleBasedActionRecognizer` is rule-based
geometric/temporal reasoning over MediaPipe landmarks + HSV-detected boxes,
and needs zero labeled data. `training/` is groundwork for a *future*
trained-model swap-in (see `docs/architecture.md`'s "why rules instead of
trained models" table), scoped exactly as far as the project spec calls
for and no further.

## Recording a dataset — `tools/record_dataset.py`

```bash
python tools/record_dataset.py --source webcam --index 0
python tools/record_dataset.py --source webcam --session my_session_01
```

This opens a live preview window. Labels are read directly from whatever
experiment is loaded (`config/config.yaml`'s `paths.experiment_config`) —
each step's `action_key` gets a number key, printed on startup and
overlaid on the video, e.g. for the sample experiment:

```
[1] PICK_RED_BOX
[2] PLACE_RED_BOX
[3] PICK_BLUE_BOX
[4] PLACE_BLUE_BOX
[5] COMPLETE_EXPERIMENT
[q] quit
```

Pressing a number key saves the **current frame** under
`data/raw/<session_id>/<label>/frame_NNNNN.jpg` and appends one record to
`data/raw/<session_id>/manifest.jsonl`:

```json
{"file": "data/raw/20260904T120000/PICK_RED_BOX/frame_00001.jpg", "timestamp": 1788500000.12, "label": "PICK_RED_BOX", "frame_index": 42}
```

The manifest is JSON Lines — one record per line, flushed to disk
immediately after each save (same crash-safety pattern as
`logging_service.py`'s run log), so a session that ends abruptly doesn't
lose earlier captures.

This is a genuinely interactive tool by design — a human decides the exact
moment a gesture is happening and presses the key. There's no headless or
automated-labeling mode (`cv2.waitKey` only receives keystrokes through an
actual GUI window), and no continuous/video recording mode: one press = one
labeled still frame.

## Known limitation: single frames, not temporal windows

`training/features.py`'s `extract_window_features()` is built to accept a
*window* of consecutive `PerceptionFrame`s so it can compute displacement
and velocity, not just a static position. But `record_dataset.py` today
only ever captures a single isolated frame per keypress — so for every
example recorded this way, the displacement/velocity features come back
as the `MISSING` sentinel (`-1.0`), and only the per-frame hand-distance
and detector-confidence features carry real signal.

If this matters for a future phase, the fix is in `record_dataset.py`
(capture a short burst of consecutive frames around each keypress instead
of one), not in `features.py` — the feature extractor already handles
multi-frame windows correctly (see `tests/test_features.py`), it just
hasn't been fed one by the current recording tool.

## Training — `training/train_baseline.py`

```bash
python training/train_baseline.py --data-dir data/raw
```

Loads every `manifest.jsonl` under `data/raw/`, re-runs
`RealPerceptionEngine` over each saved JPEG to reconstruct a
`PerceptionFrame`, extracts features, and:

- If fewer than 20 labeled frames or fewer than 2 distinct classes exist,
  writes an honest `N/A` report to `data/reports/evaluation_report.json`
  and exits — no number is printed or claimed.
- Otherwise, trains a placeholder `sklearn` `LogisticRegression` on a
  75/25 train/test split and writes the held-out accuracy — explicitly
  labeled as reflecting hand-distance/confidence features only (per the
  single-frame limitation above), never presented as a measure of real
  temporal action recognition.

## Evaluation — `training/evaluate.py`

```bash
python training/evaluate.py
```

Reads (never regenerates) `data/reports/evaluation_report.json` — the file
`train_baseline.py` writes — and prints its status. If `train_baseline.py`
has never been run, it writes and prints a placeholder `N/A` report rather
than leaving the file missing, so there's always something honest to read.
