# ASTRA — Running the Full Demo

Exact steps to run the fully-wired system end-to-end on a machine with a
real webcam. Everything here has actually been run and observed working
during development (see the phase commit messages for the specific
verification each step got) — nothing below is aspirational.

## What you need

- A machine with a **webcam** (built-in or USB) and a **display**.
- Python 3.10+ and Node.js 18+.
- Two small, solid-colored objects roughly matching **red** and **blue** —
  the sample experiment (`experiments/bas_sample_001.json`) is a pick/place
  procedure with a red container and a blue container. Anything from
  sticky notes to painted blocks works; matte, saturated colors detect
  more reliably than glossy or pale ones.
- Reasonably even lighting on your work surface — HSV color detection is
  the one part of the pipeline that's genuinely lighting-sensitive (see
  Troubleshooting below).

## 1. Backend setup

```bash
cd astra_phase0_1
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Point the camera at your webcam. Either edit `config/config.yaml`:

```yaml
camera:
  source: webcam
  camera_index: 0        # try 1, 2... if the wrong camera opens
```

or override at launch without editing the file:

```bash
# Windows (PowerShell)
$env:CAMERA_SOURCE="webcam"; $env:CAMERA_INDEX="0"; uvicorn backend.main:app --reload --port 8000

# Linux/macOS
CAMERA_SOURCE=webcam CAMERA_INDEX=0 uvicorn backend.main:app --reload --port 8000
```

If you're not sure which index is your real camera, run
`python tools/test_camera.py --source webcam --index 0` first (see
Troubleshooting) — it saves a sample frame you can open and check.

Confirm the backend is up: `curl http://localhost:8000/api/status` should
return `{"status": "IDLE", ...}`.

## 2. Frontend setup

In a second terminal:

```bash
cd astra_phase0_1/frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The header should show a green **LIVE**
pill once the WebSocket connects (it auto-reconnects if the backend isn't
up yet — no need to reload once it starts).

If your backend isn't on `localhost:8000`, copy `frontend/.env.example` to
`frontend/.env.local` and set `NEXT_PUBLIC_API_BASE` / `NEXT_PUBLIC_WS_URL`
accordingly before `npm run dev`.

## 3. Run the demo

1. Click **START**. The camera feed appears in the left panel with a live
   pose/hand skeleton overlay once a person is in frame.
2. Follow the checklist on the right: pick up the red container, place it
   in the dashed experiment-area box, pick up the blue container, place
   it in the same area.
3. Each confirmed step lights up green in the checklist and the event log
   at the bottom gets a new entry. If a step is done wrong — the wrong
   object for the current step, or a step skipped ahead — the right panel
   swaps to a red alert card showing exactly what was expected vs.
   detected; the event log entry and the spoken voice line describe why.
4. Correcting the deviation (doing the actually-expected step next) shows
   a **✓ RECOVERY VERIFIED** banner, then returns to normal.
5. After the last real step (placing the blue container) is confirmed,
   the system auto-advances through "close out the experiment" itself —
   there's no gesture for that step, it's a natural procedural conclusion,
   not a missed detection. The checklist shows all 5 steps checked and the
   panel reads **EXPERIMENT COMPLETE**.

Use **STOP** to end a run early (this also finalizes the `.mp4` recording
— see below) and **RESET** to clear the board and start over.

## Where things get saved

- `data/reports/experiment_report.json` — per-step first-try-correct vs.
  needed-correction, total deviations, duration. Overwritten each run.
- `data/reports/run_<id>.jsonl` — every event from that run, one per line.
- `data/recordings/run_<id>.mp4` — the annotated video, finalized when you
  click **STOP** (not the instant the run finishes — see
  `docs/architecture.md`'s known limitations).

## No physical boxes handy? Use the mock engines

To demo (or just sanity-check) the sequence-validation logic, voice, and
dashboard without any real detection, run the backend with the scripted
mocks instead of real perception:

```bash
# Windows (PowerShell)
$env:USE_MOCK_ENGINES="true"; $env:CAMERA_SOURCE="synthetic"; uvicorn backend.main:app --port 8000

# Linux/macOS
USE_MOCK_ENGINES=true CAMERA_SOURCE=synthetic uvicorn backend.main:app --port 8000
```

Clicking START plays the canonical scripted scenario automatically: two
correct steps, a deliberate deviation, detection, recovery, then
completion — exercising the entire CORRECT → SKIPPED_STEP → RECOVERED →
COMPLETE flow through the real dashboard with no camera or props needed.

## Troubleshooting

**Wrong camera opens, or none does.** Try `--index 1`, `2`, etc. with
`python tools/test_camera.py --source webcam --index N` (saves a sample
frame you can inspect) before changing the backend config.

**Detection is unreliable — boxes not found, or found inconsistently.**
This is almost always lighting. Run the interactive calibration tool:

```bash
python tools/calibrate_hsv.py --source webcam --color red_box
```

Drag the trackbars until the mask cleanly isolates just the box (solid
white blob, no noise), then paste the printed HSV range into
`config/config.yaml` under `perception.colors.red_box` (repeat for
`blue_box`). The shipped ranges are starting points tuned against the
synthetic test pattern, not any particular real room.

**Inference feels slow / fps is low.** Measured on development hardware
this has ranged from ~2 fps to ~21 fps depending on machine load — see
`docs/architecture.md`'s known limitations. It's a genuine CPU-bound cost
(MediaPipe pose + hands + HSV detection every frame), not a bug to chase
for the MVP.

**Voice doesn't say anything.** Check `features.enable_voice` is `true`
in `config/config.yaml` (or `ENABLE_VOICE=true`). `tools/_smoke_voice.py`
is a minimal standalone check if you want to isolate voice from the rest
of the pipeline.
