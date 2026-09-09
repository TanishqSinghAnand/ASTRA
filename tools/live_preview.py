#!/usr/bin/env python3
"""Live perception preview — the thing to actually run and test.

Runs the real HSV color detector + MediaPipe pose/hands against a camera
source and shows the annotated result live. This is Phase 2 exercised
directly, standalone, without needing the FastAPI backend running: point
it at your webcam, hold up something red or blue, and watch the detection
happen in real time.

Usage (on a machine with a webcam):
    python tools/live_preview.py --source webcam --index 0

Usage (no camera — uses the procedurally generated test pattern):
    python tools/live_preview.py --source synthetic

Press 'q' to quit the preview window. If no display is available (e.g. an
SSH session or this dev sandbox), pass --no-display and it will instead
save annotated snapshots periodically so you can inspect them as images.

NOTE ON ACCURACY: this shows perception only (objects/pose/hands) — it does
NOT yet decide what "action" is happening or validate it against the
experiment sequence. That's Phase 3/4, built on top of what this proves
works. Detection quality (especially the red/blue boxes) depends heavily on
your lighting; if it's unreliable, run tools/calibrate_hsv.py first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import CameraSettings, load_settings  # noqa: E402
from backend.perception.real_engine import RealPerceptionEngine  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402
from backend.services.display_utils import DEFAULT_DISPLAY_MAX_WIDTH, fit_for_display, gui_available  # noqa: E402
from backend.services.overlay import draw_label, draw_perception_overlay  # noqa: E402

WINDOW_NAME = "ASTRA live preview (Phase 2) — press q to quit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["webcam", "video_file", "synthetic"], default=None)
    parser.add_argument("--index", type=int, default=None, help="Camera index (webcam source)")
    parser.add_argument("--path", type=str, default=None, help="Video file path (video_file source)")
    parser.add_argument("--no-display", action="store_true", help="Headless mode: save periodic snapshots instead")
    parser.add_argument("--snapshot-every", type=float, default=1.0, help="Seconds between saved snapshots in --no-display mode")
    parser.add_argument("--snapshot-count", type=int, default=5, help="How many snapshots to save in --no-display mode")
    parser.add_argument("--out-dir", type=str, default="data/recordings/live_preview")
    parser.add_argument(
        "--display-width",
        type=int,
        default=DEFAULT_DISPLAY_MAX_WIDTH,
        help="Max on-screen window width in pixels — capture/detection always stays full resolution regardless of this",
    )
    args = parser.parse_args()

    settings = load_settings()
    overrides: dict = {}
    if args.source:
        overrides["source"] = args.source
    if args.index is not None:
        overrides["camera_index"] = args.index
    if args.path:
        overrides["video_file_path"] = args.path
    cam_settings = CameraSettings(**{**settings.camera.model_dump(), **overrides})

    repo_root = Path(__file__).resolve().parents[1]
    camera = CameraService(cam_settings, repo_root)
    engine = RealPerceptionEngine(settings.perception)

    print(f"Starting live preview: source={cam_settings.source!r} index={cam_settings.camera_index} "
          f"path={cam_settings.video_file_path!r}")
    try:
        camera.start()
    except CameraError as e:
        print(f"FAILED to start camera: {e}", file=sys.stderr)
        sys.exit(1)

    # Checked up front, not discovered by catching a failed cv2.imshow: on
    # Linux with no display server at all, that failure is a hard process
    # abort, not a catchable exception (see display_utils.py).
    display_available = (not args.no_display) and gui_available()
    if not args.no_display and not display_available:
        print("No display detected — running in snapshot mode instead.")
    if display_available:
        # WINDOW_NORMAL (not the imshow default of auto-sizing to the
        # frame) so the window is user-resizable/draggable regardless of
        # capture resolution — belt-and-suspenders alongside
        # fit_for_display() below, which handles the common case.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)  # harmless if never used (display mode)

    frame_times: list[float] = []
    last_snapshot = 0.0
    snapshots_saved = 0

    try:
        while True:
            captured = camera.read_latest()
            if captured is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            perception = engine.process(captured.frame, captured.frame_index)
            infer_ms = (time.time() - t0) * 1000
            frame_times.append(time.time())
            frame_times[:] = [t for t in frame_times if t > time.time() - 2.0]
            display_fps = len(frame_times) / 2.0

            annotated = draw_perception_overlay(captured.frame, perception)
            hud = f"frame {captured.frame_index}  infer {infer_ms:5.1f}ms  ~{display_fps:4.1f} fps  objects={len(perception.objects)} pose={'Y' if perception.pose.detected else 'N'} hands={len(perception.hands)}"
            draw_label(annotated, hud, (10, annotated.shape[0] - 16), (0, 255, 60), font_scale=0.6)

            if display_available:
                try:
                    # Detection above already ran on the full-resolution
                    # captured.frame — this resize is display-only, so
                    # nothing about perception quality is affected.
                    cv2.imshow(WINDOW_NAME, fit_for_display(annotated, args.display_width))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    display_available = False
                    print("No display backend available — switching to --no-display snapshot mode.")

            if not display_available:
                now = time.time()
                if now - last_snapshot >= args.snapshot_every and snapshots_saved < args.snapshot_count:
                    last_snapshot = now
                    snapshots_saved += 1
                    out_path = out_dir / f"snapshot_{snapshots_saved:02d}.png"
                    cv2.imwrite(str(out_path), annotated)
                    print(f"Saved {out_path}  ({hud})")
                    if snapshots_saved >= args.snapshot_count:
                        print("Snapshot count reached, exiting.")
                        break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        engine.close()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
