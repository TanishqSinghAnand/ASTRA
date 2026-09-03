#!/usr/bin/env python3
"""Sanity-check a camera source without starting the full backend.

Usage:
    python tools/test_camera.py                          # use config.yaml's source
    python tools/test_camera.py --source webcam --index 0
    python tools/test_camera.py --source synthetic --seconds 3
    python tools/test_camera.py --source video_file --path data/recordings/demo_clip.mp4

Prints measured FPS and, on a machine with a display, shows a preview
window (press 'q' to quit). In headless environments (like this dev
sandbox) pass --no-display and it will instead save one sample frame as a
PNG so the capture path can still be verified visually.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import CameraSettings, load_settings  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402
from backend.services.display_utils import gui_available  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["webcam", "video_file", "synthetic"], default=None)
    parser.add_argument("--index", type=int, default=None, help="Camera index (webcam source)")
    parser.add_argument("--path", type=str, default=None, help="Video file path (video_file source)")
    parser.add_argument("--seconds", type=float, default=3.0, help="How long to run the test")
    parser.add_argument("--no-display", action="store_true", help="Don't try to open a preview window")
    parser.add_argument("--save-sample", type=str, default="data/recordings/_camera_test_sample.png")
    args = parser.parse_args()

    settings = load_settings().camera
    overrides: dict = {}
    if args.source:
        overrides["source"] = args.source
    if args.index is not None:
        overrides["camera_index"] = args.index
    if args.path:
        overrides["video_file_path"] = args.path
    cam_settings = CameraSettings(**{**settings.model_dump(), **overrides})

    # Checked up front, not discovered by catching a failed cv2.imshow: on
    # Linux with no display server at all, that failure is a hard process
    # abort, not a catchable exception.
    if not args.no_display and not gui_available():
        print("No display detected — running with --no-display behavior instead.")
        args.no_display = True

    repo_root = Path(__file__).resolve().parents[1]
    service = CameraService(cam_settings, repo_root)

    print(f"Starting camera service: source={cam_settings.source!r} "
          f"index={cam_settings.camera_index} path={cam_settings.video_file_path!r}")
    try:
        service.start()
    except CameraError as e:
        print(f"FAILED to start camera: {e}", file=sys.stderr)
        sys.exit(1)

    last_saved = None
    deadline = time.time() + args.seconds
    frames_seen = 0
    try:
        while time.time() < deadline:
            captured = service.read_latest()
            if captured is not None:
                frames_seen += 1
                last_saved = captured.frame
                if not args.no_display:
                    try:
                        cv2.imshow("ASTRA camera test (press q to quit)", captured.frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        # No GUI backend available (headless environment) — fall back silently.
                        args.no_display = True
            time.sleep(0.03)
    finally:
        service.stop()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"Frames observed: {frames_seen}  measured_fps~={service.measured_fps:.1f}")
    if last_saved is not None:
        out_path = repo_root / args.save_sample
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), last_saved)
        print(f"Saved sample frame to {out_path}")
    else:
        print("WARNING: no frames were captured.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
