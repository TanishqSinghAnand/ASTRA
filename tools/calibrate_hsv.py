#!/usr/bin/env python3
"""Interactive HSV range tuner for the red/blue box detector.

HSV color thresholds (config/config.yaml -> perception.colors) are tuned
against the synthetic test pattern, not your actual boxes under your actual
room lighting. If tools/live_preview.py isn't reliably finding your boxes,
run this against your webcam, hold the box up, and drag the trackbars until
the mask window cleanly shows just the box (white) with everything else
black. Press 's' to print the resulting range as YAML you can paste into
config.yaml, or 'q' to quit without saving.

Usage:
    python tools/calibrate_hsv.py --source webcam --color red_box
    python tools/calibrate_hsv.py --source synthetic --color blue_box   # sanity check, no camera needed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import CameraSettings, load_settings  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402
from backend.services.display_utils import gui_available  # noqa: E402

WINDOW = "HSV calibration — drag trackbars, 's' to print range, 'q' to quit"


def nothing(_):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["webcam", "video_file", "synthetic"], default="webcam")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--color", default="red_box", help="Label only, used in the printed YAML")
    args = parser.parse_args()

    if not gui_available():
        print("No display detected (this needs a GUI window to show the mask preview). "
              "Run this on your local machine with a screen attached, not over a headless "
              "SSH session.", file=sys.stderr)
        sys.exit(1)

    settings = load_settings()
    cam_settings = CameraSettings(**{**settings.camera.model_dump(), "source": args.source, "camera_index": args.index})
    repo_root = Path(__file__).resolve().parents[1]
    camera = CameraService(cam_settings, repo_root)

    try:
        camera.start()
    except CameraError as e:
        print(f"FAILED to start camera: {e}", file=sys.stderr)
        sys.exit(1)

    cv2.namedWindow(WINDOW)

    cv2.createTrackbar("H min", WINDOW, 0, 179, nothing)
    cv2.createTrackbar("H max", WINDOW, 179, 179, nothing)
    cv2.createTrackbar("S min", WINDOW, 100, 255, nothing)
    cv2.createTrackbar("S max", WINDOW, 255, 255, nothing)
    cv2.createTrackbar("V min", WINDOW, 60, 255, nothing)
    cv2.createTrackbar("V max", WINDOW, 255, 255, nothing)

    print("Adjust the trackbars until the mask cleanly isolates just your box.")
    print("Press 's' to print the current range as YAML, 'q' to quit.\n")

    try:
        while True:
            captured = camera.read_latest()
            if captured is None:
                time.sleep(0.02)
                continue

            hsv = cv2.cvtColor(captured.frame, cv2.COLOR_BGR2HSV)
            h_min = cv2.getTrackbarPos("H min", WINDOW)
            h_max = cv2.getTrackbarPos("H max", WINDOW)
            s_min = cv2.getTrackbarPos("S min", WINDOW)
            s_max = cv2.getTrackbarPos("S max", WINDOW)
            v_min = cv2.getTrackbarPos("V min", WINDOW)
            v_max = cv2.getTrackbarPos("V max", WINDOW)

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            masked_preview = cv2.bitwise_and(captured.frame, captured.frame, mask=mask)
            stacked = np.hstack([captured.frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked_preview])
            stacked = cv2.resize(stacked, (stacked.shape[1] // 2, stacked.shape[0] // 2))
            cv2.imshow(WINDOW, stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print(f"{args.color}:")
                print("  ranges:")
                print(f"    - lower: [{h_min}, {s_min}, {v_min}]")
                print(f"      upper: [{h_max}, {s_max}, {v_max}]")
                print("(If your color wraps around hue 0, like red often does, run this again")
                print(" for the other side of the wrap and add it as a second range entry.)\n")
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
