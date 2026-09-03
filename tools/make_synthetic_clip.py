#!/usr/bin/env python3
"""Render the synthetic test pattern to an .mp4 file so it can be used as a
`video_file` camera source (config/config.yaml -> camera.video_file_path),
e.g. for Demo Mode or for testing this repo in an environment with no
webcam and no synthetic-source support in whatever is consuming the video.

Usage:
    python tools/make_synthetic_clip.py --seconds 25 --out data/recordings/demo_clip.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.synthetic_scene import generate_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=25.0, help="Clip duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=str, default="data/recordings/demo_clip.mp4")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: could not open VideoWriter for {out_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(args.seconds * args.fps)
    for i in range(total_frames):
        t = i / args.fps
        frame = generate_frame(t, args.width, args.height, i)
        writer.write(frame)
    writer.release()

    print(f"Wrote {total_frames} frames ({args.seconds:.1f}s @ {args.fps}fps) to {out_path}")


if __name__ == "__main__":
    main()
