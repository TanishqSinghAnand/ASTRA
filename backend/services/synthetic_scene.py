"""Procedural test-pattern frame generator.

Not a demo aesthetic — a plumbing-verification tool. It draws a red box, a
blue box, and an experiment-area rectangle on a plain background so that
camera_service.py, the FastAPI/WebSocket frame pipeline, and (from Phase 2
onward) the HSV color detector all have something deterministic and
color-correct to consume without any real camera attached. This is also
what tools/make_synthetic_clip.py renders to an .mp4 for the "video_file"
demo-mode source.
"""
from __future__ import annotations

import cv2
import numpy as np

BACKGROUND_BGR = (40, 40, 38)          # dark neutral "table"
RED_BOX_BGR = (40, 40, 220)             # OpenCV is BGR
BLUE_BOX_BGR = (200, 90, 30)
AREA_COLOR_BGR = (180, 180, 180)
BOX_SIZE = 90


def generate_frame(t_seconds: float, width: int = 1280, height: int = 720, frame_index: int = 0) -> np.ndarray:
    """Return one synthetic BGR frame for elapsed time t_seconds."""
    frame = np.full((height, width, 3), BACKGROUND_BGR, dtype=np.uint8)

    # Experiment area (target region), fixed, center-right of frame.
    area_x1, area_y1 = int(width * 0.55), int(height * 0.55)
    area_x2, area_y2 = int(width * 0.85), int(height * 0.85)
    cv2.rectangle(frame, (area_x1, area_y1), (area_x2, area_y2), AREA_COLOR_BGR, 2)
    # No text label drawn here on purpose — the experiment-area label is
    # drawn by the perception overlay (backend/services/overlay.py) when a
    # detector is attached, so the two don't collide when both are visible.

    # Red box: gentle bob near its supply position (bottom-left).
    red_cx = int(width * 0.20)
    red_cy = int(height * 0.70 + 6 * np.sin(t_seconds * 1.5))
    cv2.rectangle(frame, (red_cx - BOX_SIZE // 2, red_cy - BOX_SIZE // 2),
                  (red_cx + BOX_SIZE // 2, red_cy + BOX_SIZE // 2), RED_BOX_BGR, -1)

    # Blue box: gentle bob near its supply position (top-left).
    blue_cx = int(width * 0.20)
    blue_cy = int(height * 0.25 + 6 * np.cos(t_seconds * 1.5))
    cv2.rectangle(frame, (blue_cx - BOX_SIZE // 2, blue_cy - BOX_SIZE // 2),
                  (blue_cx + BOX_SIZE // 2, blue_cy + BOX_SIZE // 2), BLUE_BOX_BGR, -1)

    # Watermark so this is never mistaken for a live feed.
    cv2.putText(frame, "SYNTHETIC TEST SOURCE - NOT LIVE VIDEO", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, f"t={t_seconds:5.1f}s  frame={frame_index}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
