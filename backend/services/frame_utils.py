"""Small shared helpers for turning frames into wire-friendly payloads."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_frame_jpeg_b64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a BGR frame as a base64 JPEG data URI for WebSocket delivery.
    Kept deliberately lossy/low-res-friendly — this is a live status view,
    not the recording (recording_service.py, Phase 9, writes full quality)."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
