"""Small shared helpers for turning frames into wire-friendly payloads."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_frame_jpeg_b64(frame: np.ndarray, quality: int = 70, max_width: int | None = None) -> str:
    """Encode a BGR frame as a base64 JPEG data URI for WebSocket delivery.
    Kept deliberately lossy/low-res-friendly — this is a live status view,
    not the recording (recording_service.py, Phase 9, writes full quality).

    max_width downscales before encoding (aspect-preserving) — a live
    status view doesn't need the camera's full capture resolution, and
    encoding/base64-ing/transmitting/decoding a smaller frame is
    meaningfully cheaper per frame, which matters when this runs at
    interactive frame rates."""
    if max_width is not None and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        new_size = (max_width, max(1, int(frame.shape[0] * scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
