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


def decode_frame_jpeg_bytes(data: bytes) -> np.ndarray | None:
    """Inverse of encode_frame_jpeg_b64, minus the data-URI/base64 wrapping
    — used by the /ws/ingest browser-camera path (backend/websocket/ingest.py),
    which receives raw binary JPEG frames rather than base64 text (no reason
    to pay the ~33% base64 size tax on an inbound stream nothing else needs
    as text). Returns None for anything that fails to decode (a corrupt/
    partial frame) rather than raising — the caller should just skip it and
    wait for the next one, never crash the ingest loop over one bad frame."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if frame is not None else None
