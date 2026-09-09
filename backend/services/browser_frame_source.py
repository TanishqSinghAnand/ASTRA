"""Frame source fed by a browser's own camera instead of a local capture
device — for camera.source == "browser" (see settings.py). A cloud-hosted
backend has no webcam of its own; the visiting browser captures its own
video (getUserMedia) and pushes JPEG frames over /ws/ingest
(backend/websocket/ingest.py), which calls push_frame() here.

Deliberately duck-types CameraService's public interface (start/stop/
is_running/read_latest/measured_fps) so InferenceService (frame_utils.py's
consumer) doesn't need to know or care which kind of source it's holding.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from backend.services.camera_service import CapturedFrame

logger = logging.getLogger("astra.browser_source")

# No frame pushed in this long -> treated as "no signal" (client tab closed,
# camera permission revoked, network stall) rather than serving an
# indefinitely stale frame as if it were live.
_STALE_AFTER_S = 5.0


class BrowserFrameSource:
    def __init__(self) -> None:
        self._running = False
        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._latest_pushed_at = 0.0
        self._frame_index = 0
        self._fps_window: list[float] = []

    # -- lifecycle, matching CameraService's shape ---------------------------

    def start(self) -> None:
        self._running = True
        logger.info("BrowserFrameSource started — waiting for client frames on /ws/ingest")

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._latest = None
            self._fps_window.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return None

    # -- frame access, matching CameraService's shape ------------------------

    def read_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            if self._latest is None:
                return None
            if time.time() - self._latest_pushed_at > _STALE_AFTER_S:
                return None
            return self._latest

    @property
    def measured_fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            span = self._fps_window[-1] - self._fps_window[0]
            return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    # -- called from the ingest WebSocket handler -----------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        if not self._running:
            return
        now = time.time()
        with self._lock:
            self._latest = CapturedFrame(frame=frame, frame_index=self._frame_index, timestamp=now)
            self._latest_pushed_at = now
            self._fps_window.append(now)
            if len(self._fps_window) > 60:
                self._fps_window.pop(0)
        self._frame_index += 1
