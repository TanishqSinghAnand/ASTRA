"""Threaded video capture abstraction.

Capture runs on its own thread so a slow inference loop never stalls frame
acquisition (spec section 31: separate capture/inference/UI). Only the
*latest* frame is kept — if inference falls behind, older frames are simply
dropped rather than queued up, which is the right trade-off for a live
demo (always show "now", never a growing backlog of stale frames).

Three interchangeable sources, selected purely by config (CAMERA_SOURCE):
  webcam      - cv2.VideoCapture(camera_index), the real deal
  video_file  - a recorded clip, looped, for repeatable Demo Mode
  synthetic   - procedurally generated frames, no hardware/file required —
                this is what lets the whole pipeline run in a sandbox with
                no camera attached.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.config.settings import CameraSettings
from backend.services.synthetic_scene import generate_frame

logger = logging.getLogger("astra.camera")


class CameraError(Exception):
    """Raised when the configured camera source cannot be opened."""


@dataclass
class CapturedFrame:
    frame: np.ndarray
    frame_index: int
    timestamp: float


class CameraService:
    def __init__(self, settings: CameraSettings, repo_root: Path):
        self.settings = settings
        self.repo_root = repo_root
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._frame_index = 0
        self._start_time = 0.0
        self._fps_window: list[float] = []
        self._last_error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._running.is_set():
            return
        source = self.settings.source
        if source == "webcam":
            self._cap = cv2.VideoCapture(self.settings.camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
            if not self._cap.isOpened():
                self._last_error = f"Could not open webcam at index {self.settings.camera_index}"
                raise CameraError(self._last_error)
        elif source == "video_file":
            video_path = self.repo_root / self.settings.video_file_path
            if not video_path.exists():
                self._last_error = f"Video file not found: {video_path}"
                raise CameraError(self._last_error)
            self._cap = cv2.VideoCapture(str(video_path))
            if not self._cap.isOpened():
                self._last_error = f"Could not open video file: {video_path}"
                raise CameraError(self._last_error)
        elif source == "synthetic":
            self._cap = None  # generated on the fly, no backing capture device
        else:
            raise CameraError(f"Unknown camera source: {source!r}")

        self._last_error = None
        self._frame_index = 0
        self._start_time = time.time()
        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture")
        self._thread.start()
        logger.info("CameraService started (source=%s)", source)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("CameraService stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # -- frame access ----------------------------------------------------

    def read_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._latest

    @property
    def measured_fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            span = self._fps_window[-1] - self._fps_window[0]
            return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    # -- internal ----------------------------------------------------------

    def _capture_loop(self) -> None:
        target_dt = 1.0 / max(self.settings.target_fps, 1)
        while self._running.is_set():
            loop_start = time.time()
            frame = self._acquire_frame()
            if frame is not None:
                captured = CapturedFrame(frame=frame, frame_index=self._frame_index, timestamp=loop_start)
                with self._lock:
                    self._latest = captured
                    self._fps_window.append(loop_start)
                    if len(self._fps_window) > 60:
                        self._fps_window.pop(0)
                self._frame_index += 1
            elapsed = time.time() - loop_start
            sleep_for = target_dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _acquire_frame(self) -> Optional[np.ndarray]:
        if self.settings.source == "synthetic":
            t = time.time() - self._start_time
            return generate_frame(t, self.settings.frame_width, self.settings.frame_height, self._frame_index)

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            if self.settings.source == "video_file":
                # Loop playback for a repeatable demo (section 34).
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                self._last_error = "Webcam read failed (device disconnected?)"
                return None
        return frame
