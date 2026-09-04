"""Annotated video recording (Phase 9).

Writes the same overlaid frames the WebSocket dashboard sees to an .mp4
under paths.recordings_dir (data/recordings/), one file per run, gated by
features.enable_recording. The cv2.VideoWriter is created lazily on the
first real frame rather than at start_run() time, so it always matches the
camera's *actual* captured resolution — which, per camera_service.py, can
differ from config.yaml's configured frame_width/frame_height if the
driver doesn't honor the requested size (observed during Phase 2 hardware
verification) — instead of guessing and risking a corrupt/empty file.

Honesty note: the VideoWriter is opened at a nominal fps (camera.target_fps)
since cv2.VideoWriter needs one fixed rate up front, but frames are written
whenever the inference loop actually produces one — which measured Phase
3/4 hardware runs showed varies with real-world load (roughly 2-13 fps
depending on the machine). If the achieved rate differs from the nominal
one, the resulting clip plays back faster or slower than real time; this is
a known, documented limitation, not a bug to chase down for the MVP.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("astra.recording")


class RecordingService:
    def __init__(self, recordings_dir: Path, enabled: bool, nominal_fps: float):
        self.recordings_dir = recordings_dir
        self.enabled = enabled
        self.nominal_fps = max(1.0, nominal_fps)
        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None

    def start_run(self) -> None:
        if not self.enabled:
            return
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._path = self.recordings_dir / f"run_{run_id}.mp4"
        self._writer = None  # lazy-created on the first write_frame() call

    def write_frame(self, annotated_frame: np.ndarray) -> None:
        if not self.enabled or self._path is None:
            return
        if self._writer is None:
            h, w = annotated_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self._path), fourcc, self.nominal_fps, (w, h))
            if not writer.isOpened():
                logger.warning("Could not open VideoWriter for %s — recording disabled for this run.", self._path)
                self._path = None
                return
            self._writer = writer
        self._writer.write(annotated_frame)

    def finalize_run(self) -> Optional[Path]:
        path = self._path if self._writer is not None else None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._path = None
        return path
