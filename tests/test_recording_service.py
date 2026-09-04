"""Tests for backend/services/recording_service.py. Uses pytest's tmp_path
fixture and real cv2.VideoWriter (opencv is already a hard dependency) —
faking the writer would test nothing about whether an actual playable file
comes out."""
from __future__ import annotations

import numpy as np

from backend.services.recording_service import RecordingService


def _frame(w=64, h=48) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_disabled_service_writes_nothing(tmp_path):
    service = RecordingService(tmp_path / "recordings", enabled=False, nominal_fps=15.0)
    service.start_run()
    service.write_frame(_frame())
    assert service.finalize_run() is None
    assert list((tmp_path).rglob("*.mp4")) == []


def test_enabled_service_creates_mp4_file(tmp_path):
    recordings_dir = tmp_path / "recordings"
    service = RecordingService(recordings_dir, enabled=True, nominal_fps=15.0)
    service.start_run()
    for _ in range(5):
        service.write_frame(_frame())
    path = service.finalize_run()
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_writer_lazily_matches_actual_frame_size(tmp_path):
    # Config/nominal settings might not match the real captured frame size
    # (camera_service.py: a driver can ignore the requested resolution) —
    # the writer must use whatever size the first real frame actually is.
    recordings_dir = tmp_path / "recordings"
    service = RecordingService(recordings_dir, enabled=True, nominal_fps=10.0)
    service.start_run()
    service.write_frame(_frame(w=100, h=80))
    path = service.finalize_run()
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_finalize_without_any_frame_written_returns_none(tmp_path):
    service = RecordingService(tmp_path / "recordings", enabled=True, nominal_fps=15.0)
    service.start_run()
    assert service.finalize_run() is None


def test_finalize_is_safe_to_call_twice(tmp_path):
    service = RecordingService(tmp_path / "recordings", enabled=True, nominal_fps=15.0)
    service.start_run()
    service.write_frame(_frame())
    first = service.finalize_run()
    second = service.finalize_run()
    assert first is not None
    assert second is None


def test_second_run_gets_its_own_file(tmp_path):
    recordings_dir = tmp_path / "recordings"
    service = RecordingService(recordings_dir, enabled=True, nominal_fps=15.0)

    service.start_run()
    service.write_frame(_frame())
    first_path = service.finalize_run()

    service.start_run()
    service.write_frame(_frame())
    second_path = service.finalize_run()

    assert first_path != second_path
    assert first_path.exists() and second_path.exists()
