"""Tests for backend/services/voice_service.py.

Uses a fake engine_factory (dependency injection, see VoiceService.__init__)
instead of real pyttsx3/SAPI5 — unit tests must not depend on audio hardware
or actually produce sound. Real playback is checked manually via
tools/_smoke_voice.py against actual hardware.
"""
from __future__ import annotations

import threading
import time

from backend.experiment.experiment_loader import ExperimentDefinition
from backend.perception.base import SequenceEvent, SequenceStatus
from backend.services.voice_service import VoiceService, _message_for_event

EXPERIMENT = ExperimentDefinition(
    experiment_id="TEST",
    name="Test experiment",
    objects={
        "RED_BOX": {"label": "red container"},
        "BLUE_BOX": {"label": "blue container"},
        "EXPERIMENT_AREA": {"label": "experiment area"},
    },
    steps=[
        {
            "id": 1,
            "action": "PICK",
            "object": "RED_BOX",
            "instruction": "Pick up the RED container.",
            "voice_next": "Next, place the red container.",
        },
        {
            "id": 2,
            "action": "PLACE",
            "object": "RED_BOX",
            "target": "EXPERIMENT_AREA",
            "instruction": "Place the RED container in the experiment area.",
            "voice_next": "Next, pick up the blue container.",
        },
        {
            "id": 3,
            "action": "COMPLETE",
            "object": "EXPERIMENT",
            "instruction": "Close out the experiment.",
            "voice_next": "Experiment completed successfully.",
        },
    ],
)


class FakeEngine:
    def __init__(self):
        self.spoken: list[str] = []
        self.stopped = False

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def runAndWait(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def _event(step: int, status: SequenceStatus, expected: str = "", detected: str = "") -> SequenceEvent:
    return SequenceEvent(step=step, expected=expected, detected=detected, confidence=0.9, status=status)


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_message_for_correct_is_voice_next():
    msg = _message_for_event(EXPERIMENT, _event(1, SequenceStatus.CORRECT))
    assert msg == "Next, place the red container."


def test_message_for_recovered_is_voice_next():
    msg = _message_for_event(EXPERIMENT, _event(2, SequenceStatus.RECOVERED))
    assert msg == "Next, pick up the blue container."


def test_message_for_complete_is_voice_next():
    msg = _message_for_event(EXPERIMENT, _event(3, SequenceStatus.COMPLETE))
    assert msg == "Experiment completed successfully."


def test_message_for_wrong_object_includes_instruction():
    msg = _message_for_event(EXPERIMENT, _event(1, SequenceStatus.WRONG_OBJECT))
    assert msg == "That's the wrong object. Pick up the RED container."


def test_message_for_skipped_step_includes_instruction():
    msg = _message_for_event(EXPERIMENT, _event(2, SequenceStatus.SKIPPED_STEP))
    assert msg == "You skipped a step. Place the RED container in the experiment area."


def test_message_for_repeated_step():
    msg = _message_for_event(EXPERIMENT, _event(1, SequenceStatus.REPEATED_STEP))
    assert msg == "That step is already done. Pick up the RED container."


def test_message_for_out_of_sequence():
    msg = _message_for_event(EXPERIMENT, _event(1, SequenceStatus.OUT_OF_SEQUENCE))
    assert msg == "That wasn't recognized. Pick up the RED container."


def test_message_for_low_confidence_is_none():
    assert _message_for_event(EXPERIMENT, _event(1, SequenceStatus.LOW_CONFIDENCE)) is None


def test_disabled_service_never_starts_thread_or_speaks():
    service = VoiceService(EXPERIMENT, enabled=False, engine_factory=FakeEngine)
    service.start()
    service.speak("hello")
    service.on_sequence_event(_event(1, SequenceStatus.CORRECT))
    assert service._thread is None


def test_enabled_service_speaks_on_correct_event():
    engines: list[FakeEngine] = []

    def factory():
        e = FakeEngine()
        engines.append(e)
        return e

    service = VoiceService(EXPERIMENT, enabled=True, engine_factory=factory)
    service.start()
    try:
        service.on_sequence_event(_event(1, SequenceStatus.CORRECT))
        assert _wait_until(lambda: engines and engines[0].spoken)
        assert engines[0].spoken == ["Next, place the red container."]
    finally:
        service.stop()


def test_stop_joins_worker_thread_cleanly():
    service = VoiceService(EXPERIMENT, enabled=True, engine_factory=FakeEngine)
    service.start()
    thread = service._thread
    assert thread is not None and thread.is_alive()
    service.stop()
    assert service._thread is None
    assert not thread.is_alive()


def test_engine_init_failure_does_not_crash_or_block():
    def failing_factory():
        raise RuntimeError("no audio device")

    service = VoiceService(EXPERIMENT, enabled=True, engine_factory=failing_factory)
    service.start()
    service.speak("this should be silently dropped")
    # stop() must still return promptly even though the engine never
    # initialized — the worker's drain loop must be listening for the
    # sentinel, not stuck.
    start = time.time()
    service.stop()
    assert time.time() - start < 2.0


def test_low_confidence_event_produces_no_speech():
    engines: list[FakeEngine] = []

    def factory():
        e = FakeEngine()
        engines.append(e)
        return e

    service = VoiceService(EXPERIMENT, enabled=True, engine_factory=factory)
    service.start()
    try:
        service.on_sequence_event(_event(1, SequenceStatus.LOW_CONFIDENCE))
        time.sleep(0.1)
        assert engines == [] or engines[0].spoken == []
    finally:
        service.stop()
