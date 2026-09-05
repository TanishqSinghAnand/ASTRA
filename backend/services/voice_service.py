"""Offline voice guidance (Phase 7).

pyttsx3 runs entirely offline (SAPI5 on Windows, no network call — keeps the
"runs offline/local, always" rule the same way MediaPipe's model_complexity=1
does). Speech runs on its own background thread draining a queue.Queue: the
15Hz inference loop in inference_service.py must never block waiting for a
sentence to finish being spoken.

Two trigger points, both gated by features.enable_voice:
  - An error-status SequenceEvent (WRONG_OBJECT/SKIPPED_STEP/
    OUT_OF_SEQUENCE/REPEATED_STEP): speaks a short, spoken-friendly sentence
    built from the *expected* step's instruction — not the full multi-line
    technical `explanation` array in the event, which is meant for the
    on-screen log, not speech.
  - CORRECT/RECOVERED (and COMPLETE, a natural extension of the same rule —
    see below): proactively speaks the just-confirmed step's `voice_next`
    line from the experiment config. This is the "tells the astronaut what
    to do next without being asked" behavior the problem statement calls
    for. COMPLETE isn't explicitly listed in the spec's CORRECT/RECOVERED
    trigger, but its own voice_next line ("Experiment completed
    successfully.") exists for exactly this moment and skipping it would
    leave the demo's finale silent for no reason — included by design.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from backend.experiment.experiment_loader import ExperimentDefinition
from backend.perception.base import ERROR_STATUSES, SequenceEvent, SequenceStatus

logger = logging.getLogger("astra.voice")

_SUCCESS_STATUSES = (SequenceStatus.CORRECT, SequenceStatus.RECOVERED, SequenceStatus.COMPLETE)

_ERROR_PREFIXES = {
    SequenceStatus.WRONG_OBJECT: "That's the wrong object.",
    SequenceStatus.SKIPPED_STEP: "You skipped a step.",
    SequenceStatus.REPEATED_STEP: "That step is already done.",
    SequenceStatus.OUT_OF_SEQUENCE: "That wasn't recognized.",
    SequenceStatus.WRONG_LOCATION: "Right item, wrong spot.",
}


def _speakable_error(experiment: ExperimentDefinition, event: SequenceEvent) -> str:
    expected_step = experiment.step_by_id(event.step)
    prefix = _ERROR_PREFIXES.get(event.status, "That wasn't correct.")
    if expected_step is None:
        return prefix
    return f"{prefix} {expected_step.instruction}"


def _message_for_event(experiment: ExperimentDefinition, event: SequenceEvent) -> Optional[str]:
    if event.status in _SUCCESS_STATUSES:
        step = experiment.step_by_id(event.step)
        return step.voice_next if step and step.voice_next else None
    if event.status in ERROR_STATUSES:
        return _speakable_error(experiment, event)
    return None  # LOW_CONFIDENCE -> still "observing", stay silent


class VoiceService:
    def __init__(
        self,
        experiment: ExperimentDefinition,
        enabled: bool = True,
        engine_factory: Optional[Callable[[], object]] = None,
    ):
        self.experiment = experiment
        self.enabled = enabled
        self._engine_factory = engine_factory or _default_engine_factory
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True, name="astra-voice")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)  # sentinel: ask the worker to exit
        self._thread.join(timeout=2.0)
        self._thread = None

    # -- public API ------------------------------------------------------------

    def speak(self, text: str) -> None:
        if not self.enabled or self._thread is None or not text:
            return
        self._queue.put(text)

    def on_sequence_event(self, event: SequenceEvent) -> None:
        if not self.enabled:
            return
        text = _message_for_event(self.experiment, event)
        if text:
            self.speak(text)

    # -- worker thread -----------------------------------------------------

    def _worker(self) -> None:
        try:
            engine = self._engine_factory()
        except Exception:
            logger.warning("pyttsx3 failed to initialize — voice disabled for this run.", exc_info=True)
            self._drain_until_stopped()
            return

        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception:
                logger.exception("Voice playback failed for: %r", item)
        try:
            engine.stop()
        except Exception:
            pass

    def _drain_until_stopped(self) -> None:
        """Engine init failed — keep consuming (and discarding) queued text
        so speak()/stop() callers never block on a dead worker."""
        while True:
            item = self._queue.get()
            if item is None:
                return


def _default_engine_factory():
    import pyttsx3

    return pyttsx3.init()
