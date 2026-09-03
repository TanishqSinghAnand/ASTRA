"""Hysteresis/stability smoothing over a raw per-frame action candidate.

action_recognizer.py derives a candidate action key every frame from the
*current* interaction state (e.g. "PICK_RED_BOX" for as long as the object
is being held, "PLACE_RED_BOX" for as long as it sits placed in the
experiment area — see interaction.py's OBJECT_PLACED/OBJECT_RELEASED
persistence). This module is the debounce layer on top of that stream:
mirrors MockActionRecognizer's existing behavior (config/config.yaml's
temporal.action_stability_frames / temporal.min_confidence_duration_ms):
a candidate must hold for that many consecutive frames AND that long in
wall-clock time before it's trusted, and once emitted, the *same* action is
never emitted again on a later frame — only a genuine change (a different
action, or the same one after the candidate cleared and came back, e.g. a
REPEATED_STEP) is ever returned.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from backend.perception.base import ActionPrediction


class TemporalSmoother:
    def __init__(
        self,
        stability_frames: int,
        min_confidence_duration_ms: int,
        clock: Callable[[], float] = time.time,
    ):
        self.stability_frames = max(1, stability_frames)
        self.min_duration_s = max(0, min_confidence_duration_ms) / 1000.0
        self._clock = clock
        self._candidate: Optional[str] = None
        self._candidate_confidences: list[float] = []
        self._candidate_since: Optional[float] = None
        self._candidate_frames = 0
        self._last_emitted: Optional[str] = None

    def reset(self) -> None:
        self._candidate = None
        self._candidate_confidences = []
        self._candidate_since = None
        self._candidate_frames = 0
        self._last_emitted = None

    def update(self, candidate_key: Optional[str], candidate_confidence: float) -> ActionPrediction:
        now = self._clock()

        if candidate_key != self._candidate:
            self._candidate = candidate_key
            self._candidate_confidences = [candidate_confidence] if candidate_key else []
            self._candidate_since = now if candidate_key else None
            self._candidate_frames = 1 if candidate_key else 0
            if candidate_key is None:
                # Candidate disappeared (object no longer held/placed) — the
                # next time this action reappears it's a fresh gesture, not
                # a repeat, so it's allowed to be emitted again.
                self._last_emitted = None
        elif candidate_key is not None:
            self._candidate_confidences.append(candidate_confidence)
            self._candidate_frames += 1

        if candidate_key is None:
            return ActionPrediction(action=None, confidence=0.0, source="rule_based")

        stable = (
            self._candidate_frames >= self.stability_frames
            and self._candidate_since is not None
            and (now - self._candidate_since) >= self.min_duration_s
        )
        if stable and candidate_key != self._last_emitted:
            self._last_emitted = candidate_key
            avg_confidence = sum(self._candidate_confidences) / len(self._candidate_confidences)
            return ActionPrediction(action=candidate_key, confidence=round(avg_confidence, 3), source="rule_based")

        return ActionPrediction(action=None, confidence=0.0, source="rule_based")
