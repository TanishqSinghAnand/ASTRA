"""Abstract interfaces every perception/action/sequence implementation must
satisfy — mock or real. inference_service.py (Phase 9) only ever talks to
these interfaces, never to a concrete class, so Phase 2's real MediaPipe+HSV
perception and a future trained action model are drop-in replacements for
backend/mocks/mock_engines.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.perception.base import ActionPrediction, PerceptionFrame, SequenceEvent


class PerceptionEngine(ABC):
    """Answers: 'what do I see in this frame?'"""

    @abstractmethod
    def process(self, frame: np.ndarray, frame_index: int) -> PerceptionFrame:
        """Run detection/pose/hands on a single BGR frame and return a
        PerceptionFrame. Must not raise on a frame with nothing detected —
        return an empty-but-valid PerceptionFrame instead."""
        raise NotImplementedError


class ActionRecognizer(ABC):
    """Answers: 'what is happening, over time?' Consumes a rolling window of
    PerceptionFrame objects (temporal, never a single frame) and returns the
    current best action guess, or a low/zero-confidence guess when nothing
    conclusive is happening yet."""

    @abstractmethod
    def update(self, perception_frame: PerceptionFrame) -> ActionPrediction:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear internal temporal state (called on experiment start/reset)."""
        raise NotImplementedError


class SequenceEngine(ABC):
    """Answers: 'what does this action mean for the experiment?' Wraps the
    experiment state machine + decision engine."""

    @abstractmethod
    def submit_action(self, prediction: ActionPrediction) -> SequenceEvent | None:
        """Feed one (already temporally-smoothed) action prediction in.
        Returns a SequenceEvent when the prediction resulted in a state
        change or a decision worth reporting, else None."""
        raise NotImplementedError

    @abstractmethod
    def current_step(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
