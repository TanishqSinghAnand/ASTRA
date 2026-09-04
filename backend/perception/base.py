"""Shared data schemas for the perception -> action -> sequence pipeline.

These pydantic models are the contract between layers described in the
architecture doc: Perception answers "what do I see?", the temporal action
recognizer answers "what is happening?", and the experiment/sequence layer
answers "what does this mean for the procedure?". Every implementation of
PerceptionEngine / ActionRecognizer / SequenceEngine — mock or real — must
speak these same shapes, so swapping a mock for MediaPipe+HSV in Phase 2, or
a rule-based recognizer for an LSTM later, never touches downstream code.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObject(BaseModel):
    """One detected entity in a frame. Matches spec section 9's schema."""

    cls: str = Field(alias="class")
    confidence: float
    bbox: BBox

    model_config = {"populate_by_name": True}


class Landmark(BaseModel):
    name: str
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class PoseFrame(BaseModel):
    landmarks: list[Landmark] = Field(default_factory=list)
    detected: bool = False


class HandFrame(BaseModel):
    landmarks: list[Landmark] = Field(default_factory=list)
    handedness: Optional[str] = None  # "Left" | "Right"
    detected: bool = False


class PerceptionFrame(BaseModel):
    """Everything perception knows about a single video frame."""

    timestamp: float = Field(default_factory=time.time)
    frame_index: int = 0
    # Pixel dimensions of the source frame. Needed because MediaPipe
    # landmarks (pose/hands) come back normalized 0-1 while DetectedObject
    # bboxes (from color_detector.py) are already pixel-space — anything
    # comparing the two (Phase 3's InteractionReasoner) needs both in the
    # same units. 0 means "unknown" (e.g. a producer that predates this
    # field); consumers must treat that as "cannot do pixel-space math".
    frame_width: int = 0
    frame_height: int = 0
    objects: list[DetectedObject] = Field(default_factory=list)
    pose: PoseFrame = Field(default_factory=PoseFrame)
    hands: list[HandFrame] = Field(default_factory=list)


class InteractionState(str, Enum):
    NONE = "NONE"
    HAND_APPROACHING_OBJECT = "HAND_APPROACHING_OBJECT"
    HAND_TOUCHING_OBJECT = "HAND_TOUCHING_OBJECT"
    OBJECT_BEING_HELD = "OBJECT_BEING_HELD"
    OBJECT_MOVING_WITH_HAND = "OBJECT_MOVING_WITH_HAND"
    OBJECT_RELEASED = "OBJECT_RELEASED"
    OBJECT_PLACED = "OBJECT_PLACED"


class ActionPrediction(BaseModel):
    """Output of the temporal action recognizer for a window of frames."""

    action: Optional[str] = None  # e.g. "PICK_RED_BOX", None = no action yet
    confidence: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    source: str = "mock"  # "mock" | "rule_based" | "lstm" — for transparency in UI/logs


class SequenceStatus(str, Enum):
    CORRECT = "CORRECT"
    WRONG_OBJECT = "WRONG_OBJECT"
    SKIPPED_STEP = "SKIPPED_STEP"
    OUT_OF_SEQUENCE = "OUT_OF_SEQUENCE"
    REPEATED_STEP = "REPEATED_STEP"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    RECOVERED = "RECOVERED"
    COMPLETE = "COMPLETE"


# Statuses that represent an actual procedural deviation (as opposed to a
# clean match, a non-conclusive observation, or the auto-completion event).
# Shared by voice_service.py (which errors get spoken) and
# logging_service.py (which count toward a run's total_deviations).
ERROR_STATUSES = frozenset(
    {
        SequenceStatus.WRONG_OBJECT,
        SequenceStatus.SKIPPED_STEP,
        SequenceStatus.OUT_OF_SEQUENCE,
        SequenceStatus.REPEATED_STEP,
    }
)


class SequenceEvent(BaseModel):
    """Output of sequence validation for one recognized action. Matches the
    WebSocket 'sequence_event' shape in spec section 33 and the log schema
    in section 16."""

    timestamp: float = Field(default_factory=time.time)
    step: int
    expected: Optional[str] = None
    detected: Optional[str] = None
    confidence: float = 0.0
    status: SequenceStatus
    error_type: Optional[str] = None
    recovered: Optional[bool] = None
    explanation: list[str] = Field(default_factory=list)
