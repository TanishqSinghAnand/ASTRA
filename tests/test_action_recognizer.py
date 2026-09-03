"""Tests for backend/temporal/action_recognizer.py + temporal_smoother.py.

Same hand-constructed-PerceptionFrame approach as test_interaction.py (see
its module docstring for why: no real hand-simulation path exists through
synthetic_scene.py/MediaPipe). A FakeClock is injected into the recognizer
so stability/duration debouncing is deterministic and the tests don't
depend on real wall-clock sleeps.
"""
from __future__ import annotations

from backend.config.settings import ColorSpec, HSVRange, PerceptionSettings, TemporalSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, Landmark, PerceptionFrame
from backend.temporal.action_recognizer import RuleBasedActionRecognizer

FRAME_W, FRAME_H = 640, 480
EXPERIMENT_AREA_BBOX = BBox(x1=400, y1=300, x2=600, y2=460)
OUTSIDE_AREA = (150.0, 150.0)
INSIDE_AREA = (500.0, 380.0)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.05) -> float:
        self.t += dt
        return self.t


def _settings(colors: dict | None = None) -> PerceptionSettings:
    return PerceptionSettings(
        hand_object_distance_px=80,
        hand_object_touch_distance_px=40,
        colors=colors
        or {"red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))])},
    )


def _box(center: tuple[float, float], cls: str = "red_box", confidence: float = 0.9, half_size: float = 20.0) -> DetectedObject:
    cx, cy = center
    return DetectedObject(
        **{"class": cls},
        confidence=confidence,
        bbox=BBox(x1=cx - half_size, y1=cy - half_size, x2=cx + half_size, y2=cy + half_size),
    )


def _hand(center_px: tuple[float, float]) -> HandFrame:
    nx, ny = center_px[0] / FRAME_W, center_px[1] / FRAME_H
    return HandFrame(
        detected=True,
        handedness="Right",
        landmarks=[
            Landmark(name="wrist", x=nx, y=ny),
            Landmark(name="index_mcp", x=nx, y=ny),
            Landmark(name="pinky_mcp", x=nx, y=ny),
        ],
    )


def _frame(
    frame_index: int,
    box_center: tuple[float, float] | None,
    hand_center_px: tuple[float, float] | None,
    box_confidence: float = 0.9,
    box_cls: str = "red_box",
) -> PerceptionFrame:
    objects = [DetectedObject(**{"class": "experiment_area"}, confidence=1.0, bbox=EXPERIMENT_AREA_BBOX)]
    if box_center is not None:
        objects.append(_box(box_center, cls=box_cls, confidence=box_confidence))
    hands = [_hand(hand_center_px)] if hand_center_px is not None else []
    return PerceptionFrame(frame_index=frame_index, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects, hands=hands)


def _recognizer(stability_frames=3, min_confidence_duration_ms=100, box_confidence_hint=None):
    clock = FakeClock()
    temporal = TemporalSettings(action_stability_frames=stability_frames, min_confidence_duration_ms=min_confidence_duration_ms)
    recognizer = RuleBasedActionRecognizer(_settings(), temporal, clock=clock)
    return recognizer, clock


def _run_hold_sequence(recognizer, clock, box_center, hand_offset=10.0, frames=6, box_confidence=0.9):
    """Feeds enough touch+hold frames to get into (and stay in)
    OBJECT_BEING_HELD, advancing the fake clock each frame. Returns the list
    of ActionPredictions."""
    predictions = []
    hand = (box_center[0] + hand_offset, box_center[1])
    # Two touch frames to promote NONE -> TOUCHING -> BEING_HELD (see interaction.py).
    for i in range(2):
        clock.tick()
        predictions.append(recognizer.update(_frame(i, box_center, hand, box_confidence)))
    for i in range(2, 2 + frames):
        clock.tick()
        predictions.append(recognizer.update(_frame(i, box_center, hand, box_confidence)))
    return predictions


def test_no_candidate_yields_no_action():
    recognizer, clock = _recognizer()
    clock.tick()
    pred = recognizer.update(_frame(0, box_center=OUTSIDE_AREA, hand_center_px=(400.0, 400.0)))
    assert pred.action is None


def test_pick_emitted_exactly_once_while_held():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=8)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX"]


def test_pick_action_confidence_blends_interaction_and_hsv():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=8, box_confidence=0.8)
    fired = next(p for p in predictions if p.action == "PICK_RED_BOX")
    # 0.5 * INTERACTION_CERTAINTY(0.90) + 0.5 * hsv_confidence(0.8) = 0.85
    assert abs(fired.confidence - 0.85) < 0.01
    assert fired.source == "rule_based"


def test_pick_from_inside_area_never_fires():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    # Object is already inside the experiment area when the hold begins —
    # not a real "pick" per the module's design (latched at hold-start).
    predictions = _run_hold_sequence(recognizer, clock, INSIDE_AREA, frames=8)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == []


def test_place_emitted_after_release_into_area():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    hold_predictions = _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=4)
    assert any(p.action == "PICK_RED_BOX" for p in hold_predictions)

    # Release with the box already reported at a position inside the area.
    place_predictions = []
    for i in range(20, 20 + 6):
        clock.tick()
        place_predictions.append(recognizer.update(_frame(i, INSIDE_AREA, (50.0, 50.0))))
    actions = [p.action for p in place_predictions if p.action is not None]
    assert actions == ["PLACE_RED_BOX"]


def test_release_outside_area_emits_no_place():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=4)

    release_predictions = []
    for i in range(20, 20 + 6):
        clock.tick()
        release_predictions.append(recognizer.update(_frame(i, OUTSIDE_AREA, (400.0, 400.0))))
    actions = [p.action for p in release_predictions if p.action is not None]
    assert actions == []


def test_full_pick_then_place_sequence_in_order():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    all_predictions = []
    all_predictions += _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=4)
    for i in range(20, 26):
        clock.tick()
        all_predictions.append(recognizer.update(_frame(i, INSIDE_AREA, (50.0, 50.0))))
    actions = [p.action for p in all_predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX", "PLACE_RED_BOX"]


def test_reset_clears_recognizer_state():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=8)
    assert any(p.action == "PICK_RED_BOX" for p in predictions)

    recognizer.reset()
    clock.tick()
    pred = recognizer.update(_frame(0, box_center=None, hand_center_px=None))
    assert pred.action is None


def test_stability_below_threshold_does_not_emit():
    # Only 1 touch+hold frame, well short of stability_frames=6 default-ish
    # threshold used here — nothing should fire.
    recognizer, clock = _recognizer(stability_frames=10, min_confidence_duration_ms=100)
    predictions = _run_hold_sequence(recognizer, clock, OUTSIDE_AREA, frames=2)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == []
