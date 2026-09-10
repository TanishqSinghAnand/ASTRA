"""Tests for backend/temporal/action_recognizer.py + temporal_smoother.py.

Same hand-constructed-PerceptionFrame approach as test_interaction.py (see
its module docstring: dwell timing needs an explicit, controllable
timestamp). A single FakeClock drives both the recognizer's
TemporalSmoother debounce and each frame's `timestamp`, so dwell/stability
timing is fully deterministic without any real wall-clock sleeps.
"""
from __future__ import annotations

from backend.config.settings import ColorSpec, HSVRange, PerceptionSettings, TemporalSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, Landmark, PerceptionFrame
from backend.temporal.action_recognizer import RuleBasedActionRecognizer

FRAME_W, FRAME_H = 640, 480
ZONE_BBOX = BBox(x1=400, y1=300, x2=600, y2=460)
OUTSIDE_ZONE = (150.0, 150.0)
INSIDE_ZONE = (500.0, 380.0)
PICK_DWELL = 3.0
PLACE_DWELL = 3.0
# Comfortably more than stability_frames(3) * the default _hold_hand step
# (0.3s) — see _pick_and_hold's comment.
STABILITY_MARGIN = 1.5


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.5) -> float:
        self.t += dt
        return self.t


def _settings(colors: dict | None = None) -> PerceptionSettings:
    return PerceptionSettings(
        detector_backend="hsv",
        hand_touch_distance_px=40,
        pick_dwell_seconds=PICK_DWELL,
        place_dwell_seconds=PLACE_DWELL,
        colors=colors
        or {"red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))])},
        # Values unused (DetectedObjects built directly with absolute pixel
        # bboxes) — only the key needs to exist for the zone lookup.
        target_zones={"zone": (0.0, 0.0, 0.0, 0.0)},
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
    timestamp: float,
    box_center: tuple[float, float] | None,
    hand_center_px: tuple[float, float] | None,
    box_confidence: float = 0.9,
    box_cls: str = "red_box",
) -> PerceptionFrame:
    objects = [DetectedObject(**{"class": "zone"}, confidence=1.0, bbox=ZONE_BBOX)]
    if box_center is not None:
        objects.append(_box(box_center, cls=box_cls, confidence=box_confidence))
    hands = [_hand(hand_center_px)] if hand_center_px is not None else []
    return PerceptionFrame(frame_index=0, timestamp=timestamp, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects, hands=hands)


def _recognizer(stability_frames=3, min_confidence_duration_ms=100):
    clock = FakeClock()
    temporal = TemporalSettings(action_stability_frames=stability_frames, min_confidence_duration_ms=min_confidence_duration_ms)
    recognizer = RuleBasedActionRecognizer(_settings(), temporal, clock=clock)
    return recognizer, clock


def _hold_hand(recognizer, clock, box_center, hand_center, until_seconds, step=0.3, box_confidence=0.9):
    """Feeds frames with the hand at hand_center (or None) and the box at
    box_center, advancing the fake clock by `step` each frame, until the
    clock has advanced by at least `until_seconds` total. Returns every
    ActionPrediction produced."""
    predictions = []
    elapsed = 0.0
    while elapsed < until_seconds:
        clock.tick(step)
        elapsed += step
        predictions.append(recognizer.update(_frame(clock.t, box_center, hand_center, box_confidence)))
    return predictions


def _pick_and_hold(recognizer, clock, from_center, hold_extra_seconds=2.0, box_confidence=0.9):
    """Dwells a hand on the object long enough to confirm a pick, then
    keeps it held (box stationary, no hand needed) for a bit longer.
    Returns every ActionPrediction produced."""
    hand = (from_center[0] + 5, from_center[1])
    # +STABILITY_MARGIN: interaction.py's own dwell only takes effect on
    # the frame the timer crosses the threshold — the *smoother* then
    # needs its own several-consecutive-frame stability window on top of
    # that (module docstring: a candidate must hold for stability_frames
    # consecutive frames before it's trusted) before it actually emits
    # the action, so every wait in this file needs to run past the raw
    # dwell threshold by at least that much, not stop right at it.
    predictions = _hold_hand(recognizer, clock, from_center, hand, PICK_DWELL + STABILITY_MARGIN, box_confidence=box_confidence)
    predictions += _hold_hand(recognizer, clock, from_center, None, hold_extra_seconds, box_confidence=box_confidence)
    return predictions


def test_no_candidate_yields_no_action():
    recognizer, clock = _recognizer()
    clock.tick()
    pred = recognizer.update(_frame(clock.t, box_center=OUTSIDE_ZONE, hand_center_px=None))
    assert pred.action is None


def test_pick_emitted_exactly_once_while_held():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=2.0)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX"]


def test_pick_action_confidence_blends_interaction_and_hsv():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=2.0, box_confidence=0.8)
    fired = next(p for p in predictions if p.action == "PICK_RED_BOX")
    # 0.5 * INTERACTION_CERTAINTY(0.90) + 0.5 * hsv_confidence(0.8) = 0.85
    assert abs(fired.confidence - 0.85) < 0.01
    assert fired.source == "rule_based"


def test_pick_from_inside_zone_fires():
    """The dwell-time design has no "must start outside every zone"
    validity check: sustained hand contact is itself already strong
    evidence of a genuine pick, regardless of where the object was
    resting (a real tabletop routinely has one object's neutral resting
    spot overlap another's target zone)."""
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, INSIDE_ZONE, hold_extra_seconds=2.0)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX"]


def test_place_emitted_after_zone_dwell():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    hold_predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=0.3)
    assert any(p.action == "PICK_RED_BOX" for p in hold_predictions)

    place_predictions = _hold_hand(recognizer, clock, INSIDE_ZONE, None, PLACE_DWELL + STABILITY_MARGIN)
    actions = [p.action for p in place_predictions if p.action is not None]
    assert actions == ["PLACE_RED_BOX"]


def test_place_action_carries_the_zone_it_landed_in():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=0.3)

    place_event = None
    for pred in _hold_hand(recognizer, clock, INSIDE_ZONE, None, PLACE_DWELL + STABILITY_MARGIN):
        if pred.action == "PLACE_RED_BOX":
            place_event = pred
            break
    assert place_event is not None
    assert place_event.location == "zone"


def test_staying_outside_zone_emits_no_place():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=0.3)

    settle_predictions = _hold_hand(recognizer, clock, (280.0, 150.0), None, PLACE_DWELL + STABILITY_MARGIN)
    actions = [p.action for p in settle_predictions if p.action is not None]
    assert actions == []


def test_full_pick_then_place_sequence_in_order():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    all_predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=0.3)
    all_predictions += _hold_hand(recognizer, clock, INSIDE_ZONE, None, PLACE_DWELL + STABILITY_MARGIN)

    actions = [p.action for p in all_predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX", "PLACE_RED_BOX"]


def test_reset_clears_recognizer_state():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=2.0)
    assert any(p.action == "PICK_RED_BOX" for p in predictions)

    recognizer.reset()
    clock.tick()
    pred = recognizer.update(_frame(clock.t, box_center=None, hand_center_px=None))
    assert pred.action is None


def test_stability_below_threshold_does_not_emit():
    # Stability threshold set far above how many frames the hold sequence
    # stays OBJECT_BEING_HELD for — nothing should ever fire.
    recognizer, clock = _recognizer(stability_frames=50, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, hold_extra_seconds=0.3)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == []
