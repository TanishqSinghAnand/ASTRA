"""Tests for backend/temporal/action_recognizer.py + temporal_smoother.py.

Same position-only PerceptionFrame approach as test_interaction.py (see
its module docstring: motion-based tracking needs no hand simulation at
all). A FakeClock is injected into the recognizer so stability/duration
debouncing is deterministic and the tests don't depend on real wall-clock
sleeps.
"""
from __future__ import annotations

from backend.config.settings import ColorSpec, HSVRange, PerceptionSettings, TemporalSettings
from backend.perception.base import BBox, DetectedObject, PerceptionFrame
from backend.temporal.action_recognizer import RuleBasedActionRecognizer

FRAME_W, FRAME_H = 640, 480
ZONE_BBOX = BBox(x1=400, y1=300, x2=600, y2=460)
OUTSIDE_ZONE = (150.0, 150.0)
INSIDE_ZONE = (500.0, 380.0)


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
        detector_backend="hsv",
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


def _frame(
    frame_index: int,
    box_center: tuple[float, float] | None,
    box_confidence: float = 0.9,
    box_cls: str = "red_box",
) -> PerceptionFrame:
    objects = [DetectedObject(**{"class": "zone"}, confidence=1.0, bbox=ZONE_BBOX)]
    if box_center is not None:
        objects.append(_box(box_center, cls=box_cls, confidence=box_confidence))
    return PerceptionFrame(frame_index=frame_index, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects)


def _recognizer(stability_frames=3, min_confidence_duration_ms=100):
    clock = FakeClock()
    temporal = TemporalSettings(action_stability_frames=stability_frames, min_confidence_duration_ms=min_confidence_duration_ms)
    recognizer = RuleBasedActionRecognizer(_settings(), temporal, clock=clock)
    return recognizer, clock


def _settle(recognizer, clock, center, frames=3, start_index=0, box_confidence=0.9):
    """Feeds identical-position frames so the object is (or stays) at
    rest, advancing the fake clock each frame. Returns the list of
    ActionPredictions."""
    predictions = []
    for i in range(frames):
        clock.tick()
        predictions.append(recognizer.update(_frame(start_index + i, center, box_confidence)))
    return predictions


def _carry(recognizer, clock, path, start_index=0, box_confidence=0.9):
    """Feeds one frame per position in path — a genuine multi-step
    displacement, well over the movement threshold. Returns the list of
    ActionPredictions."""
    predictions = []
    for i, center in enumerate(path):
        clock.tick()
        predictions.append(recognizer.update(_frame(start_index + i, center, box_confidence)))
    return predictions


def _pick_and_hold(recognizer, clock, from_center, frames=6, box_confidence=0.9):
    """Rests at from_center, then carries far enough to register
    OBJECT_BEING_HELD and stays there for `frames` more frames. Returns
    every ActionPrediction produced."""
    predictions = _settle(recognizer, clock, from_center, frames=3, start_index=0, box_confidence=box_confidence)
    x, y = from_center
    path = [(x + d, y) for d in (40, 90, 140)]
    predictions += _carry(recognizer, clock, path, start_index=3, box_confidence=box_confidence)
    predictions += _settle(recognizer, clock, path[-1], frames=frames, start_index=3 + len(path), box_confidence=box_confidence)
    return predictions


def test_no_candidate_yields_no_action():
    recognizer, clock = _recognizer()
    clock.tick()
    pred = recognizer.update(_frame(0, box_center=OUTSIDE_ZONE))
    assert pred.action is None


def test_pick_emitted_exactly_once_while_held():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=8)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX"]


def test_pick_action_confidence_blends_interaction_and_hsv():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=8, box_confidence=0.8)
    fired = next(p for p in predictions if p.action == "PICK_RED_BOX")
    # 0.5 * INTERACTION_CERTAINTY(0.90) + 0.5 * hsv_confidence(0.8) = 0.85
    assert abs(fired.confidence - 0.85) < 0.01
    assert fired.source == "rule_based"


def test_pick_from_inside_zone_now_fires():
    """Design change from the old hand-proximity-based recognizer: motion
    is unambiguous evidence of a pick regardless of where the object
    started (see interaction.py's module docstring — two objects sharing
    a small real table routinely means one's neutral resting spot
    overlaps the other's target zone, which made the old "must start
    outside every zone" validity check actively counterproductive)."""
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, INSIDE_ZONE, frames=8)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX"]


def test_place_emitted_after_settling_into_zone():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    hold_predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=1)
    assert any(p.action == "PICK_RED_BOX" for p in hold_predictions)

    # Carry into the zone and settle.
    idx = len(hold_predictions)
    carry_predictions = _carry(recognizer, clock, [(300, 300), (400, 350), INSIDE_ZONE], start_index=idx)
    idx += len(carry_predictions)
    place_predictions = _settle(recognizer, clock, INSIDE_ZONE, frames=6, start_index=idx)

    actions = [p.action for p in place_predictions if p.action is not None]
    assert actions == ["PLACE_RED_BOX"]


def test_place_action_carries_the_zone_it_landed_in():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    hold_predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=1)
    idx = len(hold_predictions)
    carry_predictions = _carry(recognizer, clock, [(300, 300), (400, 350), INSIDE_ZONE], start_index=idx)
    idx += len(carry_predictions)

    place_event = None
    for pred in _settle(recognizer, clock, INSIDE_ZONE, frames=6, start_index=idx):
        if pred.action == "PLACE_RED_BOX":
            place_event = pred
            break
    assert place_event is not None
    assert place_event.location == "zone"


def test_settling_outside_zone_emits_no_place():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    hold_predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=1)
    idx = len(hold_predictions)
    # Carry elsewhere, still outside the zone, and settle there.
    carry_predictions = _carry(recognizer, clock, [(220, 150), (280, 150)], start_index=idx)
    idx += len(carry_predictions)
    settle_predictions = _settle(recognizer, clock, (280, 150), frames=6, start_index=idx)

    actions = [p.action for p in settle_predictions if p.action is not None]
    assert actions == []


def test_full_pick_then_place_sequence_in_order():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    all_predictions = []
    hold_predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=1)
    all_predictions += hold_predictions
    idx = len(hold_predictions)
    all_predictions += _carry(recognizer, clock, [(300, 300), (400, 350), INSIDE_ZONE], start_index=idx)
    idx = len(all_predictions)
    all_predictions += _settle(recognizer, clock, INSIDE_ZONE, frames=6, start_index=idx)

    actions = [p.action for p in all_predictions if p.action is not None]
    assert actions == ["PICK_RED_BOX", "PLACE_RED_BOX"]


def test_reset_clears_recognizer_state():
    recognizer, clock = _recognizer(stability_frames=3, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=8)
    assert any(p.action == "PICK_RED_BOX" for p in predictions)

    recognizer.reset()
    clock.tick()
    pred = recognizer.update(_frame(0, box_center=None))
    assert pred.action is None


def test_stability_below_threshold_does_not_emit():
    # Stability threshold set far above how long the hold sequence stays
    # OBJECT_BEING_HELD for — nothing should ever fire.
    recognizer, clock = _recognizer(stability_frames=50, min_confidence_duration_ms=100)
    predictions = _pick_and_hold(recognizer, clock, OUTSIDE_ZONE, frames=1)
    actions = [p.action for p in predictions if p.action is not None]
    assert actions == []
