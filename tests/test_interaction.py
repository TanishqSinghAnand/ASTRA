"""Tests for backend/perception/interaction.py.

interaction.py tracks each object's own position over time — no hand
simulation needed at all (a deliberate redesign after real-camera testing
showed hand-proximity tracking too unreliable; see interaction.py's module
docstring). We hand-construct PerceptionFrame sequences with a moving or
stationary DetectedObject and feed them straight into InteractionReasoner.
"""
from __future__ import annotations

from backend.config.settings import ColorSpec, HSVRange, PerceptionSettings
from backend.perception.base import BBox, DetectedObject, InteractionState, PerceptionFrame
from backend.perception.interaction import InteractionReasoner

FRAME_W, FRAME_H = 640, 480
AREA_BBOX = BBox(x1=400, y1=300, x2=600, y2=460)
AREA_CENTER = (500.0, 380.0)
OUTSIDE_AREA = (150.0, 150.0)


def _settings(**overrides) -> PerceptionSettings:
    base = dict(colors={"red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))])})
    base.update(overrides)
    return PerceptionSettings(**base)


def _box(center: tuple[float, float], cls: str = "red_box", half_size: float = 20.0) -> DetectedObject:
    cx, cy = center
    return DetectedObject(
        **{"class": cls},
        confidence=0.9,
        bbox=BBox(x1=cx - half_size, y1=cy - half_size, x2=cx + half_size, y2=cy + half_size),
    )


def _frame(
    frame_index: int,
    box_center: tuple[float, float] | None,
    include_area: bool = True,
) -> PerceptionFrame:
    objects = []
    if include_area:
        objects.append(DetectedObject(**{"class": "experiment_area"}, confidence=1.0, bbox=AREA_BBOX))
    if box_center is not None:
        objects.append(_box(box_center))
    return PerceptionFrame(frame_index=frame_index, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects)


def _hold_still(reasoner: InteractionReasoner, center: tuple[float, float], frames: int = 3, start_index: int = 0):
    """Feeds enough identical-position frames to establish a stable
    baseline (so a later displacement is unambiguously "moved from here",
    not just the first couple of samples ever seen)."""
    event = None
    for i in range(frames):
        event = reasoner.update(_frame(start_index + i, center))["red_box"]
    return event


def _move(reasoner: InteractionReasoner, path: list[tuple[float, float]], start_index: int = 0):
    """Feeds one frame per position in path, returning the last event."""
    event = None
    for i, center in enumerate(path):
        event = reasoner.update(_frame(start_index + i, center))["red_box"]
    return event


def test_stationary_object_stays_none():
    reasoner = InteractionReasoner(_settings())
    event = _hold_still(reasoner, OUTSIDE_AREA, frames=6)
    assert event.state == InteractionState.NONE


def test_small_jitter_does_not_trigger_held():
    reasoner = InteractionReasoner(_settings())
    x, y = OUTSIDE_AREA
    jitter_path = [(x, y), (x + 2, y), (x - 1, y + 2), (x, y - 1), (x + 1, y), (x, y)]
    event = _move(reasoner, jitter_path)
    assert event.state == InteractionState.NONE


def test_genuine_displacement_becomes_held():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    x, y = OUTSIDE_AREA
    events = [
        reasoner.update(_frame(3 + i, c))["red_box"]
        for i, c in enumerate([(x + 10, y), (x + 40, y), (x + 90, y)])
    ]
    assert events[-1].state == InteractionState.OBJECT_BEING_HELD
    assert any(e.changed and e.state == InteractionState.OBJECT_BEING_HELD for e in events)


def test_held_then_stops_outside_area_is_released():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    _move(reasoner, [(160, 150), (220, 150), (280, 150)], start_index=3)
    event = _move(reasoner, [(280, 150)] * 3, start_index=6)
    assert event.state == InteractionState.OBJECT_RELEASED


def test_held_then_stops_inside_area_is_placed():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    path = [(200, 200), (300, 280), (400, 340), AREA_CENTER]
    _move(reasoner, path, start_index=3)
    event = _move(reasoner, [AREA_CENTER] * 3, start_index=3 + len(path))
    assert event.state == InteractionState.OBJECT_PLACED


def test_placed_state_persists_and_is_sticky():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    path = [(250, 250), (350, 300), (450, 350), AREA_CENTER]
    _move(reasoner, path, start_index=3)
    idx = 3 + len(path)
    placed = _move(reasoner, [AREA_CENTER] * 3, start_index=idx)
    assert placed.state == InteractionState.OBJECT_PLACED
    idx += 3

    for i in range(idx, idx + 6):
        ev = reasoner.update(_frame(i, AREA_CENTER))["red_box"]
        assert ev.state == InteractionState.OBJECT_PLACED
        assert ev.changed is False


def test_resting_object_reactivates_on_movement():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    path = [(250, 250), (350, 300), (450, 350), AREA_CENTER]
    _move(reasoner, path, start_index=3)
    idx = 3 + len(path)
    _move(reasoner, [AREA_CENTER] * 3, start_index=idx)
    idx += 3

    move_again = [(x, AREA_CENTER[1]) for x in (AREA_CENTER[0] - 30, AREA_CENTER[0] - 80, AREA_CENTER[0] - 150)]
    event = _move(reasoner, move_again, start_index=idx)
    assert event.state == InteractionState.OBJECT_BEING_HELD


def test_release_caught_early_upgrades_to_placed_next_frame():
    """A release classified RELEASED on the exact settling frame (bbox
    estimate just under the area's overlap-ratio threshold) must be able
    to upgrade to PLACED on a later frame once a small jitter (well under
    the movement threshold, so not mistaken for a fresh pick) nudges the
    reported position further into the area — not frozen at the first
    reading forever."""
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    # 40x40 box (half_size=20) centered at x=385: overlap width with the
    # area (x1=400) is 5px -> ratio 5*40/1600 = 0.125, just under the 0.2
    # threshold -> RELEASED, not PLACED, on settling.
    just_outside = (385.0, 380.0)
    path = [(250, 300), (320, 340), just_outside]
    _move(reasoner, path, start_index=3)
    idx = 3 + len(path)
    first_rest = _move(reasoner, [just_outside] * 3, start_index=idx)
    assert first_rest.state == InteractionState.OBJECT_RELEASED
    idx += 3

    # A 10px jitter (comfortably under the 28px movement threshold, so
    # still "resting" not a new pick) to x=395: overlap width 15px ->
    # ratio 15*40/1600 = 0.375 -> now clears the threshold.
    settled = reasoner.update(_frame(idx, (395.0, 380.0)))["red_box"]
    assert settled.state == InteractionState.OBJECT_PLACED


def test_occlusion_gap_preserves_held_state():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    _move(reasoner, [(200, 150), (280, 150), (360, 150)], start_index=3)
    held_event = reasoner.update(_frame(6, (360, 150)))["red_box"]
    assert held_event.state == InteractionState.OBJECT_BEING_HELD

    gap_event = reasoner.update(_frame(7, box_center=None))["red_box"]
    assert gap_event.state == InteractionState.OBJECT_BEING_HELD
    assert gap_event.changed is False


def test_reset_clears_all_tracks():
    reasoner = InteractionReasoner(_settings())
    _hold_still(reasoner, OUTSIDE_AREA, frames=3)
    event = _move(reasoner, [(200, 150), (280, 150), (360, 150)], start_index=3)
    assert event.state == InteractionState.OBJECT_BEING_HELD

    reasoner.reset()
    event = reasoner.update(_frame(99, box_center=None))["red_box"]
    assert event.state == InteractionState.NONE


def test_tracked_classes_come_from_settings_colors():
    settings = _settings(
        colors={
            "red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))]),
            "blue_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))]),
        }
    )
    reasoner = InteractionReasoner(settings)
    events = reasoner.update(_frame(0, box_center=None))
    assert set(events.keys()) == {"red_box", "blue_box"}


def test_unknown_frame_dimensions_are_safe_noop():
    reasoner = InteractionReasoner(_settings())
    frame = PerceptionFrame(frame_index=0, frame_width=0, frame_height=0)
    events = reasoner.update(frame)
    assert events["red_box"].state == InteractionState.NONE
    assert events["red_box"].changed is False


def test_full_pick_move_place_trajectory():
    """End-to-end: resting -> picked up -> carried -> placed, asserting the
    state at each stage and that OBJECT_PLACED only fires once, on the
    frame the box actually comes to rest inside the area."""
    reasoner = InteractionReasoner(_settings())
    states = []

    states.append(_hold_still(reasoner, (100, 100), frames=3))
    path = [(140, 150), (220, 220), (320, 300), (420, 350), (480, 370)]
    for i, center in enumerate(path, start=3):
        events = reasoner.update(_frame(i, center))
        states.append(events["red_box"])
    idx = 3 + len(path)
    place_event = _move(reasoner, [AREA_CENTER] * 3, start_index=idx)

    assert states[0].state == InteractionState.NONE
    assert any(s.state == InteractionState.OBJECT_BEING_HELD for s in states[1:])
    assert place_event.state == InteractionState.OBJECT_PLACED
