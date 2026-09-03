"""Tests for backend/perception/interaction.py.

Per the Phase 3 testing note: synthetic_scene.py only draws static colored
squares (no simulated hand) and MediaPipe can't detect a hand/pose from a
procedurally generated blob anyway, so this cannot be exercised through the
real camera pipeline. Instead we hand-construct PerceptionFrame sequences
directly (fabricated DetectedObject/HandFrame/Landmark trajectories) and
feed them straight into InteractionReasoner, bypassing MediaPipe entirely.
"""
from __future__ import annotations

from backend.config.settings import ColorSpec, HSVRange, PerceptionSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, InteractionState, Landmark, PerceptionFrame
from backend.perception.interaction import InteractionReasoner

FRAME_W, FRAME_H = 640, 480
EXPERIMENT_AREA_BBOX = BBox(x1=400, y1=300, x2=600, y2=460)
OUTSIDE_EXPERIMENT_AREA = (150.0, 150.0)


def _settings(**overrides) -> PerceptionSettings:
    base = dict(
        hand_object_distance_px=80,
        hand_object_touch_distance_px=40,
        colors={"red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))])},
    )
    base.update(overrides)
    return PerceptionSettings(**base)


def _box(center: tuple[float, float], cls: str = "red_box", half_size: float = 20.0) -> DetectedObject:
    cx, cy = center
    return DetectedObject(
        **{"class": cls},
        confidence=0.9,
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
    include_area: bool = True,
) -> PerceptionFrame:
    objects = []
    if include_area:
        objects.append(DetectedObject(**{"class": "experiment_area"}, confidence=1.0, bbox=EXPERIMENT_AREA_BBOX))
    if box_center is not None:
        objects.append(_box(box_center))
    hands = [_hand(hand_center_px)] if hand_center_px is not None else []
    return PerceptionFrame(frame_index=frame_index, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects, hands=hands)


def test_hand_far_from_object_is_none():
    reasoner = InteractionReasoner(_settings())
    events = reasoner.update(_frame(0, box_center=(100, 100), hand_center_px=(500, 400)))
    assert events["red_box"].state == InteractionState.NONE


def test_hand_within_approach_threshold():
    reasoner = InteractionReasoner(_settings())
    # distance = 60, between touch(40) and approach(80)
    events = reasoner.update(_frame(0, box_center=(100, 100), hand_center_px=(160, 100)))
    assert events["red_box"].state == InteractionState.HAND_APPROACHING_OBJECT


def test_touching_requires_two_frames_before_held():
    reasoner = InteractionReasoner(_settings())
    box = (100.0, 100.0)
    hand = (115.0, 100.0)  # distance = 15, within touch threshold (40)

    first = reasoner.update(_frame(0, box, hand))
    assert first["red_box"].state == InteractionState.HAND_TOUCHING_OBJECT
    assert first["red_box"].changed is True

    second = reasoner.update(_frame(1, box, hand))
    assert second["red_box"].state == InteractionState.OBJECT_BEING_HELD
    assert second["red_box"].changed is True


def test_held_object_moving_with_hand_becomes_moving():
    reasoner = InteractionReasoner(_settings())
    hand_offset = 10.0  # hand stays 10px from box center throughout -> always "touching"

    # Get into OBJECT_BEING_HELD first.
    reasoner.update(_frame(0, (100, 100), (100 + hand_offset, 100)))
    reasoner.update(_frame(1, (100, 100), (100 + hand_offset, 100)))

    # Now walk the box+hand together by a large displacement over a few
    # frames -> should flip to OBJECT_MOVING_WITH_HAND once the box's first
    # vs. latest tracked position exceeds the noise floor.
    last_event = None
    for i, x in enumerate([120, 160, 200, 260], start=2):
        last_event = reasoner.update(_frame(i, (x, 100), (x + hand_offset, 100)))["red_box"]
    assert last_event.state == InteractionState.OBJECT_MOVING_WITH_HAND


def test_release_outside_experiment_area():
    reasoner = InteractionReasoner(_settings())
    reasoner.update(_frame(0, OUTSIDE_EXPERIMENT_AREA, (OUTSIDE_EXPERIMENT_AREA[0] + 10, OUTSIDE_EXPERIMENT_AREA[1])))
    reasoner.update(_frame(1, OUTSIDE_EXPERIMENT_AREA, (OUTSIDE_EXPERIMENT_AREA[0] + 10, OUTSIDE_EXPERIMENT_AREA[1])))
    # Hand pulls far away while the box stays put and outside the experiment area.
    event = reasoner.update(_frame(2, OUTSIDE_EXPERIMENT_AREA, (400.0, 400.0)))["red_box"]
    assert event.state == InteractionState.OBJECT_RELEASED


def test_release_inside_experiment_area_is_placed():
    reasoner = InteractionReasoner(_settings())
    area_center = (500.0, 380.0)  # inside EXPERIMENT_AREA_BBOX
    reasoner.update(_frame(0, area_center, (area_center[0] + 10, area_center[1])))
    reasoner.update(_frame(1, area_center, (area_center[0] + 10, area_center[1])))
    event = reasoner.update(_frame(2, area_center, (50.0, 50.0)))["red_box"]
    assert event.state == InteractionState.OBJECT_PLACED


def test_placed_state_persists_until_retouched():
    reasoner = InteractionReasoner(_settings())
    area_center = (500.0, 380.0)
    reasoner.update(_frame(0, area_center, (area_center[0] + 10, area_center[1])))
    reasoner.update(_frame(1, area_center, (area_center[0] + 10, area_center[1])))
    placed = reasoner.update(_frame(2, area_center, (50.0, 50.0)))["red_box"]
    assert placed.state == InteractionState.OBJECT_PLACED

    # Hand stays far away, box hasn't moved -> still placed, several frames later.
    for i in range(3, 8):
        ev = reasoner.update(_frame(i, area_center, (50.0, 50.0)))["red_box"]
        assert ev.state == InteractionState.OBJECT_PLACED
        assert ev.changed is False

    # Hand comes back and touches it -> restarts the touch/hold cycle rather
    # than staying stuck in PLACED forever.
    retouch = reasoner.update(_frame(8, area_center, (area_center[0] + 10, area_center[1])))["red_box"]
    assert retouch.state == InteractionState.HAND_TOUCHING_OBJECT


def test_released_state_persists_until_retouched():
    reasoner = InteractionReasoner(_settings())
    reasoner.update(_frame(0, OUTSIDE_EXPERIMENT_AREA, (OUTSIDE_EXPERIMENT_AREA[0] + 10, OUTSIDE_EXPERIMENT_AREA[1])))
    reasoner.update(_frame(1, OUTSIDE_EXPERIMENT_AREA, (OUTSIDE_EXPERIMENT_AREA[0] + 10, OUTSIDE_EXPERIMENT_AREA[1])))
    released = reasoner.update(_frame(2, OUTSIDE_EXPERIMENT_AREA, (400.0, 400.0)))["red_box"]
    assert released.state == InteractionState.OBJECT_RELEASED

    ev = reasoner.update(_frame(3, OUTSIDE_EXPERIMENT_AREA, (400.0, 400.0)))["red_box"]
    assert ev.state == InteractionState.OBJECT_RELEASED
    assert ev.changed is False


def test_occlusion_gap_preserves_held_state():
    reasoner = InteractionReasoner(_settings())
    box = (100.0, 100.0)
    hand = (110.0, 100.0)
    reasoner.update(_frame(0, box, hand))
    reasoner.update(_frame(1, box, hand))  # now OBJECT_BEING_HELD
    # Box briefly undetected (hand occluding its own color blob) — state
    # must not reset to NONE/RELEASED just because of a missing detection.
    gap_event = reasoner.update(_frame(2, box_center=None, hand_center_px=hand))["red_box"]
    assert gap_event.state == InteractionState.OBJECT_BEING_HELD
    assert gap_event.changed is False


def test_reset_clears_all_tracks():
    reasoner = InteractionReasoner(_settings())
    box = (100.0, 100.0)
    hand = (110.0, 100.0)
    reasoner.update(_frame(0, box, hand))
    reasoner.update(_frame(1, box, hand))
    assert reasoner.update(_frame(2, box, hand))["red_box"].state == InteractionState.OBJECT_BEING_HELD

    reasoner.reset()
    event = reasoner.update(_frame(3, box_center=None, hand_center_px=None))["red_box"]
    assert event.state == InteractionState.NONE


def test_tracked_classes_come_from_settings_colors():
    settings = _settings(
        colors={
            "red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))]),
            "blue_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))]),
        }
    )
    reasoner = InteractionReasoner(settings)
    events = reasoner.update(_frame(0, box_center=None, hand_center_px=None))
    assert set(events.keys()) == {"red_box", "blue_box"}


def test_unknown_frame_dimensions_are_safe_noop():
    reasoner = InteractionReasoner(_settings())
    frame = PerceptionFrame(frame_index=0, frame_width=0, frame_height=0)
    events = reasoner.update(frame)
    assert events["red_box"].state == InteractionState.NONE
    assert events["red_box"].changed is False


def test_full_pick_move_place_trajectory():
    """End-to-end: approach -> touch -> hold -> move -> place, asserting
    the state at each stage and that OBJECT_PLACED only fires once, on the
    frame the box actually lands inside the experiment area."""
    reasoner = InteractionReasoner(_settings())
    hand_offset = 10.0
    states = []

    # Approach
    states.append(reasoner.update(_frame(0, (100, 100), (170, 100)))["red_box"].state)  # dist=70 -> approaching
    # Touch (first + second frame)
    states.append(reasoner.update(_frame(1, (100, 100), (100 + hand_offset, 100)))["red_box"].state)
    states.append(reasoner.update(_frame(2, (100, 100), (100 + hand_offset, 100)))["red_box"].state)
    # Move together toward the experiment area
    events = None
    for i, x in enumerate([140, 200, 300, 400, 480], start=3):
        events = reasoner.update(_frame(i, (x, 380), (x + hand_offset, 380)))
        states.append(events["red_box"].state)
    # Release inside the experiment area (500, 380 is inside EXPERIMENT_AREA_BBOX)
    place_event = reasoner.update(_frame(9, (500, 380), (50, 50)))["red_box"]

    assert states[0] == InteractionState.HAND_APPROACHING_OBJECT
    assert states[1] == InteractionState.HAND_TOUCHING_OBJECT
    assert states[2] == InteractionState.OBJECT_BEING_HELD
    assert InteractionState.OBJECT_MOVING_WITH_HAND in states[3:]
    assert place_event.state == InteractionState.OBJECT_PLACED
    assert place_event.changed is True
