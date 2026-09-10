"""Tests for backend/perception/interaction.py.

interaction.py confirms PICK/PLACE by dwell time (continuous seconds of
hand contact / zone occupancy), not frame counts — see its module
docstring for the design history. Frames are hand-constructed with an
explicit `timestamp` so dwell timing is deterministic in tests, without
any real wall-clock sleeps.
"""
from __future__ import annotations

from backend.config.settings import ColorSpec, HSVRange, PerceptionSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, InteractionState, Landmark, PerceptionFrame
from backend.perception.interaction import InteractionReasoner

FRAME_W, FRAME_H = 640, 480
ZONE_BBOX = BBox(x1=400, y1=300, x2=600, y2=460)
ZONE_CENTER = (500.0, 380.0)
OUTSIDE_ZONE = (150.0, 150.0)
PICK_DWELL = 3.0
PLACE_DWELL = 3.0


def _settings(**overrides) -> PerceptionSettings:
    base = dict(
        detector_backend="hsv",
        hand_touch_distance_px=40,
        pick_dwell_seconds=PICK_DWELL,
        place_dwell_seconds=PLACE_DWELL,
        colors={"red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))])},
        # Values are unused by these tests (DetectedObjects are constructed
        # directly with absolute pixel bboxes in _frame()) — only the key
        # needs to exist so interaction.py recognizes "zone" as a valid
        # zone name to look up.
        target_zones={"zone": (0.0, 0.0, 0.0, 0.0)},
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
    timestamp: float,
    box_center: tuple[float, float] | None,
    hand_center_px: tuple[float, float] | None,
    include_zone: bool = True,
    zone_bbox: BBox = ZONE_BBOX,
    zone_cls: str = "zone",
    frame_index: int = 0,
) -> PerceptionFrame:
    objects = []
    if include_zone:
        objects.append(DetectedObject(**{"class": zone_cls}, confidence=1.0, bbox=zone_bbox))
    if box_center is not None:
        objects.append(_box(box_center))
    hands = [_hand(hand_center_px)] if hand_center_px is not None else []
    return PerceptionFrame(
        frame_index=frame_index, timestamp=timestamp, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects, hands=hands
    )


def test_hand_far_from_object_is_none():
    reasoner = InteractionReasoner(_settings())
    event = reasoner.update(_frame(0.0, OUTSIDE_ZONE, (500.0, 400.0)))["red_box"]
    assert event.state == InteractionState.NONE


def test_brief_touch_under_dwell_does_not_pick():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    event = reasoner.update(_frame(PICK_DWELL - 0.5, OUTSIDE_ZONE, hand))["red_box"]
    assert event.state == InteractionState.NONE


def test_touch_sustained_for_dwell_becomes_held():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    event = reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))["red_box"]
    assert event.state == InteractionState.OBJECT_BEING_HELD
    assert event.changed is True


def test_touch_broken_resets_dwell_timer():
    reasoner = InteractionReasoner(_settings())
    hand_near = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    hand_far = (500.0, 400.0)
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand_near))
    # Contact clearly breaks (hand observed far away) partway through.
    reasoner.update(_frame(1.5, OUTSIDE_ZONE, hand_far))
    # Total elapsed since the very first touch would exceed the dwell, but
    # contact was broken in between -> must restart, not still be held.
    event = reasoner.update(_frame(PICK_DWELL + 0.2, OUTSIDE_ZONE, hand_near))["red_box"]
    assert event.state == InteractionState.NONE

    # Continuing from here for the full dwell duration does confirm it.
    event = reasoner.update(_frame(PICK_DWELL + 0.2 + PICK_DWELL + 0.1, OUTSIDE_ZONE, hand_near))["red_box"]
    assert event.state == InteractionState.OBJECT_BEING_HELD


def test_no_hand_detected_gap_preserves_dwell_progress():
    """A frame with no hand detected at all (vs. one showing the hand
    clearly elsewhere) is ambiguous — detector miss, not necessarily a
    broken grip — and must not reset an in-progress dwell."""
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    # No hand detected this frame at all.
    gap_event = reasoner.update(_frame(1.0, OUTSIDE_ZONE, None))["red_box"]
    assert gap_event.state == InteractionState.NONE
    # Hand reappears close, and the *original* start time still counts.
    event = reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))["red_box"]
    assert event.state == InteractionState.OBJECT_BEING_HELD


def test_held_in_zone_for_dwell_becomes_placed():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    held = reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))["red_box"]
    assert held.state == InteractionState.OBJECT_BEING_HELD

    t0 = PICK_DWELL + 0.1
    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, None))  # carried into the zone
    event = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL + 0.1, ZONE_CENTER, None))["red_box"]
    assert event.state == InteractionState.OBJECT_PLACED
    assert event.zone == "zone"


def test_held_in_zone_briefly_not_enough_stays_held():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))
    t0 = PICK_DWELL + 0.1
    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, None))
    event = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL - 0.5, ZONE_CENTER, None))["red_box"]
    assert event.state == InteractionState.OBJECT_BEING_HELD


def test_zone_dwell_resets_if_object_leaves_zone():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))
    t0 = PICK_DWELL + 0.1

    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, None))  # enters zone
    reasoner.update(_frame(t0 + 1.5, OUTSIDE_ZONE, None))  # leaves before dwell completes
    # Total elapsed since first entering the zone would exceed the dwell,
    # but it left in between -> must restart.
    event = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL + 0.1, ZONE_CENTER, None))["red_box"]
    assert event.state == InteractionState.OBJECT_BEING_HELD


def test_hand_in_zone_does_not_disrupt_place_dwell():
    """A hand lingering in/near the zone while the object dwells there
    must not affect the place timer at all — only the object's own
    position matters once held."""
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))
    t0 = PICK_DWELL + 0.1

    hand_in_zone = (ZONE_CENTER[0] + 5, ZONE_CENTER[1])
    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, hand_in_zone))
    event = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL + 0.1, ZONE_CENTER, hand_in_zone))["red_box"]
    assert event.state == InteractionState.OBJECT_PLACED


def test_multiple_zones_report_which_one_was_landed_in():
    zone_a_bbox = BBox(x1=0, y1=0, x2=200, y2=200)
    zone_b_bbox = BBox(x1=400, y1=300, x2=600, y2=460)
    settings = _settings(target_zones={"zone_a": (0.0, 0.0, 0.0, 0.0), "zone_b": (0.0, 0.0, 0.0, 0.0)})
    reasoner = InteractionReasoner(settings)

    def frame_with_zones(ts, box_center, hand_center):
        objects = [
            DetectedObject(**{"class": "zone_a"}, confidence=1.0, bbox=zone_a_bbox),
            DetectedObject(**{"class": "zone_b"}, confidence=1.0, bbox=zone_b_bbox),
            _box(box_center),
        ]
        hands = [_hand(hand_center)] if hand_center is not None else []
        return PerceptionFrame(frame_index=0, timestamp=ts, frame_width=FRAME_W, frame_height=FRAME_H, objects=objects, hands=hands)

    pick_point = (300.0, 380.0)  # outside both zones
    hand = (pick_point[0] + 5, pick_point[1])
    reasoner.update(frame_with_zones(0.0, pick_point, hand))
    reasoner.update(frame_with_zones(PICK_DWELL + 0.1, pick_point, hand))
    t0 = PICK_DWELL + 0.1

    place_in_b = (500.0, 380.0)  # inside zone_b only
    reasoner.update(frame_with_zones(t0 + 0.1, place_in_b, None))
    event = reasoner.update(frame_with_zones(t0 + 0.1 + PLACE_DWELL + 0.1, place_in_b, None))["red_box"]

    assert event.state == InteractionState.OBJECT_PLACED
    assert event.zone == "zone_b"


def test_placed_state_persists_until_retouched():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))
    t0 = PICK_DWELL + 0.1
    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, None))
    placed = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL + 0.1, ZONE_CENTER, None))["red_box"]
    assert placed.state == InteractionState.OBJECT_PLACED
    t1 = t0 + 0.1 + PLACE_DWELL + 0.1

    # No hand nearby, box hasn't moved -> still placed, much later.
    ev = reasoner.update(_frame(t1 + 10.0, ZONE_CENTER, None))["red_box"]
    assert ev.state == InteractionState.OBJECT_PLACED
    assert ev.changed is False

    # A hand merely passing near (not literally over the box) doesn't
    # reactivate a resting/placed object.
    nearby_not_touching = (ZONE_CENTER[0] + 100, ZONE_CENTER[1])
    ev = reasoner.update(_frame(t1 + 11.0, ZONE_CENTER, nearby_not_touching))["red_box"]
    assert ev.state == InteractionState.OBJECT_PLACED

    # A hand actually on the box, sustained for the pick dwell, restarts
    # the cycle.
    on_box = ZONE_CENTER
    reasoner.update(_frame(t1 + 12.0, ZONE_CENTER, on_box))
    retouch = reasoner.update(_frame(t1 + 12.0 + PICK_DWELL + 0.1, ZONE_CENTER, on_box))["red_box"]
    assert retouch.state == InteractionState.OBJECT_BEING_HELD


def test_retouch_of_placed_object_tolerates_hand_just_outside_raw_bbox():
    """Regression: a real grasp's landmark centroid (wrist + index_mcp +
    pinky_mcp) routinely lands just outside an already-placed object's own
    detected bbox -- e.g. gripping from below/the side, wrist trailing
    below the box -- even though the fingers are plainly in contact.
    Observed live: re-picking an object placed in the wrong zone silently
    never started its dwell timer because of this, leaving PLACE stuck
    forever. The re-pick check now pads the resting object's bbox by half
    touch_threshold rather than requiring strict containment."""
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))
    t0 = PICK_DWELL + 0.1
    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, None))
    placed = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL + 0.1, ZONE_CENTER, None))["red_box"]
    assert placed.state == InteractionState.OBJECT_PLACED
    t1 = t0 + 0.1 + PLACE_DWELL + 0.1

    # _box()'s bbox around ZONE_CENTER=(500, 380) with half_size=20 is
    # x:[480,520] y:[360,400] -- this hand is 15px below the raw bbox
    # (outside it) but within the padded region (touch_threshold=40 -> pad
    # 20), simulating a wrist trailing just below a held box.
    just_outside_raw_bbox = (500.0, 415.0)
    reasoner.update(_frame(t1 + 1.0, ZONE_CENTER, just_outside_raw_bbox))
    retouch = reasoner.update(
        _frame(t1 + 1.0 + PICK_DWELL + 0.1, ZONE_CENTER, just_outside_raw_bbox)
    )["red_box"]
    assert retouch.state == InteractionState.OBJECT_BEING_HELD


def test_reset_clears_all_tracks():
    reasoner = InteractionReasoner(_settings())
    hand = (OUTSIDE_ZONE[0] + 5, OUTSIDE_ZONE[1])
    reasoner.update(_frame(0.0, OUTSIDE_ZONE, hand))
    event = reasoner.update(_frame(PICK_DWELL + 0.1, OUTSIDE_ZONE, hand))["red_box"]
    assert event.state == InteractionState.OBJECT_BEING_HELD

    reasoner.reset()
    event = reasoner.update(_frame(999.0, box_center=None, hand_center_px=None))["red_box"]
    assert event.state == InteractionState.NONE


def test_tracked_classes_come_from_settings_colors():
    settings = _settings(
        colors={
            "red_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))]),
            "blue_box": ColorSpec(ranges=[HSVRange(lower=(0, 0, 0), upper=(1, 1, 1))]),
        }
    )
    reasoner = InteractionReasoner(settings)
    events = reasoner.update(_frame(0.0, box_center=None, hand_center_px=None))
    assert set(events.keys()) == {"red_box", "blue_box"}


def test_unknown_frame_dimensions_are_safe_noop():
    reasoner = InteractionReasoner(_settings())
    frame = PerceptionFrame(frame_index=0, frame_width=0, frame_height=0)
    events = reasoner.update(frame)
    assert events["red_box"].state == InteractionState.NONE
    assert events["red_box"].changed is False


def test_full_pick_dwell_then_place_dwell_trajectory():
    """End-to-end: resting -> hand dwells -> picked up -> carried into the
    zone -> position dwells -> placed."""
    reasoner = InteractionReasoner(_settings())

    hand = (100.0 + 5, 100.0)
    initial = reasoner.update(_frame(0.0, (100.0, 100.0), hand))["red_box"]
    assert initial.state == InteractionState.NONE

    picked = reasoner.update(_frame(PICK_DWELL + 0.1, (100.0, 100.0), hand))["red_box"]
    assert picked.state == InteractionState.OBJECT_BEING_HELD
    t0 = PICK_DWELL + 0.1

    reasoner.update(_frame(t0 + 0.1, ZONE_CENTER, None))
    place_event = reasoner.update(_frame(t0 + 0.1 + PLACE_DWELL + 0.1, ZONE_CENTER, None))["red_box"]

    assert place_event.state == InteractionState.OBJECT_PLACED
    assert place_event.zone == "zone"
