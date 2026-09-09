"""Hand-object interaction reasoning (Phase 3; multi-zone placement v2.0).

Still perception, not experiment logic: this module only answers "is a hand
approaching/touching/holding/placing an object?" for each tracked box, and
(v2.0) "which configured zone did it land in?" — not whether that's the
*right* zone for the current step, which is the sequence layer's job
(backend/experiment/validator.py). It knows nothing about PICK/PLACE
experiment steps or the sequence — Phase 4 (backend/temporal/) is what
turns an InteractionState transition into an action_key like
"PICK_RED_BOX".

Units: MediaPipe hand landmarks are normalized (0-1 of frame size);
DetectedObject bboxes from color_detector.py/object_detector.py are
already pixel-space. Both are needed in the same units to compare, which
is why this reads PerceptionFrame.frame_width/frame_height (see base.py)
rather than assuming a fixed resolution.

Hand position is approximated as the average of wrist + index_mcp +
pinky_mcp (hands.py's own "palm-center" landmark set) — fingertips are too
noisy for distance-based reasoning.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, InteractionState, PerceptionFrame

# Per-frame box displacement below this (pixels) is treated as detection/
# hand jitter, not the box actually moving with the hand.
_DISPLACEMENT_NOISE_FLOOR_PX = 6.0

# How many recent (hand, box) position samples to correlate when deciding
# whether a held object is actively moving.
_HISTORY_LEN = 8

_PALM_LANDMARKS = {"wrist", "index_mcp", "pinky_mcp"}

# hand_object_distance_px/hand_object_touch_distance_px are absolute pixel
# values, calibrated against this reference frame width (matches the
# hand-constructed test fixtures in tests/test_interaction.py, so those
# keep behaving identically at scale=1.0). A real webcam frequently
# delivers a different actual resolution than config.yaml's requested
# camera.frame_width — observed capturing 1920x1080 despite a 1280x720
# request — and an absolute threshold tuned for one width is wildly wrong
# at another (a 90px "touch" radius is generous at 640px wide, effectively
# invisible at 1920px wide). Scaling by actual frame_width/this reference
# keeps the *proportional* reach the config values were meant to express.
_THRESHOLD_REFERENCE_FRAME_WIDTH = 640

# How many *consecutive* frames with zero hands detected anywhere in frame
# a HELD/MOVING object tolerates before actually releasing. Real-world
# testing showed MediaPipe's dedicated Hands model dropping detection
# entirely for a majority of individual frames even while a hand
# genuinely, continuously gripped an object for 10+ seconds — not just
# rare isolated flicker. Releasing on the very first such frame (the
# original fix here) meant almost every real hold got prematurely cut
# short before the object had even moved, immediately re-requiring a
# fresh 2-consecutive-frame touch to resume — which, given that same drop
# rate, routinely never completed either. This tolerates a real gap
# (roughly half a second at a typical 15-20fps loop) before concluding the
# hand has actually left, not just that one frame missed it.
_HAND_ABSENCE_GRACE_FRAMES = 8


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def hand_center_px(hand: HandFrame, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    pts = [lm for lm in hand.landmarks if lm.name in _PALM_LANDMARKS]
    if len(pts) < len(_PALM_LANDMARKS):
        return None
    x = sum(p.x for p in pts) / len(pts) * frame_w
    y = sum(p.y for p in pts) / len(pts) * frame_h
    return (x, y)



def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


@dataclass
class InteractionEvent:
    """Per-object result of one InteractionReasoner.update() call."""

    object_class: str
    state: InteractionState
    changed: bool  # True only on the frame the state actually transitioned
    # v2.0: which settings.target_zones key the object is sitting in, set
    # whenever state is OBJECT_PLACED (and persists while it stays resting
    # there — see _update_track's "currently_resting" branch). None
    # otherwise, or if it's outside every configured zone.
    zone: Optional[str] = None


@dataclass
class _ObjectTrack:
    state: InteractionState = InteractionState.NONE
    hand_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    box_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    zone: Optional[str] = None
    # Consecutive frames (this object HELD/MOVING) with zero hands detected
    # anywhere in frame. See _HAND_ABSENCE_GRACE_FRAMES.
    no_hand_streak: int = 0


def tracked_classes_for(settings: PerceptionSettings) -> list[str]:
    if settings.detector_backend == "yolo":
        from backend.perception.object_detector import normalize_class_name

        return [normalize_class_name(c) for c in settings.yolo_classes]
    return list(settings.colors.keys())


# Fraction of the object's own bbox area that must overlap a zone's bbox
# for the object to count as "placed there". A single point test (center
# or base) is fragile against real-camera perspective: an object's actual
# footprint on the table routinely doesn't line up pixel-perfectly with a
# zone rectangle calibrated ahead of time, especially near a zone's edge
# or with the camera at an angle. Overlap area matches the physical
# intuition ("is the object sitting over the marked zone") far better.
_ZONE_OVERLAP_RATIO = 0.2


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def zone_overlapping(obj_bbox: BBox, zones: list[DetectedObject]) -> Optional[str]:
    """Which settings.target_zones key obj_bbox substantially overlaps
    (>= _ZONE_OVERLAP_RATIO of its own area), or None. Public — also used
    by action_recognizer.py's pick-validity check, for the same reason."""
    obj_area = _bbox_area(obj_bbox)
    if obj_area <= 0:
        return None
    best_zone: Optional[str] = None
    best_ratio = _ZONE_OVERLAP_RATIO
    for zone in zones:
        ratio = _bbox_intersection_area(obj_bbox, zone.bbox) / obj_area
        if ratio > best_ratio:
            best_ratio = ratio
            best_zone = zone.cls
    return best_zone


class InteractionReasoner:
    """Maintains per-object InteractionState (and, v2.0, current zone)
    across frames.

    Tracked object classes come from whichever detector backend is
    configured (settings.colors for HSV, settings.yolo_classes for YOLO)
    rather than hardcoded, so a different experiment's object set doesn't
    require touching this code.
    """

    def __init__(self, settings: PerceptionSettings):
        self.settings = settings
        self._tracked_classes = tracked_classes_for(settings)
        self._tracks: dict[str, _ObjectTrack] = {cls: _ObjectTrack() for cls in self._tracked_classes}

    def reset(self) -> None:
        self._tracks = {cls: _ObjectTrack() for cls in self._tracked_classes}

    def update(self, frame: PerceptionFrame) -> dict[str, InteractionEvent]:
        if frame.frame_width <= 0 or frame.frame_height <= 0:
            # No known pixel space this frame (e.g. a producer that doesn't
            # populate frame_width/height) — nothing safe to compute.
            return {
                cls: InteractionEvent(cls, track.state, changed=False, zone=track.zone)
                for cls, track in self._tracks.items()
            }

        objects_by_class = {obj.cls: obj for obj in frame.objects}
        zones = [objects_by_class[name] for name in self.settings.target_zones if name in objects_by_class]
        hand_centers = [
            c
            for h in frame.hands
            if (c := hand_center_px(h, frame.frame_width, frame.frame_height)) is not None
        ]

        scale = frame.frame_width / _THRESHOLD_REFERENCE_FRAME_WIDTH
        touch_threshold = self.settings.hand_object_touch_distance_px * scale
        approach_threshold = self.settings.hand_object_distance_px * scale

        events: dict[str, InteractionEvent] = {}
        for cls in self._tracked_classes:
            track = self._tracks[cls]
            new_state, new_zone = self._update_track(
                track, objects_by_class.get(cls), hand_centers, zones, touch_threshold, approach_threshold
            )
            changed = new_state != track.state
            track.state = new_state
            track.zone = new_zone
            events[cls] = InteractionEvent(object_class=cls, state=new_state, changed=changed, zone=new_zone)
        return events

    # -- internals -----------------------------------------------------------

    def _update_track(
        self,
        track: _ObjectTrack,
        obj: DetectedObject | None,
        hand_centers: list[tuple[float, float]],
        zones: list[DetectedObject],
        touch_threshold: float,
        approach_threshold: float,
    ) -> tuple[InteractionState, Optional[str]]:
        # MediaPipe's hand detector flickers frame-to-frame in practice (a
        # real capture routinely shows several consecutive "no hand"
        # frames interleaved with good detections, even while a hand is
        # genuinely resting on/near the object) — so a single dropped
        # detection must not discard in-progress engagement (touching or
        # approaching), or a pick can never accumulate the two consecutive
        # touching frames it needs to promote to HELD.
        _GAP_TOLERANT_STATES = (
            InteractionState.OBJECT_BEING_HELD,
            InteractionState.OBJECT_MOVING_WITH_HAND,
            InteractionState.HAND_TOUCHING_OBJECT,
            InteractionState.HAND_APPROACHING_OBJECT,
        )

        if obj is None:
            # The object itself wasn't detected this frame — most often a
            # hand closing around it occludes its own color/shape signature.
            # Don't spontaneously release just because of a missing
            # detection; hold the state and wait for the next real
            # observation of the object.
            if track.state in _GAP_TOLERANT_STATES:
                return track.state, None
            if track.state in (InteractionState.OBJECT_RELEASED, InteractionState.OBJECT_PLACED):
                return track.state, track.zone
            return InteractionState.NONE, None

        box_center = bbox_center(obj.bbox)

        if not hand_centers:
            # The object IS clearly visible here — unlike the occlusion case
            # above — but no hand exists anywhere in frame this frame. Only
            # an already-HELD/MOVING object should actually release (the
            # hand genuinely moved away, e.g. out of the camera's view after
            # placing it) — a merely touching/approaching object just keeps
            # its state and waits for the hand detector to catch up, same
            # flicker tolerance as the obj-is-None branch above.
            if track.state in (InteractionState.OBJECT_BEING_HELD, InteractionState.OBJECT_MOVING_WITH_HAND):
                track.no_hand_streak += 1
                if track.no_hand_streak <= _HAND_ABSENCE_GRACE_FRAMES:
                    return track.state, None
                zone_name = zone_overlapping(obj.bbox, zones)
                if zone_name is not None:
                    return InteractionState.OBJECT_PLACED, zone_name
                return InteractionState.OBJECT_RELEASED, None
            if track.state in (InteractionState.HAND_TOUCHING_OBJECT, InteractionState.HAND_APPROACHING_OBJECT):
                return track.state, None
            if track.state in (InteractionState.OBJECT_RELEASED, InteractionState.OBJECT_PLACED):
                return self._resting_zone_state(track.state, track.zone, obj.bbox, zones)
            return InteractionState.NONE, None

        track.no_hand_streak = 0
        nearest_hand = min(hand_centers, key=lambda h: _dist(h, box_center))
        distance = _dist(nearest_hand, box_center)

        track.hand_history.append(nearest_hand)
        track.box_history.append(box_center)

        currently_held = track.state in (
            InteractionState.OBJECT_BEING_HELD,
            InteractionState.OBJECT_MOVING_WITH_HAND,
        )
        currently_resting = track.state in (
            InteractionState.OBJECT_RELEASED,
            InteractionState.OBJECT_PLACED,
        )

        if currently_held:
            if distance <= touch_threshold:
                state = InteractionState.OBJECT_MOVING_WITH_HAND if self._is_moving(track) else InteractionState.OBJECT_BEING_HELD
                return state, None
            # Hand pulled away while holding -> released, either into a
            # configured target zone (placed) or elsewhere (just
            # dropped/released).
            zone_name = zone_overlapping(obj.bbox, zones)
            if zone_name is not None:
                return InteractionState.OBJECT_PLACED, zone_name
            return InteractionState.OBJECT_RELEASED, None

        if currently_resting:
            # A placed/released object doesn't spontaneously go back to
            # NONE just because the hand that let go of it moved away — it
            # physically stays where it was left. Only a hand actually
            # touching it again restarts the touch/hold cycle; this also
            # gives Phase 4's temporal smoother something that holds stable
            # across frames to debounce, instead of a one-frame edge event.
            #
            # Reactivation deliberately uses a tighter test than the fresh-
            # pick path below (hand point literally inside the object's
            # bbox, not just within the generous touch_threshold radius):
            # objects placed close together on a real table mean a hand
            # reaching for a *different* object routinely passes within
            # touch_threshold of one that's already resting/placed, which
            # would otherwise re-trigger it as picked up again every time.
            if point_in_bbox(nearest_hand, obj.bbox):
                return InteractionState.HAND_TOUCHING_OBJECT, None
            return self._resting_zone_state(track.state, track.zone, obj.bbox, zones)

        if distance <= touch_threshold:
            # Require a HAND_TOUCHING_OBJECT frame before counting the
            # object as held, so a hand merely brushing past doesn't
            # register as a pick.
            if track.state == InteractionState.HAND_TOUCHING_OBJECT:
                return InteractionState.OBJECT_BEING_HELD, None
            return InteractionState.HAND_TOUCHING_OBJECT, None
        if distance <= approach_threshold:
            return InteractionState.HAND_APPROACHING_OBJECT, None
        return InteractionState.NONE, None

    def _is_moving(self, track: _ObjectTrack) -> bool:
        if len(track.box_history) < 2:
            return False
        return _dist(track.box_history[0], track.box_history[-1]) > _DISPLACEMENT_NOISE_FLOOR_PX

    @staticmethod
    def _resting_zone_state(
        current_state: InteractionState, current_zone: Optional[str], obj_bbox: BBox, zones: list[DetectedObject]
    ) -> tuple[InteractionState, Optional[str]]:
        """PLACED/RELEASED + zone for a resting object. RELEASED is
        re-derived from the object's *current* bbox every frame rather than
        frozen at the moment it was released — a release caught a frame
        early (object still settling, or the hand's final position not
        quite overlapping the zone yet) would otherwise permanently
        misclassify it as RELEASED even once it's sitting squarely inside
        the zone in every subsequent frame.
        PLACED, once reached, is sticky and never auto-reverts, though: a
        bbox estimate hovering right at a zone's overlap-ratio threshold
        would otherwise flicker PLACED/RELEASED frame to frame, and the
        temporal smoother (needs several *consecutive* stable frames) would
        never accumulate enough of the same candidate to ever confirm the
        PLACE action — leaving a genuinely, visibly placed object stuck on
        "observing..." forever. Only a fresh touch (which exits this
        resting state entirely) resets that."""
        if current_state == InteractionState.OBJECT_PLACED:
            return InteractionState.OBJECT_PLACED, current_zone
        zone_name = zone_overlapping(obj_bbox, zones)
        return (InteractionState.OBJECT_PLACED, zone_name) if zone_name else (InteractionState.OBJECT_RELEASED, None)
