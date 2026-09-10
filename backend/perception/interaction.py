"""Dwell-time interaction reasoning (Phase 3; multi-zone placement v2.0;
redesigned twice now after real-camera testing — see below).

Still perception, not experiment logic: this module only answers "has
this object been picked up, and if so, has it been sitting in a
configured zone long enough to count as placed?" for each tracked box. It
knows nothing about PICK/PLACE experiment steps or the sequence —
Phase 4 (backend/temporal/) is what turns an InteractionState transition
into an action_key like "PICK_RED_BOX".

Design history:
  1. Hand-to-object distance, frame-debounced (2 consecutive touching
     frames). Failed in real testing: MediaPipe's Hands model dropped
     detection on a majority of individual frames even during a
     continuous, genuine grip, so the 2-frame debounce routinely never
     completed.
  2. Object motion instead of hand proximity — track the object's own
     position, PICK = starts moving, PLACE = comes back to rest in a
     zone. Removed the Hands dependency entirely. Worked better, but
     real testing (now across machines — Windows and macOS both) still
     showed object detection itself dropping frames/jittering enough
     that motion-vs-jitter was hard to call reliably in every lighting
     setup.
  3. This version — dwell time, wall-clock seconds not frame counts (so
     behavior doesn't depend on a given machine's actual frame rate):
     PICK requires a hand to stay within hand_touch_distance_px of the
     object *continuously* for pick_dwell_seconds; PLACE requires the
     object's own position to stay inside a configured zone
     *continuously* for place_dwell_seconds, independent of whether a
     hand is anywhere nearby during that window. Sustained dwell is a
     much more forgiving signal than either single-frame proximity or
     frame-to-frame motion: a brief detection gap doesn't reset it (see
     _update_track's gap-tolerance), only a clear break (hand genuinely
     elsewhere, object genuinely leaves the zone) does.

Units: DetectedObject bboxes from color_detector.py/object_detector.py
are pixel-space; MediaPipe hand landmarks are normalized (0-1 of frame
size) — both are needed in the same units to compare, which is why this
reads PerceptionFrame.frame_width/frame_height (see base.py). Dwell
timers use PerceptionFrame.timestamp (wall-clock, set per-frame by
whatever produced the frame) rather than a separately injected clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, InteractionState, PerceptionFrame

# Reference frame width hand_touch_distance_px is calibrated against
# (matches tests/test_interaction.py's fixtures at scale=1.0). A real
# webcam frequently delivers a different actual resolution than
# config.yaml's requested camera.frame_width, so this scales the
# threshold to whatever the camera's *actual* capture width turns out to
# be — see interaction.py's git history for the same reasoning applied
# across every version of this module.
_THRESHOLD_REFERENCE_FRAME_WIDTH = 640

# Fraction of the object's own bbox area that must overlap a zone's bbox
# for the object to count as "in" it. A single point test (center or
# base) is fragile against real-camera perspective: an object's actual
# footprint on the table routinely doesn't line up pixel-perfectly with a
# zone rectangle calibrated ahead of time, especially near a zone's edge
# or with the camera at an angle. Overlap area matches the physical
# intuition ("is the object sitting over the marked zone") far better.
_ZONE_OVERLAP_RATIO = 0.2

_PALM_LANDMARKS = {"wrist", "index_mcp", "pinky_mcp"}


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
    # there). None otherwise, or if it's outside every configured zone.
    zone: Optional[str] = None
    # Progress of whichever dwell timer is currently running (elapsed
    # seconds, seconds required) — the hand-contact timer while watching
    # for a pick, or the zone-occupancy timer while held and inside a
    # zone. None, None when no timer is currently running. Introspection
    # only (tools/live_demo.py's debug HUD) — not read by any recognition
    # logic.
    dwell_elapsed: Optional[float] = None
    dwell_required: Optional[float] = None


@dataclass
class _ObjectTrack:
    state: InteractionState = InteractionState.NONE
    zone: Optional[str] = None
    # Wall-clock timestamp (PerceptionFrame.timestamp) a hand first came
    # within hand_touch_distance_px, continuously since — reset to None
    # the moment a frame clearly shows the hand elsewhere. None means "not
    # currently in contact".
    touch_started_at: Optional[float] = None
    # Wall-clock timestamp the object first entered `dwelling_zone`,
    # continuously since — reset to None the moment it's observed outside
    # every zone. None means "not currently dwelling in a zone".
    zone_entered_at: Optional[float] = None
    dwelling_zone: Optional[str] = None


def tracked_classes_for(settings: PerceptionSettings) -> list[str]:
    if settings.detector_backend == "yolo":
        from backend.perception.object_detector import normalize_class_name

        return [normalize_class_name(c) for c in settings.yolo_classes]
    return list(settings.colors.keys())


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
    (>= _ZONE_OVERLAP_RATIO of its own area), or None."""
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
    across frames, from dwell timers on hand-to-object distance (pick) and
    object-to-zone overlap (place).

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
        touch_threshold = self.settings.hand_touch_distance_px * scale

        events: dict[str, InteractionEvent] = {}
        for cls in self._tracked_classes:
            track = self._tracks[cls]
            new_state, new_zone = self._update_track(
                track, objects_by_class.get(cls), hand_centers, zones, touch_threshold, frame.timestamp
            )
            changed = new_state != track.state
            track.state = new_state
            track.zone = new_zone
            dwell_elapsed, dwell_required = self._dwell_progress(track, frame.timestamp)
            events[cls] = InteractionEvent(
                object_class=cls,
                state=new_state,
                changed=changed,
                zone=new_zone,
                dwell_elapsed=dwell_elapsed,
                dwell_required=dwell_required,
            )
        return events

    def _dwell_progress(self, track: _ObjectTrack, now: float) -> tuple[Optional[float], Optional[float]]:
        if track.state == InteractionState.OBJECT_BEING_HELD and track.zone_entered_at is not None:
            return now - track.zone_entered_at, self.settings.place_dwell_seconds
        if track.touch_started_at is not None:
            return now - track.touch_started_at, self.settings.pick_dwell_seconds
        return None, None

    # -- internals -----------------------------------------------------------

    def _update_track(
        self,
        track: _ObjectTrack,
        obj: DetectedObject | None,
        hand_centers: list[tuple[float, float]],
        zones: list[DetectedObject],
        touch_threshold: float,
        now: float,
    ) -> tuple[InteractionState, Optional[str]]:
        if track.state == InteractionState.OBJECT_BEING_HELD:
            return self._update_held(track, obj, zones, now)
        return self._update_watching_for_pick(track, obj, hand_centers, touch_threshold, now)

    def _update_watching_for_pick(
        self,
        track: _ObjectTrack,
        obj: DetectedObject | None,
        hand_centers: list[tuple[float, float]],
        touch_threshold: float,
        now: float,
    ) -> tuple[InteractionState, Optional[str]]:
        """State is NONE, OBJECT_RELEASED, or OBJECT_PLACED — watching for
        a hand to dwell near the object long enough to confirm a fresh
        pick. A resting (RELEASED/PLACED) object requires the hand to be
        *inside* its bbox to start the dwell timer, not just within the
        broader touch_threshold radius — objects placed close together on
        a real table mean a hand reaching for a *different* object
        routinely passes within touch_threshold of one that's already
        resting, which would otherwise restart its pick timer constantly.
        A never-touched object (state NONE) uses the more generous
        touch_threshold, since there's no "already resting nearby" object
        for a passing hand to be confused with yet.
        """
        resting = track.state in (InteractionState.OBJECT_RELEASED, InteractionState.OBJECT_PLACED)

        if obj is None or not hand_centers:
            # No object and/or no hand observed this frame — genuinely
            # ambiguous (detector miss vs. hand truly elsewhere), not
            # evidence contact broke. Preserve whatever dwell progress
            # exists rather than resetting it on every dropped frame.
            return track.state, track.zone

        box_center = bbox_center(obj.bbox)
        nearest_hand = min(hand_centers, key=lambda h: _dist(h, box_center))

        in_contact = point_in_bbox(nearest_hand, obj.bbox) if resting else _dist(nearest_hand, box_center) <= touch_threshold

        if not in_contact:
            track.touch_started_at = None
            return track.state, track.zone

        if track.touch_started_at is None:
            track.touch_started_at = now
        elif now - track.touch_started_at >= self.settings.pick_dwell_seconds:
            # Pick confirmed — start this hold with a clean zone-dwell
            # slate (any zone dwell tracked while it was previously
            # resting is irrelevant to this new pick).
            track.touch_started_at = None
            track.zone_entered_at = None
            track.dwelling_zone = None
            return InteractionState.OBJECT_BEING_HELD, None
        return track.state, track.zone

    def _update_held(
        self,
        track: _ObjectTrack,
        obj: DetectedObject | None,
        zones: list[DetectedObject],
        now: float,
    ) -> tuple[InteractionState, Optional[str]]:
        """State is OBJECT_BEING_HELD — watching for the object's own
        position to dwell inside a zone long enough to confirm a place.
        Hand position is irrelevant here by design (a hand lingering in
        or near the zone while setting the object down must not disrupt
        this); only where the *object* is matters."""
        if obj is None:
            # Not detected this frame (often a hand occluding it
            # mid-carry) — preserve dwell progress rather than resetting.
            return InteractionState.OBJECT_BEING_HELD, None

        zone_name = zone_overlapping(obj.bbox, zones)
        if zone_name is None:
            track.zone_entered_at = None
            track.dwelling_zone = None
            return InteractionState.OBJECT_BEING_HELD, None

        if track.dwelling_zone != zone_name:
            # Entered this zone (or switched from a different one) —
            # restart the dwell timer for it.
            track.dwelling_zone = zone_name
            track.zone_entered_at = now
            return InteractionState.OBJECT_BEING_HELD, None

        if now - track.zone_entered_at >= self.settings.place_dwell_seconds:
            placed_zone = track.dwelling_zone
            track.zone_entered_at = None
            track.dwelling_zone = None
            return InteractionState.OBJECT_PLACED, placed_zone
        return InteractionState.OBJECT_BEING_HELD, None
