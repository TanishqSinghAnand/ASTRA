"""Object-motion interaction reasoning (Phase 3; redesigned to
motion-based tracking after real-camera testing).

Still perception, not experiment logic: this module only answers "is this
object currently being carried, or resting inside the experiment area?"
for each tracked box. It knows nothing about PICK/PLACE experiment steps
or the sequence — Phase 4 (backend/temporal/) is what turns an
InteractionState transition into an action_key like "PICK_RED_BOX".

Design note — why motion, not hand proximity: the original design (see
git history) inferred PICK/PLACE from hand-to-object distance, using
MediaPipe's dedicated Hands model to locate the hand each frame. Real-
camera testing showed that model dropping detection entirely on a
majority of individual frames even during a continuous, genuine grip —
not rare flicker — which no amount of threshold tuning or gap-tolerance
patching made reliably solvable, because the signal it depended on simply
wasn't there most frames. The color detector, by contrast, was rock solid
throughout that same testing. So: track the *object's own position*
instead. A PICK is the object transitioning from resting to moving; a
PLACE is it coming back to rest inside the experiment area. Hand
landmarks are computed only for the on-screen overlay now (see
backend/services/overlay.py) — nothing here reads them.

Units: DetectedObject bboxes from color_detector.py are pixel-space;
frame.frame_width/frame_height (see base.py) is used only to scale the
movement threshold proportionally to actual capture resolution (a real
webcam frequently delivers a different resolution than config.yaml
requested).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, InteractionState, PerceptionFrame

# How many of the most recent box-center samples must all sit within
# move_threshold of each other to confirm the object has come to rest.
# Deliberately short: a long window would keep comparing against a stale
# sample from mid-motion for many frames after the object actually
# stopped, delaying "placed" detection long after it visibly settled.
_STILL_WINDOW = 3

# Reference frame width settings.movement_threshold_px is calibrated
# against (matches tests/test_interaction.py's fixtures at scale=1.0). A
# real webcam frequently delivers a different actual resolution than
# config.yaml's requested camera.frame_width, so this scales the
# threshold to whatever the camera's *actual* capture width turns out to
# be — see interaction.py's git history for the same reasoning applied to
# the old hand-distance thresholds.
_THRESHOLD_REFERENCE_FRAME_WIDTH = 640

# Fraction of the object's own bbox area that must overlap the experiment
# area's bbox for the object to count as "placed there". A single point
# test (center or base) is fragile against real-camera perspective: an
# object's actual footprint on the table routinely doesn't line up
# pixel-perfectly with an area rectangle calibrated ahead of time,
# especially near its edge or with the camera at an angle. Overlap area
# matches the physical intuition ("is the object sitting over the marked
# area") far better.
_AREA_OVERLAP_RATIO = 0.2

_PALM_LANDMARKS = {"wrist", "index_mcp", "pinky_mcp"}


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def hand_center_px(hand: HandFrame, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    """Kept for the on-screen overlay/debug HUD only (tools/live_preview.py,
    backend/services/overlay.py) — no longer read by any interaction
    logic in this module. See the module docstring for why."""
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


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def area_overlapping(obj_bbox: BBox, area_bbox: Optional[BBox]) -> bool:
    """Does obj_bbox substantially overlap area_bbox (>= _AREA_OVERLAP_RATIO
    of the object's own area)?"""
    if area_bbox is None:
        return False
    obj_area = _bbox_area(obj_bbox)
    if obj_area <= 0:
        return False
    return (_bbox_intersection_area(obj_bbox, area_bbox) / obj_area) >= _AREA_OVERLAP_RATIO


@dataclass
class InteractionEvent:
    """Per-object result of one InteractionReasoner.update() call."""

    object_class: str
    state: InteractionState
    changed: bool  # True only on the frame the state actually transitioned


@dataclass
class _ObjectTrack:
    state: InteractionState = InteractionState.NONE
    # Last _STILL_WINDOW positions — used only to detect settling back to
    # rest while currently held (short on purpose, see _STILL_WINDOW).
    recent: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=_STILL_WINDOW))
    # The object's last confirmed-resting position — the reference point a
    # fresh pick's displacement is measured against. Unlike `recent`, this
    # doesn't age out of a sliding window, so a slow, deliberate lift over
    # many small-per-frame steps still accumulates a real displacement
    # from it instead of only ever being compared to a few frames ago.
    anchor: Optional[tuple[float, float]] = None


class InteractionReasoner:
    """Maintains per-object InteractionState across frames, purely from
    each tracked object's own position history.

    Tracked object classes are read from settings.colors (i.e. whatever
    color_detector.py is configured to detect — "red_box"/"blue_box" for the
    sample experiment) rather than hardcoded, so a different experiment's
    object set doesn't require touching this code.
    """

    def __init__(self, settings: PerceptionSettings):
        self.settings = settings
        self._tracked_classes = list(settings.colors.keys())
        self._tracks: dict[str, _ObjectTrack] = {cls: _ObjectTrack() for cls in self._tracked_classes}

    def reset(self) -> None:
        self._tracks = {cls: _ObjectTrack() for cls in self._tracked_classes}

    def update(self, frame: PerceptionFrame) -> dict[str, InteractionEvent]:
        if frame.frame_width <= 0 or frame.frame_height <= 0:
            # No known pixel space this frame (e.g. a producer that doesn't
            # populate frame_width/height) — nothing safe to compute.
            return {
                cls: InteractionEvent(cls, track.state, changed=False)
                for cls, track in self._tracks.items()
            }

        objects_by_class = {obj.cls: obj for obj in frame.objects}
        experiment_area = objects_by_class.get("experiment_area")
        move_threshold = self.settings.movement_threshold_px * (frame.frame_width / _THRESHOLD_REFERENCE_FRAME_WIDTH)

        events: dict[str, InteractionEvent] = {}
        for cls in self._tracked_classes:
            track = self._tracks[cls]
            new_state = self._update_track(track, objects_by_class.get(cls), experiment_area, move_threshold)
            changed = new_state != track.state
            track.state = new_state
            events[cls] = InteractionEvent(object_class=cls, state=new_state, changed=changed)
        return events

    # -- internals -----------------------------------------------------------

    def _update_track(
        self,
        track: _ObjectTrack,
        obj: DetectedObject | None,
        experiment_area: DetectedObject | None,
        move_threshold: float,
    ) -> InteractionState:
        if obj is None:
            # Not detected this frame — most often a hand briefly occluding
            # it mid-carry, or a momentary detector miss. Hold the last
            # known state rather than resetting; the object hasn't been
            # observed to have actually changed.
            return track.state

        box_center = bbox_center(obj.bbox)
        track.recent.append(box_center)
        if track.anchor is None:
            track.anchor = box_center

        currently_held = track.state == InteractionState.OBJECT_BEING_HELD
        currently_resting = track.state in (InteractionState.OBJECT_RELEASED, InteractionState.OBJECT_PLACED)

        if currently_held:
            if not self._has_settled(track, move_threshold):
                return InteractionState.OBJECT_BEING_HELD
            # Just came to rest — this hold's natural end: settled either
            # into the experiment area (placed) or elsewhere (just
            # released/dropped). This position becomes the new anchor a
            # future pick's displacement will be measured against.
            track.anchor = box_center
            if area_overlapping(obj.bbox, experiment_area.bbox if experiment_area else None):
                return InteractionState.OBJECT_PLACED
            return InteractionState.OBJECT_RELEASED

        if _dist(box_center, track.anchor) > move_threshold:
            return InteractionState.OBJECT_BEING_HELD

        if currently_resting:
            return self._resting_state(track.state, obj.bbox, experiment_area)

        return InteractionState.NONE

    @staticmethod
    def _has_settled(track: _ObjectTrack, move_threshold: float) -> bool:
        """True once the last _STILL_WINDOW observed positions are all
        close together — i.e. the object has stopped moving, regardless of
        how it was moving before that window."""
        if len(track.recent) < _STILL_WINDOW:
            return False
        newest = track.recent[-1]
        return all(_dist(p, newest) <= move_threshold for p in track.recent)

    @staticmethod
    def _resting_state(
        current_state: InteractionState, obj_bbox: BBox, experiment_area: DetectedObject | None
    ) -> InteractionState:
        """PLACED/RELEASED for a resting object. RELEASED is re-derived
        from the object's *current* bbox every frame rather than frozen at
        the moment it stopped moving — settling caught a frame early
        (bbox estimate still stabilizing) would otherwise permanently
        misclassify it as RELEASED even once it's sitting squarely inside
        the area in every subsequent frame.
        PLACED, once reached, is sticky and never auto-reverts, though: a
        bbox estimate hovering right at the area's overlap-ratio threshold
        would otherwise flicker PLACED/RELEASED frame to frame, and the
        temporal smoother (needs several *consecutive* stable frames)
        would never accumulate enough of the same candidate to ever
        confirm the PLACE action. Only fresh movement (which exits this
        resting state entirely) resets that."""
        if current_state == InteractionState.OBJECT_PLACED:
            return InteractionState.OBJECT_PLACED
        if area_overlapping(obj_bbox, experiment_area.bbox if experiment_area else None):
            return InteractionState.OBJECT_PLACED
        return InteractionState.OBJECT_RELEASED
