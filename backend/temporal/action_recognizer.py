"""Rule-based temporal action recognizer (Phase 4).

Implements the ActionRecognizer interface (interfaces.py) by running each
PerceptionFrame through Phase 3's InteractionReasoner and mapping the
resulting per-object interaction state into a discrete action candidate,
then debouncing that candidate through TemporalSmoother before it's ever
returned. This is rule-based geometric/temporal reasoning, not a trained
classifier — same "no labeled dataset exists yet" rationale as
color_detector.py's HSV thresholding (see training/ for the swap-in point
once one exists).

Action mapping (only PICK/PLACE — this experiment has no other gesture):
  - While an object is held (OBJECT_BEING_HELD or OBJECT_MOVING_WITH_HAND)
    AND it was picked up from outside every configured target zone, the
    candidate is "PICK_<CLASS>" (e.g. PICK_CELL_PHONE) for as long as the
    hold lasts. The "outside every zone" check is latched at the moment
    the hold begins (not re-checked every frame) so carrying the object
    across a zone's edge mid-motion can't flip the candidate. This mirrors
    the class naming color_detector.py/object_detector.py already use
    ("cell_phone") upper-cased to match ExperimentStep.action_key's
    convention ("CELL_PHONE") — no explicit per-object mapping table
    needed.
  - While an object sits OBJECT_PLACED (persists until re-touched, per
    interaction.py), the candidate is "PLACE_<CLASS>", and (v2.0) the
    emitted ActionPrediction.location names which configured zone it
    landed in — see zones.py and validator.py's WRONG_LOCATION handling.
  - OBJECT_RELEASED (dropped outside every zone) maps to no action — this
    sample experiment has no "drop"/failure gesture, only PICK/PLACE.

COMPLETE_EXPERIMENT (design decision, per spec): there is no physical
gesture for "close out the experiment" — step 5 in bas_sample_001.json has
no PICK/PLACE behind it. This recognizer deliberately never emits
"COMPLETE_EXPERIMENT". The sequence engine (Phase 5/6 completion) is
expected to auto-advance through a trailing COMPLETE-typed step once the
prior step is confirmed CORRECT, rather than waiting for a visually
detected action that can't exist. Documented here since it constrains what
Phase 5/6 must do with validator.py.

Confidence is a documented heuristic, not a calibrated probability (same
honesty rule as color_detector.py): it blends the interaction reasoner's
own certainty — fixed high, since crossing a hard distance/displacement
threshold is a deterministic geometric signal, not a probabilistic one —
with the HSV detector's confidence for that object, averaged over a small
rolling window (which genuinely varies with lighting/blob quality and is
the more informative half of the blend).
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from backend.config.settings import PerceptionSettings, TemporalSettings
from backend.perception.base import ActionPrediction, InteractionState, PerceptionFrame
from backend.perception.interaction import InteractionReasoner, tracked_classes_for, bbox_center, point_in_bbox
from backend.perception.interfaces import ActionRecognizer
from backend.temporal.temporal_smoother import TemporalSmoother

# Fixed heuristic weight for the interaction reasoner's own certainty in the
# blended confidence score — see module docstring.
INTERACTION_CERTAINTY = 0.90
_CONFIDENCE_WINDOW_LEN = 10

_HELD_STATES = (InteractionState.OBJECT_BEING_HELD, InteractionState.OBJECT_MOVING_WITH_HAND)


class RuleBasedActionRecognizer(ActionRecognizer):
    def __init__(
        self,
        perception_settings: PerceptionSettings,
        temporal_settings: TemporalSettings,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.reasoner = InteractionReasoner(perception_settings)
        self.smoother = TemporalSmoother(
            stability_frames=temporal_settings.action_stability_frames,
            min_confidence_duration_ms=temporal_settings.min_confidence_duration_ms,
            **({"clock": clock} if clock is not None else {}),
        )
        self.perception_settings = perception_settings
        self._tracked_classes = tracked_classes_for(perception_settings)
        self._confidence_windows: dict[str, deque[float]] = {
            cls: deque(maxlen=_CONFIDENCE_WINDOW_LEN) for cls in self._tracked_classes
        }
        # Latched per-class: was the hold currently in progress picked up
        # from outside the experiment area? Decided once, at the frame the
        # hold begins — see module docstring.
        self._pick_valid: dict[str, bool] = {}

    def reset(self) -> None:
        self.reasoner.reset()
        self.smoother.reset()
        for window in self._confidence_windows.values():
            window.clear()
        self._pick_valid.clear()

    def update(self, perception_frame: PerceptionFrame) -> ActionPrediction:
        events = self.reasoner.update(perception_frame)
        objects_by_class = {obj.cls: obj for obj in perception_frame.objects}
        zones = [objects_by_class[name] for name in self.perception_settings.target_zones if name in objects_by_class]

        candidate_key: Optional[str] = None
        candidate_conf = 0.0
        candidate_location: Optional[str] = None

        for cls, event in events.items():
            obj = objects_by_class.get(cls)
            if obj is not None:
                self._confidence_windows[cls].append(obj.confidence)

            if event.changed and event.state in _HELD_STATES:
                self._pick_valid[cls] = self._picked_from_outside_zones(obj, zones)

            action_key: Optional[str] = None
            if event.state in _HELD_STATES and self._pick_valid.get(cls, True):
                action_key = f"PICK_{cls.upper()}"
            elif event.state == InteractionState.OBJECT_PLACED:
                action_key = f"PLACE_{cls.upper()}"

            if action_key is not None and candidate_key is None:
                candidate_key = action_key
                candidate_conf = self._combined_confidence(cls)
                candidate_location = event.zone

        return self.smoother.update(candidate_key, candidate_conf, location=candidate_location)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _picked_from_outside_zones(obj, zones: list) -> bool:
        if obj is None or not zones:
            return True
        center = bbox_center(obj.bbox)
        return not any(point_in_bbox(center, zone.bbox) for zone in zones)

    def _combined_confidence(self, cls: str) -> float:
        window = self._confidence_windows.get(cls)
        hsv_confidence = (sum(window) / len(window)) if window else 0.0
        return round(min(0.99, 0.5 * INTERACTION_CERTAINTY + 0.5 * hsv_confidence), 3)
