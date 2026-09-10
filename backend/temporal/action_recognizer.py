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
  - While an object is OBJECT_BEING_HELD (interaction.py: a hand dwelled
    near it continuously for pick_dwell_seconds), the candidate is
    "PICK_<CLASS>" (e.g. PICK_CELL_PHONE) for as long as the hold lasts.
    No separate "was it picked up from outside a zone" validity check —
    the dwell timer requiring sustained *contact* is itself already
    enough evidence of a genuine pick regardless of where the object
    happened to be resting beforehand (which, for a real tabletop, is not
    always "neatly outside every target zone" — two boxes sharing a small
    table routinely means one's neutral resting spot overlaps the other's
    target zone). This mirrors the class naming
    color_detector.py/object_detector.py already use ("cell_phone")
    upper-cased to match ExperimentStep.action_key's convention
    ("CELL_PHONE") — no explicit per-object mapping table needed.
  - While an object sits OBJECT_PLACED (its own position dwelled inside a
    zone continuously for place_dwell_seconds, persisting until a fresh
    pick, per interaction.py), the candidate is "PLACE_<CLASS>", and
    (v2.0) the emitted ActionPrediction.location names which configured
    zone it landed in — see zones.py and validator.py's WRONG_LOCATION
    handling.

Multi-object candidate selection (_active_class): with more than one
tracked class, a naive "first class with a non-None action wins this
frame" picks whichever class happens to iterate first — and OBJECT_PLACED
persists *indefinitely* once reached (interaction.py's sticky-placed
design), so an object placed during an earlier step would otherwise win
that slot on literally every subsequent frame, permanently starving any
other class's own PICK/PLACE from ever reaching the smoother (it never
even gets offered as a candidate, so stability_frames of consecutive
identical candidates can never accumulate for it — observed directly:
holding a second object well past its dwell threshold never registered
anything, because a first object placed earlier kept winning first).
_active_class only ever changes on a fresh transition (event.changed)
into HELD or PLACED, then stays pinned to that class — reporting its
*current* state every frame, changed or not — until some other class has
a fresh transition of its own. A real transition always immediately
outranks however long another class has been quietly resting.

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
own certainty — fixed high, since clearing a dwell-time threshold is a
deterministic timing signal, not a probabilistic one — with the
detector's own confidence for that object, averaged over a small rolling
window (which genuinely varies with lighting/blob quality and is the more
informative half of the blend).
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from backend.config.settings import PerceptionSettings, TemporalSettings
from backend.perception.base import ActionPrediction, InteractionState, PerceptionFrame
from backend.perception.interaction import InteractionReasoner, tracked_classes_for
from backend.perception.interfaces import ActionRecognizer
from backend.temporal.temporal_smoother import TemporalSmoother

# Fixed heuristic weight for the interaction reasoner's own certainty in the
# blended confidence score — see module docstring.
INTERACTION_CERTAINTY = 0.90
_CONFIDENCE_WINDOW_LEN = 10
_CANDIDATE_STATES = (InteractionState.OBJECT_BEING_HELD, InteractionState.OBJECT_PLACED)


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
        # Per-class InteractionEvent from the most recent update() call —
        # introspection for callers that want to show *why* (tools/
        # live_demo.py's debug HUD), not read by any recognition logic
        # here.
        self.last_events: dict[str, object] = {}
        # Which class currently "owns" the candidate slot — see
        # _active_class in the module docstring. Also exposed for
        # introspection (inference_service.py's dwell-progress broadcast
        # shows whichever object is actually driving the candidate, not
        # just the first one alphabetically/config-order that happens to
        # have some dwell timer running).
        self.active_class: Optional[str] = None

    def reset(self) -> None:
        self.reasoner.reset()
        self.smoother.reset()
        for window in self._confidence_windows.values():
            window.clear()
        self.last_events = {}
        self.active_class = None

    def update(self, perception_frame: PerceptionFrame) -> ActionPrediction:
        events = self.reasoner.update(perception_frame)
        self.last_events = events
        objects_by_class = {obj.cls: obj for obj in perception_frame.objects}

        for cls, event in events.items():
            obj = objects_by_class.get(cls)
            if obj is not None:
                self._confidence_windows[cls].append(obj.confidence)
            if event.changed and event.state in _CANDIDATE_STATES:
                self.active_class = cls

        active_event = events.get(self.active_class) if self.active_class else None
        candidate_key: Optional[str] = None
        candidate_conf = 0.0
        candidate_location: Optional[str] = None
        if active_event is not None and active_event.state in _CANDIDATE_STATES:
            prefix = "PICK" if active_event.state == InteractionState.OBJECT_BEING_HELD else "PLACE"
            candidate_key = f"{prefix}_{self.active_class.upper()}"
            candidate_conf = self._combined_confidence(self.active_class)
            candidate_location = active_event.zone
        else:
            self.active_class = None

        return self.smoother.update(candidate_key, candidate_conf, location=candidate_location)

    # -- internals -----------------------------------------------------------

    def _combined_confidence(self, cls: str) -> float:
        window = self._confidence_windows.get(cls)
        hsv_confidence = (sum(window) / len(window)) if window else 0.0
        return round(min(0.99, 0.5 * INTERACTION_CERTAINTY + 0.5 * hsv_confidence), 3)
