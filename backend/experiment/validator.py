"""Sequence validation — deterministic procedural logic, not ML, per the
architecture principle in spec section 50: the procedure is known, so
correctness is decided by explicit rules against the experiment config, and
the action recognizer's job is only to say *what* it saw.

Implements the full decision set: CORRECT / WRONG_OBJECT / SKIPPED_STEP /
OUT_OF_SEQUENCE / REPEATED_STEP / RECOVERED / LOW_CONFIDENCE / COMPLETE
(Phase 5/6 completion — this used to merge the middle three into a single
catch-all OUT_OF_SEQUENCE; see below for how they're now told apart).

Mismatch classification (a detected action that isn't the expected one):
  - REPEATED_STEP: matches a step already completed (id < current step).
  - WRONG_OBJECT: matches a step not yet completed, whose action TYPE
    (PICK/PLACE) is the same as the currently expected step's — e.g.
    expected PICK_RED_BOX, got PICK_BLUE_BOX. Same gesture, wrong item:
    grabbed the wrong object for *this* step, not a jump to a later one.
  - SKIPPED_STEP: matches a step not yet completed, whose action TYPE
    differs from what's expected — e.g. expected PLACE_RED_BOX (still
    holding red), got PICK_BLUE_BOX. That action does belong to a real,
    later step, but it isn't a same-type substitution for the current one —
    the current step was skipped over entirely.
  - OUT_OF_SEQUENCE: true fallback — detected_key doesn't match any known
    step's action_key at all. Shouldn't normally happen (the recognizer
    only emits known action_keys), kept defensively.

COMPLETE_EXPERIMENT: per backend/temporal/action_recognizer.py's documented
design decision, the recognizer never emits this — there's no physical
gesture for "close out the experiment". So this engine auto-advances
through a trailing COMPLETE-typed step itself, the instant the prior real
step is confirmed, rather than waiting for an action that will never
arrive. Both the just-confirmed step's event and the auto-generated
COMPLETE event are appended to `history`, but submit_action can only
return one SequenceEvent per call (interfaces.py's SequenceEngine
contract), so the COMPLETE event is what's returned/live-broadcast; the
granular "prior step CORRECT" entry is still there in `history` for
logging/reporting (Phase 8).
"""
from __future__ import annotations

from backend.experiment.experiment_loader import ExperimentDefinition, ExperimentStep
from backend.perception.base import ActionPrediction, SequenceEvent, SequenceStatus
from backend.perception.interfaces import SequenceEngine


class RuleBasedSequenceEngine(SequenceEngine):
    """Validates a stream of ActionPrediction against an ExperimentDefinition."""

    def __init__(self, experiment: ExperimentDefinition, confidence_threshold: float = 0.60):
        self.experiment = experiment
        self.confidence_threshold = confidence_threshold
        self._step_index = 0  # pointer into experiment.steps (0-based)
        self._last_was_error = False
        self._finished = False
        self.history: list[SequenceEvent] = []

    # -- SequenceEngine interface -----------------------------------------

    def current_step(self) -> int:
        if self._finished:
            return self.experiment.total_steps()
        return self.experiment.steps[self._step_index].id

    def is_finished(self) -> bool:
        return self._finished

    def reset(self) -> None:
        self._step_index = 0
        self._last_was_error = False
        self._finished = False
        self.history.clear()

    def submit_action(self, prediction: ActionPrediction) -> SequenceEvent | None:
        if prediction.action is None:
            return None  # "observing" — nothing conclusive yet, no event
        if self._finished:
            return None

        expected_step: ExperimentStep = self.experiment.steps[self._step_index]
        expected_key = expected_step.action_key
        detected_key = prediction.action

        if prediction.confidence < self.confidence_threshold:
            event = SequenceEvent(
                step=expected_step.id,
                expected=expected_key,
                detected=detected_key,
                confidence=prediction.confidence,
                status=SequenceStatus.LOW_CONFIDENCE,
                explanation=[
                    f"Detected '{detected_key}' but confidence "
                    f"{prediction.confidence:.0%} is below the "
                    f"{self.confidence_threshold:.0%} threshold — observing "
                    f"for more evidence before deciding."
                ],
            )
            self.history.append(event)
            return event

        if detected_key == expected_key:
            return self._handle_correct(expected_step, prediction)

        return self._handle_mismatch(expected_step, prediction)

    # -- correct / auto-complete --------------------------------------------

    def _handle_correct(self, expected_step: ExperimentStep, prediction: ActionPrediction) -> SequenceEvent:
        was_recovery = self._last_was_error
        self._last_was_error = False
        self._step_index += 1
        # A directly-submitted COMPLETE-typed step (defensive: normal flow
        # always auto-advances into it below before it's ever "expected").
        just_finished = expected_step.action == "COMPLETE" or self._step_index >= self.experiment.total_steps()
        if just_finished:
            self._finished = True
        event = SequenceEvent(
            step=expected_step.id,
            expected=expected_step.action_key,
            detected=expected_step.action_key,
            confidence=prediction.confidence,
            status=SequenceStatus.RECOVERED if was_recovery else (
                SequenceStatus.COMPLETE if just_finished else SequenceStatus.CORRECT
            ),
            recovered=was_recovery or None,
            explanation=self._explain_success(expected_step, was_recovery),
        )
        self.history.append(event)

        if not self._finished and self._step_index < self.experiment.total_steps():
            next_step = self.experiment.steps[self._step_index]
            if next_step.action == "COMPLETE":
                return self._auto_complete(expected_step, next_step, prediction)

        return event

    def _auto_complete(
        self, prior_step: ExperimentStep, complete_step: ExperimentStep, prediction: ActionPrediction
    ) -> SequenceEvent:
        self._step_index += 1
        self._finished = True
        event = SequenceEvent(
            step=complete_step.id,
            expected=complete_step.action_key,
            detected=complete_step.action_key,
            confidence=prediction.confidence,
            status=SequenceStatus.COMPLETE,
            explanation=[
                f"'{prior_step.action_key}' confirmed — auto-advancing through "
                f"'{complete_step.action_key}' (no gesture expected for this step).",
                "Experiment complete.",
            ],
        )
        self.history.append(event)
        return event

    # -- mismatch classification ---------------------------------------------

    def _handle_mismatch(self, expected_step: ExperimentStep, prediction: ActionPrediction) -> SequenceEvent:
        detected_key = prediction.action
        matching_step = next(
            (s for s in self.experiment.steps if s.action_key == detected_key), None
        )

        if matching_step is None:
            status = SequenceStatus.OUT_OF_SEQUENCE
        elif matching_step.id < expected_step.id:
            status = SequenceStatus.REPEATED_STEP
        elif matching_step.action == expected_step.action:
            status = SequenceStatus.WRONG_OBJECT
        else:
            status = SequenceStatus.SKIPPED_STEP

        self._last_was_error = True
        event = SequenceEvent(
            step=expected_step.id,
            expected=expected_step.action_key,
            detected=detected_key,
            confidence=prediction.confidence,
            status=status,
            error_type=status.value,
            recovered=False,
            explanation=self._explain_error(expected_step, detected_key, status, matching_step),
        )
        self.history.append(event)
        return event

    # -- explanations (spec section 51) ------------------------------------

    def _explain_success(self, step: ExperimentStep, was_recovery: bool) -> list[str]:
        lines = [f"Expected action: {step.action_key}", f"Detected action: {step.action_key} — match."]
        if was_recovery:
            lines.append("Previous deviation resolved — sequence back on track.")
        return lines

    def _explain_error(
        self,
        expected: ExperimentStep,
        detected_key: str,
        status: SequenceStatus,
        matching_step: ExperimentStep | None,
    ) -> list[str]:
        lines = [
            f"Expected object/action: {expected.action_key}",
            f"Detected object/action: {detected_key}",
            f"Current required step: {expected.id} — {expected.instruction}",
            f"Classified as: {status.value}",
        ]
        if status == SequenceStatus.WRONG_OBJECT:
            lines.append("Same action as expected, but the wrong object — grab the correct one for this step.")
        elif status == SequenceStatus.SKIPPED_STEP and matching_step is not None:
            lines.append(f"That belongs to a later step (step {matching_step.id}) — complete step {expected.id} first.")
        return lines
