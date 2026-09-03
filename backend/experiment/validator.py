"""Sequence validation — deterministic procedural logic, not ML, per the
architecture principle in spec section 50: the procedure is known, so
correctness is decided by explicit rules against the experiment config, and
the action recognizer's job is only to say *what* it saw.

NOTE (scope): this is the MVP-simplified version wired up in Phase 0/1 so the
backend has genuine sequence intelligence from the very first runnable
increment, not just a scripted fake. It implements CORRECT / OUT_OF_SEQUENCE
/ RECOVERED / LOW_CONFIDENCE / REPEATED_STEP / COMPLETE. The finer-grained
split of "out of sequence" into WRONG_OBJECT vs SKIPPED_STEP vs true
OUT_OF_SEQUENCE (spec sections 13-14) — which needs to reason about *why* a
mismatch happened, not just *that* one happened — is completed in the
dedicated state_machine.py + fuller validator pass (Phase 5/6).
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
            was_recovery = self._last_was_error
            self._last_was_error = False
            self._step_index += 1
            if expected_step.action == "COMPLETE" or self._step_index >= self.experiment.total_steps():
                self._finished = True
            event = SequenceEvent(
                step=expected_step.id,
                expected=expected_key,
                detected=detected_key,
                confidence=prediction.confidence,
                status=SequenceStatus.RECOVERED if was_recovery else (
                    SequenceStatus.COMPLETE if self._finished else SequenceStatus.CORRECT
                ),
                recovered=was_recovery or None,
                explanation=self._explain_success(expected_step, was_recovery),
            )
            self.history.append(event)
            return event

        # Mismatch: detected something valid for a *different* step.
        already_done = detected_key in {
            s.action_key for s in self.experiment.steps[: self._step_index]
        }
        status = SequenceStatus.REPEATED_STEP if already_done else SequenceStatus.OUT_OF_SEQUENCE
        self._last_was_error = True
        event = SequenceEvent(
            step=expected_step.id,
            expected=expected_key,
            detected=detected_key,
            confidence=prediction.confidence,
            status=status,
            error_type=status.value,
            recovered=False,
            explanation=self._explain_error(expected_step, detected_key, status),
        )
        self.history.append(event)
        return event

    # -- explanations (spec section 51) ------------------------------------

    def _explain_success(self, step: ExperimentStep, was_recovery: bool) -> list[str]:
        lines = [f"Expected action: {step.action_key}", f"Detected action: {step.action_key} — match."]
        if was_recovery:
            lines.append("Previous deviation resolved — sequence back on track.")
        return lines

    def _explain_error(self, expected: ExperimentStep, detected_key: str, status: SequenceStatus) -> list[str]:
        return [
            f"Expected object/action: {expected.action_key}",
            f"Detected object/action: {detected_key}",
            f"Current required step: {expected.id} — {expected.instruction}",
            f"Classified as: {status.value}",
        ]
