"""Loads and validates an experiment definition (e.g.
experiments/bas_sample_001.json) into typed objects. This is the ONLY place
that reads experiment JSON — the state machine, UI, and logger all consume
the ExperimentDefinition object this produces, never raw JSON directly. That
is what makes "load a different experiment" a config change instead of a
code change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExperimentObject(BaseModel):
    label: str
    color_hint: Optional[str] = None


class ExperimentStep(BaseModel):
    id: int
    action: str  # "PICK" | "PLACE" | "COMPLETE" (extensible)
    object: str
    target: Optional[str] = None
    instruction: str
    voice_next: str = ""

    @property
    def action_key(self) -> str:
        """The canonical action name this step expects, e.g. PICK_RED_BOX,
        PLACE_RED_BOX, or COMPLETE_EXPERIMENT. This is what the action
        recognizer's output is compared against."""
        if self.action == "COMPLETE":
            return "COMPLETE_EXPERIMENT"
        return f"{self.action}_{self.object}"


class ExperimentDefinition(BaseModel):
    experiment_id: str
    name: str
    description: str = ""
    objects: dict[str, ExperimentObject] = Field(default_factory=dict)
    steps: list[ExperimentStep]

    @field_validator("steps")
    @classmethod
    def _steps_must_be_sequential_and_nonempty(cls, steps: list[ExperimentStep]) -> list[ExperimentStep]:
        if not steps:
            raise ValueError("Experiment must define at least one step.")
        ids = [s.id for s in steps]
        expected = list(range(1, len(steps) + 1))
        if ids != expected:
            raise ValueError(
                f"Experiment steps must be numbered sequentially starting at 1 "
                f"(got {ids}, expected {expected})."
            )
        return steps

    def step_by_id(self, step_id: int) -> Optional[ExperimentStep]:
        return next((s for s in self.steps if s.id == step_id), None)

    def total_steps(self) -> int:
        return len(self.steps)


class ExperimentLoadError(Exception):
    """Raised when an experiment config file is missing or invalid."""


def load_experiment(path: str | Path) -> ExperimentDefinition:
    p = Path(path)
    if not p.exists():
        raise ExperimentLoadError(f"Experiment config not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ExperimentLoadError(f"Experiment config at {p} is not valid JSON: {e}") from e
    try:
        return ExperimentDefinition(**raw)
    except Exception as e:  # pydantic ValidationError, re-raised with context
        raise ExperimentLoadError(f"Experiment config at {p} failed validation: {e}") from e
