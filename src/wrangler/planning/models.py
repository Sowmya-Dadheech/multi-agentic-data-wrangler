"""The structured cleaning plan.

The planner (LLM or heuristic) must answer in this schema. Constraining ``action`` to a
``Literal`` is itself a guardrail: the model picks from a menu of vetted operations and can
only write free-form code through the explicit ``custom`` action, which is AST-checked.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Action = Literal[
    "standardize_nulls",
    "strip",
    "cast_numeric",
    "parse_dates",
    "to_bool",
    "canonicalize",
    "rename",
    "custom",
]

# order in which actions are applied to a column (renames always last)
ACTION_ORDER = {
    "standardize_nulls": 0,
    "strip": 1,
    "canonicalize": 2,
    "cast_numeric": 3,
    "parse_dates": 3,
    "to_bool": 3,
    "custom": 4,
    "rename": 9,
}

# dtype family each action is expected to produce (used by the validator)
EXPECTED_KIND = {"cast_numeric": "numeric", "parse_dates": "datetime", "to_bool": "bool"}


class PlanStep(BaseModel):
    column: str = Field(description="Existing source column name")
    action: Action
    params: dict[str, Any] = Field(default_factory=dict)
    strategy: int = Field(default=0, ge=0, le=3, description="Escalation level for retries")
    rationale: str = ""


class CleaningPlan(BaseModel):
    steps: list[PlanStep]
    confidence: float = Field(ge=0, le=1, description="How sure the planner is (0-1)")
    notes: str = ""

    def ordered(self) -> list[PlanStep]:
        return sorted(self.steps, key=lambda s: (ACTION_ORDER[s.action], s.column))

    def renames(self) -> dict[str, str]:
        return {s.column: s.params["to"] for s in self.steps if s.action == "rename"}

    def final_name(self, column: str) -> str:
        return self.renames().get(column, column)
