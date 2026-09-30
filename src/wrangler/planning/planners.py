"""Planner agents.

* ``HeuristicPlanner`` - deterministic, offline, zero-cost. Maps detected issues to vetted
  actions and repairs failed plans by escalating strategies based on validator evidence.
  Used for tests, CI and the reproducible benchmark.
* ``LLMPlanner`` - any LangChain chat model (Anthropic, OpenAI, Ollama...). Receives a
  compact profile of the *problem columns only*, the rule-based suggestions and any hint
  from memory, and must answer with a ``CleaningPlan`` (structured output). On failure it
  gets the validator's concrete error messages and examples back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from ..profiling.profiler import compact_profile
from ..sandbox import helpers as wr
from .models import CleaningPlan, PlanStep


@dataclass
class PlanResult:
    plan: CleaningPlan
    usage: dict[str, int] = field(default_factory=dict)
    note: str = ""
    exhausted: bool = False  # the planner has no further idea how to fix this


class Planner(Protocol):
    name: str

    def plan(self, profile: dict, issues: list[dict], sample: pd.DataFrame,
             target: dict | None = None, hint: dict | None = None) -> PlanResult: ...

    def repair(self, plan: CleaningPlan, problems: list[dict], sample: pd.DataFrame,
               profile: dict) -> PlanResult: ...


# ============================================================================ heuristic


def canonical_mapping(s: pd.Series) -> dict[str, str]:
    t = s.astype("string").str.strip().dropna()
    t = t[t != ""]
    if t.empty:
        return {}
    df = pd.DataFrame({"k": t.str.lower(), "v": t})
    return {k: str(v.value_counts().index[0]) for k, v in df.groupby("k")["v"]}


def steps_from_issues(issues: list[dict], sample: pd.DataFrame) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for issue in issues:
        col, action, d = issue["column"], issue["action"], issue.get("detail", {})
        if action in ("none", None) or col not in sample.columns:
            continue
        if action == "standardize_nulls":
            steps.append(PlanStep(column=col, action=action, params={"tokens": d.get("tokens", [])},
                                  rationale=f"null tokens {d.get('tokens')}"))
        elif action == "strip":
            steps.append(PlanStep(column=col, action=action, rationale="leading/trailing whitespace"))
        elif action == "cast_numeric":
            types = d.get("types", {})
            messy = bool(types.get("numeric_text") or types.get("str"))
            steps.append(PlanStep(column=col, action=action, params={"integer": bool(d.get("integer"))},
                                  strategy=1 if messy else 0,
                                  rationale=f"numbers stored as text {types}"))
        elif action == "parse_dates":
            fmts = d.get("formats", [])
            simple = fmts in (["date:iso"], ["datetime:iso"])
            steps.append(PlanStep(column=col, action=action, params={"dayfirst": False},
                                  strategy=0 if simple else 1, rationale=f"date formats {fmts}"))
        elif action == "to_bool":
            steps.append(PlanStep(column=col, action=action, rationale="boolean words stored as text"))
        elif action == "canonicalize":
            steps.append(PlanStep(column=col, action=action,
                                  params={"mapping": canonical_mapping(sample[col])},
                                  rationale="case / whitespace variants of the same category"))
        elif action == "rename":
            steps.append(PlanStep(column=col, action=action, params={"to": d["to"]},
                                  rationale=f"naming ({d.get('level')})"))
    return steps


def heuristic_confidence(issues: list[dict]) -> float:
    conf = 0.9
    if any(i["issue"] == "mixed_shapes" for i in issues):
        conf -= 0.35  # needs a human decision about nested vs scalar
    if any(i["issue"] == "unmatched_column" for i in issues):
        conf -= 0.1
    return round(max(conf, 0.1), 2)


class HeuristicPlanner:
    name = "heuristic"

    def plan(self, profile, issues, sample, target=None, hint=None) -> PlanResult:
        steps = steps_from_issues(issues, sample)
        note = ""
        if hint:
            # reuse strategies that worked for a similar schema (hint only - still validated)
            learned = {(s["column"], s["action"]): s for s in hint.get("plan", {}).get("steps", [])}
            for st in steps:
                prev = learned.get((st.column, st.action))
                if prev and prev.get("strategy", 0) > st.strategy:
                    st.strategy = prev["strategy"]
                    st.params.update({k: v for k, v in prev.get("params", {}).items() if k == "dayfirst"})
            note = f"used hint from similar schema {hint.get('source')}"
        return PlanResult(CleaningPlan(steps=steps, confidence=heuristic_confidence(issues)), note=note)

    def repair(self, plan, problems, sample, profile) -> PlanResult:
        new = plan.model_copy(deep=True)
        changes: list[str] = []
        for prob in problems:
            col = prob.get("column")
            if prob["check"] not in ("new_nulls", "dtype") or col is None:
                continue
            for step in new.steps:
                if step.column != col or step.action not in ("cast_numeric", "parse_dates", "to_bool"):
                    continue
                if step.action == "cast_numeric" and step.strategy < 1:
                    step.strategy = 1
                    changes.append(f"cast_numeric({col}): extract numbers from text (strategy 1)")
                elif step.action == "parse_dates" and step.strategy < 1:
                    step.strategy = 1
                    changes.append(f"parse_dates({col}): try each known format explicitly (strategy 1)")
                elif step.action == "parse_dates" and step.strategy >= 1:
                    evidence = pd.concat([sample[col].astype("string"),
                                          pd.Series(prob.get("examples", []), dtype="string")],
                                         ignore_index=True)
                    cleaned = wr.strip(wr.standardize_nulls(evidence))
                    dayfirst = wr.infer_dayfirst(cleaned)
                    if step.strategy == 2 and bool(step.params.get("dayfirst")) == dayfirst:
                        continue  # evidence agrees with what already failed: nothing left to try
                    step.strategy = 2
                    step.params["dayfirst"] = dayfirst
                    changes.append(f"parse_dates({col}): day/month order from evidence -> dayfirst={dayfirst}")
        if not changes:
            return PlanResult(new, note="no further repair strategy", exhausted=True)
        new.confidence = round(max(0.1, new.confidence - 0.05), 2)
        return PlanResult(new, note="; ".join(changes))


# ============================================================================ LLM

PLANNER_SYSTEM = """You are the planning agent in a data-cleaning pipeline.
Given a compact profile of the columns that have problems, the issues a rule engine
detected, and optionally a previously approved plan for a similar dataset, return the
MINIMAL cleaning plan.

Rules:
- Only reference columns that exist in the profile.
- Prefer the vetted actions: standardize_nulls, strip, cast_numeric, parse_dates, to_bool,
  canonicalize, rename. Use "custom" only when none fits; then put the body of the code in
  params.code as pandas statements that modify `df` in place (no imports, no I/O, no loops
  over rows, no underscores in attribute names). Available names: df, pd, np, wr.
- Never drop rows. Coerce unparseable values to null rather than deleting data.
- cast_numeric: set params.integer=true for whole numbers; strategy 1 extracts numbers from
  text like "$1,200" or "42 yrs"; strategy 0 is a plain numeric cast.
- parse_dates: strategy 0 = pandas default (only safe for a single ISO format),
  1 = try every known format (month-first for a/b/yyyy), 2 = honour params.dayfirst.
- rename: params.to is the new snake_case (or target-schema) name.
- confidence: your honest probability that the plan is correct without human edits."""

REPAIR_SYSTEM = """You are the repair agent in a data-cleaning pipeline. The previous plan ran
in a sandbox and failed validation. Using the concrete errors and the example values that
were lost, return a corrected COMPLETE plan. Escalate strategies, set dayfirst from evidence,
or fall back to a small "custom" step. Keep everything that already worked unchanged."""


def estimate_prompt_tokens(profile: dict, issues: list[dict], draft: CleaningPlan) -> int:
    """Rough size (chars / 4) of the prompt LLMPlanner would send. Used by the offline
    benchmark to report what memory hits save when no real model is attached."""
    problem_cols = sorted({i["column"] for i in issues if i["column"] in profile})
    payload = {
        "profile": compact_profile(profile, problem_cols),
        "issues": [{k: i[k] for k in ("column", "issue", "detail")} for i in issues],
        "rule_based_draft": draft.model_dump(),
    }
    return (len(PLANNER_SYSTEM) + len(json.dumps(payload, default=str, indent=1))) // 4


def _usage(raw: Any) -> dict[str, int]:
    meta = getattr(raw, "usage_metadata", None) or {}
    return {"llm_calls": 1, "input_tokens": int(meta.get("input_tokens", 0)),
            "output_tokens": int(meta.get("output_tokens", 0))}


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: Any, fallback: HeuristicPlanner | None = None):
        self.llm = llm
        self.fallback = fallback or HeuristicPlanner()
        self._structured = llm.with_structured_output(CleaningPlan, include_raw=True)

    def _call(self, system: str, payload: dict) -> tuple[CleaningPlan | None, dict[str, int], str]:
        msgs = [("system", system), ("human", json.dumps(payload, default=str, indent=1))]
        out = self._structured.invoke(msgs)
        usage = _usage(out.get("raw"))
        if out.get("parsing_error") or out.get("parsed") is None:
            return None, usage, f"structured output failed: {out.get('parsing_error')}"
        return out["parsed"], usage, ""

    def _sanitize(self, plan: CleaningPlan, sample: pd.DataFrame) -> CleaningPlan:
        # hallucination guard: drop steps on columns that don't exist
        steps = [s for s in plan.steps if s.column in sample.columns]
        for s in steps:
            if s.action == "canonicalize" and not s.params.get("mapping"):
                s.params["mapping"] = canonical_mapping(sample[s.column])
            if s.action == "rename" and not s.params.get("to"):
                s.action = "strip"
        return CleaningPlan(steps=steps, confidence=plan.confidence, notes=plan.notes)

    def plan(self, profile, issues, sample, target=None, hint=None) -> PlanResult:
        problem_cols = sorted({i["column"] for i in issues if i["column"] in profile})
        draft = self.fallback.plan(profile, issues, sample, target, hint).plan
        payload = {
            "profile": compact_profile(profile, problem_cols),
            "issues": [{k: i[k] for k in ("column", "issue", "detail")} for i in issues],
            "rule_based_draft": draft.model_dump(),
            "target_schema": target or {},
            "hint_from_similar_dataset": (hint or {}).get("plan"),
        }
        plan, usage, err = self._call(PLANNER_SYSTEM, payload)
        if plan is None:
            return PlanResult(draft, usage, note=f"{err}; fell back to rule-based draft")
        return PlanResult(self._sanitize(plan, sample), usage, note=plan.notes)

    def repair(self, plan, problems, sample, profile) -> PlanResult:
        payload = {
            "previous_plan": plan.model_dump(),
            "validation_errors": problems,
            "profile": compact_profile(profile, sorted({p["column"] for p in problems if p.get("column")})),
        }
        new, usage, err = self._call(REPAIR_SYSTEM, payload)
        if new is None:
            res = self.fallback.repair(plan, problems, sample, profile)
            res.usage, res.note = usage, f"{err}; {res.note}"
            return res
        new = self._sanitize(new, sample)
        exhausted = new.model_dump() == plan.model_dump()
        return PlanResult(new, usage, note=new.notes or "LLM repair", exhausted=exhausted)
