"""Post-condition validation: did the transformation do what the plan said, without damage?

The most important check is **new nulls**. ``errors="coerce"`` silently turns values it can't
parse into NaN/NaT, so a "successful" run can quietly destroy a column. Here we count only
nulls that did NOT already exist (true nulls, null tokens, whitespace-only strings) and fail
the run - with concrete examples of the lost values - if too many appear.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from .planning.models import EXPECTED_KIND, CleaningPlan
from .profiling.profiler import infer_value_type

_KIND_CHECK = {
    "numeric": ptypes.is_numeric_dtype,
    "datetime": ptypes.is_datetime64_any_dtype,
    "bool": ptypes.is_bool_dtype,
}


def _expected_null_mask(s: pd.Series) -> pd.Series:
    return s.map(lambda v: infer_value_type(v) in ("null", "null_token")).astype(bool)


def _as_text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("<NA>")


def validate(
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: CleaningPlan,
    max_new_null_ratio: float = 0.02,
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    warnings: list[str] = []
    column_stats: dict[str, dict] = {}
    n = len(before)

    if len(after) != n:
        problems.append({"column": None, "check": "row_count",
                         "message": f"row count changed {n} -> {len(after)}; plans must never drop rows"})
        return {"passed": False, "problems": problems, "warnings": warnings, "columns": column_stats}

    renames = plan.renames()
    expected_cols = [renames.get(c, c) for c in before.columns]
    missing = [c for c in expected_cols if c not in after.columns]
    if missing:
        problems.append({"column": None, "check": "columns", "message": f"missing columns after transform: {missing}"})
    extra = [c for c in after.columns if c not in expected_cols]
    if extra:
        warnings.append(f"transform added columns: {extra}")

    before = before.reset_index(drop=True)
    after = after.reset_index(drop=True)
    touched = {s.column for s in plan.steps if s.action != "rename"}
    actions_by_col: dict[str, list[str]] = {}
    for s in plan.steps:
        actions_by_col.setdefault(s.column, []).append(s.action)

    for col in before.columns:
        out_col = renames.get(col, col)
        if out_col not in after.columns:
            continue
        b, a = before[col], after[out_col]
        if col not in touched:
            if not _as_text(b).equals(_as_text(a)):
                problems.append({"column": col, "check": "untouched_changed",
                                 "message": f"{col} was not in the plan but its values changed"})
            continue

        expected_null = _expected_null_mask(b)
        new_null = a.isna().to_numpy() & ~expected_null.to_numpy()
        ratio = float(new_null.sum()) / n if n else 0.0
        changed = int((_as_text(b) != _as_text(a)).sum())
        column_stats[col] = {"new_null_ratio": round(ratio, 4), "changed_cells": changed,
                             "dtype_after": str(a.dtype)}
        if ratio > max_new_null_ratio:
            lost = b[new_null].astype("string").dropna().unique()[:5].tolist()
            problems.append({
                "column": col, "check": "new_nulls", "ratio": round(ratio, 4),
                "actions": actions_by_col.get(col, []),
                "examples": lost,
                "message": (f"{col}: transform turned {ratio:.1%} of values into nulls "
                            f"(limit {max_new_null_ratio:.0%}); lost values like {lost}"),
            })
        for action in actions_by_col.get(col, []):
            kind = EXPECTED_KIND.get(action)
            if kind and not _KIND_CHECK[kind](a):
                problems.append({"column": col, "check": "dtype", "actions": [action],
                                 "message": f"{col}: expected a {kind} column after {action}, got {a.dtype}"})
        if ptypes.is_datetime64_any_dtype(a):
            years = a.dropna().dt.year
            odd = float(((years < 1900) | (years > 2100)).mean()) if len(years) else 0.0
            if odd > 0.01:
                warnings.append(f"{col}: {odd:.1%} of dates fall outside 1900-2100")

    return {"passed": not problems, "problems": problems, "warnings": warnings, "columns": column_stats}
