"""Turn a profile into a list of concrete inconsistencies.

Rules first, LLM second: these deterministic checks are free, fast, testable and never
hallucinate. The LLM planner is used for what rules can't do well (semantics, long tail).
"""

from __future__ import annotations

from typing import Any

from ..naming import is_snake_case, normalize
from .matching import match_columns

ID_COLUMNS = {"_id"}
DOMINANCE = 0.8  # a family must cover >= 80% of non-null values to be "the" type


def normalize_target(target: dict | None) -> dict[str, dict[str, Any]]:
    if not target:
        return {}
    out = {}
    for col, spec in target.items():
        out[col] = {"type": spec} if isinstance(spec, str) else dict(spec)
    return out


def plan_renames(
    profile: dict[str, dict], target: dict[str, dict] | None
) -> tuple[dict[str, dict], list[str]]:
    """source column -> {"to": new_name, "level": how it was matched}."""
    renames: dict[str, dict] = {}
    cols = [c for c in profile if c not in ID_COLUMNS]
    if target:
        src = {c: _examples(profile[c]) for c in cols}
        tgt = {c: [str(x) for x in spec.get("examples", [])] for c, spec in target.items()}
        matches, unresolved = match_columns(src, tgt)
        for m in matches:
            if m.source != m.target:
                renames[m.source] = {"to": m.target, "level": m.level, "score": m.score}
        return renames, unresolved
    for c in cols:
        if not is_snake_case(c):
            renames[c] = {"to": normalize(c), "level": "normalized", "score": 1.0}
    return renames, []


def _examples(p: dict) -> list[str]:
    ex: list[str] = []
    for vals in p.get("examples", {}).values():
        ex.extend(vals)
    return ex[:5]


def detect_issues(
    profile: dict[str, dict],
    target: dict | None = None,
    declared: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    target = normalize_target(target)
    issues: list[dict[str, Any]] = []
    renames, unresolved = plan_renames(profile, target)

    for col, p in profile.items():
        if col in ID_COLUMNS:
            continue
        final_name = renames.get(col, {}).get("to", col)
        want = target.get(final_name, {}).get("type") if target else None
        fam = p["families"]
        nn = p["non_null_types"]
        text_like = p["dtype"] in ("object", "string", "str") or p["dtype"].startswith("string")

        def add(issue: str, action: str, severity: str = "error", **detail: Any) -> None:
            issues.append({"column": col, "issue": issue, "action": action,
                           "severity": severity, "detail": detail})

        if p["null_tokens"]:
            add("null_tokens", "standardize_nulls", tokens=list(p["null_tokens"]))
        if p["whitespace_ratio"] > 0:
            add("whitespace", "strip", ratio=p["whitespace_ratio"])

        number = fam.get("number", 0)
        date = fam.get("date", 0)
        boolean = fam.get("bool", 0)

        if (want in ("int", "float") or (want is None and number >= DOMINANCE)) and number > 0 and text_like:
            kind = "mixed_types" if nn.get("numeric_text", 0) or fam.get("str", 0) else "numeric_as_text"
            integer = want == "int" or (want is None and bool(p.get("integral")))
            add(kind, "cast_numeric", types=nn, integer=integer)
        elif (want in ("date", "datetime") or (want is None and date >= DOMINANCE)) and date > 0 and text_like:
            kinds = sorted(k for k in nn if k.startswith(("date:", "datetime:")))
            add("date_formats" if len(kinds) > 1 else "date_as_text", "parse_dates", formats=kinds)
        elif (want == "bool" or (want is None and boolean >= DOMINANCE)) and boolean > 0 and text_like:
            add("bool_as_text", "to_bool", types=nn)
        elif fam.get("str", 0) >= DOMINANCE and text_like and want in (None, "category", "string"):
            non_null = max(1, round(p["count"] * (1 - p["null_ratio"])))
            dn = p["distinct_normalized"]
            categorical = dn <= 50 and dn / non_null < 0.5
            if categorical and dn < p.get("distinct_stripped", p["distinct"]):
                add("inconsistent_categories", "canonicalize",
                    distinct=p["distinct"], distinct_normalized=p["distinct_normalized"])

        if col in renames:
            r = renames[col]
            add("naming", "rename", severity="warning", to=r["to"], level=r["level"], score=r["score"])

    for col in unresolved:
        issues.append({"column": col, "issue": "unmatched_column", "action": "none",
                       "severity": "warning", "detail": {"note": "no target column matched"}})
    if target:
        mapped = {renames.get(c, {}).get("to", c) for c in profile}
        for t in target:
            if t not in mapped:
                issues.append({"column": t, "issue": "missing_column", "action": "none",
                               "severity": "warning", "detail": {}})
    return issues
