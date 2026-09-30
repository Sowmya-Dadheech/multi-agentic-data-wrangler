"""Schema drift: compare this run's profile with the baseline saved from the last approved run.

Structural drift (columns added/dropped, dominant type changed) also changes the schema
fingerprint, so it naturally causes a memory miss. Value-level drift (null-ratio jumps, new
categories) does NOT change the fingerprint - which is why these checks run even on an
exact memory hit.
"""

from __future__ import annotations

from typing import Any


def detect_drift(
    baseline: dict[str, dict], current: dict[str, dict], null_jump: float = 0.20
) -> list[dict[str, Any]]:
    drift: list[dict[str, Any]] = []
    old, new = set(baseline), set(current)
    for c in sorted(new - old):
        drift.append({"column": c, "kind": "new_column", "structural": True})
    for c in sorted(old - new):
        drift.append({"column": c, "kind": "dropped_column", "structural": True})
    for c in sorted(old & new):
        b, n = baseline[c], current[c]
        if b.get("dominant_family") != n.get("dominant_family"):
            drift.append({
                "column": c, "kind": "type_change", "structural": True,
                "detail": f"{b.get('dominant_family')} -> {n.get('dominant_family')}",
            })
        jump = n.get("null_ratio", 0) - b.get("null_ratio", 0)
        if jump > null_jump:
            drift.append({
                "column": c, "kind": "null_jump", "structural": False,
                "detail": f"{b.get('null_ratio', 0):.0%} -> {n.get('null_ratio', 0):.0%}",
            })
        is_categorical = b.get("distinct", 999) <= 20 and b.get("dominant_family") == "str"
        if is_categorical:
            new_vals = set(n.get("top_values", {})) - set(b.get("top_values", {}))
            if new_vals:
                drift.append({
                    "column": c, "kind": "new_categories", "structural": False,
                    "detail": sorted(new_vals)[:5],
                })
    return drift
