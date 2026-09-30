"""Schema inference for MongoDB.

Mongo has no stored schema, so we *infer* one from sampled documents: for every field path
we count which BSON/Python types appear and how often the field is present. The result is a
distribution over document shapes rather than a fixed contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return type(v).__name__


def _walk(doc: dict, prefix: str, out: dict[str, Counter]) -> None:
    for k, v in doc.items():
        path = f"{prefix}.{k}" if prefix else k
        out[path][_type_name(v)] += 1
        if isinstance(v, dict):
            _walk(v, path, out)
        elif isinstance(v, list):
            for item in v[:3]:  # peek into array elements
                if isinstance(item, dict):
                    _walk(item, path + "[]", out)


def infer_mongo_schema(docs: list[dict]) -> dict[str, dict]:
    hist: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        _walk(d, "", hist)
    n = max(len(docs), 1)
    return {
        path: {
            "presence": round(sum(c.values()) / n, 4),
            "types": dict(c.most_common()),
        }
        for path, c in sorted(hist.items())
        if path != "_id"
    }


def shape_conflicts(schema: dict[str, dict]) -> list[dict]:
    """Fields that are an object in some documents and a scalar in others."""
    issues = []
    for path, info in schema.items():
        types = set(info["types"]) - {"null"}
        if "object" in types and types - {"object"}:
            issues.append({
                "column": path,
                "issue": "mixed_shapes",
                "action": "none",
                "detail": info["types"],
                "severity": "warning",
                "note": "field is nested in some documents and scalar in others; needs a human decision",
            })
        if "array" in types and types - {"array"}:
            issues.append({
                "column": path,
                "issue": "mixed_shapes",
                "action": "none",
                "detail": info["types"],
                "severity": "warning",
                "note": "field is an array in some documents and scalar in others",
            })
    return issues
