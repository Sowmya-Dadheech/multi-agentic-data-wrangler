"""Column profiling: what does each column *actually* contain?

Type inference runs on the column's unique values (weighted by frequency), so profiling a
5,000-row sample with 30 columns takes milliseconds.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "null", "none", "nil", "nan", "-", "--", "?", "unknown",
    "missing", "#n/a", "-999", "999999", "(blank)",
}

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
# numbers wrapped in currency symbols, thousands separators or unit suffixes
_NUMERIC_TEXT = re.compile(
    r"^(?:[$€£₹]|usd|eur|inr)?\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:%|[a-z]{1,6}\.?)?$",
    re.IGNORECASE,
)
_BOOL = {"true", "false", "yes", "no", "y", "n", "t", "f"}
DATE_PATTERNS: dict[str, re.Pattern] = {
    "date:iso": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "datetime:iso": re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"),
    "date:us": re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"),
    "date:dotted": re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"),
    "date:text_dmy": re.compile(r"^\d{1,2} [A-Za-z]{3,9},? \d{4}$"),
    "date:text_mdy": re.compile(r"^[A-Za-z]{3,9} \d{1,2},? \d{4}$"),
}

TYPE_FAMILY = {
    "int": "number", "float": "number", "numeric_text": "number",
    "bool": "bool", "str": "str", "json": "json",
}


def family(type_name: str) -> str:
    if type_name.startswith(("date:", "datetime:")):
        return "date"
    return TYPE_FAMILY.get(type_name, type_name)


def infer_value_type(v: Any) -> str:
    if v is None or v is pd.NA or v is pd.NaT:
        return "null"
    if isinstance(v, float) and v != v:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, (pd.Timestamp,)):
        return "datetime:iso"
    s = str(v).strip()
    if s.lower() in NULL_TOKENS:
        return "null_token"
    if _INT.match(s):
        # leading-zero codes (zip codes, ids) are identifiers, not numbers
        return "str" if len(s) > 1 and s.lstrip("+-").startswith("0") else "int"
    if _FLOAT.match(s):
        return "float"
    for name, pat in DATE_PATTERNS.items():
        if pat.match(s):
            return name
    if s.lower() in _BOOL:
        return "bool"
    if _NUMERIC_TEXT.match(s) and any(ch.isdigit() for ch in s):
        return "numeric_text"
    if s[:1] in "{[" and s[-1:] in "}]":
        return "json"
    return "str"


def _clip(v: Any, n: int = 60) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def profile_column(s: pd.Series, top_k: int = 8) -> dict[str, Any]:
    n = len(s)
    vc = s.value_counts(dropna=False)
    type_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    whitespace = 0
    null_tokens_seen: Counter[str] = Counter()
    ints_are_01 = True
    for value, count in vc.items():
        t = infer_value_type(value)
        if t == "int" and str(value).strip().lstrip("+") not in ("0", "1"):
            ints_are_01 = False
        type_counts[t] += int(count)
        if t not in ("null",) and len(examples.setdefault(t, [])) < 3:
            examples[t].append(_clip(value))
        if t == "null_token":
            null_tokens_seen[str(value)] += int(count)
        if isinstance(value, str) and value != value.strip() and value.strip():
            whitespace += int(count)

    dist = {k: round(v / n, 4) for k, v in type_counts.most_common()} if n else {}
    nulls = type_counts.get("null", 0) + type_counts.get("null_token", 0)
    non_null = n - nulls
    nn_dist = (
        {k: round(v / non_null, 4) for k, v in type_counts.items() if k not in ("null", "null_token")}
        if non_null
        else {}
    )
    fam: Counter[str] = Counter()
    for k, v in nn_dist.items():
        fam[family(k)] += v
    if ints_are_01 and fam.get("bool", 0) > 0 and "int" in nn_dist:
        # 1/0 mixed with yes/no/true/false is a boolean column, not a number column
        fam["bool"] += nn_dist["int"]
        fam["number"] -= nn_dist["int"]
        if fam["number"] <= 1e-9:
            del fam["number"]
    dominant = max(nn_dist, key=nn_dist.get) if nn_dist else "null"
    dominant_family = fam.most_common(1)[0][0] if fam else "null"

    text_values = s.dropna()
    text_values = text_values[text_values.map(lambda x: isinstance(x, str))]
    distinct = int(s.nunique(dropna=True))
    distinct_norm = int(text_values.str.strip().str.lower().nunique()) if len(text_values) else distinct
    distinct_stripped = int(text_values.str.strip().nunique()) if len(text_values) else distinct

    integral = None
    if dominant_family == "number":
        nums = pd.to_numeric(
            s.dropna().astype(str).str.extract(r"([+-]?\d[\d,]*\.?\d*)")[0].str.replace(",", ""),
            errors="coerce",
        ).dropna()
        integral = bool(len(nums)) and bool((nums % 1 == 0).all())

    return {
        "dtype": str(s.dtype),
        "count": n,
        "type_distribution": dist,
        "non_null_types": nn_dist,
        "families": {k: round(v, 4) for k, v in fam.items()},
        "dominant_type": dominant,
        "dominant_family": dominant_family,
        "null_ratio": round(nulls / n, 4) if n else 0.0,
        "true_null_ratio": round(type_counts.get("null", 0) / n, 4) if n else 0.0,
        "null_tokens": dict(null_tokens_seen.most_common(5)),
        "whitespace_ratio": round(whitespace / n, 4) if n else 0.0,
        "distinct": distinct,
        "distinct_normalized": distinct_norm,
        "distinct_stripped": distinct_stripped,
        "integral": integral,
        "ints_are_01": ints_are_01 and "int" in nn_dist,
        "top_values": {_clip(k): int(v) for k, v in s.astype("string").value_counts().head(top_k).items()},
        "examples": examples,
    }


def profile_frame(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {col: profile_column(df[col]) for col in df.columns}


def compact_profile(profile: dict[str, dict], columns: list[str] | None = None) -> dict[str, dict]:
    """A token-cheap view of the profile for the LLM (no raw data beyond a few examples)."""
    cols = columns if columns is not None else list(profile)
    out = {}
    for c in cols:
        p = profile[c]
        out[c] = {
            "types": p["non_null_types"],
            "null_ratio": p["null_ratio"],
            "null_tokens": list(p["null_tokens"])[:3],
            "distinct": p["distinct"],
            "examples": {k: v[:2] for k, v in p["examples"].items() if k != "null_token"},
        }
    return out
