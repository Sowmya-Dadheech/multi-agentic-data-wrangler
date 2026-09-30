"""Vetted transformation helpers, injected into the sandbox as ``wr``.

Generated code calls these instead of re-implementing parsing logic, which keeps LLM-written
code short, reviewable and far less error-prone ("templates first, free code last").
This module must only depend on pandas / numpy / re: it is loaded inside the sandbox.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "null", "none", "nil", "nan", "-", "--", "?", "unknown",
    "missing", "#n/a", "-999", "999999", "(blank)",
}
TRUE_WORDS = {"true", "t", "yes", "y", "1"}
FALSE_WORDS = {"false", "f", "no", "n", "0"}
_NUMBER = re.compile(r"([+-]?\d[\d,]*\.?\d*)")

# (regex, strptime format) tried in order by parse_dates(strategy>=1)
DATE_FORMATS = [
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    (r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}$", "ISO8601"),
    (r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", "ISO8601"),
    (r"^\d{1,2}/\d{1,2}/\d{4}$", "%m/%d/%Y"),
    (r"^\d{1,2}\.\d{1,2}\.\d{4}$", "%d.%m.%Y"),
    (r"^\d{1,2} [A-Za-z]{3},? \d{4}$", "%d %b %Y"),
    (r"^\d{1,2} [A-Za-z]{4,9},? \d{4}$", "%d %B %Y"),
    (r"^[A-Za-z]{3} \d{1,2},? \d{4}$", "%b %d %Y"),
    (r"^[A-Za-z]{4,9} \d{1,2},? \d{4}$", "%B %d %Y"),
]


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string")


def strip(s: pd.Series) -> pd.Series:
    t = _text(s).str.strip()
    return t.mask(t == "", pd.NA)


def standardize_nulls(s: pd.Series, tokens: list[str] | None = None) -> pd.Series:
    toks = NULL_TOKENS | {str(x).strip().lower() for x in (tokens or [])}
    t = _text(s)
    return t.mask(t.str.strip().str.lower().isin(toks), pd.NA)


def cast_numeric(s: pd.Series, integer: bool = False, strategy: int = 0) -> pd.Series:
    """strategy 0: plain ``to_numeric`` (what most people write first).
    strategy 1+: pull the number out of text such as "$1,200.50", "42 yrs", "70kg"."""
    if strategy <= 0:
        out = pd.to_numeric(_text(s).str.strip(), errors="coerce")
    else:
        raw = _text(s).str.strip()
        neg_paren = raw.str.match(r"^\(.*\)$", na=False)
        num = raw.str.extract(_NUMBER, expand=False).str.replace(",", "", regex=False)
        out = pd.to_numeric(num, errors="coerce")
        out = out.where(~neg_paren, -out.abs())
    out = out.astype("Float64")
    if integer:
        rounded = out.round()
        if bool(((out - rounded).abs().fillna(0) < 1e-9).all()):
            return rounded.astype("Int64")
    return out


def parse_dates(s: pd.Series, strategy: int = 0, dayfirst: bool = False) -> pd.Series:
    """strategy 0: plain ``to_datetime`` - pandas infers ONE format from the first value, so
    every other format silently becomes NaT (a classic wrangling bug).
    strategy 1: try each known format explicitly, assuming month-first for dd/mm vs mm/dd.
    strategy 2: same, but honour ``dayfirst`` (decided from evidence, e.g. a first part > 12).
    """
    t = _text(s).str.strip()
    if strategy <= 0:
        return pd.to_datetime(t, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for pattern, fmt in DATE_FORMATS:
        if dayfirst and strategy >= 2 and fmt == "%m/%d/%Y":
            fmt = "%d/%m/%Y"
        mask = t.str.match(pattern, na=False) & out.isna()
        if mask.any():
            vals = t[mask].str.replace(",", "", regex=False)
            parsed = pd.to_datetime(vals, format=fmt.replace(",", ""), errors="coerce")
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_convert(None)
            out.loc[mask] = parsed.astype("datetime64[ns]")
    return out


def to_bool(s: pd.Series) -> pd.Series:
    t = _text(s).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[t.isin(TRUE_WORDS)] = True
    out[t.isin(FALSE_WORDS)] = False
    return out


def canonicalize(s: pd.Series, mapping: dict[str, str] | None = None) -> pd.Series:
    """Map case / whitespace variants onto one canonical spelling.

    ``mapping`` is {normalized key -> canonical value}; unseen variants fall back to the
    most frequent spelling observed for that key in this batch."""
    t = _text(s).str.strip()
    key = t.str.lower()
    mapping = dict(mapping or {})
    observed = (
        pd.DataFrame({"k": key, "v": t}).dropna().groupby("k")["v"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    for k, v in observed.items():
        mapping.setdefault(k, v)
    return key.map(mapping).astype("string").where(t.notna(), pd.NA)


def infer_dayfirst(s: pd.Series) -> bool:
    """Evidence for dd/mm/yyyy: some slash date has a first part > 12."""
    t = _text(s).str.strip()
    parts = t[t.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)].str.split("/")
    if parts.empty:
        return False
    first = parts.str[0].astype(int)
    second = parts.str[1].astype(int)
    return bool((first > 12).any() and not (second > 12).any())


__all__ = [
    "np", "pd", "strip", "standardize_nulls", "cast_numeric", "parse_dates",
    "to_bool", "canonicalize", "infer_dayfirst",
]
