"""Compile one dialect-agnostic plan into pandas, PostgreSQL or a MongoDB pipeline.

Keeping the *plan* independent of the dialect means the same approved fix can run in-memory
on a sample (pandas, inside the sandbox) and be pushed down to the database for the full
table (SQL / aggregation pipeline), so data never has to leave the database.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .models import CleaningPlan, PlanStep


class UnsupportedForDialect(Exception):
    """A step (usually ``custom``) cannot be expressed in the target dialect."""


# ============================================================================ pandas


def _q(v) -> str:
    return json.dumps(v)


def to_pandas(plan: CleaningPlan) -> str:
    lines = ["def transform(df):"]
    for step in plan.ordered():
        c = _q(step.column)
        p = step.params
        a = step.action
        if a == "standardize_nulls":
            lines.append(f"    df[{c}] = wr.standardize_nulls(df[{c}], tokens={_q(p.get('tokens', []))})")
        elif a == "strip":
            lines.append(f"    df[{c}] = wr.strip(df[{c}])")
        elif a == "cast_numeric":
            lines.append(
                f"    df[{c}] = wr.cast_numeric(df[{c}], integer={bool(p.get('integer'))}, "
                f"strategy={step.strategy})"
            )
        elif a == "parse_dates":
            lines.append(
                f"    df[{c}] = wr.parse_dates(df[{c}], strategy={step.strategy}, "
                f"dayfirst={bool(p.get('dayfirst'))})"
            )
        elif a == "to_bool":
            lines.append(f"    df[{c}] = wr.to_bool(df[{c}])")
        elif a == "canonicalize":
            lines.append(f"    df[{c}] = wr.canonicalize(df[{c}], mapping={_q(p.get('mapping', {}))})")
        elif a == "custom":
            code = str(p.get("code", "")).strip("\n")
            lines.append(f"    # custom step for {step.column}: {step.rationale[:60]}")
            lines.extend("    " + ln for ln in code.splitlines())
    renames = plan.renames()
    if renames:
        lines.append(f"    df = df.rename(columns={_q(renames)})")
    lines.append("    return df")
    return "\n".join(lines) + "\n"


# ============================================================================ PostgreSQL


def _sql_str(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


_PG_NUM = r"[+-]?[0-9][0-9,]*\.?[0-9]*"
# plain TRIM() only removes spaces; Python's str.strip() also removes tabs / newlines
_WS = "' ' || CHR(9) || CHR(10) || CHR(13)"


def _trim(expr: str) -> str:
    return f"BTRIM({expr}, {_WS})"


def _pg_expr(expr: str, step: PlanStep) -> str:
    a, p = step.action, step.params
    if a == "standardize_nulls":
        from ..sandbox.helpers import NULL_TOKENS

        toks = sorted(NULL_TOKENS | {str(t).strip().lower() for t in p.get("tokens", [])})
        lst = ", ".join(_sql_str(t) for t in toks)
        return f"CASE WHEN LOWER({_trim(expr)}) IN ({lst}) THEN NULL ELSE {expr} END"
    if a == "strip":
        return f"NULLIF({_trim(expr)}, '')"
    if a == "canonicalize":
        mapping = p.get("mapping", {})
        if not mapping:
            return f"{_trim(expr)}"
        whens = " ".join(f"WHEN {_sql_str(k)} THEN {_sql_str(v)}" for k, v in sorted(mapping.items()))
        return f"CASE LOWER({_trim(expr)}) {whens} ELSE {_trim(expr)} END"
    if a == "cast_numeric":
        if step.strategy <= 0:
            clean = f"{_trim(expr)}"
        else:
            clean = f"REPLACE(SUBSTRING({expr} FROM '{_PG_NUM}'), ',', '')"
        num = f"CASE WHEN {clean} ~ '^[+-]?([0-9]+\\.?[0-9]*|\\.[0-9]+)$' THEN CAST({clean} AS DOUBLE PRECISION) END"
        if step.strategy >= 1:
            num = f"CASE WHEN {_trim(expr)} ~ '^\\(.*\\)$' THEN -ABS({num}) ELSE {num} END"
        if p.get("integer"):
            return f"CAST(ROUND({num}) AS BIGINT)"
        return num
    if a == "parse_dates":
        if step.strategy <= 0:
            raise UnsupportedForDialect("strategy 0 date parsing is pandas-only")
        slash = "DD/MM/YYYY" if (p.get("dayfirst") and step.strategy >= 2) else "MM/DD/YYYY"
        e = f"{_trim(expr)}"
        return (
            "CASE"
            f" WHEN {e} ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$' THEN TO_TIMESTAMP({e}, 'YYYY-MM-DD')"
            f" WHEN {e} ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}[T ][0-9]{{2}}:[0-9]{{2}}' THEN CAST(SUBSTRING({e} FROM 1 FOR 19) AS TIMESTAMP)"
            f" WHEN {e} ~ '^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{4}}$' THEN TO_TIMESTAMP({e}, '{slash}')"
            f" WHEN {e} ~ '^[0-9]{{1,2}}\\.[0-9]{{1,2}}\\.[0-9]{{4}}$' THEN TO_TIMESTAMP({e}, 'DD.MM.YYYY')"
            f" WHEN {e} ~ '^[0-9]{{1,2}} [A-Za-z]{{3,9}},? [0-9]{{4}}$' THEN TO_TIMESTAMP(REPLACE({e}, ',', ''), 'DD Mon YYYY')"
            f" WHEN {e} ~ '^[A-Za-z]{{3,9}} [0-9]{{1,2}},? [0-9]{{4}}$' THEN TO_TIMESTAMP(REPLACE({e}, ',', ''), 'Mon DD YYYY')"
            " END::timestamp"
        )
    if a == "to_bool":
        t = ", ".join(_sql_str(x) for x in ["true", "t", "yes", "y", "1"])
        f = ", ".join(_sql_str(x) for x in ["false", "f", "no", "n", "0"])
        return (f"CASE WHEN LOWER({_trim(expr)}) IN ({t}) THEN TRUE "
                f"WHEN LOWER({_trim(expr)}) IN ({f}) THEN FALSE END")
    if a == "custom":
        raise UnsupportedForDialect("custom (LLM-written) steps run in the pandas sandbox")
    return expr


def to_postgres(plan: CleaningPlan, columns: list[str], source_table: str, target_table: str) -> str:
    by_col: dict[str, list[PlanStep]] = defaultdict(list)
    for s in plan.ordered():
        if s.action != "rename":
            by_col[s.column].append(s)
    renames = plan.renames()
    selects = []
    for col in columns:
        steps = by_col.get(col, [])
        expr = f"CAST({_ident(col)} AS TEXT)" if steps else _ident(col)
        for step in steps:
            expr = _pg_expr(expr, step)
        selects.append(f"  {expr} AS {_ident(renames.get(col, col))}")
    body = ",\n".join(selects)
    return (
        f"DROP TABLE IF EXISTS {_ident(target_table)};\n"
        f"CREATE TABLE {_ident(target_table)} AS\nSELECT\n{body}\nFROM {_ident(source_table)}\n"
        # keep the source's physical row order so row-by-row validation lines up
        "ORDER BY ctid"
    )


# ============================================================================ MongoDB

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _m_str(field: str) -> dict:
    return {"$trim": {"input": {"$toString": f"${field}"}}}


def _mongo_expr(step: PlanStep) -> dict:
    f, a, p = step.column, step.action, step.params
    ref = f"${f}"
    if a == "standardize_nulls":
        from ..sandbox.helpers import NULL_TOKENS

        toks = sorted(NULL_TOKENS | {str(t).strip().lower() for t in p.get("tokens", [])})
        return {"$cond": [{"$in": [{"$toLower": _m_str(f)}, toks]}, None, ref]}
    if a == "strip":
        return {"$cond": [{"$eq": [{"$type": ref}, "string"]}, {"$trim": {"input": ref}}, ref]}
    if a == "canonicalize":
        branches = [
            {"case": {"$eq": [{"$toLower": _m_str(f)}, k]}, "then": v}
            for k, v in sorted(p.get("mapping", {}).items())
        ]
        return {"$switch": {"branches": branches, "default": ref}} if branches else ref
    if a == "cast_numeric":
        match = {"$regexFind": {"input": {"$toString": ref}, "regex": r"[+-]?[0-9][0-9,]*\.?[0-9]*"}}
        num = {"$convert": {
            "input": {"$replaceAll": {"input": "$$m.match", "find": ",", "replacement": ""}},
            "to": "double", "onError": None, "onNull": None,
        }}
        expr = {"$let": {"vars": {"m": match}, "in": num}}
        return {"$round": [expr, 0]} if p.get("integer") else expr
    if a == "parse_dates":
        s = _m_str(f)
        slash = "%d/%m/%Y" if (p.get("dayfirst") and step.strategy >= 2) else "%m/%d/%Y"
        parts = {"$split": [s, " "]}
        text_dmy = {"$dateFromParts": {
            "year": {"$toInt": {"$arrayElemAt": [parts, 2]}},
            "month": {"$add": [{"$indexOfArray": [_MONTHS, {"$toLower": {"$substrCP": [{"$arrayElemAt": [parts, 1]}, 0, 3]}}]}, 1]},
            "day": {"$toInt": {"$arrayElemAt": [parts, 0]}},
        }}

        def dfs(fmt: str) -> dict:
            return {"$dateFromString": {"dateString": s, "format": fmt, "onError": None, "onNull": None}}

        return {"$switch": {"branches": [
            {"case": {"$regexMatch": {"input": s, "regex": r"^\d{4}-\d{2}-\d{2}$"}}, "then": dfs("%Y-%m-%d")},
            {"case": {"$regexMatch": {"input": s, "regex": r"^\d{4}-\d{2}-\d{2}[T ]"}},
             "then": {"$dateFromString": {"dateString": s, "onError": None, "onNull": None}}},
            {"case": {"$regexMatch": {"input": s, "regex": r"^\d{1,2}/\d{1,2}/\d{4}$"}}, "then": dfs(slash)},
            {"case": {"$regexMatch": {"input": s, "regex": r"^\d{1,2}\.\d{1,2}\.\d{4}$"}}, "then": dfs("%d.%m.%Y")},
            {"case": {"$regexMatch": {"input": s, "regex": r"^\d{1,2} [A-Za-z]{3,9} \d{4}$"}}, "then": text_dmy},
        ], "default": None}}
    if a == "to_bool":
        low = {"$toLower": _m_str(f)}
        return {"$switch": {"branches": [
            {"case": {"$in": [low, ["true", "t", "yes", "y", "1"]]}, "then": True},
            {"case": {"$in": [low, ["false", "f", "no", "n", "0"]]}, "then": False},
        ], "default": None}}
    raise UnsupportedForDialect(f"{a} runs in the pandas sandbox")


def to_mongo(plan: CleaningPlan, target_collection: str) -> list[dict]:
    pipeline: list[dict] = []
    for step in plan.ordered():
        if step.action == "rename":
            continue
        pipeline.append({"$set": {step.column: _mongo_expr(step)}})
    renames = plan.renames()
    if renames:
        pipeline.append({"$set": {new: f"${old}" for old, new in renames.items()}})
        pipeline.append({"$unset": list(renames)})
    pipeline.append({"$merge": {"into": target_collection, "whenMatched": "replace", "whenNotMatched": "insert"}})
    return pipeline
