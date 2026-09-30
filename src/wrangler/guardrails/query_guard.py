"""Guards for generated SQL (parsed with sqlglot) and MongoDB pipelines (operator allowlist).

A real parser beats regex: regex is fooled by comments, casing and string literals.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

from .python_guard import GuardReport

_FORBIDDEN_SQL = (
    exp.Delete, exp.Update, exp.Insert, exp.Alter, exp.Command, exp.Grant,
    exp.TruncateTable, exp.Merge, exp.Copy,
)
_FORBIDDEN_FUNCS = {"pg_read_file", "pg_read_binary_file", "pg_ls_dir", "lo_import", "lo_export",
                    "dblink", "dblink_exec", "pg_sleep", "load_extension", "readfile", "writefile"}


def check_sql(sql: str, *, dialect: str = "postgres", output_prefix: str = "clean_",
              source_tables: set[str] | None = None) -> GuardReport:
    violations: list[str] = []
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        return GuardReport(False, [f"SQL parse error: {str(e).splitlines()[0]}"])
    if not statements:
        return GuardReport(False, ["empty SQL"])

    for stmt in statements:
        for bad in stmt.find_all(*_FORBIDDEN_SQL):
            violations.append(f"forbidden statement: {bad.key.upper()}")
        for fn in stmt.find_all(exp.Anonymous, exp.Func):
            name = (fn.name or "").lower()
            if name in _FORBIDDEN_FUNCS:
                violations.append(f"forbidden function: {name}")

        if isinstance(stmt, exp.Drop):
            tables = [t.name for t in stmt.find_all(exp.Table)]
            if (stmt.args.get("kind") or "").upper() != "TABLE" or not tables or not all(
                t.startswith(output_prefix) for t in tables
            ):
                violations.append(f"DROP is only allowed on {output_prefix}* tables (got {tables})")
        elif isinstance(stmt, exp.Create):
            target = stmt.this.name if stmt.this else ""
            if (stmt.args.get("kind") or "").upper() != "TABLE":
                violations.append("only CREATE TABLE ... AS SELECT is allowed")
            if not target.startswith(output_prefix):
                violations.append(f"CREATE target must start with {output_prefix!r} (got {target!r})")
            if not isinstance(stmt.expression, exp.Select):
                violations.append("CREATE TABLE must be CREATE TABLE ... AS SELECT")
        elif not isinstance(stmt, exp.Select):
            violations.append(f"statement type {stmt.key.upper()} is not allowed")

        if source_tables is not None:
            for tbl in stmt.find_all(exp.Table):
                name = tbl.name
                if name and not name.startswith(output_prefix) and name not in source_tables:
                    violations.append(f"query touches table {name!r} outside this job")

    return GuardReport(not violations, sorted(set(violations)))


ALLOWED_MONGO_STAGES = {"$match", "$set", "$addFields", "$project", "$unset", "$replaceRoot",
                        "$replaceWith", "$unwind", "$group", "$sort", "$limit", "$merge"}
# $where / $function / $accumulator run server-side JavaScript; $out replaces a collection
BANNED_MONGO_OPERATORS = {"$where", "$function", "$accumulator", "$out", "$lookup",
                          "$graphLookup", "$unionWith", "$currentOp", "$listSessions"}


def check_mongo(pipeline: list[dict[str, Any]], *, output_prefix: str = "clean_") -> GuardReport:
    violations: list[str] = []

    def scan(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k in BANNED_MONGO_OPERATORS:
                    violations.append(f"{path}: operator {k} is not allowed")
                scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, x in enumerate(obj):
                scan(x, f"{path}[{i}]")

    if not isinstance(pipeline, list):
        return GuardReport(False, ["pipeline must be a list of stages"])
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict) or len(stage) != 1:
            violations.append(f"stage {i}: each stage must have exactly one operator")
            continue
        (name, body), = stage.items()
        if name not in ALLOWED_MONGO_STAGES:
            violations.append(f"stage {i}: {name} is not allowed")
        if name == "$merge":
            into = body.get("into") if isinstance(body, dict) else body
            target = into.get("coll") if isinstance(into, dict) else into
            if not str(target).startswith(output_prefix):
                violations.append(f"stage {i}: $merge target must start with {output_prefix!r}")
            if i != len(pipeline) - 1:
                violations.append(f"stage {i}: $merge must be the last stage")
        scan(body, f"stage {i}.{name}")

    return GuardReport(not violations, sorted(set(violations)))
