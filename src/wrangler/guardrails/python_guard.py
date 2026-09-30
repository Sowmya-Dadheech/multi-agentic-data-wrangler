"""Static analysis of generated Python with the ``ast`` module.

This is a *filter*, not a security boundary: Python is too dynamic to prove safe by reading
it. It rejects the obvious and common problems cheaply, with precise error messages that are
fed back to the code generator. The sandbox is the real boundary.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


class GuardError(Exception):
    """Raised when generated code or a query violates a guardrail."""


# names the generated code may reference (anything else is rejected)
ALLOWED_NAMES = {
    "df", "pd", "np", "wr", "transform",
    "len", "str", "int", "float", "bool", "range", "min", "max", "abs", "round",
    "list", "dict", "set", "tuple", "sorted", "enumerate", "zip", "any", "all", "sum",
    "isinstance", "True", "False", "None",
}
BANNED_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "help", "exit", "quit", "memoryview",
}
# attribute names that read/write files, run shells, or evaluate hidden string expressions
BANNED_ATTRS = {
    "to_pickle", "read_pickle", "to_sql", "read_sql", "read_sql_query", "read_sql_table",
    "to_csv", "read_csv", "to_parquet", "read_parquet", "to_json", "read_json", "to_excel",
    "read_excel", "to_hdf", "read_hdf", "to_feather", "read_feather", "to_html", "read_html",
    "to_clipboard", "read_clipboard", "to_stata", "to_gbq", "read_gbq", "load", "save",
    "loadtxt", "savetxt", "fromfile", "tofile", "memmap", "system", "popen", "spawn",
    "eval", "query", "pipe", "ctypeslib", "lib", "io", "os", "sys", "subprocess",
    "builtins", "socket", "shutil", "pathlib", "request", "urlopen",
}


@dataclass
class GuardReport:
    ok: bool
    violations: list[str]


def check_python(code: str, max_len: int = 20_000) -> GuardReport:
    violations: list[str] = []
    if len(code) > max_len:
        return GuardReport(False, [f"code longer than {max_len} characters"])
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return GuardReport(False, [f"syntax error: {e.msg} (line {e.lineno})"])

    local_names = _locally_bound_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append(f"line {node.lineno}: imports are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"line {node.lineno}: global/nonlocal is not allowed")
        elif isinstance(node, (ast.While, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            violations.append(f"line {node.lineno}: {type(node).__name__} is not allowed")
        elif isinstance(node, (ast.ClassDef, ast.Lambda)):
            violations.append(f"line {node.lineno}: {type(node).__name__} is not allowed")
        elif isinstance(node, (ast.Try, ast.With)):
            violations.append(f"line {node.lineno}: {type(node).__name__} blocks are not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                violations.append(f"line {node.lineno}: attribute {node.attr!r} (underscore/dunder) not allowed")
            elif node.attr in BANNED_ATTRS:
                violations.append(f"line {node.lineno}: attribute {node.attr!r} not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                violations.append(f"line {node.lineno}: name {node.id!r} not allowed")
            elif node.id in BANNED_CALLS:
                violations.append(f"line {node.lineno}: {node.id!r} is not allowed")
            elif node.id not in ALLOWED_NAMES and node.id not in local_names:
                violations.append(f"line {node.lineno}: unknown name {node.id!r}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            violations.append(f"line {getattr(node, 'lineno', '?')}: dunder inside a string literal")

    body = tree.body
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != "transform":
        violations.append("code must consist of exactly one function: def transform(df)")
    else:
        fn = body[0]
        if [a.arg for a in fn.args.args] != ["df"] or fn.args.vararg or fn.args.kwarg:
            violations.append("transform must take exactly one argument named df")
        if fn.decorator_list:
            violations.append("decorators are not allowed")

    return GuardReport(not violations, sorted(set(violations)))


def _locally_bound_names(tree: ast.AST) -> set[str]:
    """Names the code itself assigns (loop vars, local variables, comprehension targets)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names
