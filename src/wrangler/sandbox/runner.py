"""Entry point executed INSIDE the sandbox process / container.

    python -I runner.py <helpers.py> <code.py> <input.parquet> <output.parquet>

It never imports the rest of the package: the sandbox only needs pandas, numpy and the
vetted helpers file. Untrusted code runs with a restricted ``__builtins__`` and with
networking disabled at the socket level (a belt-and-braces layer on top of process /
container isolation).
"""

import importlib.util
import json
import sys
import time


def _disable_network() -> None:
    import socket

    def _blocked(*_a, **_k):
        raise PermissionError("network access is disabled in the sandbox")

    # Must stay a *class*: modules imported later (e.g. ssl) subclass socket.socket, and
    # replacing it with a plain function makes those imports crash with a confusing error.
    class BlockedSocket(socket.socket):
        def __init__(self, *_a, **_k):
            _blocked()

    socket.socket = BlockedSocket  # type: ignore[misc]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]


SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "len", "str", "int", "float", "bool", "range", "min", "max", "abs", "round", "list",
        "dict", "set", "tuple", "sorted", "enumerate", "zip", "any", "all", "sum", "isinstance",
        "ValueError", "TypeError", "KeyError", "Exception", "True", "False", "None",
    )
    if (name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name))
}


def main() -> int:
    helpers_path, code_path, in_path, out_path = sys.argv[1:5]
    import numpy as np
    import pandas as pd

    spec = importlib.util.spec_from_file_location("wr", helpers_path)
    wr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wr)

    df = pd.read_parquet(in_path)  # read BEFORE untrusted code runs
    with open(code_path) as fh:
        code = fh.read()

    _disable_network()
    namespace = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "wr": wr}
    t0 = time.perf_counter()
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - this IS the sandbox
    out = namespace["transform"](df.copy())
    elapsed = time.perf_counter() - t0
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"transform must return a DataFrame, got {type(out).__name__}")
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(out_path, index=False)
    print(json.dumps({"rows_in": len(df), "rows_out": len(out), "cols_out": list(out.columns),
                      "seconds": round(elapsed, 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
