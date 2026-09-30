import sys

import pandas as pd
import pytest

from wrangler.sandbox import SubprocessExecutor


@pytest.fixture
def io(tmp_path):
    inp = tmp_path / "in.parquet"
    pd.DataFrame({"a": pd.Series(["$1", "2 yrs", None], dtype="string")}).to_parquet(inp)
    return str(inp), str(tmp_path / "out.parquet")


def run(code, io, **kw):
    return SubprocessExecutor(**{"timeout_s": 20, "cpu_s": 5, **kw}).run(code, *io)


def test_runs_helpers(io):
    r = run("def transform(df):\n    df['a'] = wr.cast_numeric(df['a'], strategy=1)\n    return df\n", io)
    assert r.ok, r.error
    assert pd.read_parquet(r.output_path)["a"].tolist()[:2] == [1.0, 2.0]


def test_cpu_limit_kills_runaway_code(io):
    code = "def transform(df):\n    x = 0\n    for i in range(10**12):\n        x += i\n    return df\n"
    r = run(code, io, cpu_s=2, timeout_s=15)
    assert not r.ok and ("signal" in r.error or "timed out" in r.error)


def test_imports_unavailable_even_if_guard_is_bypassed(io):
    r = run("def transform(df):\n    import os\n    return df\n", io)
    assert not r.ok and "__import__" in r.error


def test_network_is_disabled(io):
    # bypasses the AST guard on purpose: this checks the sandbox itself (public pandas API,
    # so it doesn't depend on pandas internals that change between versions)
    code = "def transform(df):\n    wr.pd.read_csv('http://example.com/data.csv')\n    return df\n"
    r = run(code, io)
    assert not r.ok
    assert "network access is disabled" in r.error or "PermissionError" in r.error


def test_no_inherited_secrets(io, monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    code = "def transform(df):\n    df['env'] = str(len(wr.pd.io.common.os.environ.get('DB_PASSWORD', '')))\n    return df\n"
    r = run(code, io)
    assert r.ok, r.error
    assert set(pd.read_parquet(r.output_path)["env"]) == {"0"}


def test_must_return_dataframe(io):
    r = run("def transform(df):\n    return 5\n", io)
    assert not r.ok and "DataFrame" in r.error


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS is not enforced on macOS")
def test_memory_limit(io):
    code = "def transform(df):\n    x = [0] * (10**10)\n    return df\n"
    r = run(code, io, mem_mb=512)
    assert not r.ok
