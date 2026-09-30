"""Helper semantics, plus: SQL pushdown must produce the same result as the pandas path."""

import pandas as pd
import pytest
import sqlglot

from wrangler.planning import CleaningPlan, PlanStep, to_postgres
from wrangler.sandbox import helpers as wr


def test_naive_parse_silently_loses_data_and_strategy_fixes_it():
    s = pd.Series(["2026-01-02", "01/05/2026", "3 Feb 2026", "Mar 5, 2026"])
    assert wr.parse_dates(s, 0).isna().sum() == 3  # the classic bug
    assert wr.parse_dates(s, 1).notna().all()


def test_dayfirst_from_evidence():
    s = pd.Series(["13/04/2026", "01/05/2026"])
    assert wr.infer_dayfirst(s) is True
    out = wr.parse_dates(s, 2, dayfirst=True)
    assert out.tolist() == [pd.Timestamp("2026-04-13"), pd.Timestamp("2026-05-01")]


def test_cast_numeric_strategies():
    s = pd.Series(["$1,200.50", "42 yrs", "(5)", "abc", None])
    assert wr.cast_numeric(s, strategy=0).isna().all()
    assert wr.cast_numeric(s, strategy=1).tolist()[:3] == [1200.5, 42.0, -5.0]
    assert str(wr.cast_numeric(pd.Series(["1", "2 yrs"]), integer=True, strategy=1).dtype) == "Int64"


def test_canonicalize_and_bool_and_nulls():
    assert wr.canonicalize(pd.Series(["Pro", " pro", "PRO", "Pro"])).tolist() == ["Pro"] * 4
    assert wr.to_bool(pd.Series(["Y", "no", "1", "maybe"])).tolist()[:3] == [True, False, True]
    assert wr.standardize_nulls(pd.Series(["N/A", "x", " null "])).isna().tolist() == [True, False, True]


PLAN = CleaningPlan(confidence=0.9, steps=[
    PlanStep(column="Age", action="standardize_nulls", params={"tokens": ["N/A"]}),
    PlanStep(column="Age", action="cast_numeric", params={"integer": True}, strategy=1),
    PlanStep(column="price", action="cast_numeric", strategy=1),
    PlanStep(column="signup", action="parse_dates", strategy=2, params={"dayfirst": True}),
    PlanStep(column="plan", action="strip"),
    PlanStep(column="plan", action="canonicalize", params={"mapping": {"pro": "Pro", "basic": "Basic"}}),
    PlanStep(column="active", action="to_bool"),
    PlanStep(column="Age", action="rename", params={"to": "age"}),
])

DIRTY = pd.DataFrame({
    "id": ["1", "2", "3", "4", "5"],
    "Age": ["42", " 37 yrs", "N/A", "51", None],
    "price": ["$1,200.50", "13", "7.5", "(2.25)", "-"],
    "signup": ["2026-01-02", "13/04/2026", "3 Feb 2026", "Mar 5, 2026", "2025-12-31T08:00:00"],
    "plan": ["\tpro", "Pro", "BASIC", "basic\n", None],
    "active": ["yes", "N", "1", "false", "?"],
})


def test_postgres_sql_parses():
    sql = to_postgres(PLAN, list(DIRTY.columns), "customers", "clean_customers")
    assert all(sqlglot.parse(sql, read="postgres"))


@pytest.fixture(scope="module")
def pg_engine(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    pytest.importorskip("psycopg2")
    from sqlalchemy import create_engine

    srv = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="stop")
    eng = create_engine(srv.get_uri().replace("postgresql://", "postgresql+psycopg2://"))
    yield eng
    eng.dispose()


def test_pushdown_matches_pandas(pg_engine):
    from sqlalchemy import text

    DIRTY.to_sql("customers", pg_engine, if_exists="replace", index=False)
    sql = to_postgres(PLAN, list(DIRTY.columns), "customers", "clean_customers")
    with pg_engine.begin() as con:
        for stmt in sql.split(";\n"):
            con.execute(text(stmt))
        via_sql = pd.read_sql(text('SELECT * FROM "clean_customers" ORDER BY id'), con).convert_dtypes()

    ns = {"wr": wr, "pd": pd}
    from wrangler.planning import to_pandas

    exec(to_pandas(PLAN), ns)
    via_pandas = ns["transform"](DIRTY.copy())
    for col in via_pandas.columns:
        a = via_pandas[col].astype("string").fillna("<NA>").tolist()
        b = via_sql[col].astype("string").fillna("<NA>").tolist()
        if col == "signup":
            a = [x[:10] for x in a]
            b = [x[:10] for x in b]
        assert a == b, col
