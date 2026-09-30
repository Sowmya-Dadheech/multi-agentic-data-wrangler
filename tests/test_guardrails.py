"""Red-team the static guardrails: every snippet here must be blocked."""

import pytest

from wrangler.guardrails import check_mongo, check_python, check_sql
from wrangler.guardrails.redteam import MALICIOUS_MONGO, MALICIOUS_PYTHON, MALICIOUS_SQL
from wrangler.planning import CleaningPlan, PlanStep, to_mongo, to_pandas, to_postgres


@pytest.mark.parametrize("name", list(MALICIOUS_PYTHON))
def test_python_attacks_blocked(name):
    assert not check_python(MALICIOUS_PYTHON[name]).ok, name


@pytest.mark.parametrize("name", list(MALICIOUS_SQL))
def test_sql_attacks_blocked(name):
    assert not check_sql(MALICIOUS_SQL[name], source_tables={"customers"}).ok, name


@pytest.mark.parametrize("name", list(MALICIOUS_MONGO))
def test_mongo_attacks_blocked(name):
    assert not check_mongo(MALICIOUS_MONGO[name]).ok, name


FULL_PLAN = CleaningPlan(confidence=0.9, steps=[
    PlanStep(column="Age", action="standardize_nulls", params={"tokens": ["N/A"]}),
    PlanStep(column="Age", action="strip"),
    PlanStep(column="Age", action="cast_numeric", params={"integer": True}, strategy=1),
    PlanStep(column="signup", action="parse_dates", strategy=2, params={"dayfirst": True}),
    PlanStep(column="plan", action="canonicalize", params={"mapping": {"pro": "Pro", "it's": "It's"}}),
    PlanStep(column="active", action="to_bool"),
    PlanStep(column="Age", action="rename", params={"to": "age"}),
])


def test_compiled_plans_pass_every_guard():
    assert check_python(to_pandas(FULL_PLAN)).ok
    sql = to_postgres(FULL_PLAN, ["id", "Age", "signup", "plan", "active"], "customers", "clean_customers")
    assert check_sql(sql, source_tables={"customers"}).ok, check_sql(sql).violations
    assert check_mongo(to_mongo(FULL_PLAN, "clean_customers")).ok


def test_benign_custom_code_passes():
    code = ("def transform(df):\n"
            "    df['total'] = df['qty'] * df['price']\n"
            "    for c in ['a', 'b']:\n"
            "        df[c] = df[c].str.upper()\n"
            "    return df\n")
    assert check_python(code).ok
