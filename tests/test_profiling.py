import pandas as pd
import pytest

from wrangler.naming import normalize
from wrangler.profiling import (
    detect_drift, detect_issues, infer_mongo_schema, infer_value_type, match_columns,
    profile_frame, shape_conflicts,
)


@pytest.mark.parametrize("value,expected", [
    ("42", "int"), (" 42 ", "int"), ("3.14", "float"), ("$1,200.50", "numeric_text"),
    ("42 yrs", "numeric_text"), ("70kg", "numeric_text"), ("2026-04-01", "date:iso"),
    ("04/01/2026", "date:us"), ("1 Apr 2026", "date:text_dmy"), ("Apr 1, 2026", "date:text_mdy"),
    ("2026-04-01T10:30:00", "datetime:iso"), ("yes", "bool"), ("N/A", "null_token"),
    ("", "null_token"), (None, "null"), ("Seattle", "str"), ("02134", "str"), (7, "int"),
])
def test_infer_value_type(value, expected):
    assert infer_value_type(value) == expected


@pytest.mark.parametrize("raw,norm", [
    ("customerId", "customer_id"), ("Customer-ID", "customer_id"), (" signup date ", "signup_date"),
    ("LTV_usd", "ltv_usd"), ("HTTPStatus", "http_status"),
])
def test_normalize(raw, norm):
    assert normalize(raw) == norm


def _issues(df):
    return {(i["column"], i["issue"]) for i in detect_issues(profile_frame(df))}


def test_detects_the_classic_problems():
    df = pd.DataFrame({
        "Age": ["42", " 37", "42 yrs", "N/A"] * 10,
        "signup": ["2026-01-02", "01/05/2026", "3 Feb 2026", "2026-03-01"] * 10,
        "plan": ["Pro", "pro", "Basic", "BASIC"] * 10,
        "active": ["yes", "no", "1", "0"] * 10,
    }, dtype="string")
    found = _issues(df)
    assert ("Age", "mixed_types") in found
    assert ("Age", "null_tokens") in found
    assert ("Age", "whitespace") in found
    assert ("Age", "naming") in found
    assert ("signup", "date_formats") in found
    assert ("plan", "inconsistent_categories") in found
    assert ("active", "bool_as_text") in found  # 1/0 mixed with yes/no is boolean


def test_clean_frame_has_no_errors():
    df = pd.DataFrame({"id": ["a1", "a2"], "amount": [1.5, 2.0], "note": ["x", "y"]})
    assert not [i for i in detect_issues(profile_frame(df)) if i["severity"] == "error"]


def test_mongo_schema_and_shape_conflicts():
    docs = [
        {"_id": 1, "price": 12.5, "address": {"city": "Seattle"}},
        {"_id": 2, "price": "$12.99", "address": "Mumbai"},
        {"_id": 3, "address": {"city": "Austin"}},
    ]
    schema = infer_mongo_schema(docs)
    assert schema["price"]["types"] == {"float": 1, "str": 1}
    assert schema["price"]["presence"] == pytest.approx(2 / 3, abs=1e-3)
    assert "address.city" in schema
    assert [c["column"] for c in shape_conflicts(schema)] == ["address"]


def test_matching_cascade_levels():
    src = {"customerId": ["C1"], "dob": ["1990-04-12"], "emial": ["a@x.com"], "phone_no": ["206-555-0101"]}
    tgt = {"customer_id": ["C9"], "date_of_birth": ["1985-11-03"], "email": ["b@y.com"],
           "contact_number": ["(206) 555 0199"]}
    matches, unresolved = match_columns(src, tgt)
    levels = {m.source: m.level for m in matches}
    assert levels["customerId"] == "normalized"
    assert levels["dob"] == "synonym"
    assert levels["emial"] in ("fuzzy", "embedding")
    # pure synonyms with different formatting are beyond the cheap levels: that is
    # exactly what the (optional) LLM level is for
    assert unresolved == ["phone_no"]
    matches, unresolved = match_columns(src, tgt, llm_fn=lambda p, t: {"phone_no": "contact_number"})
    assert {m.source: m.level for m in matches}["phone_no"] == "llm" and not unresolved


def test_matching_escalates_to_llm_only_for_leftovers():
    calls = []

    def fake_llm(pending, targets):
        calls.append((pending, targets))
        return {"zz_code": "region"}

    matches, unresolved = match_columns({"id": ["1"], "zz_code": ["WA"]},
                                        {"id": ["2"], "region": ["northwest"]}, llm_fn=fake_llm)
    assert calls == [(["zz_code"], ["region"])]
    assert {m.level for m in matches} == {"exact", "llm"} and not unresolved


def test_drift_detection():
    base = profile_frame(pd.DataFrame({"a": ["1", "2", "3"] * 10, "s": ["x", "y", "x"] * 10}, dtype="string"))
    cur = profile_frame(pd.DataFrame({"a": ["1", None, None] * 10, "s": ["x", "z", "x"] * 10,
                                      "new": ["q"] * 30}, dtype="string"))
    kinds = {(d["column"], d["kind"]) for d in detect_drift(base, cur)}
    assert ("new", "new_column") in kinds
    assert ("a", "null_jump") in kinds
    assert ("s", "new_categories") in kinds
