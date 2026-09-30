import pandas as pd
from langgraph.store.memory import InMemoryStore

from wrangler.datagen import CorruptionProfile, make_dataset
from wrangler.memory import TransformMemory, schema_fingerprint
from wrangler.planning import CleaningPlan, PlanStep
from wrangler.profiling import profile_frame
from wrangler.sandbox import helpers as wr
from wrangler.validation import validate

BEFORE = pd.DataFrame({"price": ["$12.50", "13", "N/A", "$7"] * 25, "note": ["a", "b", "c", "d"] * 25},
                      dtype="string")


def _plan(strategy):
    return CleaningPlan(confidence=0.9, steps=[PlanStep(column="price", action="cast_numeric", strategy=strategy)])


def test_validator_catches_silent_coercion():
    after = BEFORE.copy()
    after["price"] = wr.cast_numeric(BEFORE["price"], strategy=0)  # "$12.50" -> NaN, silently
    v = validate(BEFORE, after, _plan(0))
    assert not v["passed"]
    prob = v["problems"][0]
    assert prob["check"] == "new_nulls" and prob["ratio"] == 0.5
    assert "$12.50" in prob["examples"]  # concrete evidence is fed back to the planner


def test_validator_passes_correct_fix_and_ignores_expected_nulls():
    after = BEFORE.copy()
    after["price"] = wr.cast_numeric(BEFORE["price"], strategy=1)
    assert validate(BEFORE, after, _plan(1))["passed"]  # "N/A" -> null is expected


def test_validator_flags_untouched_changes_and_row_loss():
    after = BEFORE.copy()
    after["price"] = wr.cast_numeric(BEFORE["price"], strategy=1)
    after["note"] = after["note"].str.upper()
    assert any(p["check"] == "untouched_changed" for p in validate(BEFORE, after, _plan(1))["problems"])
    assert validate(BEFORE, after.iloc[:-1], _plan(1))["problems"][0]["check"] == "row_count"


def test_fingerprint_is_stable_across_batches_but_changes_on_drift():
    cp = CorruptionProfile(rate=0.08, date_style="mixed_us")
    a = profile_frame(make_dataset("orders", 2000, 1, cp).dirty)
    b = profile_frame(make_dataset("orders", 2000, 99, cp).dirty)
    assert schema_fingerprint(a) == schema_fingerprint(b)
    drifted = make_dataset("orders", 2000, 99, cp).dirty.assign(coupon_code="X")
    assert schema_fingerprint(profile_frame(drifted)) != schema_fingerprint(a)


def test_memory_versioning_similarity_and_demotion():
    mem = TransformMemory(InMemoryStore())
    prof = profile_frame(BEFORE)
    fp = schema_fingerprint(prof)
    kw = dict(source="csv:prices", fingerprint=fp, profile=prof, plan=_plan(1).model_dump(),
              code="def transform(df):\n    return df\n", dialect_artifacts={}, approved_by="auto", run_id="r1")
    assert mem.save(**kw)["version"] == 1
    assert mem.save(**{**kw, "run_id": "r2"})["version"] == 2
    assert mem.lookup_exact("csv:prices", fp)["history"][0]["run_id"] == "r1"

    similar, score = mem.find_similar(prof, threshold=0.6, exclude=("csv:other", "x"))
    assert similar and score == 1.0

    mem.demote("csv:prices", fp, "failed on new data")
    assert mem.lookup_exact("csv:prices", fp) is None
    assert mem.find_similar(prof, threshold=0.6)[0] is None
