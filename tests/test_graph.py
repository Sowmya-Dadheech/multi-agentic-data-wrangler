"""End-to-end behaviour of the LangGraph workflow."""

import pandas as pd
from langchain_core.messages import AIMessage

from wrangler import Wrangler
from wrangler.datagen import CorruptionProfile, load_into, make_dataset
from wrangler.planning import CleaningPlan, HeuristicPlanner, LLMPlanner, PlanStep
from wrangler.planning.planners import PlanResult


def nodes(result):
    return [n for n, _ in result.updates]


def test_happy_path_writes_clean_output(wrangler, messy_csv):
    ds, spec = messy_csv(date_style="iso")
    r = wrangler.run(spec)
    assert r.status == "succeeded", r.state.get("errors")
    out = pd.read_csv(r.state["output_ref"])
    assert len(out) == len(ds.dirty)
    assert "customer_id" in out.columns  # renamed from customerId
    assert r.state["memory_hit"] == "none"
    assert "guard" in nodes(r) and nodes(r).index("guard") < nodes(r).index("execute")


def test_retry_loop_repairs_day_first_dates(wrangler, messy_csv):
    _, spec = messy_csv(date_style="mixed_eu")
    r = wrangler.run(spec)
    assert r.status == "succeeded"
    assert r.state["attempts"] == 2 and "repair" in nodes(r)
    step = next(s for s in r.state["plan"]["steps"] if s["action"] == "parse_dates")
    assert step["params"]["dayfirst"] is True and step["strategy"] == 2


def test_second_batch_hits_memory_and_skips_planning(wrangler, tmp_path):
    cp = CorruptionProfile(date_style="mixed_us")
    for seed in (1, 2):
        ds = make_dataset("orders", 1500, seed, cp)
        spec = load_into(ds, "sql", "orders", str(tmp_path))
        r = wrangler.run(spec)
        assert r.status == "succeeded"
    assert r.state["memory_hit"] == "exact"
    assert "plan" not in nodes(r) and "codegen" not in nodes(r)
    assert {"guard", "execute", "validate"} <= set(nodes(r))  # safety checks still run
    assert wrangler.memory.list_all()[0]["uses"] == 1


def test_mongo_source_and_cross_source_hint(wrangler, tmp_path):
    cp = CorruptionProfile(date_style="mixed_eu")
    load = lambda kind, seed: load_into(make_dataset("customers", 1500, seed, cp, as_text=kind != "mongo"),
                                        kind, "customers", str(tmp_path), mongo_uri="mongomock://t")
    assert wrangler.run(load("csv", 1)).status == "succeeded"
    r = wrangler.run(load("mongo", 2))
    assert r.status == "succeeded"
    assert r.state["memory_hit"] == "similar"
    assert r.state["attempts"] == 1  # the hint carried dayfirst=True over: no retry needed


def test_value_drift_forces_replanning(wrangler, tmp_path):
    cp = CorruptionProfile(date_style="iso")
    ds = make_dataset("sensors", 1500, 1, cp)
    wrangler.run(load_into(ds, "csv", "sensors", str(tmp_path)))
    ds2 = make_dataset("sensors", 1500, 2, cp)
    ds2.dirty.loc[ds2.dirty.index[:900], "humidity_pct"] = None  # upstream broke the field
    r = wrangler.run(load_into(ds2, "csv", "sensors", str(tmp_path)))
    assert r.state["memory_hit"] == "similar"
    assert any(d["kind"] == "null_jump" for d in r.state["drift"])


class LowConfidencePlanner(HeuristicPlanner):
    def plan(self, *a, **kw):
        res = super().plan(*a, **kw)
        res.plan.confidence = 0.3
        return res


def test_low_confidence_pauses_for_human_then_resumes(settings, executor, messy_csv):
    w = Wrangler(settings, planner=LowConfidencePlanner(), executor=executor, hitl="interrupt")
    _, spec = messy_csv(date_style="iso")
    r = w.run(spec)
    assert r.interrupted and "confidence" in r.interrupt["reason"]
    assert "output_ref" not in r.state  # nothing written before approval
    r = w.resume(r.run_id, {"action": "approve"})
    assert r.status == "succeeded" and r.state["approved_by"] == "human"
    assert w.memory.list_all()[0]["approved_by"] == "human"
    assert len(list(w.history(r.run_id))) > 5  # every step checkpointed


class EvilPlanner(HeuristicPlanner):
    """Simulates a compromised / confused model that keeps emitting dangerous code."""

    def plan(self, profile, issues, sample, target=None, hint=None):
        col = sample.columns[1]
        steps = [PlanStep(column=col, action="custom", params={"code": "import os\nos.system('rm -rf /')"})]
        return PlanResult(CleaningPlan(steps=steps, confidence=0.99))

    def repair(self, plan, problems, sample, profile):
        return PlanResult(plan, note="same thing again")


def test_guard_blocks_and_human_cannot_wave_it_through(settings, executor, messy_csv):
    w = Wrangler(settings, planner=EvilPlanner(), executor=executor, hitl="interrupt")
    _, spec = messy_csv()
    r = w.run(spec)
    assert r.interrupted and "import" in r.interrupt["reason"]
    assert "execute" not in nodes(r)  # never reached the sandbox
    r = w.resume(r.run_id, {"action": "approve"})
    assert r.status == "rejected"
    assert "output_ref" not in r.state


def test_escalate_mode_ends_with_needs_human(wrangler, messy_csv):
    wrangler.planner = EvilPlanner()
    _, spec = messy_csv()
    r = wrangler.run(spec)
    assert r.status == "needs_human" and not r.interrupted


# ---------------------------------------------------------------- LLM planner (fake model)


class FakeStructured:
    def __init__(self, plans):
        self.plans, self.calls = list(plans), []

    def invoke(self, messages):
        self.calls.append(messages)
        raw = AIMessage(content="", usage_metadata={"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500})
        return {"raw": raw, "parsed": self.plans.pop(0), "parsing_error": None}


class FakeChatModel:
    def __init__(self, plans):
        self.structured = FakeStructured(plans)

    def with_structured_output(self, schema, include_raw=False):
        return self.structured


def test_llm_planner_sanitizes_and_counts_tokens(settings, executor, messy_csv):
    ds, spec = messy_csv(date_style="iso")
    plan = CleaningPlan(confidence=0.9, steps=[
        PlanStep(column="age", action="cast_numeric", params={"integer": True}, strategy=1),
        PlanStep(column="plan", action="canonicalize"),
        PlanStep(column="does_not_exist", action="strip"),  # hallucinated column
    ])
    fake = FakeChatModel([plan])
    w = Wrangler(settings, planner=LLMPlanner(fake), executor=executor, hitl="escalate")
    r = w.run(spec)
    assert r.status == "succeeded", r.state.get("errors")
    cols = {s["column"] for s in r.state["plan"]["steps"]}
    assert "does_not_exist" not in cols
    assert next(s for s in r.state["plan"]["steps"] if s["action"] == "canonicalize")["params"]["mapping"]
    assert r.state["usage"] == {"llm_calls": 1, "input_tokens": 1200, "output_tokens": 300}
    payload = fake.structured.calls[0][1][1]
    assert "rule_based_draft" in payload and "LTV_usd" in payload  # compact profile, not raw rows
