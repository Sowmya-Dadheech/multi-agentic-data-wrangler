"""The LangGraph workflow.

    ingest -> profile -> memory_lookup -+-(exact hit)-----------------------> guard
                                        +-(miss / similar)-> plan -> codegen -> guard
    guard   -> execute (sandbox, sample) -> validate
    validate -(pass)-> [human_review if low confidence] -> apply (full data) -> save -> END
    any failure -> repair (planner gets the concrete errors) -> codegen   (bounded retries)
    retries exhausted -> human_review (interrupt) -> approve / edit / reject

Routing is deterministic Python, not an LLM: "code must always pass the guard before it
runs" is guaranteed by the graph's edges, not by an agent remembering to do it.
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from . import connectors as cx
from .config import Settings
from .connectors.base import SourceRef
from .guardrails import check_mongo, check_python, check_sql
from .memory import TransformMemory, schema_fingerprint
from .planning import CleaningPlan, UnsupportedForDialect, to_mongo, to_pandas, to_postgres
from .planning.planners import Planner
from .profiling import detect_drift, detect_issues, infer_mongo_schema, profile_frame, shape_conflicts
from .validation import validate


# ============================================================================ state


def _merge_usage(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


class WrangleState(TypedDict, total=False):
    run_id: str
    source_id: str  # only an id: connection strings never enter checkpointed state
    target_schema: dict | None
    # ingestion
    sample_path: str
    row_count: int
    declared_schema: dict
    shape_issues: list[dict]
    # profiling
    profile: dict
    issues: list[dict]
    fingerprint: str
    drift: list[dict]
    # memory
    memory_hit: Literal["exact", "similar", "none"]
    memory_similarity: float
    hint: dict | None
    # planning / codegen
    plan: dict
    plan_source: Literal["planner", "memory", "human"]
    code: str
    artifacts: dict
    attempts: int
    exhausted: bool
    # execution / validation
    guard: dict
    exec: dict
    validation: dict
    full_validation: dict
    # outcome
    approved_by: str
    output_ref: str
    apply_mode: str
    status: Literal["running", "succeeded", "needs_human", "rejected", "failed"]
    # append-only channels (reducers merge updates instead of overwriting)
    errors: Annotated[list[str], operator.add]
    events: Annotated[list[dict], operator.add]
    usage: Annotated[dict, _merge_usage]


@dataclass
class Context:
    """Per-invocation dependencies (not checkpointed)."""

    settings: Settings
    sources: dict[str, SourceRef]
    planner: Planner
    executor: Any
    hitl: Literal["interrupt", "escalate"] = "interrupt"
    extra: dict = field(default_factory=dict)


def _ev(node: str, msg: str, **data: Any) -> list[dict]:
    return [{"node": node, "msg": msg, "t": round(time.time(), 3), **data}]


def _run_dir(ctx: Context, state: WrangleState) -> Path:
    d = ctx.settings.runs_dir / state["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _memory(runtime: Runtime[Context]) -> TransformMemory:
    return TransformMemory(runtime.store)


# ============================================================================ nodes


def ingest_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    source = ctx.sources[state["source_id"]]
    res = cx.ingest(source, ctx.settings.sample_size)
    path = _run_dir(ctx, state) / "sample.parquet"
    res.sample.to_parquet(path, index=False)
    shape = shape_conflicts(infer_mongo_schema(res.raw_docs)) if res.raw_docs is not None else []
    return {
        "sample_path": str(path),
        "row_count": res.row_count,
        "declared_schema": res.declared_schema,
        "shape_issues": shape,
        "status": "running",
        "attempts": 0,
        "events": _ev("ingest", f"sampled {len(res.sample):,} of {res.row_count:,} rows "
                      f"from {source.kind}:{source.name}", rows=len(res.sample)),
    }


def profile_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    sample = pd.read_parquet(state["sample_path"])
    profile = profile_frame(sample)
    issues = detect_issues(profile, state.get("target_schema"), state.get("declared_schema"))
    issues += state.get("shape_issues", [])
    fp = schema_fingerprint(profile)
    source = ctx.sources[state["source_id"]]
    baseline = _memory(runtime).baseline(source.id)
    drift = detect_drift(baseline, profile, ctx.settings.null_jump_threshold) if baseline else []
    n_err = sum(i["severity"] == "error" for i in issues)
    return {
        "profile": profile,
        "issues": issues,
        "fingerprint": fp,
        "drift": drift,
        "events": _ev("profile", f"{len(profile)} columns, {n_err} issues, {len(issues) - n_err} warnings"
                      + (f", {len(drift)} drift signals" if drift else ""), fingerprint=fp),
    }


def memory_lookup_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    mem = _memory(runtime)
    source = ctx.sources[state["source_id"]]
    hit = mem.lookup_exact(source.id, state["fingerprint"])
    value_drift = [d for d in state.get("drift", []) if not d["structural"]]
    if hit and not value_drift:
        return {
            "memory_hit": "exact", "memory_similarity": 1.0, "hint": None,
            "plan": hit["plan"], "plan_source": "memory", "code": hit["code"],
            "artifacts": hit.get("artifacts", {}), "attempts": 1,
            "events": _ev("memory", f"exact hit (v{hit['version']}, approved by {hit['approved_by']}) "
                          "-> reusing fix, skipping the LLM; safety checks still run"),
        }
    if hit and value_drift:
        kinds = sorted({d["kind"] for d in value_drift})
        return {
            "memory_hit": "similar", "memory_similarity": 1.0, "hint": hit, "plan_source": "planner",
            "events": _ev("memory", f"schema matches but data drifted ({', '.join(kinds)}) "
                          "-> re-planning with the stored fix as a hint"),
        }
    similar, score = mem.find_similar(state["profile"], ctx.settings.similar_threshold,
                                      exclude=(source.id, state["fingerprint"]))
    if similar:
        return {
            "memory_hit": "similar", "memory_similarity": score, "hint": similar, "plan_source": "planner",
            "events": _ev("memory", f"similar schema from {similar['source']} (similarity {score:.2f}) "
                          "-> passing its fix to the planner as a hint"),
        }
    return {"memory_hit": "none", "memory_similarity": score, "hint": None, "plan_source": "planner",
            "events": _ev("memory", "no stored fix for this schema -> planning from scratch")}


def plan_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    sample = pd.read_parquet(state["sample_path"])
    res = ctx.planner.plan(state["profile"], state["issues"], sample,
                           state.get("target_schema"), state.get("hint"))
    msg = f"{ctx.planner.name} planner: {len(res.plan.steps)} steps, confidence {res.plan.confidence:.2f}"
    return {
        "plan": res.plan.model_dump(), "plan_source": "planner", "exhausted": False,
        "usage": res.usage, "events": _ev("plan", msg + (f" ({res.note})" if res.note else "")),
    }


def codegen_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    source = ctx.sources[state["source_id"]]
    plan = CleaningPlan.model_validate(state["plan"])
    code = to_pandas(plan)
    artifacts: dict[str, Any] = {}
    sample_cols = list(pd.read_parquet(state["sample_path"]).columns)
    target = cx.registry.output_name(source, ctx.settings.output_prefix)
    try:
        if source.kind == "sql":
            artifacts["postgres_sql"] = to_postgres(plan, sample_cols, source.name, target)
        elif source.kind == "mongo":
            artifacts["mongo_pipeline"] = to_mongo(plan, target)
    except UnsupportedForDialect as e:
        artifacts["pushdown_unsupported"] = str(e)
    attempts = state.get("attempts", 0) + 1
    return {"code": code, "artifacts": artifacts, "attempts": attempts,
            "validation": {}, "full_validation": {}, "guard": {}, "exec": {},
            "events": _ev("codegen", f"attempt {attempts}: compiled plan to pandas"
                          + (" + " + ", ".join(k for k in artifacts if k != "pushdown_unsupported")
                             if artifacts else ""))}


def guard_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    source = ctx.sources[state["source_id"]]
    violations = list(check_python(state["code"]).violations)
    arts = state.get("artifacts", {})
    if "postgres_sql" in arts:
        violations += check_sql(arts["postgres_sql"], output_prefix=ctx.settings.output_prefix,
                                source_tables={source.name}).violations
    if "mongo_pipeline" in arts:
        violations += check_mongo(arts["mongo_pipeline"], output_prefix=ctx.settings.output_prefix).violations
    ok = not violations
    return {
        "guard": {"ok": ok, "violations": violations},
        "errors": [f"guard: {v}" for v in violations],
        "events": _ev("guard", "AST + query checks passed" if ok else f"BLOCKED: {violations[:3]}"),
    }


def execute_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    out = _run_dir(ctx, state) / f"sample_clean_{state['attempts']}.parquet"
    res = ctx.executor.run(state["code"], state["sample_path"], str(out))
    if res.ok:
        return {"exec": {"ok": True, "output_path": res.output_path, **res.stats},
                "events": _ev("sandbox", f"ran on sample in isolated {ctx.executor.name} "
                              f"({res.stats.get('seconds', 0):.3f}s)")}
    return {"exec": {"ok": False, "error": res.error}, "errors": [f"sandbox: {res.error}"],
            "events": _ev("sandbox", f"execution failed: {str(res.error)[:160]}")}


def validate_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    before = pd.read_parquet(state["sample_path"])
    after = pd.read_parquet(state["exec"]["output_path"])
    plan = CleaningPlan.model_validate(state["plan"])
    v = validate(before, after, plan, ctx.settings.max_new_null_ratio)
    update: dict[str, Any] = {"validation": v}
    if v["passed"]:
        update["events"] = _ev("validate", "sample checks passed (row count, new nulls, dtypes, untouched columns)")
    else:
        update["errors"] = [p["message"] for p in v["problems"]]
        update["events"] = _ev("validate", f"FAILED: {v['problems'][0]['message'][:160]}")
        if state.get("plan_source") == "memory":
            source = ctx.sources[state["source_id"]]
            _memory(runtime).demote(source.id, state["fingerprint"], v["problems"][0]["message"])
            update["events"] += _ev("memory", "cached fix failed on new data -> demoted, re-planning")
            update["plan_source"] = "planner"
    return update


def repair_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    sample = pd.read_parquet(state["sample_path"])
    plan = CleaningPlan.model_validate(state["plan"])
    problems = _current_problems(state)
    res = ctx.planner.repair(plan, problems, sample, state["profile"])
    return {"plan": res.plan.model_dump(), "plan_source": "planner", "exhausted": res.exhausted,
            "usage": res.usage, "events": _ev("repair", res.note or "revised plan")}


def human_review_node(state: WrangleState, runtime: Runtime[Context]) -> Command:
    ctx = runtime.context
    reason = _review_reason(state, ctx.settings)
    if ctx.hitl == "escalate":
        return Command(goto=END, update={"status": "needs_human",
                                         "events": _ev("human", f"escalated to a human: {reason}")})
    decision = interrupt({
        "reason": reason,
        "plan": state.get("plan"),
        "code": state.get("code"),
        "problems": _current_problems(state),
        "confidence": (state.get("plan") or {}).get("confidence"),
    })
    action = (decision or {}).get("action", "reject")
    if action == "approve":
        if not (state.get("guard") or {}).get("ok") or not (state.get("exec") or {}).get("ok"):
            # a human can accept imperfect *results*, but can never wave through code that
            # failed the guardrails or crashed in the sandbox
            return Command(goto=END, update={"status": "rejected", "events": _ev(
                "human", "cannot approve code that failed guardrails or the sandbox; edit the plan instead")})
        return Command(goto="apply", update={"approved_by": "human",
                                             "events": _ev("human", "approved by reviewer")})
    if action == "edit" and decision.get("plan"):
        return Command(goto="codegen", update={
            "plan": decision["plan"], "plan_source": "human", "attempts": 0, "exhausted": False,
            "approved_by": "human", "events": _ev("human", "reviewer edited the plan -> re-running checks")})
    return Command(goto=END, update={"status": "rejected", "events": _ev("human", "rejected by reviewer")})


def apply_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    """Apply the validated fix to the FULL dataset, writing to a new clean_* table."""
    ctx = runtime.context
    source = ctx.sources[state["source_id"]]
    target = cx.registry.output_name(source, ctx.settings.output_prefix)
    plan = CleaningPlan.model_validate(state["plan"])
    before = cx.load_full(source)
    arts = state.get("artifacts", {})

    pushdown = (source.kind == "sql" and "postgres_sql" in arts
                and cx.registry.sql_engine(source.staging_uri or source.uri).dialect.name == "postgresql")
    if pushdown:
        cx.run_sql(source, arts["postgres_sql"])
        after = cx.registry.read_table(source, target)
        eng = cx.registry.sql_engine(source.staging_uri or source.uri)
        mode, ref = "sql pushdown", f"{eng.url.render_as_string(hide_password=True)}::{target}"
    else:
        full_in = _run_dir(ctx, state) / "full.parquet"
        before.to_parquet(full_in, index=False)
        full_out = _run_dir(ctx, state) / "full_clean.parquet"
        res = ctx.executor.run(state["code"], str(full_in), str(full_out))
        if not res.ok:
            return {"full_validation": {"passed": False, "problems": [
                        {"column": None, "check": "runtime", "message": f"full run failed: {res.error}"}]},
                    "errors": [f"full run: {res.error}"],
                    "events": _ev("apply", f"full-data run failed: {str(res.error)[:120]}")}
        after = pd.read_parquet(full_out)
        mode, ref = f"sandbox ({ctx.executor.name})", None

    v = validate(before, after, plan, ctx.settings.max_new_null_ratio)
    if not v["passed"] and state.get("approved_by") != "human":
        return {"full_validation": v, "errors": [p["message"] for p in v["problems"]],
                "events": _ev("apply", f"full-data validation FAILED: {v['problems'][0]['message'][:140]}")}
    if ref is None:
        ref = cx.write_frame(source, after, target, ctx.settings.output_prefix)
    return {"full_validation": v, "output_ref": ref, "apply_mode": mode,
            "events": _ev("apply", f"cleaned {len(after):,} rows via {mode} -> {ref}")}


def save_node(state: WrangleState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    mem = _memory(runtime)
    source = ctx.sources[state["source_id"]]
    if state.get("plan_source") == "memory":
        mem.record_use(source.id, state["fingerprint"], state["profile"])
        msg = "reused stored fix (use count incremented)"
        approved = "memory"
    else:
        approved = state.get("approved_by") or "auto"
        saved = mem.save(source=source.id, fingerprint=state["fingerprint"], profile=state["profile"],
                         plan=state["plan"], code=state["code"], dialect_artifacts=state.get("artifacts", {}),
                         approved_by=approved, run_id=state["run_id"])
        msg = f"saved fix to long-term memory as v{saved['version']} (key {state['fingerprint']})"
    return {"status": "succeeded", "approved_by": approved, "events": _ev("save", msg)}


# ============================================================================ routing


def _current_problems(state: WrangleState) -> list[dict]:
    g = state.get("guard") or {}
    if g and not g.get("ok", True):
        return [{"column": None, "check": "guard", "message": v} for v in g["violations"]]
    ex = state.get("exec") or {}
    if ex and not ex.get("ok", True):
        return [{"column": None, "check": "runtime", "message": ex.get("error", "")}]
    v = state.get("validation") or {}
    if v and not v.get("passed", True):
        return v.get("problems", [])
    fv = state.get("full_validation") or {}
    if fv and not fv.get("passed", True):
        return fv.get("problems", [])
    return []


def _review_reason(state: WrangleState, settings: Settings) -> str:
    if _current_problems(state):
        return f"checks still failing after {state.get('attempts', 0)} attempt(s): " + \
               _current_problems(state)[0]["message"][:160]
    conf = (state.get("plan") or {}).get("confidence", 1)
    return f"planner confidence {conf:.2f} below {settings.min_confidence}"


def _retry_or_human(state: WrangleState, runtime: Runtime[Context]) -> str:
    return "repair" if state.get("attempts", 0) < runtime.context.settings.max_attempts else "human_review"


def route_after_memory(state: WrangleState) -> str:
    return "guard" if state["memory_hit"] == "exact" else "plan"


def route_after_guard(state: WrangleState, runtime: Runtime[Context]) -> str:
    return "execute" if state["guard"]["ok"] else _retry_or_human(state, runtime)


def route_after_execute(state: WrangleState, runtime: Runtime[Context]) -> str:
    return "validate" if state["exec"]["ok"] else _retry_or_human(state, runtime)


def route_after_validate(state: WrangleState, runtime: Runtime[Context]) -> str:
    if not state["validation"]["passed"]:
        return _retry_or_human(state, runtime)
    conf = (state.get("plan") or {}).get("confidence", 1.0)
    if (state.get("plan_source") != "memory" and state.get("approved_by") != "human"
            and conf < runtime.context.settings.min_confidence):
        return "human_review"
    return "apply"


def route_after_repair(state: WrangleState) -> str:
    return "human_review" if state.get("exhausted") else "codegen"


def route_after_apply(state: WrangleState, runtime: Runtime[Context]) -> str:
    if state.get("output_ref"):
        return "save"
    return _retry_or_human(state, runtime)


# ============================================================================ build


def build_graph() -> StateGraph:
    g = StateGraph(WrangleState, context_schema=Context)
    g.add_node("ingest", ingest_node)
    g.add_node("profile", profile_node)
    g.add_node("memory_lookup", memory_lookup_node)
    g.add_node("plan", plan_node)
    g.add_node("codegen", codegen_node)
    g.add_node("guard", guard_node)
    g.add_node("execute", execute_node)
    g.add_node("validate", validate_node)
    g.add_node("repair", repair_node)
    g.add_node("human_review", human_review_node, destinations=("apply", "codegen", END))
    g.add_node("apply", apply_node)
    g.add_node("save", save_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "profile")
    g.add_edge("profile", "memory_lookup")
    g.add_conditional_edges("memory_lookup", route_after_memory, ["guard", "plan"])
    g.add_edge("plan", "codegen")
    g.add_edge("codegen", "guard")
    g.add_conditional_edges("guard", route_after_guard, ["execute", "repair", "human_review"])
    g.add_conditional_edges("execute", route_after_execute, ["validate", "repair", "human_review"])
    g.add_conditional_edges("validate", route_after_validate, ["apply", "repair", "human_review"])
    g.add_conditional_edges("repair", route_after_repair, ["codegen", "human_review"])
    g.add_conditional_edges("apply", route_after_apply, ["save", "repair", "human_review"])
    g.add_edge("save", END)
    return g


def new_run_id(source_id: str) -> str:
    return f"{source_id.replace(':', '-')}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
