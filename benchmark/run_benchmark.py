"""Reproducible benchmark with ground truth.

    python benchmark/run_benchmark.py                 # offline, heuristic planner (default)
    python benchmark/run_benchmark.py --llm anthropic  # same benchmark with a real model

For each synthetic dataset we know every injected problem and every correct value, so the
numbers below are measured, not estimated (except "est. prompt tokens" in offline mode,
which is the size of the prompt the LLM planner *would* have been sent).

Passes
  1. first contact  - fresh memory, every dataset is new
  2. repeat batch   - a new batch of each source with the same schema and quirks
  3. drift          - a third of the sources change upstream (new column, format change,
                      broken field); memory must notice instead of blindly reusing fixes
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrangler import Settings, Wrangler  # noqa: E402
from wrangler.connectors.registry import mongo_client, sql_engine  # noqa: E402
from wrangler.datagen import (  # noqa: E402
    DOMAINS, CorruptionProfile, MessyDataset, corrupt, generate_clean, load_into, target_schema,
)
from wrangler.guardrails import check_mongo, check_python, check_sql  # noqa: E402
from wrangler.guardrails.redteam import MALICIOUS_MONGO, MALICIOUS_PYTHON, MALICIOUS_SQL  # noqa: E402
from wrangler.planning import CleaningPlan, make_planner  # noqa: E402
from wrangler.planning.planners import estimate_prompt_tokens  # noqa: E402
from wrangler.sandbox import SubprocessExecutor  # noqa: E402

ISSUE_KIND = {
    "null_tokens": "null_tokens", "whitespace": "whitespace", "mixed_types": "numeric",
    "numeric_as_text": "numeric", "date_formats": "dates", "date_as_text": "dates",
    "bool_as_text": "bool", "inconsistent_categories": "categories", "naming": "naming",
}


# ============================================================================ helpers


def norm_value(v, kind: str):
    """Canonical comparable form of a value, whatever the storage gave back."""
    if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        if kind in ("int", "float"):
            f = float(v)
            return None if np.isnan(f) else round(f, 6)
        if kind == "date":
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        if kind == "datetime":
            return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
        if kind == "bool":
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, float, np.integer, np.floating)) and float(v) in (0.0, 1.0):
                return bool(v)  # SQLite stores booleans as 0 / 1
            s = str(v).strip().lower()
            return True if s in ("true", "1") else False if s in ("false", "0") else f"?{s}"
    except (ValueError, TypeError, OverflowError):
        return f"?{v}"
    return str(v)


def read_output(ref: str, kind: str) -> pd.DataFrame:
    if kind == "csv":
        return pd.read_csv(ref, dtype=str, keep_default_na=False, na_values=[""])
    if kind in ("sql", "pg"):
        uri, table = ref.rsplit("::", 1)
        with sql_engine_for(uri).connect() as con:
            from sqlalchemy import text

            order = " ORDER BY ctid" if kind == "pg" else ""
            return pd.read_sql(text(f'SELECT * FROM "{table}"{order}'), con)
    if kind == "mongo":
        db, coll = ref.split(".", 1)
        docs = list(mongo_client("mongomock://bench")[db][coll].find({}, {"_id": 0}))
        return pd.json_normalize(docs)
    raise ValueError(kind)


_ENGINES: dict = {}


def sql_engine_for(rendered_uri: str):
    return _ENGINES[rendered_uri]


def score_quality(ds: MessyDataset, out: pd.DataFrame, plan: dict) -> dict:
    """Cell-level repair accuracy and collateral damage against ground truth."""
    renames = {s["column"]: s["params"]["to"] for s in plan.get("steps", []) if s["action"] == "rename"}
    dirty_to_target = {v: k for k, v in ds.rename_map.items()}
    rep_ok = rep_n = col_bad = col_n = garb_nulled = garb_n = 0
    mapped = mapped_ok = 0
    per_kind: dict[str, list[int]] = {}
    specs = {c.name: c for c in DOMAINS[ds.domain]}
    # align output rows to ground truth by the primary key (databases don't promise order)
    key = DOMAINS[ds.domain][0].name
    key_out = renames.get(ds.rename_map.get(key, key), ds.rename_map.get(key, key))
    if key_out in out.columns and out[key_out].astype(str).str.strip().is_unique:
        out = out.set_index(out[key_out].astype(str).str.strip())
        out = out.reindex(ds.clean[key].astype(str)).reset_index(drop=True)
    for dirty_col in ds.dirty.columns:
        tgt = dirty_to_target.get(dirty_col, dirty_col)
        if tgt not in specs:
            continue  # e.g. a column added by drift: no ground truth
        out_col = renames.get(dirty_col, dirty_col)
        if tgt != dirty_col:
            mapped += 1
            mapped_ok += out_col == tgt
        if out_col not in out.columns:
            continue
        kind = specs[tgt].kind
        k = "int" if kind == "int" else "float" if kind == "float" else kind
        truth = ds.clean[tgt].reset_index(drop=True)
        got = out[out_col].reset_index(drop=True)
        mask = ds.dirty_mask[tgt].reset_index(drop=True)
        garbage = ds.garbage_mask[tgt].reset_index(drop=True)
        for i in range(len(truth)):
            t = norm_value(truth.iat[i], k)
            g = norm_value(got.iat[i], k)
            if garbage.iat[i]:
                garb_n += 1
                garb_nulled += g is None
            elif mask.iat[i]:
                rep_n += 1
                rep_ok += g == t
                per_kind.setdefault(kind, [0, 0])
                per_kind[kind][0] += g == t
                per_kind[kind][1] += 1
            else:
                col_n += 1
                col_bad += g != t
    return {
        "dirty_cells": rep_n, "repaired": rep_ok, "clean_cells": col_n, "collateral": col_bad,
        "garbage_cells": garb_n, "garbage_nulled": garb_nulled,
        "renames_needed": mapped, "renames_correct": mapped_ok, "per_kind": per_kind,
    }


def score_detection(ds: MessyDataset, issues: list[dict]) -> tuple[int, int, int]:
    dirty_to_target = {v: k for k, v in ds.rename_map.items()}
    detected = {(dirty_to_target.get(i["column"], i["column"]), ISSUE_KIND[i["issue"]])
                for i in issues if i["issue"] in ISSUE_KIND}
    truth = set(ds.injected)
    tp = len(detected & truth)
    return tp, len(detected), len(truth)


# ============================================================================ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=6, help="datasets per domain (6 domains)")
    ap.add_argument("--llm", default="heuristic")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "benchmark" / "results"))
    ap.add_argument("--no-postgres", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    work = out_dir / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)

    pg_uri = None
    if not args.no_postgres:
        try:
            import pgserver

            srv = pgserver.get_server(str(work / "pg"), cleanup_mode="stop")
            pg_uri = srv.get_uri().replace("postgresql://", "postgresql+psycopg2://")
        except Exception as e:  # pragma: no cover - optional
            print(f"(postgres unavailable: {e}; using csv/sqlite/mongo only)")
    kinds = ["csv", "sql", "mongo"] + (["pg"] if pg_uri else [])
    sqlite_uri = f"sqlite:///{work / 'bench.db'}"
    for uri in [sqlite_uri] + ([pg_uri] if pg_uri else []):
        eng = sql_engine(uri)
        _ENGINES[eng.url.render_as_string(hide_password=True)] = eng

    settings = Settings(home=work / ".wrangler", sample_size=5000, max_attempts=3)
    planner = make_planner(args.llm)
    w = Wrangler(settings, planner=planner, executor=SubprocessExecutor(timeout_s=120, cpu_s=60),
                 hitl="interrupt")

    # ---------------------------------------------------------------- dataset specs
    specs = []
    for domain in DOMAINS:
        for j in range(args.per_domain):
            specs.append({
                "name": f"{domain}_{j}", "domain": domain, "kind": kinds[len(specs) % len(kinds)],
                "rows": int(rng.choice([2000, 5000, 8000, 12000])),
                "cp": CorruptionProfile.random(rng), "seed": int(rng.integers(1, 10**6)),
            })
    drift_names = {s["name"] for s in specs[::3]}

    def make(spec: dict, seed: int, cp: CorruptionProfile) -> MessyDataset:
        clean = generate_clean(spec["domain"], spec["rows"], seed)
        return corrupt(clean, spec["domain"], seed + 7, cp, as_text=spec["kind"] != "mongo")

    def run_one(spec: dict, ds: MessyDataset, pass_name: str, drift: str | None = None) -> dict:
        kind = "sql" if spec["kind"] == "pg" else spec["kind"]
        src = load_into(ds, kind, spec["name"], str(work), mongo_uri="mongomock://bench",
                        sql_uri=pg_uri if spec["kind"] == "pg" else sqlite_uri)
        t0 = time.perf_counter()
        r = w.run(src, target_schema(spec["domain"], ds.clean))
        reviewed = r.interrupted
        review_reason = (r.interrupt or {}).get("reason", "")
        if r.interrupted:
            # simulated reviewer: approves whatever passed the guard + sandbox. The approval
            # is stored in memory, so the next batch of this source runs unattended.
            r = w.resume(r.run_id, {"action": "approve"})
        secs = time.perf_counter() - t0
        st_ = r.state
        planner_calls = sum(1 for n, _ in r.updates if n in ("plan", "repair"))
        est_tokens = 0
        if planner_calls and st_.get("profile"):
            est_tokens = estimate_prompt_tokens(st_["profile"], st_["issues"],
                                                CleaningPlan.model_validate(st_["plan"])) * planner_calls
        row = {
            "pass": pass_name, "dataset": spec["name"], "domain": spec["domain"], "source": spec["kind"],
            "rows": spec["rows"], "drift": drift, "status": r.status, "attempts": st_.get("attempts", 0),
            "human_review": reviewed, "review_reason": review_reason[:80],
            "memory_hit": st_.get("memory_hit"), "planner_calls": planner_calls,
            "llm_calls": (st_.get("usage") or {}).get("llm_calls", 0),
            "tokens": (st_.get("usage") or {}).get("input_tokens", 0) + (st_.get("usage") or {}).get("output_tokens", 0),
            "est_prompt_tokens": est_tokens, "seconds": round(secs, 3),
            "apply_mode": st_.get("apply_mode"),
            "demoted": any("demoted" in e["msg"] for e in st_.get("events", [])),
        }
        tp, det, truth = score_detection(ds, st_.get("issues", []))
        row.update({"det_tp": tp, "det_found": det, "det_truth": truth})
        if r.status == "succeeded":
            row.update(score_quality(ds, read_output(st_["output_ref"], spec["kind"]), st_["plan"]))
        print(f"  [{pass_name:<6}] {spec['name']:<14} {spec['kind']:<5} {r.status:<11} "
              f"mem={row['memory_hit']:<7} attempts={row['attempts']} {secs:5.1f}s"
              + ("  [human review]" if reviewed else "") + (f"  drift={drift}" if drift else ""))
        return row

    rows = []
    print(f"\nPass 1 - first contact ({len(specs)} datasets)")
    for s in specs:
        rows.append(run_one(s, make(s, s["seed"], s["cp"]), "first"))
    print("\nPass 2 - repeat batch (same sources, new rows)")
    for s in specs:
        rows.append(run_one(s, make(s, s["seed"] + 1000, s["cp"]), "repeat"))
    print(f"\nPass 3 - drift on {len(drift_names)} sources")
    drift_kinds = ["new_column", "format_change", "broken_field"]
    for i, s in enumerate(x for x in specs if x["name"] in drift_names):
        kind = drift_kinds[i % 3]
        cp = s["cp"]
        if kind == "format_change":
            cp = CorruptionProfile(**{**cp.__dict__, "date_style": "mixed_eu", "numeric_text": True})
        ds = make(s, s["seed"] + 2000, cp)
        if kind == "new_column":
            ds.dirty["loyalty_tier"] = pd.Series(np.random.default_rng(1).choice(["gold", "silver"], len(ds.dirty)))
        elif kind == "broken_field":
            victim = ds.dirty.columns[2]
            ds.dirty.loc[ds.dirty.index[: int(len(ds.dirty) * 0.5)], victim] = None
            ds.clean.loc[ds.clean.index[: int(len(ds.clean) * 0.5)], ds.clean.columns[2]] = None
            ds.dirty_mask.loc[ds.dirty_mask.index[: int(len(ds.dirty) * 0.5)], ds.dirty_mask.columns[2]] = False
            ds.garbage_mask.loc[ds.garbage_mask.index[: int(len(ds.dirty) * 0.5)], ds.garbage_mask.columns[2]] = False
        rows.append(run_one(s, ds, "drift", kind))

    # ---------------------------------------------------------------- guardrail red team
    blocked = (sum(not check_python(c).ok for c in MALICIOUS_PYTHON.values())
               + sum(not check_sql(c, source_tables={"customers"}).ok for c in MALICIOUS_SQL.values())
               + sum(not check_mongo(c).ok for c in MALICIOUS_MONGO.values()))
    attacks = len(MALICIOUS_PYTHON) + len(MALICIOUS_SQL) + len(MALICIOUS_MONGO)
    generated_code = [r for r in rows if r["attempts"] > 0]

    summary = summarize(rows, blocked, attacks, len(generated_code), planner.name, len(specs))
    names = {"csv": "CSV", "sql": "SQLite", "pg": "PostgreSQL", "mongo": "MongoDB"}
    summary["sources"] = [names[k] for k in kinds]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["per_kind"], errors="ignore").to_csv(out_dir / "runs.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "RESULTS.md").write_text(render_markdown(summary))
    try:
        make_charts(summary, rows, ROOT / "docs" / "assets")
    except ImportError:
        print("(matplotlib not installed - skipping charts)")
    print("\n" + render_markdown(summary))
    shutil.rmtree(work, ignore_errors=True)  # generated data is reproducible from the seed
    return 0


def _pct(a: float, b: float) -> float:
    return round(100 * a / b, 1) if b else 0.0


def summarize(rows, blocked, attacks, guarded_runs, planner_name, n_datasets) -> dict:
    df = pd.DataFrame(rows)
    out: dict = {"planner": planner_name, "datasets": n_datasets,
                 "total_rows_first_pass": int(df[df["pass"] == "first"]["rows"].sum())}
    for p in ("first", "repeat", "drift"):
        d = df[df["pass"] == p]
        ok = d[d["status"] == "succeeded"]
        pk: dict[str, list[int]] = {}
        for r in [r for r in rows if r["pass"] == p and r.get("per_kind")]:
            for k, (a, b) in r["per_kind"].items():
                pk.setdefault(k, [0, 0])
                pk[k][0] += a
                pk[k][1] += b
        out[p] = {
            "runs": len(d),
            "auto_success_pct": _pct(((d["status"] == "succeeded") & ~d["human_review"]).sum(), len(d)),
            "needs_human_pct": _pct(d["human_review"].sum(), len(d)),
            "final_success_pct": _pct((d["status"] == "succeeded").sum(), len(d)),
            "first_try_pct": _pct(((d["status"] == "succeeded") & (d["attempts"] == 1) & ~d["human_review"]).sum(), len(d)),
            "mean_attempts": round(float(d["attempts"].mean()), 2) if len(d) else 0,
            "memory_exact_pct": _pct((d["memory_hit"] == "exact").sum(), len(d)),
            "memory_similar_pct": _pct((d["memory_hit"] == "similar").sum(), len(d)),
            "planner_calls": int(d["planner_calls"].sum()),
            "llm_calls": int(d["llm_calls"].sum()),
            "tokens": int(d["tokens"].sum()),
            "est_prompt_tokens": int(d["est_prompt_tokens"].sum()),
            "median_seconds": round(float(d["seconds"].median()), 2) if len(d) else 0,
            "detection_precision_pct": _pct(d["det_tp"].sum(), d["det_found"].sum()),
            "detection_recall_pct": _pct(d["det_tp"].sum(), d["det_truth"].sum()),
            "repair_accuracy_pct": _pct(ok["repaired"].sum(), ok["dirty_cells"].sum()) if len(ok) else None,
            "collateral_pct": round(100 * ok["collateral"].sum() / ok["clean_cells"].sum(), 3) if len(ok) else None,
            "garbage_nulled_pct": _pct(ok["garbage_nulled"].sum(), ok["garbage_cells"].sum()) if len(ok) else None,
            "rename_accuracy_pct": _pct(ok["renames_correct"].sum(), ok["renames_needed"].sum()) if len(ok) else None,
            "dirty_cells": int(ok["dirty_cells"].sum()) if len(ok) else 0,
            "repair_by_kind_pct": {k: _pct(a, b) for k, (a, b) in sorted(pk.items())},
            "demoted": int(d["demoted"].sum()) if "demoted" in d else 0,
            "pushdown_runs": int((d["apply_mode"] == "sql pushdown").sum()),
        }
    f, r = out["first"], out["repeat"]
    out["memory_effect"] = {
        "planner_calls_first": f["planner_calls"], "planner_calls_repeat": r["planner_calls"],
        "planner_call_reduction_pct": _pct(f["planner_calls"] - r["planner_calls"], f["planner_calls"]),
        "est_tokens_first": f["est_prompt_tokens"] or f["tokens"],
        "est_tokens_repeat": r["est_prompt_tokens"] or r["tokens"],
        "token_reduction_x": round((f["est_prompt_tokens"] or f["tokens"]) /
                                   max(1, r["est_prompt_tokens"] or r["tokens"]), 1),
        "median_seconds_first": f["median_seconds"], "median_seconds_repeat": r["median_seconds"],
    }
    out["guardrails"] = {"attacks": attacks, "blocked": blocked, "catch_rate_pct": _pct(blocked, attacks),
                         "generated_programs_checked": guarded_runs}
    d = df[df["pass"] == "drift"]
    out["drift_detail"] = d[["dataset", "drift", "memory_hit", "status", "attempts", "demoted"]].to_dict("records")
    hr = df[df["human_review"]]
    out["review_reasons"] = hr["review_reason"].str.split(":").str[0].value_counts().to_dict()
    return out


def render_markdown(s: dict) -> str:
    f, r, d, m, g = s["first"], s["repeat"], s["drift"], s["memory_effect"], s["guardrails"]
    lines = [
        f"# Benchmark results (planner: `{s['planner']}`)",
        "",
        f"{s['datasets']} synthetic datasets across 6 domains and {len(s.get('sources', []))} source types "
        f"({', '.join(s.get('sources', []))}), {s['total_rows_first_pass']:,} rows in the first pass, with injected "
        "corruption and full ground truth. Reproduce with `python benchmark/run_benchmark.py`.",
        "",
        "| metric | first contact | repeat batch | drift |",
        "|---|---:|---:|---:|",
    ]

    def row(label, key, fmt="{}"):
        vals = [fmt.format(x[key]) if x.get(key) is not None else "-" for x in (f, r, d)]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    row("runs", "runs")
    row("succeeded without a human", "auto_success_pct", "{}%")
    row("succeeded on the first attempt, no human", "first_try_pct", "{}%")
    row("paused for human review", "needs_human_pct", "{}%")
    row("succeeded after review (simulated approver)", "final_success_pct", "{}%")
    row("mean attempts", "mean_attempts")
    row("exact memory hits", "memory_exact_pct", "{}%")
    row("similar-schema hints", "memory_similar_pct", "{}%")
    row("planner calls", "planner_calls")
    row("issue detection precision", "detection_precision_pct", "{}%")
    row("issue detection recall", "detection_recall_pct", "{}%")
    row("dirty cells repaired correctly", "repair_accuracy_pct", "{}%")
    row("clean cells damaged (collateral)", "collateral_pct", "{}%")
    row("unrecoverable junk nulled (not guessed)", "garbage_nulled_pct", "{}%")
    row("column renames mapped correctly", "rename_accuracy_pct", "{}%")
    row("cached fixes demoted", "demoted")
    row("median seconds per run", "median_seconds")
    lines += [
        "",
        "**Memory effect (pass 1 → pass 2):** planner calls "
        f"{m['planner_calls_first']} → {m['planner_calls_repeat']} "
        f"({m['planner_call_reduction_pct']}% fewer); prompt tokens "
        f"{m['est_tokens_first']:,} → {m['est_tokens_repeat']:,} (~{m['token_reduction_x']}× fewer).",
        "",
        f"**Guardrails:** {g['blocked']}/{g['attacks']} red-team programs blocked "
        f"({g['catch_rate_pct']}%); every one of the {g['generated_programs_checked']} generated "
        "transforms passed through the AST/query guard and the sandbox before touching data.",
        "",
        "**Why runs paused for review:** "
        + (", ".join(f"{k} ({v})" for k, v in s["review_reasons"].items()) or "none"),
        "",
        "**Repair accuracy by column type (first pass):** "
        + ", ".join(f"{k} {v}%" for k, v in f["repair_by_kind_pct"].items()),
        "",
        "**Drift pass:**",
        "",
        "| dataset | drift | memory | status | attempts | cached fix demoted |",
        "|---|---|---|---|---:|---|",
    ]
    for x in s["drift_detail"]:
        lines.append(f"| {x['dataset']} | {x['drift']} | {x['memory_hit']} | {x['status']} | "
                     f"{x['attempts']} | {'yes' if x['demoted'] else ''} |")
    if s["planner"] == "heuristic":
        lines += ["", "_Offline mode: token figures are the size of the prompt the LLM planner would "
                  "have been sent (chars/4); no model was called._"]
    return "\n".join(lines) + "\n"


def make_charts(s: dict, rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12})
    ink, muted, accent, accent2 = "#1f2937", "#9ca3af", "#2563eb", "#10b981"

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    f, r = s["first"], s["repeat"]
    # 1. memory effect on planner calls & tokens
    ax = axes[0]
    labels = ["planner calls", "prompt tokens (k)"]
    first = [f["planner_calls"], (f["est_prompt_tokens"] or f["tokens"]) / 1000]
    rep = [r["planner_calls"], (r["est_prompt_tokens"] or r["tokens"]) / 1000]
    x = np.arange(2)
    b1 = ax.bar(x - 0.18, first, 0.36, color=muted, label="first contact")
    b2 = ax.bar(x + 0.18, rep, 0.36, color=accent, label="repeat batch")
    ax.set_xticks(x, labels)
    ax.set_title("Memory: work avoided on repeat runs")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=9)
    ax.set_yticks([])
    # 2. quality
    ax = axes[1]
    names = ["detection\nrecall", "detection\nprecision", "dirty cells\nrepaired", "auto\nsuccess"]
    vals = [f["detection_recall_pct"], f["detection_precision_pct"], f["repair_accuracy_pct"] or 0,
            f["auto_success_pct"]]
    bars = ax.bar(names, vals, color=accent2, width=0.6)
    ax.set_ylim(0, 110)
    ax.set_yticks([])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Quality vs ground truth (first pass, n={f['runs']})")
    # 3. attempts distribution
    ax = axes[2]
    d = pd.DataFrame(rows)
    d = d[d["pass"] == "first"]
    counts = d["attempts"].value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values, color=[accent if i == 1 else "#f59e0b" for i in counts.index],
           width=0.55)
    ax.set_title("Attempts until validation passed")
    ax.set_xlabel("attempts (retries use validator feedback)")
    for i, v in enumerate(counts.values):
        ax.annotate(str(v), (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
