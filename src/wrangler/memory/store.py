"""Long-term memory of approved transformations (LangGraph ``BaseStore``).

* Short-term memory = the LangGraph *checkpointer* (one run / thread: pause, resume, replay).
* Long-term memory  = this *store*, shared across runs: "what have we learned so far?"

Lookup is two-tier:
1. **Exact** - key = schema fingerprint (normalized column names + dominant type family).
   A hit skips the LLM entirely, but the cached code still goes through the guard, the
   sandbox and validation: memory skips the model, never the safety checks.
2. **Similar** - no exact hit, but a stored schema overlaps strongly. The old plan is passed
   to the planner as a *hint* only; it is never executed blindly.

Only validated or human-approved transformations are written (memory-poisoning defence),
entries are versioned instead of overwritten, and a cached fix that later fails validation
is demoted so it can't be served again.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

from langgraph.store.base import BaseStore

from ..naming import normalize

NS_TRANSFORMS = "transforms"
NS_BASELINES = "baselines"


def signature(profile: dict[str, dict]) -> list[str]:
    return sorted(
        f"{normalize(col)}:{p.get('dominant_family', 'null')}"
        for col, p in profile.items()
        if col != "_id"
    )


def schema_fingerprint(profile: dict[str, dict]) -> str:
    """Stable across batches of the same logical schema (sample noise doesn't matter:
    only names and the dominant type family are hashed), but changes on structural drift."""
    raw = json.dumps(signature(profile))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def similarity(sig_a: list[str], sig_b: list[str]) -> float:
    a, b = set(sig_a), set(sig_b)
    names_a = {s.split(":")[0] for s in a}
    names_b = {s.split(":")[0] for s in b}
    j_sig = len(a & b) / len(a | b) if a | b else 0.0
    j_names = len(names_a & names_b) / len(names_a | names_b) if names_a | names_b else 0.0
    return round(0.7 * j_sig + 0.3 * j_names, 4)


def slim_profile(profile: dict[str, dict]) -> dict[str, dict]:
    keep = ("dominant_family", "null_ratio", "distinct", "top_values", "families")
    return {c: {k: p.get(k) for k in keep} for c, p in profile.items()}


class TransformMemory:
    def __init__(self, store: BaseStore):
        self.store = store

    # ------------------------------------------------------------------ reads
    def lookup_exact(self, source: str, fingerprint: str) -> dict | None:
        item = self.store.get((NS_TRANSFORMS, source), fingerprint)
        if item and item.value.get("status") == "active":
            return item.value
        return None

    def find_similar(self, profile: dict[str, dict], threshold: float,
                     exclude: tuple[str, str] | None = None) -> tuple[dict | None, float]:
        sig = signature(profile)
        best, best_score = None, 0.0
        for item in self._all():
            v = item.value
            if v.get("status") != "active":
                continue
            if exclude and (item.namespace[-1], item.key) == exclude:
                continue
            score = similarity(sig, v.get("signature", []))
            if score > best_score:
                best, best_score = v, score
        return (best, best_score) if best_score >= threshold else (None, best_score)

    def baseline(self, source: str) -> dict | None:
        item = self.store.get((NS_BASELINES, source), "latest")
        return item.value.get("profile") if item else None

    def list_all(self) -> list[dict]:
        return [dict(item.value, _source=item.namespace[-1]) for item in self._all()]

    def _all(self):
        out, offset = [], 0
        while True:
            batch = self.store.search((NS_TRANSFORMS,), limit=100, offset=offset)
            out.extend(batch)
            if len(batch) < 100:
                return out
            offset += 100

    # ------------------------------------------------------------------ writes
    def save(self, *, source: str, fingerprint: str, profile: dict, plan: dict, code: str,
             dialect_artifacts: dict[str, Any], approved_by: str, run_id: str) -> dict:
        ns = (NS_TRANSFORMS, source)
        prev = self.store.get(ns, fingerprint)
        history: list[dict] = []
        version = 1
        if prev:
            old = prev.value
            version = int(old.get("version", 0)) + 1
            history = (old.get("history", []) + [{k: old.get(k) for k in
                       ("version", "plan", "code", "approved_by", "saved_at", "run_id")}])[-5:]
        value = {
            "source": source,
            "fingerprint": fingerprint,
            "signature": signature(profile),
            "plan": plan,
            "code": code,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "artifacts": dialect_artifacts,
            "approved_by": approved_by,
            "run_id": run_id,
            "version": version,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "active",
            "uses": 0,
            "failures": 0,
            "history": history,
        }
        self.store.put(ns, fingerprint, value)
        self.store.put((NS_BASELINES, source), "latest",
                       {"profile": slim_profile(profile), "fingerprint": fingerprint})
        return value

    def record_use(self, source: str, fingerprint: str, profile: dict | None = None) -> None:
        item = self.store.get((NS_TRANSFORMS, source), fingerprint)
        if item:
            v = dict(item.value)
            v["uses"] = int(v.get("uses", 0)) + 1
            v["last_used_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.store.put((NS_TRANSFORMS, source), fingerprint, v)
        if profile is not None:
            self.store.put((NS_BASELINES, source), "latest",
                           {"profile": slim_profile(profile), "fingerprint": fingerprint})

    def demote(self, source: str, fingerprint: str, reason: str) -> None:
        item = self.store.get((NS_TRANSFORMS, source), fingerprint)
        if item:
            v = dict(item.value)
            v["failures"] = int(v.get("failures", 0)) + 1
            v["status"] = "demoted"
            v["demoted_reason"] = reason[:300]
            self.store.put((NS_TRANSFORMS, source), fingerprint, v)


def open_sqlite_store(path: str) -> BaseStore:
    from langgraph.store.sqlite import SqliteStore

    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store
