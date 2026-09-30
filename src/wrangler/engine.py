"""High-level API around the compiled graph.

    from wrangler import Wrangler
    w = Wrangler()                                    # heuristic planner, sqlite memory
    result = w.run("sqlite:///shop.db::customers")
    if result.interrupted:
        result = w.resume(result.run_id, {"action": "approve"})
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from langgraph.types import Command

from .config import Settings
from .connectors.base import SourceRef, parse_source
from .graph import Context, build_graph, new_run_id
from .memory import TransformMemory, open_sqlite_store
from .planning import make_planner
from .sandbox import make_executor


@dataclass
class RunResult:
    run_id: str
    state: dict[str, Any]
    interrupt: dict | None = None
    updates: list[tuple[str, dict]] = field(default_factory=list)

    @property
    def interrupted(self) -> bool:
        return self.interrupt is not None

    @property
    def status(self) -> str:
        return "awaiting_review" if self.interrupted else self.state.get("status", "unknown")


class Wrangler:
    def __init__(self, settings: Settings | None = None, planner: Any = None, executor: Any = None,
                 hitl: str = "interrupt", persistent: bool = True):
        self.settings = (settings or Settings()).ensure_dirs()
        self.planner = planner or make_planner()
        self.executor = executor or make_executor(
            "subprocess", timeout_s=self.settings.sandbox_timeout_s,
            cpu_s=self.settings.sandbox_cpu_s, mem_mb=self.settings.sandbox_mem_mb)
        self.hitl = hitl
        if persistent:
            from langgraph.checkpoint.sqlite import SqliteSaver

            self.store = open_sqlite_store(self.settings.memory_db)
            conn = sqlite3.connect(self.settings.checkpoint_db, check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)
        else:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.store.memory import InMemoryStore

            self.store, self.checkpointer = InMemoryStore(), InMemorySaver()
        self.graph = build_graph().compile(checkpointer=self.checkpointer, store=self.store)
        self.sources: dict[str, SourceRef] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def memory(self) -> TransformMemory:
        return TransformMemory(self.store)

    def _ctx(self) -> Context:
        return Context(settings=self.settings, sources=self.sources, planner=self.planner,
                       executor=self.executor, hitl=self.hitl)  # type: ignore[arg-type]

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 80}

    def _drive(self, payload: Any, run_id: str,
               on_update: Callable[[str, dict], None] | None) -> RunResult:
        updates: list[tuple[str, dict]] = []
        interrupt_payload = None
        for chunk in self.graph.stream(payload, self._config(run_id), context=self._ctx(),
                                       stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupt_payload = update[0].value if update else {}
                    continue
                updates.append((node, update or {}))
                if on_update:
                    on_update(node, update or {})
        state = self.graph.get_state(self._config(run_id)).values
        return RunResult(run_id, dict(state), interrupt_payload, updates)

    # ------------------------------------------------------------------ public
    def run(self, source: SourceRef | str, target_schema: dict | None = None,
            run_id: str | None = None,
            on_update: Callable[[str, dict], None] | None = None) -> RunResult:
        src = parse_source(source) if isinstance(source, str) else source
        self.sources[src.id] = src
        run_id = run_id or new_run_id(src.id)
        payload = {"run_id": run_id, "source_id": src.id, "target_schema": target_schema,
                   "errors": [], "events": [], "usage": {}}
        return self._drive(payload, run_id, on_update)

    def resume(self, run_id: str, decision: dict,
               on_update: Callable[[str, dict], None] | None = None) -> RunResult:
        """Continue a run paused at human review: {"action": "approve" | "reject" | "edit", "plan": {...}}"""
        return self._drive(Command(resume=decision), run_id, on_update)

    def history(self, run_id: str) -> Iterator[Any]:
        """Every checkpoint of a run (time-travel / debugging)."""
        return self.graph.get_state_history(self._config(run_id))
