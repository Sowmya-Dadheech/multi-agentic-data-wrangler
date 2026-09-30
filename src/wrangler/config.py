"""Runtime settings. Everything is overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # where checkpoints, memory and sandbox scratch files live
    home: Path = field(default_factory=lambda: Path(_env("WRANGLER_HOME", ".wrangler")))
    sample_size: int = int(_env("WRANGLER_SAMPLE_SIZE", "5000"))
    max_attempts: int = int(_env("WRANGLER_MAX_ATTEMPTS", "3"))
    # validator: max share of a column that a fix may newly turn into nulls
    max_new_null_ratio: float = float(_env("WRANGLER_MAX_NEW_NULLS", "0.02"))
    # planner confidence below this goes to a human
    min_confidence: float = float(_env("WRANGLER_MIN_CONFIDENCE", "0.6"))
    # drift: a null-ratio jump bigger than this (absolute) counts as drift
    null_jump_threshold: float = float(_env("WRANGLER_NULL_JUMP", "0.20"))
    # similarity threshold for "similar schema" memory hints
    similar_threshold: float = float(_env("WRANGLER_SIMILAR_THRESHOLD", "0.6"))
    # sandbox limits
    sandbox_timeout_s: int = int(_env("WRANGLER_SANDBOX_TIMEOUT", "60"))
    sandbox_cpu_s: int = int(_env("WRANGLER_SANDBOX_CPU", "30"))
    sandbox_mem_mb: int = int(_env("WRANGLER_SANDBOX_MEM_MB", "2048"))
    # prefix every output table / collection must use (guardrail)
    output_prefix: str = _env("WRANGLER_OUTPUT_PREFIX", "clean_")

    def ensure_dirs(self) -> "Settings":
        (self.home / "runs").mkdir(parents=True, exist_ok=True)
        return self

    @property
    def checkpoint_db(self) -> str:
        return str(self.home / "checkpoints.sqlite")

    @property
    def memory_db(self) -> str:
        return str(self.home / "memory.sqlite")

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"
