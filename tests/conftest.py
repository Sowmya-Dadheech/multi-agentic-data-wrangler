from __future__ import annotations

from pathlib import Path

import pytest

from wrangler import Settings, Wrangler
from wrangler.datagen import CorruptionProfile, load_into, make_dataset
from wrangler.planning import HeuristicPlanner
from wrangler.sandbox import SubprocessExecutor


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / ".wrangler", sample_size=2000, max_attempts=3)


@pytest.fixture
def executor() -> SubprocessExecutor:
    return SubprocessExecutor(timeout_s=30, cpu_s=20)


@pytest.fixture
def wrangler(settings, executor) -> Wrangler:
    return Wrangler(settings, planner=HeuristicPlanner(), executor=executor, hitl="escalate")


@pytest.fixture
def messy_csv(tmp_path: Path):
    def _make(domain: str = "customers", seed: int = 1, n: int = 1500, **cp):
        ds = make_dataset(domain, n, seed, CorruptionProfile(**cp) if cp else CorruptionProfile())
        return ds, load_into(ds, "csv", domain, str(tmp_path))

    return _make
