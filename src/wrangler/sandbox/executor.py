"""Run generated code in isolation.

* ``SubprocessExecutor`` (default): separate interpreter in isolated mode (``-I``), empty
  environment (no inherited secrets), throwaway working directory, CPU / memory / file-size
  rlimits and a wall-clock timeout, plus a socket-level network kill switch.
* ``DockerExecutor``: the same runner inside a container with ``--network none``, a
  read-only root filesystem, a non-root user, dropped capabilities and cgroup limits. Use
  this for anything multi-tenant or production-like.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).parent
RUNNER = _HERE / "runner.py"
HELPERS = _HERE / "helpers.py"


@dataclass
class ExecResult:
    ok: bool
    output_path: str | None = None
    stats: dict = field(default_factory=dict)
    error: str | None = None


def _tail(text: str, n: int = 1500) -> str:
    text = text.strip()
    return text if len(text) <= n else "…" + text[-n:]


class SubprocessExecutor:
    name = "subprocess"

    def __init__(self, timeout_s: int = 60, cpu_s: int = 30, mem_mb: int = 2048):
        self.timeout_s, self.cpu_s, self.mem_mb = timeout_s, cpu_s, mem_mb

    def _limits(self):  # runs in the child just before exec
        import resource

        def set_limit(res, value):
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass  # e.g. RLIMIT_AS is not enforceable on macOS

        set_limit(resource.RLIMIT_CPU, self.cpu_s)
        set_limit(resource.RLIMIT_AS, self.mem_mb * 1024 * 1024)
        set_limit(resource.RLIMIT_FSIZE, 2 * 1024**3)
        os.setsid()  # own process group, so a timeout kills everything it spawned

    def run(self, code: str, input_path: str, output_path: str) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="wrangler-sbx-") as tmp:
            code_file = Path(tmp) / "generated.py"
            code_file.write_text(code)
            out_tmp = Path(tmp) / "out.parquet"
            cmd = [sys.executable, "-I", str(RUNNER), str(HELPERS), str(code_file),
                   str(Path(input_path).resolve()), str(out_tmp)]
            env = {"PATH": os.defpath, "HOME": tmp, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout_s, cwd=tmp, env=env,
                    preexec_fn=self._limits if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired:
                return ExecResult(False, error=f"timed out after {self.timeout_s}s")
            if proc.returncode < 0:
                return ExecResult(False, error=f"killed by signal {-proc.returncode} "
                                  f"(resource limit: {self.cpu_s}s CPU / {self.mem_mb} MB)")
            if proc.returncode != 0:
                return ExecResult(False, error=_tail(proc.stderr) or f"exit code {proc.returncode}")
            try:
                stats = json.loads(proc.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                return ExecResult(False, error="sandbox produced no result")
            shutil.copyfile(out_tmp, output_path)
            return ExecResult(True, output_path, stats)


class DockerExecutor:
    name = "docker"

    def __init__(self, image: str = "wrangler-sandbox:latest", timeout_s: int = 120,
                 mem_mb: int = 2048, cpus: float = 1.0):
        self.image, self.timeout_s, self.mem_mb, self.cpus = image, timeout_s, mem_mb, cpus

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0

    def run(self, code: str, input_path: str, output_path: str) -> ExecResult:
        with tempfile.TemporaryDirectory(prefix="wrangler-sbx-") as tmp:
            job = Path(tmp)
            shutil.copy(RUNNER, job / "runner.py")
            shutil.copy(HELPERS, job / "helpers.py")
            shutil.copy(input_path, job / "in.parquet")
            (job / "generated.py").write_text(code)
            os.chmod(tmp, 0o777)
            cmd = [
                "docker", "run", "--rm", "--network", "none", "--read-only",
                "--memory", f"{self.mem_mb}m", "--cpus", str(self.cpus), "--pids-limit", "64",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
                "--tmpfs", "/tmp:size=64m", "-v", f"{tmp}:/job", "-w", "/job", self.image,
                "python", "-I", "/job/runner.py", "/job/helpers.py", "/job/generated.py",
                "/job/in.parquet", "/job/out.parquet",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return ExecResult(False, error=f"timed out after {self.timeout_s}s")
            if proc.returncode != 0:
                return ExecResult(False, error=_tail(proc.stderr) or f"exit code {proc.returncode}")
            stats = json.loads(proc.stdout.strip().splitlines()[-1])
            shutil.copyfile(job / "out.parquet", output_path)
            return ExecResult(True, output_path, stats)


def make_executor(kind: str = "subprocess", **kw):
    if kind == "docker":
        return DockerExecutor(**kw)
    return SubprocessExecutor(**kw)
