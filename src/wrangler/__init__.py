"""Agentic Data Wrangler: a LangGraph multi-agent data-cleaning pipeline."""

__version__ = "0.1.0"

from .config import Settings  # noqa: E402
from .engine import RunResult, Wrangler  # noqa: E402

__all__ = ["Settings", "Wrangler", "RunResult", "__version__"]
