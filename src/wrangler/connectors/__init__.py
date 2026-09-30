"""Source connectors: SQL (SQLAlchemy), MongoDB (PyMongo / mongomock) and CSV.

Ingestion is deliberately deterministic Python - no LLM is involved in reading data.
"""

from .base import IngestResult, SourceRef, parse_source
from .registry import ingest, load_full, write_frame, run_sql, run_mongo_pipeline

__all__ = [
    "IngestResult",
    "SourceRef",
    "parse_source",
    "ingest",
    "load_full",
    "write_frame",
    "run_sql",
    "run_mongo_pipeline",
]
