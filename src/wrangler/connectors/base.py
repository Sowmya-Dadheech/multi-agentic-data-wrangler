from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

SourceKind = Literal["sql", "mongo", "csv"]


@dataclass(frozen=True)
class SourceRef:
    """Where the data lives.

    Only the *id* of a source goes into LangGraph state (state is checkpointed to disk),
    so connection strings - which may carry credentials - never end up in checkpoints.
    """

    kind: SourceKind
    name: str  # table / collection / file stem
    uri: str  # SQLAlchemy URL, Mongo URI or CSV path
    database: str | None = None  # Mongo database
    staging_uri: str | None = None  # optional separate *writable* connection for outputs

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_source(spec: str) -> SourceRef:
    """Parse a CLI-style source spec.

    * ``data/orders.csv``                       -> CSV file
    * ``sqlite:///shop.db::customers``          -> SQL table (any SQLAlchemy URL)
    * ``postgresql://u:p@host/db::customers``   -> SQL table
    * ``mongodb://localhost:27017::shop.orders``-> Mongo collection
    * ``mongomock://demo::shop.orders``         -> in-memory Mongo (demos / tests)
    """
    if "::" in spec:
        uri, obj = spec.rsplit("::", 1)
        if uri.startswith(("mongodb://", "mongodb+srv://", "mongomock://")):
            db, _, coll = obj.partition(".")
            if not coll:
                raise ValueError("Mongo sources need '::<database>.<collection>'")
            return SourceRef("mongo", coll, uri, database=db)
        return SourceRef("sql", obj, uri)
    if spec.lower().endswith((".csv", ".tsv", ".txt")):
        from pathlib import Path

        return SourceRef("csv", Path(spec).stem, spec)
    raise ValueError(f"Unrecognised source spec: {spec!r}")


@dataclass
class IngestResult:
    sample: pd.DataFrame
    row_count: int
    declared_schema: dict[str, str] = field(default_factory=dict)
    raw_docs: list[dict] | None = None  # Mongo only: kept for shape analysis


def to_transport(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame safe to hand across the sandbox boundary as Parquet.

    Mixed-type object columns (``42`` and ``"42 yrs"`` in one column - very common in
    Mongo) can't be written to Parquet, so every non-null value in an object column is
    rendered as text. Nested dicts / lists become JSON strings.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype == object or str(s.dtype).startswith("string"):
            out[col] = s.map(_render).astype("string")
    return out


def _render(v: Any) -> Any:
    if v is None:
        return pd.NA
    if isinstance(v, float) and pd.isna(v):
        return pd.NA
    if v is pd.NA or v is pd.NaT:
        return pd.NA
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str, sort_keys=True)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
