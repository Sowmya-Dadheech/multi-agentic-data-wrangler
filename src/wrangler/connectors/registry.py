from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .base import IngestResult, SourceRef, to_transport

# --------------------------------------------------------------------------- engines

_ENGINES: dict[str, Engine] = {}
_MONGOMOCK_CLIENTS: dict[str, Any] = {}


def sql_engine(uri: str) -> Engine:
    if uri not in _ENGINES:
        # pool_pre_ping replaces pooled connections that died (DB restart, idle timeout)
        _ENGINES[uri] = create_engine(uri, pool_pre_ping=True)
    return _ENGINES[uri]


def mongo_client(uri: str):
    if uri.startswith("mongomock://"):
        import mongomock

        # one shared in-memory server per URI so demos/tests can seed then read
        return _MONGOMOCK_CLIENTS.setdefault(uri, mongomock.MongoClient())
    from pymongo import MongoClient

    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def is_mongomock(source: SourceRef) -> bool:
    return source.uri.startswith("mongomock://")


def _check_table(engine: Engine, table: str) -> None:
    # Table names cannot be bind parameters, so allowlist them against the catalog
    # before interpolating them into SQL (prevents SQL injection through the name).
    if table not in inspect(engine).get_table_names():
        raise ValueError(f"Unknown table {table!r}")


# --------------------------------------------------------------------------- ingest


def ingest(source: SourceRef, sample_n: int = 5000, seed: int = 7) -> IngestResult:
    if source.kind == "sql":
        return _ingest_sql(source, sample_n)
    if source.kind == "mongo":
        return _ingest_mongo(source, sample_n)
    if source.kind == "csv":
        return _ingest_csv(source, sample_n, seed)
    raise ValueError(source.kind)


def _ingest_sql(source: SourceRef, sample_n: int) -> IngestResult:
    engine = sql_engine(source.uri)
    _check_table(engine, source.name)
    declared = {c["name"]: str(c["type"]) for c in inspect(engine).get_columns(source.name)}
    q = f'"{source.name}"'
    with engine.connect() as con:
        row_count = int(con.execute(text(f"SELECT COUNT(*) FROM {q}")).scalar() or 0)
        if row_count <= sample_n:
            sql = f"SELECT * FROM {q}"
        elif engine.dialect.name == "postgresql":
            # TABLESAMPLE reads a fraction of storage pages instead of sorting the whole
            # table like ORDER BY random(); oversample x2 then LIMIT.
            pct = min(100.0, math.ceil(sample_n / row_count * 200 * 100) / 100)
            sql = f"SELECT * FROM {q} TABLESAMPLE SYSTEM ({pct}) LIMIT {int(sample_n)}"
        else:
            sql = f"SELECT * FROM {q} ORDER BY RANDOM() LIMIT {int(sample_n)}"
        df = pd.read_sql(text(sql), con)
    return IngestResult(to_transport(df), row_count, declared)


def _ingest_mongo(source: SourceRef, sample_n: int) -> IngestResult:
    coll = mongo_client(source.uri)[source.database][source.name]
    row_count = int(coll.estimated_document_count())  # O(1): read from metadata
    if row_count <= sample_n:
        docs = list(coll.find({}))
    else:
        docs = list(coll.aggregate([{"$sample": {"size": int(sample_n)}}]))
    for d in docs:
        d["_id"] = str(d.get("_id"))
    df = pd.json_normalize(docs) if docs else pd.DataFrame()
    return IngestResult(to_transport(df), row_count, {}, raw_docs=docs)


def _ingest_csv(source: SourceRef, sample_n: int, seed: int) -> IngestResult:
    # dtype=str: keep raw text so the profiler sees exactly what was in the file
    df = pd.read_csv(source.uri, dtype=str, keep_default_na=False, na_values=[""])
    row_count = len(df)
    if row_count > sample_n:
        df = df.sample(sample_n, random_state=seed)
    return IngestResult(to_transport(df.reset_index(drop=True)), row_count, {})


# --------------------------------------------------------------------------- full load / write


def _physical_order(engine: Engine) -> str:
    # Row-by-row validation needs both sides in the same order. Postgres gives no order
    # guarantee without ORDER BY (e.g. synchronized seq scans); ctid is physical order, and a
    # CREATE TABLE AS SELECT writes rows in the order the source was scanned.
    return " ORDER BY ctid" if engine.dialect.name == "postgresql" else ""


def load_full(source: SourceRef) -> pd.DataFrame:
    if source.kind == "sql":
        engine = sql_engine(source.uri)
        _check_table(engine, source.name)
        with engine.connect() as con:
            sql = f'SELECT * FROM "{source.name}"{_physical_order(engine)}'
            return to_transport(pd.read_sql(text(sql), con))
    if source.kind == "mongo":
        coll = mongo_client(source.uri)[source.database][source.name]
        docs = list(coll.find({}))
        for d in docs:
            d["_id"] = str(d.get("_id"))
        return to_transport(pd.json_normalize(docs)) if docs else pd.DataFrame()
    if source.kind == "csv":
        df = pd.read_csv(source.uri, dtype=str, keep_default_na=False, na_values=[""])
        return to_transport(df)
    raise ValueError(source.kind)


def output_name(source: SourceRef, prefix: str) -> str:
    return f"{prefix}{source.name}"


def write_frame(source: SourceRef, df: pd.DataFrame, target: str, prefix: str) -> str:
    """Write cleaned data to a NEW table / collection / file. Never in place."""
    if not target.startswith(prefix):
        raise PermissionError(f"refusing to write to {target!r}: outputs must start with {prefix!r}")
    if source.kind == "sql":
        engine = sql_engine(source.staging_uri or source.uri)
        df.to_sql(target, engine, if_exists="replace", index=False)
        return f"{engine.url.render_as_string(hide_password=True)}::{target}"
    if source.kind == "mongo":
        db = mongo_client(source.staging_uri or source.uri)[source.database]
        db[target].drop()
        records = [_unflatten(_mongo_safe(r)) for r in df.to_dict(orient="records")]
        if records:
            db[target].insert_many(records)
        return f"{source.database}.{target}"
    if source.kind == "csv":
        path = Path(source.uri).with_name(f"{target}.csv")
        df.to_csv(path, index=False)
        return str(path)
    raise ValueError(source.kind)


def run_sql(source: SourceRef, sql: str) -> None:
    engine = sql_engine(source.staging_uri or source.uri)
    with engine.begin() as con:
        for stmt in [s for s in sql.split(";\n") if s.strip()]:
            con.execute(text(stmt))


def read_table(source: SourceRef, table: str) -> pd.DataFrame:
    engine = sql_engine(source.staging_uri or source.uri)
    with engine.connect() as con:
        # convert_dtypes: nullable BOOLEAN / BIGINT columns come back as object / float
        sql = f'SELECT * FROM "{table}"{_physical_order(engine)}'
        return pd.read_sql(text(sql), con).convert_dtypes()


def run_mongo_pipeline(source: SourceRef, pipeline: list[dict]) -> None:
    coll = mongo_client(source.uri)[source.database][source.name]
    list(coll.aggregate(pipeline))


# --------------------------------------------------------------------------- helpers


def _mongo_safe(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
            out[k] = None
        elif isinstance(v, pd.Timestamp):
            out[k] = v.to_pydatetime()
        elif hasattr(v, "item"):  # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _unflatten(record: dict) -> dict:
    """{"address.city": "x"} -> {"address": {"city": "x"}} (json_normalize in reverse)."""
    out: dict = {}
    for key, val in record.items():
        if key == "_id" or "." not in key:
            if not (key in out and isinstance(out[key], dict)):
                out[key] = val
            continue
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {} if node.get(p) is None else {"_value": node[p]}
            node = node[p]
        node[parts[-1]] = val
    return out
