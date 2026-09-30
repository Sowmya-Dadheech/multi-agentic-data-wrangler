"""Synthetic messy-data generator with ground truth.

Used by the demo, the tests and the benchmark. Each dataset is generated *clean* first
(typed values, a few genuine nulls), then corrupted in the ways real data is messy. Because
the corruption is injected, we know exactly which cells were dirtied and what their correct
value is - which is what makes the benchmark metrics honest.

All data here is synthetic; no real people or records.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

FIRST = ["Aarav", "Maya", "Liam", "Priya", "Noah", "Zara", "Ethan", "Ananya", "Lucas", "Mei",
         "Omar", "Sofia", "Ravi", "Emma", "Kenji", "Leila", "Diego", "Ines", "Arjun", "Chloe"]
LAST = ["Shah", "Nguyen", "Garcia", "Iyer", "Smith", "Kim", "Okafor", "Rossi", "Patel", "Chen",
        "Silva", "Novak", "Haddad", "Brown", "Tanaka", "Mehta", "Lopez", "Muller", "Rao", "Khan"]
CITIES = ["Seattle", "Mumbai", "Austin", "London", "Toronto", "Berlin", "Singapore", "Sydney"]


# ============================================================================ column specs


@dataclass
class Col:
    name: str
    kind: str  # id | str | int | float | date | datetime | bool | category
    gen: Callable[[np.random.Generator, int], list]
    null_rate: float = 0.0
    unit: str | None = None  # suffix used by the "units" corruption ("yrs", "kg"...)
    currency: bool = False
    categories: list[str] | None = None


def _ids(prefix: str, width: int = 5):
    return lambda rng, n: [f"{prefix}{i:0{width}d}" for i in rng.permutation(n) + 1]


def _ints(lo: int, hi: int):
    return lambda rng, n: rng.integers(lo, hi, n).tolist()


def _floats(lo: float, hi: float, nd: int = 2):
    return lambda rng, n: np.round(rng.uniform(lo, hi, n), nd).tolist()


def _dates(start: str, days: int):
    base = dt.date.fromisoformat(start)
    return lambda rng, n: [base + dt.timedelta(days=int(d)) for d in rng.integers(0, days, n)]


def _datetimes(start: str, days: int):
    base = dt.datetime.fromisoformat(start)
    return lambda rng, n: [base + dt.timedelta(minutes=int(m)) for m in rng.integers(0, days * 1440, n)]


def _choice(values: list):
    return lambda rng, n: rng.choice(values, n).tolist()


def _bools(p: float = 0.5):
    return lambda rng, n: (rng.random(n) < p).tolist()


def _names(rng, n):
    return [f"{a} {b}" for a, b in zip(rng.choice(FIRST, n), rng.choice(LAST, n))]


def _emails(rng, n):
    return [f"{a.lower()}.{b.lower()}{i}@example.com"
            for i, (a, b) in enumerate(zip(rng.choice(FIRST, n), rng.choice(LAST, n)))]


DOMAINS: dict[str, list[Col]] = {
    "customers": [
        Col("customer_id", "id", _ids("C")),
        Col("full_name", "str", _names),
        Col("email", "str", _emails, null_rate=0.03),
        Col("age", "int", _ints(18, 90), null_rate=0.04, unit="yrs"),
        Col("signup_date", "date", _dates("2019-01-01", 2400), null_rate=0.02),
        Col("plan", "category", _choice(["Free", "Pro", "Enterprise"]), categories=["Free", "Pro", "Enterprise"]),
        Col("is_active", "bool", _bools(0.7)),
        Col("city", "category", _choice(CITIES), categories=CITIES),
        Col("lifetime_value", "float", _floats(0, 5000), null_rate=0.05, currency=True),
    ],
    "orders": [
        Col("order_id", "id", _ids("O", 6)),
        Col("customer_id", "id", _ids("C")),
        Col("order_date", "date", _dates("2022-01-01", 1300)),
        Col("quantity", "int", _ints(1, 20), unit="units"),
        Col("unit_price", "float", _floats(1, 900), currency=True),
        Col("status", "category", _choice(["Pending", "Shipped", "Delivered", "Returned"]),
            categories=["Pending", "Shipped", "Delivered", "Returned"]),
        Col("is_gift", "bool", _bools(0.15)),
        Col("country", "category", _choice(["US", "IN", "UK", "CA", "DE"]), categories=["US", "IN", "UK", "CA", "DE"]),
    ],
    "patients": [
        Col("patient_id", "id", _ids("P")),
        Col("birth_date", "date", _dates("1940-01-01", 25000), null_rate=0.02),
        Col("height_cm", "float", _floats(140, 200, 1), unit="cm"),
        Col("weight_kg", "float", _floats(40, 140, 1), null_rate=0.05, unit="kg"),
        Col("visit_date", "date", _dates("2023-01-01", 600)),
        Col("department", "category", _choice(["Radiology", "Cardiology", "Oncology", "Neurology"]),
            categories=["Radiology", "Cardiology", "Oncology", "Neurology"]),
        Col("smoker", "bool", _bools(0.2), null_rate=0.03),
        Col("systolic_bp", "int", _ints(90, 180), null_rate=0.04, unit="mmHg"),
    ],
    "sensors": [
        Col("sensor_id", "id", _ids("S", 4)),
        Col("reading_time", "datetime", _datetimes("2025-06-01", 90)),
        Col("temperature_c", "float", _floats(-10, 45, 2), null_rate=0.03, unit="C"),
        Col("humidity_pct", "float", _floats(5, 100, 1), null_rate=0.03, unit="%"),
        Col("status", "category", _choice(["OK", "Warning", "Fault"]), categories=["OK", "Warning", "Fault"]),
        Col("battery_level", "int", _ints(0, 101), unit="%"),
    ],
    "employees": [
        Col("employee_id", "id", _ids("E", 4)),
        Col("full_name", "str", _names),
        Col("hire_date", "date", _dates("2010-01-01", 5000)),
        Col("salary", "float", _floats(40000, 250000, 0), currency=True),
        Col("department", "category", _choice(["Engineering", "Sales", "Finance", "Design", "Support"]),
            categories=["Engineering", "Sales", "Finance", "Design", "Support"]),
        Col("is_remote", "bool", _bools(0.4)),
        Col("rating", "int", _ints(1, 6), null_rate=0.06),
        Col("office", "category", _choice(CITIES[:5]), categories=CITIES[:5]),
    ],
    "products": [
        Col("sku", "id", _ids("SKU-", 5)),
        Col("product_name", "str", lambda rng, n: [f"Item {i}" for i in rng.integers(100, 999, n)]),
        Col("category", "category", _choice(["Books", "Electronics", "Home", "Toys", "Garden"]),
            categories=["Books", "Electronics", "Home", "Toys", "Garden"]),
        Col("price", "float", _floats(2, 1500), currency=True),
        Col("stock", "int", _ints(0, 5000), null_rate=0.03, unit="pcs"),
        Col("launch_date", "date", _dates("2015-01-01", 3500)),
        Col("discontinued", "bool", _bools(0.1)),
        Col("rating", "float", _floats(1, 5, 1), null_rate=0.08),
    ],
}

# how each domain's source system mangles column names (structural corruption)
RENAMES: dict[str, dict[str, str]] = {
    "customers": {"customer_id": "customerId", "signup_date": "SignupDate", "is_active": "isActive",
                  "lifetime_value": "LTV_usd"},
    "orders": {"order_id": "OrderID", "unit_price": "unitPrice", "is_gift": "IsGift", "quantity": "qty"},
    "patients": {"birth_date": "dob", "height_cm": "HeightCm", "systolic_bp": "systolicBP"},
    "sensors": {"reading_time": "readingTime", "temperature_c": "Temperature_C", "battery_level": "batteryLevel"},
    "employees": {"employee_id": "EmployeeId", "hire_date": "HireDate", "is_remote": "remote"},
    "products": {"product_name": "ProductName", "launch_date": "launchDate", "discontinued": "isDiscontinued"},
}


def target_schema(domain: str, clean: pd.DataFrame | None = None) -> dict[str, dict]:
    out = {}
    for c in DOMAINS[domain]:
        t = {"id": "string", "str": "string", "category": "category"}.get(c.kind, c.kind)
        spec: dict[str, Any] = {"type": t}
        if clean is not None:
            spec["examples"] = [str(v) for v in clean[c.name].dropna().head(3)]
        out[c.name] = spec
    return out


# ============================================================================ generation


def generate_clean(domain: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for c in DOMAINS[domain]:
        vals = c.gen(rng, n)
        if c.null_rate:
            mask = rng.random(n) < c.null_rate
            vals = [None if m else v for v, m in zip(vals, mask)]
        data[c.name] = vals
    df = pd.DataFrame(data)
    for c in DOMAINS[domain]:
        if c.kind in ("date", "datetime"):
            df[c.name] = pd.to_datetime(df[c.name])
        elif c.kind == "int":
            df[c.name] = pd.array(df[c.name], dtype="Int64")
        elif c.kind == "float":
            df[c.name] = pd.array(df[c.name], dtype="Float64")
        elif c.kind == "bool":
            df[c.name] = pd.array(df[c.name], dtype="boolean")
        else:
            df[c.name] = df[c.name].astype("string")
    return df


@dataclass
class CorruptionProfile:
    rate: float = 0.08  # share of cells hit by each cell-level corruption
    date_style: str = "mixed_us"  # iso | mixed_us | mixed_eu
    numeric_text: bool = True
    null_tokens: bool = True
    whitespace: bool = True
    case_variants: bool = True
    bool_words: bool = True
    rename: bool = True
    garbage_rate: float = 0.004  # unrecoverable junk ("#REF!", "??") - keeps the benchmark honest
    # semantic traps the pipeline is NOT expected to solve on its own:
    unit_mix: bool = False  # some weights / heights silently recorded in lb / in
    nested_shapes: bool = False  # Mongo only: `address` is an object in some docs, a string in others

    @classmethod
    def random(cls, rng: np.random.Generator) -> "CorruptionProfile":
        return cls(
            rate=float(rng.choice([0.04, 0.08, 0.12])),
            date_style=str(rng.choice(["iso", "mixed_us", "mixed_us", "mixed_eu"])),
            numeric_text=bool(rng.random() < 0.85),
            null_tokens=bool(rng.random() < 0.8),
            whitespace=bool(rng.random() < 0.7),
            case_variants=bool(rng.random() < 0.8),
            bool_words=bool(rng.random() < 0.85),
            rename=bool(rng.random() < 0.8),
            garbage_rate=float(rng.choice([0.0, 0.003, 0.006])),
            unit_mix=bool(rng.random() < 0.3),
            nested_shapes=bool(rng.random() < 0.3),
        )


@dataclass
class MessyDataset:
    domain: str
    clean: pd.DataFrame  # ground truth, target column names, typed
    dirty: pd.DataFrame  # what the pipeline sees
    dirty_mask: pd.DataFrame  # True where a cell was corrupted (target column names)
    garbage_mask: pd.DataFrame  # True where the value was replaced by unrecoverable junk
    rename_map: dict[str, str]  # target -> dirty name
    injected: set[tuple[str, str]] = field(default_factory=set)  # (target column, issue kind)
    profile: CorruptionProfile = field(default_factory=CorruptionProfile)


NULL_TOKEN_POOL = ["N/A", "null", "-", "NA", "none", "?", "", "n/a", "NULL"]
GARBAGE_POOL = ["#REF!", "??", "ERR", "see notes", "tbd"]


def _fmt_date(d: pd.Timestamp, style: str, rng: np.random.Generator, is_dt: bool) -> str:
    if is_dt:
        choices = ["iso_t", "iso_space"] if style == "iso" else ["iso_t", "iso_space", "slash"]
        f = rng.choice(choices)
        if f == "iso_t":
            return d.strftime("%Y-%m-%dT%H:%M:%S")
        if f == "iso_space":
            return d.strftime("%Y-%m-%d %H:%M:%S")
        return d.strftime("%Y-%m-%d") if style == "iso" else d.strftime("%Y-%m-%dT%H:%M")
    if style == "iso":
        return d.strftime("%Y-%m-%d")
    f = rng.choice(["iso", "iso", "slash", "text", "text2"])
    if f == "iso":
        return d.strftime("%Y-%m-%d")
    if f == "slash":
        return d.strftime("%d/%m/%Y") if style == "mixed_eu" else d.strftime("%m/%d/%Y")
    if f == "text":
        return f"{d.day} {d.strftime('%b %Y')}"
    return d.strftime("%b %d, %Y")


def corrupt(clean: pd.DataFrame, domain: str, seed: int, cp: CorruptionProfile | None = None,
            as_text: bool = True) -> MessyDataset:
    """Inject realistic mess. ``as_text=True`` mimics CSV / VARCHAR-everything tables;
    ``as_text=False`` keeps native types where untouched (Mongo-style mixed types)."""
    rng = np.random.default_rng(seed)
    cp = cp or CorruptionProfile()
    n = len(clean)
    cols = {c.name: c for c in DOMAINS[domain]}
    dirty: dict[str, list] = {}
    mask = pd.DataFrame(False, index=clean.index, columns=clean.columns)
    garbage = pd.DataFrame(False, index=clean.index, columns=clean.columns)
    injected: set[tuple[str, str]] = set()

    for name, spec in cols.items():
        truth = clean[name]
        out: list[Any] = []
        hit = rng.random(n) < cp.rate
        for i, v in enumerate(truth):
            is_null = v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
            cell: Any
            changed = False
            if is_null:
                if cp.null_tokens and rng.random() < 0.7:
                    cell, changed = str(rng.choice(NULL_TOKEN_POOL)), True
                    injected.add((name, "null_tokens"))
                else:
                    cell = None
            elif spec.kind in ("int", "float"):
                num = int(v) if spec.kind == "int" else float(v)
                if cp.unit_mix and spec.unit in ("kg", "cm") and rng.random() < 0.05:
                    # the value is right, the unit is wrong: 70 kg recorded as "154.3 lb"
                    factor, unit = (2.20462, "lb") if spec.unit == "kg" else (0.393701, "in")
                    cell, changed = f"{round(num * factor, 1)} {unit}", True
                    injected.add((name, "numeric"))
                elif cp.numeric_text and hit[i]:
                    if spec.currency:
                        cell = f"${num:,.2f}" if spec.kind == "float" else f"${num:,}"
                    elif spec.unit:
                        cell = f"{num} {spec.unit}" if rng.random() < 0.5 else f"{num}{spec.unit}"
                    else:
                        cell = f"{num:,}" if abs(num) >= 1000 else f" {num} "
                    changed = True
                    injected.add((name, "numeric"))
                else:
                    cell = str(num) if as_text else num
                    if as_text:
                        injected.add((name, "numeric"))
            elif spec.kind in ("date", "datetime"):
                style = cp.date_style
                cell = _fmt_date(pd.Timestamp(v), style, rng, spec.kind == "datetime")
                iso = pd.Timestamp(v).strftime("%Y-%m-%d")
                changed = not cell.startswith(iso)
                injected.add((name, "dates"))
            elif spec.kind == "bool":
                if cp.bool_words:
                    words = ["yes", "no"] if rng.random() < 0.5 else ["Y", "N"]
                    pick = rng.choice([words, ["TRUE", "FALSE"], ["true", "false"], ["1", "0"]])
                    cell = pick[0] if v else pick[1]
                    changed = cell not in ("True", "False")
                else:
                    cell = str(bool(v)) if as_text else bool(v)
                injected.add((name, "bool"))
            elif spec.kind == "category":
                s = str(v)
                if cp.case_variants and hit[i]:
                    s2 = rng.choice([s.lower(), s.upper(), s.capitalize() if s.isupper() else s.swapcase()])
                    cell, changed = str(s2), str(s2) != s
                    if changed:
                        injected.add((name, "categories"))
                else:
                    cell = s
            else:
                cell = str(v)

            if cp.whitespace and isinstance(cell, str) and cell and not is_null and rng.random() < cp.rate / 2:
                cell = rng.choice([" ", "  ", "\t"]) + cell + rng.choice(["", " "])
                changed = True
                injected.add((name, "whitespace"))
            if (not is_null and cp.garbage_rate and spec.kind in ("int", "float", "date", "datetime")
                    and rng.random() < cp.garbage_rate):
                cell, changed = str(rng.choice(GARBAGE_POOL)), True
                garbage.iat[i, clean.columns.get_loc(name)] = True
            out.append(cell)
            if changed:
                mask.iat[i, clean.columns.get_loc(name)] = True
        dirty[name] = out

    rename_map: dict[str, str] = {}
    if cp.rename:
        for tgt, src in RENAMES[domain].items():
            rename_map[tgt] = src
            injected.add((tgt, "naming"))
    frame = pd.DataFrame({rename_map.get(k, k): v for k, v in dirty.items()})
    if cp.nested_shapes and not as_text:
        city = rng.choice(CITIES, n)
        frame["address"] = [
            {"city": str(c), "zip": f"{rng.integers(10000, 99999)}"} if rng.random() < 0.8 else f"{c}, somewhere"
            for c in city
        ]
    if as_text:
        frame = frame.astype("string")
    else:
        frame = frame.astype(object).where(frame.notna(), None)
    return MessyDataset(domain, clean, frame, mask, garbage, rename_map, injected, cp)


def make_dataset(domain: str, n: int, seed: int, cp: CorruptionProfile | None = None,
                 as_text: bool = True) -> MessyDataset:
    clean = generate_clean(domain, n, seed)
    return corrupt(clean, domain, seed + 1, cp, as_text)


# ============================================================================ loading into sources


def to_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for r in df.to_dict(orient="records"):
        recs.append({k: (None if (v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))) else v)
                     for k, v in r.items()})
    return recs


def load_into(ds: MessyDataset, kind: str, name: str, workdir: str,
              sql_uri: str | None = None, mongo_uri: str = "mongomock://bench") -> str:
    """Write the dirty frame to a source and return its spec string."""
    from pathlib import Path

    from sqlalchemy import create_engine

    from .connectors.registry import mongo_client

    if kind == "csv":
        path = Path(workdir) / f"{name}.csv"
        ds.dirty.to_csv(path, index=False)
        return str(path)
    if kind == "sql":
        uri = sql_uri or f"sqlite:///{Path(workdir) / 'bench.db'}"
        eng = create_engine(uri)
        ds.dirty.astype("string").to_sql(name, eng, if_exists="replace", index=False)
        eng.dispose()
        return f"{uri}::{name}"
    if kind == "mongo":
        db = mongo_client(mongo_uri)["bench"]
        db[name].drop()
        db[name].insert_many(to_records(ds.dirty))
        return f"{mongo_uri}::bench.{name}"
    raise ValueError(kind)
