"""Column-name normalization shared by the profiler, matcher and fingerprinting."""

from __future__ import annotations

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(name: str) -> str:
    """`customerId` / `Customer-ID` / ` customer id ` -> `customer_id`."""
    name = _CAMEL.sub("_", str(name).strip())
    name = _NON_ALNUM.sub("_", name.lower())
    return name.strip("_") or "col"


def is_snake_case(name: str) -> bool:
    return normalize(name) == name
