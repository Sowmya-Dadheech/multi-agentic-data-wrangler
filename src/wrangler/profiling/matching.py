"""Cross-source column matching with a cost-ordered cascade.

    exact -> normalized -> fuzzy string -> lexical embedding (name + sample values) -> LLM

Cheapest checks run first; only columns still unresolved escalate to the next level, so the
expensive LLM level sees very few columns (often none).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from rapidfuzz import fuzz

from ..naming import normalize

# small, hand-curated synonym groups; the embedding / LLM levels cover the long tail
SYNONYM_GROUPS = [
    {"date_of_birth", "birth_date", "dob", "birthdate", "birthday"},
    {"customer_id", "cust_id", "client_id", "customerid"},
    {"phone", "phone_number", "tel", "telephone", "mobile"},
    {"email", "e_mail", "email_address", "mail"},
    {"quantity", "qty", "units"},
    {"amount", "amt"},
    {"postal_code", "zip", "zipcode", "zip_code", "postcode"},
    {"product_id", "sku", "prod_id"},
    {"is_remote", "remote", "works_remotely"},
    {"is_discontinued", "discontinued"},
]
SYNONYMS: dict[str, set[str]] = {}
for _group in SYNONYM_GROUPS:
    for _name in _group:
        SYNONYMS.setdefault(_name, set()).update(_group - {_name})


@dataclass
class Match:
    source: str
    target: str
    level: str  # exact | normalized | synonym | fuzzy | embedding | llm
    score: float


def _ngrams(text: str, n: int = 3) -> Counter:
    t = f"  {text.lower()}  "
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def lexical_embedding(name: str, samples: Iterable[str] = ()) -> Counter:
    """A cheap stand-in for a neural embedding: char-trigrams of the name plus the *shape*
    of sample values (digits -> 9, letters -> a), so `dob: 1990-04-12` and
    `date_of_birth: 1985-11-03` land close together. Swap in a real embedding model via
    ``embed_fn`` in :func:`match_columns`.
    """
    vec = _ngrams(normalize(name).replace("_", " "))
    for s in list(samples)[:5]:
        s = str(s)
        digits = sum(c.isdigit() for c in s)
        letters = sum(c.isalpha() for c in s)
        # coarse value-shape features: robust to punctuation / formatting differences
        vec.update({
            f"shape:{''.join('9' if c.isdigit() else 'a' if c.isalpha() else c for c in s)[:16]}": 2,
            f"digits:{digits}": 3,
            f"letters:{min(letters, 12) // 4}": 1,
        })
        if "@" in s:
            vec["has_at"] += 3
    return vec


def match_columns(
    source_cols: dict[str, list[str]],
    target_cols: dict[str, list[str]],
    fuzzy_threshold: float = 88,
    embed_threshold: float = 0.5,
    embed_fn: Callable[[str, list[str]], Counter] | None = None,
    llm_fn: Callable[[list[str], list[str]], dict[str, str]] | None = None,
) -> tuple[list[Match], list[str]]:
    """Map source columns onto target columns.

    ``source_cols`` / ``target_cols``: column name -> a few sample values.
    Returns (matches, unresolved source columns).
    """
    embed = embed_fn or lexical_embedding
    matches: list[Match] = []
    free_targets = dict(target_cols)
    pending = list(source_cols)

    def take(src: str, tgt: str, level: str, score: float) -> None:
        matches.append(Match(src, tgt, level, round(score, 3)))
        free_targets.pop(tgt, None)
        pending.remove(src)

    # 1. exact
    for src in list(pending):
        if src in free_targets:
            take(src, src, "exact", 1.0)
    # 2. normalized (+ synonyms)
    norm_targets = {normalize(t): t for t in free_targets}
    for src in list(pending):
        n = normalize(src)
        if n in norm_targets and norm_targets[n] in free_targets:
            take(src, norm_targets[n], "normalized", 1.0)
            continue
        for syn in sorted(SYNONYMS.get(n, ())):
            if syn in norm_targets and norm_targets[syn] in free_targets:
                take(src, norm_targets[syn], "synonym", 0.95)
                break
    # 3. fuzzy string (edit distance)
    for src in list(pending):
        best = max(
            ((t, fuzz.token_sort_ratio(normalize(src), normalize(t))) for t in free_targets),
            key=lambda x: x[1],
            default=(None, 0),
        )
        if best[0] is not None and best[1] >= fuzzy_threshold:
            take(src, best[0], "fuzzy", best[1] / 100)
    # 4. embedding similarity of name + sample values (global greedy: best pairs first)
    tvecs = {t: embed(t, v) for t, v in free_targets.items()}
    pairs = sorted(
        ((_cosine(embed(src, source_cols[src]), tv), src, t) for src in pending for t, tv in tvecs.items()),
        reverse=True,
    )
    for score, src, t in pairs:
        if score < embed_threshold:
            break
        if src in pending and t in free_targets:
            take(src, t, "embedding", score)
    # 5. LLM judgment for whatever is left
    if llm_fn and pending and free_targets:
        for src, tgt in (llm_fn(list(pending), list(free_targets)) or {}).items():
            if src in pending and tgt in free_targets:
                take(src, tgt, "llm", 0.7)
    return matches, pending
