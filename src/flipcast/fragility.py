"""plans that can flip today, from nothing but analyze sampling noise.

analyze reads a sample of 300 x statistics_target rows, so every fraction it
stores has sampling error. a plan sitting near a crossover can flip on the
next analyze with no growth at all. clerk's february 2026 outage was that
kind: after an automatic analyze the planner assumed a column held almost no
non-null values, and the query returned over 17,000.

for each query, this nudges the stats of the columns its plan filters on by
about two standard errors of a sample of that size, one change at a time,
and re-plans. any change of shape is reported. n_distinct gets a wider
range (half and double), because analyze's estimator for it is known to be
far less stable than its fractions. correlation moves by its own sampling
error, which is zero for a column stored in perfect order.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import psycopg

from . import plans
from .advise import scans
from .shadow import Shadow
from .snapshot import ColumnStats, Snapshot, TableStats
from .sweep import Point, explain_point, force
from .workload import Query

log = logging.getLogger(__name__)

Z = 2.0


@dataclass
class Fragile:
    query: Query
    table: str
    column: str
    change: str  # what was nudged, e.g. "null_frac 0.950 -> 0.996"
    headline: str
    shape: str
    cost_now: float
    cost_flipped: float | None  # the flipped plan priced with the real stats


def sample_size(t: TableStats, default_target: int) -> int:
    targets = [c.stattarget for c in t.columns if c.stattarget is not None and c.stattarget >= 0]
    target = max(targets) if targets else default_target
    return max(1, min(int(t.reltuples), 300 * target))


def _sigma(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-9) / n)


def nudges(t: TableStats, c: ColumnStats, n: int) -> Iterator[tuple[str, ColumnStats]]:
    """(description, nudged copy) pairs for one column."""
    p = c.null_frac
    s = _sigma(p, n)
    for q in (p - Z * s, p + Z * s):
        q = min(max(q, 0.0), 1.0)
        if abs(q - p) > 1e-6:
            yield f"null_frac {p:.3f} -> {q:.3f}", replace(c, null_frac=q)
    if p > 0.9 and p < 1 - 1 / n:
        # the clerk case: the sample saw almost no non-null values at all
        q = 1 - 1 / n
        yield f"null_frac {p:.3f} -> {q:.4f}", replace(c, null_frac=q)

    if c.mcv_freqs:
        for sign in (-1, 1):
            freqs = [min(max(f + sign * Z * _sigma(f, n), 0.0), 1.0) for f in c.mcv_freqs]
            total = sum(freqs)
            if total > 1 - c.null_frac:
                freqs = [f * (1 - c.null_frac) / total for f in freqs]
            word = "higher" if sign > 0 else "lower"
            yield (
                f"common value shares {word} (top one {c.mcv_freqs[0]:.3f} -> {freqs[0]:.3f})",
                replace(c, mcv_freqs=freqs),
            )

    nd = c.n_distinct
    if nd > 0:
        for k in (0.5, 2.0):
            v = max(1.0, min(nd * k, t.reltuples))
            if v != nd:
                yield f"n_distinct {nd:,.0f} -> {v:,.0f}", replace(c, n_distinct=v)
    elif -1 < nd < 0:
        for k in (0.5, 2.0):
            v = max(-1.0, nd * k)
            yield f"n_distinct {nd:.3f} -> {v:.3f}", replace(c, n_distinct=v)

    if c.correlation is not None:
        # standard error of a rank correlation from n pairs is about
        # (1 - rho^2) / sqrt(n): a perfectly ordered column reads 1.0 every time
        rho = c.correlation
        d = Z * (1 - rho * rho) / math.sqrt(n)
        if d >= 0.005:
            for v in (rho - d, rho + d):
                v = min(max(v, -1.0), 1.0)
                yield f"correlation {rho:.3f} -> {v:.3f}", replace(c, correlation=v)


def check(
    shadow: Shadow,
    conn: psycopg.Connection[Any],
    queries: list[Query],
    snap: Snapshot,
    baseline: dict[str, Point],
) -> list[Fragile]:
    try:
        default_target = int(snap.settings.get("default_statistics_target", "100"))
    except ValueError:
        default_target = 100
    found: list[Fragile] = []
    shadow.load(conn, snap)
    for q in queries:
        base = baseline.get(q.name)
        if base is None or base.plan is None or base.shape is None:
            continue
        cols: list[tuple[str, str]] = []
        for s in scans(base.plan, snap):
            for c in s.cond_cols:
                if (f"{s.schema}.{s.table}", c) not in cols:
                    cols.append((f"{s.schema}.{s.table}", c))
        hit = _first_flip(shadow, conn, q, snap, base, cols, default_target)
        if hit is not None:
            found.append(hit)
    return found


def _first_flip(
    shadow: Shadow,
    conn: psycopg.Connection[Any],
    q: Query,
    snap: Snapshot,
    base: Point,
    cols: list[tuple[str, str]],
    default_target: int,
) -> Fragile | None:
    for qual, name in cols:
        t = snap.table(qual)
        if t is None:
            continue
        col = t.column(name)
        if col is None:
            continue
        n = sample_size(t, default_target)
        for what, nudged in nudges(t, col, n):
            # only this one column changes. a shallow copy is enough for
            # rendering params, and the shadow gets a single-column restore
            alt_t = replace(t, columns=[nudged if c.key == col.key else c for c in t.columns])
            alt = replace(snap, tables=[alt_t if x is t else x for x in snap.tables])
            shadow.load_column(conn, qual, nudged, snap.server_version_num)
            pt = explain_point(conn, q, alt, 0)
            shadow.load_column(conn, qual, col, snap.server_version_num)
            if pt.shape is None or pt.shape == base.shape:
                continue
            assert pt.plan is not None and base.plan is not None
            # what the flipped plan costs with the real stats
            real = force(conn, q, snap, pt.plan, base.plan)
            return Fragile(
                query=q,
                table=qual,
                column=name,
                change=what,
                headline=plans.headline(base.plan, pt.plan),
                shape=pt.shape,
                cost_now=base.cost,
                cost_flipped=real,
            )
    return None
