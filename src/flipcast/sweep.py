"""Run every tracked query against every step and find where plans change.

A sweep takes a list of statistics (one per step), loads each into the
shadow, and EXPLAINs each query. Planning against an empty database is
cheap, a few milliseconds per query, so a 12-month forecast in three growth
scenarios is a few hundred EXPLAINs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg

from . import plans
from .shadow import Shadow
from .snapshot import Snapshot
from .workload import ParamError, Query, render

log = logging.getLogger(__name__)


@dataclass
class Point:
    step: int
    at: datetime
    shape: str | None = None
    cost: float = 0.0
    rows: float = 0.0
    plan: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class Switch:
    query: Query
    step: int
    before: Point
    after: Point
    changes: list[str] = field(default_factory=list)
    headline: str = ""


def sweep(
    shadow: Shadow,
    conn: psycopg.Connection[Any],
    queries: list[Query],
    steps: list[Snapshot],
    *,
    keep_plans: bool = True,
) -> dict[str, list[Point]]:
    out: dict[str, list[Point]] = {q.name: [] for q in queries}
    for i, stats in enumerate(steps):
        shadow.load(conn, stats)
        for q in queries:
            out[q.name].append(explain_point(conn, q, stats, i, keep_plans=keep_plans))
    return out


def explain_point(
    conn: psycopg.Connection[Any], q: Query, stats: Snapshot, step: int, *, keep_plans: bool = True
) -> Point:
    pt = Point(step=step, at=stats.taken_at)
    try:
        p = plans.explain(conn, render(q, stats.taken_at, stats))
    except (ParamError, plans.ExplainError) as e:
        pt.error = str(e)
        return pt
    pt.shape = plans.shape(p)
    pt.cost = plans.cost(p)
    pt.rows = float(p["Plan Rows"])
    pt.plan = p if keep_plans else None
    return pt


def switches(q: Query, points: list[Point]) -> list[Switch]:
    out: list[Switch] = []
    prev: Point | None = None
    for pt in points:
        if pt.shape is None:
            continue
        if prev is not None and pt.shape != prev.shape:
            sw = Switch(query=q, step=pt.step, before=prev, after=pt)
            if prev.plan is not None and pt.plan is not None:
                sw.changes = plans.describe_change(prev.plan, pt.plan)
                sw.headline = plans.headline(prev.plan, pt.plan)
            out.append(sw)
        prev = pt
    return out


def first_step_with(points: list[Point], shape: str, after: int = 0) -> int | None:
    for pt in points:
        if pt.step > after and pt.shape == shape:
            return pt.step
    return None


def force(
    conn: psycopg.Connection[Any],
    q: Query,
    stats: Snapshot,
    target: dict[str, Any],
    other: dict[str, Any],
) -> float | None:
    """Cost of `target`'s shape at these stats, by switching off what `other`
    uses. None if the planner can't be pushed back to that exact shape.

    since PG 18 disabled nodes don't get a 1e10 cost penalty anymore (they are
    counted separately), so the reported cost is the plan's real cost.
    """
    want = plans.shape(target)
    for toggles in (plans.toggles_against(target, other), plans.toggles_except(target)):
        if not toggles:
            continue
        try:
            with conn.transaction():
                for g in toggles:
                    val = "0" if g == "max_parallel_workers_per_gather" else "off"
                    conn.execute("select set_config(%s, %s, true)", (g, val))
                p = plans.explain(conn, render(q, stats.taken_at, stats))
        except (ParamError, plans.ExplainError, psycopg.Error) as e:
            log.debug("forcing %s failed: %s", q.name, e)
            continue
        if plans.shape(p) == want:
            return plans.cost(p)
    return None


def tables_in(plan: dict[str, Any]) -> set[str]:
    out = set()
    for n in plans.walk(plan):
        if "Relation Name" in n:
            schema = n.get("Schema", "public")
            out.add(f"{schema}.{n['Relation Name']}")
    return out


def rows_at(stats: Snapshot, tables: Iterable[str]) -> dict[str, float]:
    out = {}
    for t in tables:
        ts = stats.table(t)
        if ts is not None:
            out[t] = ts.reltuples
    return out
