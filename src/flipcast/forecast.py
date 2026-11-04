"""Fit, project, sweep, explain the switches. The `flipcast forecast` command."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from psycopg import sql

from . import advise, plans
from .advise import Fix
from .collect import tracked
from .config import Config
from .growth import MONTH_DAYS, GrowthModel
from .shadow import Shadow
from .snapshot import Snapshot
from .sweep import Point, Switch, explain_point, first_step_with, force, rows_at, sweep, switches
from .sweep import tables_in as plan_tables
from .workload import Query

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    query: Query
    switch: Switch
    month: float
    # step of the same switch under faster and slower growth. None: not
    # within the horizon
    fast_step: int | None
    slow_step: int | None
    rows_then: dict[str, float]
    rows_now: dict[str, float]
    # (plan we have today, cost of the new shape forced today)
    early: tuple[float, float] | None = None
    # (old shape forced at the switch, new shape at the switch)
    late: tuple[float, float] | None = None
    ms_now: float | None = None
    ms_then: float | None = None
    fixes: list[Fix] = field(default_factory=list)

    @property
    def early_penalty(self) -> float | None:
        if self.early is None or self.early[0] <= 0:
            return None
        return self.early[1] / self.early[0]


@dataclass
class Forecast:
    database: str
    first_snapshot: datetime
    last_snapshot: datetime
    n_snapshots: int
    months: int
    step_days: float
    band: float
    steps: list[datetime]
    model_lines: list[str]
    predictions: list[Prediction]
    stable: list[tuple[Query, str]]
    failed: list[tuple[Query, str]]
    tracks: dict[str, list[Point]]


def run(
    cfg: Config,
    snaps: list[Snapshot],
    *,
    months: int | None = None,
    step_days: float | None = None,
    scenarios: bool | None = None,
    verify: bool = True,
    keep_shadow: bool = False,
) -> Forecast:
    if not snaps:
        raise ValueError("no snapshots to forecast from; run `flipcast collect` first")
    months = cfg.months if months is None else months
    step_days = cfg.step_days if step_days is None else step_days
    scenarios = cfg.scenarios if scenarios is None else scenarios

    model = GrowthModel(snaps)
    latest = model.latest
    times = model.step_times(months, step_days)
    base_steps = [latest] + [model.project(t) for t in times[1:]]
    queries = tracked(cfg.queries, latest)

    shadow = Shadow(cfg.shadow_dsn, cfg.shadow_database)
    shadow.build(latest)
    conn = shadow.session(latest.settings)
    try:
        tracks = sweep(shadow, conn, queries, base_steps)
        band = model.band()
        fast: dict[str, list[Point]] = {}
        slow: dict[str, list[Point]] = {}
        if scenarios:
            fast = sweep(
                shadow,
                conn,
                queries,
                [latest] + [model.project(t, 1 + band) for t in times[1:]],
                keep_plans=False,
            )
            slow = sweep(
                shadow,
                conn,
                queries,
                [latest] + [model.project(t, 1 - band) for t in times[1:]],
                keep_plans=False,
            )

        preds: list[Prediction] = []
        stable: list[tuple[Query, str]] = []
        failed: list[tuple[Query, str]] = []
        for q in queries:
            pts = tracks[q.name]
            if pts[0].shape is None:
                failed.append((q, pts[0].error or "no plan"))
                continue
            sws = switches(q, pts)
            if not sws:
                stable.append((q, pts[0].shape))
            for sw in sws:
                after_shape = sw.after.shape or ""
                fs = first_step_with(fast[q.name], after_shape, 0) if fast else None
                ss = first_step_with(slow[q.name], after_shape, 0) if slow else None
                tabs = plan_tables(sw.after.plan or {}) | plan_tables(sw.before.plan or {})
                preds.append(
                    Prediction(
                        query=q,
                        switch=sw,
                        month=sw.step * step_days / MONTH_DAYS,
                        fast_step=fs,
                        slow_step=ss,
                        rows_then=rows_at(base_steps[sw.step], tabs),
                        rows_now=rows_at(latest, tabs),
                    )
                )

        for p in preds:
            _costs(shadow, conn, p, base_steps)
            p.fixes = advise.suggest(p.switch, base_steps[p.switch.step], latest.settings)
            if verify:
                for f in p.fixes:
                    if f.kind in ("index", "setting"):
                        _verify(shadow, conn, p, f, base_steps, step_days)
                # fixes that were checked and work first, ones that don't help last
                p.fixes.sort(key=lambda f: {True: 0, None: 1, False: 2}[f.holds])
        preds.sort(key=lambda p: (p.switch.step, p.query.name))
    finally:
        conn.close()
        if not keep_shadow:
            shadow.drop()

    return Forecast(
        database=latest.database,
        first_snapshot=model.snaps[0].taken_at,
        last_snapshot=latest.taken_at,
        n_snapshots=len(model.snaps),
        months=months,
        step_days=step_days,
        band=band if scenarios else 0.0,
        steps=times,
        model_lines=model.describe(months * step_days),
        predictions=preds,
        stable=stable,
        failed=failed,
        tracks=tracks,
    )


def _costs(
    shadow: Shadow, conn: psycopg.Connection[Any], p: Prediction, steps: list[Snapshot]
) -> None:
    sw = p.switch
    old, new = sw.before.plan, sw.after.plan
    if old is None or new is None:
        return
    # what if the planner flipped today, e.g. after an unlucky ANALYZE sample
    shadow.load(conn, steps[0])
    first = steps[0]
    forced_new_now = force(conn, p.query, first, new, old)
    now_pt = explain_point(conn, p.query, first, 0, keep_plans=False)
    if forced_new_now is not None and now_pt.shape is not None:
        p.early = (now_pt.cost, forced_new_now)
    # how close the call is at the switch itself
    shadow.load(conn, steps[sw.step])
    forced_old_then = force(conn, p.query, steps[sw.step], old, new)
    if forced_old_then is not None:
        p.late = (forced_old_then, sw.after.cost)
    q = p.query
    if q.mean_ms and now_pt.cost > 0:
        per_cost = q.mean_ms / now_pt.cost
        p.ms_now = q.mean_ms
        p.ms_then = per_cost * sw.after.cost


def _verify(
    shadow: Shadow,
    conn: psycopg.Connection[Any],
    p: Prediction,
    f: Fix,
    steps: list[Snapshot],
    step_days: float,
) -> None:
    """Re-run the sweep for this one query with the fix in place."""
    bad = p.switch.after.shape
    idx_name = f"flipcast_fix_{abs(hash((p.query.name, f.sql))) % 10**8}"
    pts: list[Point] = []
    try:
        if f.kind == "index" and f.index is not None:
            schema, table, cols, include = f.index
            stmt = sql.SQL("CREATE INDEX {} ON {}.{} ({})").format(
                sql.Identifier(idx_name),
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.SQL(", ").join(sql.Identifier(c) for c in cols),
            )
            if include:
                stmt = sql.Composed(
                    [
                        stmt,
                        sql.SQL(" INCLUDE ({})").format(
                            sql.SQL(", ").join(sql.Identifier(c) for c in include)
                        ),
                    ]
                )
            conn.execute(stmt)
        elif f.kind == "setting" and f.setting is not None:
            conn.execute("select set_config(%s, %s, false)", f.setting)
        for i, st in enumerate(steps):
            shadow.load(conn, st)
            if f.kind == "index" and f.index is not None:
                _size_fix_index(conn, idx_name, f.index, st)
            pts.append(explain_point(conn, p.query, st, i, keep_plans=False))
    except psycopg.Error as e:
        f.verified = f"could not test: {str(e).strip().splitlines()[0]}"
        return
    finally:
        if f.kind == "index":
            conn.execute(
                sql.SQL("DROP INDEX IF EXISTS {}.{}").format(
                    sql.Identifier(f.index[0] if f.index else "public"), sql.Identifier(idx_name)
                )
            )
        elif f.kind == "setting" and f.setting is not None:
            conn.execute(sql.SQL("RESET {}").format(sql.Identifier(f.setting[0])))

    k = p.switch.step
    hit = next((pt.step for pt in pts if pt.shape == bad), None)
    cost_then = pts[k].cost if pts[k].shape else None
    horizon = (len(steps) - 1) * step_days / MONTH_DAYS
    if hit is None:
        f.holds = True
        f.verified = f"switch gone through month {horizon:.0f}"
    elif hit > k:
        f.holds = True
        f.verified = f"switch moves out to month {hit * step_days / MONTH_DAYS:.0f}"
    else:
        f.holds = False
        f.verified = f"does not help, the switch still happens at month {p.month:.0f}"
    if cost_then is not None and p.switch.after.cost > 0 and f.holds:
        f.verified += (
            f"; cost at month {p.month:.0f}: {p.switch.after.cost:,.0f} -> {cost_then:,.0f}"
        )
        f.extra["cost_then"] = cost_then
        if f.holds and cost_then > p.switch.after.cost * 1.05 and hit is None:
            # the planner keeps the old shape, but it costs more than the plan
            # it would have switched to. still a fix (it stops the flip) but say so
            f.verified += " (keeps the old plan, at a higher estimated cost)"
    if pts and pts[0].shape and pts[0].shape != p.switch.before.shape:
        # the fix changes today's plan as well, usually for the better
        f.extra["changes_today"] = True


def _size_fix_index(
    conn: psycopg.Connection[Any],
    name: str,
    spec: tuple[str, str, list[str], list[str]],
    st: Snapshot,
) -> None:
    schema, table, cols, include = spec
    ts = st.table(f"{schema}.{table}")
    if ts is None:
        return
    pages, tuples = advise.index_size(ts, cols, include, ts.reltuples)
    conn.execute(
        "select pg_catalog.pg_restore_relation_stats('schemaname', %s::text, 'relname', %s::text,"
        " 'relpages', %s::integer, 'reltuples', %s::real)",
        (schema, name, pages, tuples),
    )


def describe_shape(plan: dict[str, Any] | None, shape: str | None) -> str:
    if plan is None:
        return shape or "?"
    acc = plans.accesses(plan)
    return "; ".join(f"{rel}: {' + '.join(m)}" for rel, m in sorted(acc.items())) or (shape or "")
