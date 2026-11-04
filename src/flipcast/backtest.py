"""would the forecast have worked? replay history and check.

for each origin k (with at least min_train snapshots up to and including k):
fit the growth models on snapshots[0..k], project to the exact times of the
later snapshots, and plan every query against those projections. ground
truth is the plan each query gets with the stats analyze actually produced
at those later times, replayed through the same shadow.

replaying the truth through the shadow (instead of only trusting the plans
the collector observed on the source) means the comparison isolates the
statistics forecast. the observed plans are still used: replayed vs
observed agreement measures how faithful the shadow itself is.

scores:
  shape agreement   share of (origin, later snapshot, query) where the
                    predicted shape is the true shape
  switch detection  per (origin, query): did the first switch after the
                    origin get predicted, within `tolerance_days`?
  timing error      for detected switches, predicted minus actual date
  row error         |predicted - actual| / actual for table row counts

the same thing runs for a rows-only baseline (scale row counts, freeze the
column stats), which is what you'd get from "just multiply by growth".
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .collect import tracked
from .config import Config
from .growth import MONTH_DAYS, GrowthModel
from .shadow import Shadow
from .snapshot import Snapshot
from .sweep import explain_point
from .workload import Query

log = logging.getLogger(__name__)


@dataclass
class Case:
    query: str
    origin: datetime
    truth_at: datetime | None
    pred_at: datetime | None
    truth_shape: str | None
    pred_shape: str | None
    outcome: str  # hit, miss, false_alarm, quiet, mismatch (wrong time or wrong new plan)

    @property
    def error_days(self) -> float | None:
        if self.truth_at is None or self.pred_at is None:
            return None
        return (self.pred_at - self.truth_at).total_seconds() / 86400


@dataclass
class Score:
    name: str
    agree: int = 0
    total: int = 0
    cases: list[Case] = field(default_factory=list)
    row_err: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))

    def count(self, outcome: str) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def precision(self) -> float | None:
        hits = self.count("hit")
        alarms = hits + self.count("false_alarm") + self.count("mismatch")
        return hits / alarms if alarms else None

    @property
    def recall(self) -> float | None:
        hits = self.count("hit")
        real = hits + self.count("miss") + self.count("mismatch")
        return hits / real if real else None

    @property
    def timing_mae_days(self) -> float | None:
        errs = [
            abs(c.error_days) for c in self.cases if c.outcome == "hit" and c.error_days is not None
        ]
        return sum(errs) / len(errs) if errs else None

    @property
    def shape_accuracy(self) -> float | None:
        return self.agree / self.total if self.total else None


@dataclass
class Result:
    n_snapshots: int
    origins: int
    tolerance_days: float
    queries: list[str]
    model: Score
    baseline: Score
    fidelity_agree: int
    fidelity_total: int
    skipped_schema_change: int


def run(
    cfg: Config,
    snaps: list[Snapshot],
    *,
    min_train: int = 4,
    horizon_months: float = 12,
    tolerance_days: float = 45,
) -> Result:
    snaps = sorted(snaps, key=lambda s: s.taken_at)
    if len(snaps) < min_train + 1:
        raise ValueError(f"backtest needs at least {min_train + 1} snapshots, have {len(snaps)}")
    latest = snaps[-1]
    queries = tracked(cfg.queries, latest)
    shadow = Shadow(cfg.shadow_dsn, cfg.shadow_database + "_bt")
    shadow.build(latest)
    conn = shadow.session(latest.settings)
    try:
        truth: list[dict[str, str | None]] = []
        fid_agree = fid_total = 0
        for s in snaps:
            row = _plan_all(shadow, conn, queries, s)
            truth.append(row)
            for q in queries:
                obs = s.plans.get(q.name)
                if obs is not None and not obs.error and row.get(q.name):
                    fid_total += 1
                    fid_agree += obs.shape == row[q.name]

        good = [q.name for q in queries if any(t.get(q.name) for t in truth)]
        model, base = Score("flipcast"), Score("rows only")
        skipped = 0
        origins = range(min_train - 1, len(snaps) - 1)
        for k in origins:
            later = [
                j
                for j in range(k + 1, len(snaps))
                if (snaps[j].taken_at - snaps[k].taken_at).days <= horizon_months * MONTH_DAYS
            ]
            if any(snaps[j].schema_hash != snaps[k].schema_hash for j in later):
                skipped += 1
                later = [j for j in later if snaps[j].schema_hash == snaps[k].schema_hash]
            for score, rows_only in ((model, False), (base, True)):
                gm = GrowthModel(snaps[: k + 1], rows_only=rows_only)
                preds: dict[int, dict[str, str | None]] = {}
                for j in later:
                    proj = gm.project(snaps[j].taken_at)
                    preds[j] = _plan_all(shadow, conn, queries, proj)
                    h = round((snaps[j].taken_at - snaps[k].taken_at).days / MONTH_DAYS)
                    for t in snaps[j].tables:
                        p = proj.table(t.qualname)
                        if p is not None and t.reltuples > 1000:
                            score.row_err[h].append(abs(p.reltuples - t.reltuples) / t.reltuples)
                for name in good:
                    base_shape = truth[k].get(name)
                    if base_shape is None:
                        continue
                    for j in later:
                        ts, ps = truth[j].get(name), preds[j].get(name)
                        if ts is not None and ps is not None:
                            score.total += 1
                            score.agree += ts == ps
                    score.cases.append(
                        _case(name, snaps, k, later, truth, preds, base_shape, tolerance_days)
                    )
    finally:
        conn.close()
        shadow.drop()

    return Result(
        n_snapshots=len(snaps),
        origins=len(origins),
        tolerance_days=tolerance_days,
        queries=good,
        model=model,
        baseline=base,
        fidelity_agree=fid_agree,
        fidelity_total=fid_total,
        skipped_schema_change=skipped,
    )


def _plan_all(
    shadow: Shadow, conn: Any, queries: list[Query], stats: Snapshot
) -> dict[str, str | None]:
    shadow.load(conn, stats)
    return {q.name: explain_point(conn, q, stats, 0, keep_plans=False).shape for q in queries}


def _case(
    name: str,
    snaps: list[Snapshot],
    k: int,
    later: list[int],
    truth: list[dict[str, str | None]],
    preds: dict[int, dict[str, str | None]],
    base_shape: str,
    tol: float,
) -> Case:
    t_j = next((j for j in later if truth[j].get(name) not in (None, base_shape)), None)
    p_j = next((j for j in later if preds[j].get(name) not in (None, base_shape)), None)
    t_at = snaps[t_j].taken_at if t_j is not None else None
    p_at = snaps[p_j].taken_at if p_j is not None else None
    t_shape = truth[t_j].get(name) if t_j is not None else None
    p_shape = preds[p_j].get(name) if p_j is not None else None
    if t_j is None and p_j is None:
        outcome = "quiet"
    elif t_j is None:
        outcome = "false_alarm"
    elif p_j is None:
        outcome = "miss"
    else:
        assert t_at is not None and p_at is not None
        close = abs((p_at - t_at).total_seconds()) <= tol * 86400
        outcome = "hit" if close and t_shape == p_shape else "mismatch"
    return Case(name, snaps[k].taken_at, t_at, p_at, t_shape, p_shape, outcome)
