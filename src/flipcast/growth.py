"""Per-table and per-column models of how planner statistics change over time.

Fitted from a series of snapshots, then asked for the statistics at any
future time. What gets modeled:

  rows          constant, linear or exponential in time
  pages         linear in rows (bytes per row stays put unless it trends)
  null_frac     constant or linear, clamped to [0, 1]
  n_distinct    constant, or a power law in rows (D = c * R^k)
  MCV list      each value's share follows a logistic trend, then the list is
                rebuilt with the same cutoff rule ANALYZE uses
  histogram     "append" for columns whose top bound keeps moving (ids,
                created_at): old buckets are kept and new rows are spread
                over the extended range. otherwise per-quantile drift or
                static, whichever forecasts better
  correlation   constant or linear, clamped to [-1, 1]

Everything else (array element stats, range stats, avg_width) is carried
forward from the latest snapshot.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from . import fit, pgarray
from .fit import Curve
from .snapshot import ColumnStats, IndexStats, Snapshot, TableStats
from .values import Codec, codec_for, decode_all

DAY_S = 86400.0
MONTH_DAYS = 30.0
# ANALYZE keeps a value in the MCV list only if it is this much more common
# than an average non-MCV value (see analyze_mcv_list in analyze.c)
MCV_CUTOFF = 1.25
# ANALYZE stores n_distinct as a negative fraction once it passes this share
NDISTINCT_RATIO_SWITCH = 0.10


def days(dt: datetime) -> float:
    return dt.astimezone(UTC).timestamp() / DAY_S


def from_days(t: float) -> datetime:
    return datetime.fromtimestamp(t * DAY_S, tz=UTC)


@dataclass(frozen=True)
class Clock:
    """cal is the calendar time being planned. eff is how far growth has
    gotten by then; in the base scenario they are equal."""

    cal: float
    eff: float


@dataclass
class Series:
    t: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def add(self, t: float, y: float) -> None:
        self.t.append(t)
        self.y.append(y)

    def __len__(self) -> int:
        return len(self.t)


def _pct_per_month(f: Curve, t0: float) -> float:
    a, b = f(t0), f(t0 + MONTH_DAYS)
    return (b / a - 1.0) * 100 if a > 0 else 0.0


def _fmt_count(x: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.1f}{unit}"
    return f"{x:.0f}"


class RowModel:
    def __init__(self, s: Series):
        self.last_t = s.t[-1]
        self.last = s.y[-1]
        self.kind, self.curve = fit.choose(
            s.t,
            s.y,
            [("constant", fit.constant), ("linear", fit.linear), ("exponential", fit.exponential)],
        )
        self.rel_se = _relative_slope_se(s, self.kind)
        # anchor the curve on the latest point. a fitted line that misses
        # today's value by 3% would put a fake jump between step 0 and step 1.
        fitted = self.curve(self.last_t)
        self.shift = self.last - fitted
        self.scale = self.last / fitted if fitted > 0 else 1.0

    def __call__(self, t: float) -> float:
        if self.kind == "constant":
            return self.last
        if self.kind == "exponential":
            return max(0.0, self.curve(t) * self.scale)
        return max(0.0, self.curve(t) + self.shift)

    def describe(self) -> str:
        if self.kind == "constant":
            return f"rows flat at {_fmt_count(self.last)}"
        return f"rows {self.kind}, {_pct_per_month(self, self.last_t):+.1f}%/month"


def _relative_slope_se(s: Series, kind: str) -> float:
    if len(s) < 3 or kind == "constant":
        return math.inf
    ys = [math.log(y) for y in s.y] if kind == "exponential" and min(s.y) > 0 else s.y
    ln = fit.linfit(s.t, ys)
    if ln.b == 0:
        return math.inf
    return abs(ln.se_b / ln.b)


class RatioModel:
    """y as a function of rows, e.g. relpages. Linear when there is enough
    spread to fit one, otherwise the latest ratio."""

    def __init__(self, rows: Sequence[float], ys: Sequence[float]):
        self.last_rows, self.last = rows[-1], ys[-1]
        self.line: fit.Line | None = None
        if len(rows) >= 3 and max(rows) > 1.2 * max(min(rows), 1.0):
            ln = fit.linfit(rows, ys)
            if ln.b > 0:
                self.line = ln

    def __call__(self, r: float) -> float:
        if self.last_rows <= 0:
            return self.last
        if self.line is not None:
            # same anchoring trick as RowModel
            shift = self.last - self.line(self.last_rows)
            return max(1.0, self.line(r) + shift) if r > 0 else 0.0
        return self.last * r / self.last_rows


class ColumnModel:
    def __init__(
        self,
        history: list[tuple[float, float, ColumnStats]],
        latest: ColumnStats,
        rows: RowModel,
        default_target: int,
    ):
        # history: (t, table rows, stats) in time order, latest last
        self.latest = latest
        self.rows = rows
        self.codec: Codec | None = codec_for(latest.type_oid)
        self.notes: list[str] = []
        self.target = (
            latest.stattarget
            if latest.stattarget is not None and latest.stattarget >= 0
            else default_target
        )
        self.mcv_array_type = latest.type_name.endswith("]")
        ts = [h[0] for h in history]
        rws = [h[1] for h in history]
        cs = [h[2] for h in history]

        self.null_kind, self.null_curve = fit.choose(
            ts,
            [c.null_frac for c in cs],
            [("constant", fit.constant), ("linear", fit.linear)],
            scale=1.0,
        )
        self._fit_ndistinct(ts, rws, cs)
        self._fit_mcv(ts, cs)
        self._fit_histogram(ts, rws, cs)
        corr = [
            (t, c.correlation) for t, c in zip(ts, cs, strict=True) if c.correlation is not None
        ]
        if corr:
            self.corr_kind, self.corr_curve = fit.choose(
                [t for t, _ in corr],
                [c for _, c in corr],
                [("constant", fit.constant), ("linear", fit.linear)],
                scale=1.0,
            )
        else:
            self.corr_kind, self.corr_curve = "none", fit.constant([0.0], [0.0])

    # n_distinct

    def _fit_ndistinct(self, ts: list[float], rws: list[float], cs: list[ColumnStats]) -> None:
        self.nd_negative = self.latest.n_distinct < 0
        if self.nd_negative:
            # ANALYZE already decided this column scales with the table
            self.nd_kind = "proportional"
            self.nd_ratio = -self.latest.n_distinct
            return
        absd = [
            c.n_distinct if c.n_distinct > 0 else -c.n_distinct * r
            for c, r in zip(cs, rws, strict=True)
        ]
        pairs = [(r, d) for r, d in zip(rws, absd, strict=True) if r > 0 and d > 0]
        self.nd_last = absd[-1]
        if len(pairs) < 2 or max(p[0] for p in pairs) < 1.05 * min(p[0] for p in pairs):
            self.nd_kind = "constant"
            self.nd_curve: Curve = fit.constant([0.0], [self.nd_last])
            return

        def power(xs: Sequence[float], ys: Sequence[float]) -> Curve:
            a, k = _power_fit(xs, ys)
            return lambda r: math.exp(a + k * math.log(max(r, 1.0)))

        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        self.nd_kind, self.nd_curve = fit.choose(
            xs, ys, [("constant", fit.constant), ("power law", power)]
        )
        self.nd_exp = _power_fit(xs, ys)[1]

    def distinct(self, r: float) -> float:
        if self.nd_kind == "proportional":
            return self.nd_ratio * r
        d = self.nd_curve(r)
        if self.nd_kind == "power law":
            d *= self.nd_last / max(self.nd_curve(self.rows.last), 1e-9)
        return max(1.0, d)

    # MCV list

    def _fit_mcv(self, ts: list[float], cs: list[ColumnStats]) -> None:
        self.mcv_values: list[str | None] = []
        self.mcv_curves: list[Curve] = []
        self.mcv_kinds: list[str] = []
        self.mcv_closed = False
        if not self.latest.mcv or not self.latest.mcv_freqs:
            return
        self.mcv_values = pgarray.parse(self.latest.mcv)
        # a categorical column: the MCV list is the whole distribution
        mass = sum(self.latest.mcv_freqs)
        self.mcv_closed = (
            self.latest.histogram is None and mass >= (1 - self.latest.null_frac) - 0.01
        )

        past: list[dict[str | None, float]] = []
        for c in cs:
            if c.mcv and c.mcv_freqs:
                past.append(dict(zip(pgarray.parse(c.mcv), c.mcv_freqs, strict=False)))
            else:
                past.append({})
        for v in self.mcv_values:
            s = Series()
            for t, d in zip(ts, past, strict=True):
                if v in d:
                    s.add(t, d[v])
            kind, curve = fit.choose(
                s.t, s.y, [("constant", fit.constant), ("logistic", fit.logistic)]
            )
            # anchor on today's share
            if kind != "constant":
                now = curve(s.t[-1])
                base = fit.logit(s.y[-1]) - fit.logit(now)
                curve = _shift_logit(curve, base)
            self.mcv_kinds.append(kind)
            self.mcv_curves.append(curve)

    def mcv_at(
        self, t_eff: float, null_frac: float, distinct: float
    ) -> tuple[list[str | None], list[float]] | None:
        if not self.mcv_values:
            return None
        if self.mcv_array_type:
            # can't safely re-print arrays of arrays, keep the list as is
            return self.mcv_values, list(self.latest.mcv_freqs or [])
        cap = max(0.0, 1.0 - null_frac)
        pairs = [
            (v, min(max(c(t_eff), 0.0), 1.0))
            for v, c in zip(self.mcv_values, self.mcv_curves, strict=True)
        ]
        total = sum(f for _, f in pairs)
        if self.mcv_closed and total > 0:
            pairs = [(v, f * cap / total) for v, f in pairs]
        elif total > 0.999 * cap and total > 0:
            pairs = [(v, f * 0.999 * cap / total) for v, f in pairs]
        pairs.sort(key=lambda p: -p[1])
        pairs = pairs[: self.target]
        if not self.mcv_closed:
            # drop values that are no longer common enough to be listed
            while pairs:
                rest = cap - sum(f for _, f in pairs)
                others = distinct - len(pairs)
                if others < 1:
                    break
                avg = rest / others
                if pairs[-1][1] >= MCV_CUTOFF * avg:
                    break
                pairs.pop()
        pairs = [p for p in pairs if p[1] > 0]
        if not pairs:
            return None
        return [p[0] for p in pairs], [p[1] for p in pairs]

    # histogram

    def _fit_histogram(self, ts: list[float], rws: list[float], cs: list[ColumnStats]) -> None:
        self.hist_kind = "none"
        if not self.latest.histogram:
            return
        self.hist_kind = "static"
        if self.codec is None:
            return
        latest = decode_all(self.codec, pgarray.parse(self.latest.histogram))
        if latest is None or len(latest) < 2:
            return
        self.h_latest = latest
        nb = len(latest) - 1
        hs: list[tuple[float, list[float]]] = []
        hrows: list[float] = []
        for t, r, c in zip(ts, rws, cs, strict=True):
            if not c.histogram:
                continue
            b = decode_all(self.codec, pgarray.parse(c.histogram))
            if b is not None and len(b) >= 2:
                hs.append((t, _resample(b, nb)))
                hrows.append(r)
        self.h_series = hs
        if len(hs) < 2:
            return

        tops = [h[-1] for _, h in hs]
        span = latest[-1] - latest[0]
        recent = tops[-4:]
        rising = all(b > a for a, b in itertools.pairwise(recent))
        corr = self.latest.correlation if self.latest.correlation is not None else 0.0
        if rising and corr >= 0.5 and span > 0 and (tops[-1] - tops[0]) > 0.01 * span:
            self.hist_kind = "append"
            # a created_at column moves with the calendar. an id column moves
            # with the row count: under exponential growth a line in time
            # would fall further behind every month.
            xs = [t for t, _ in hs] if self.codec.temporal else hrows
            self.hi_line = fit.linfit(xs, tops)
            self.lo_line = fit.linfit(xs, [h[0] for h in hs])
            self.lo_moves = self.hi_line.b > 0 and self.lo_line.b > 0.05 * self.hi_line.b
            self.t_latest_hist = hs[-1][0]
            self.rows_latest_hist = hrows[-1]
            return

        if len(hs) >= 3 and span > 0 and _drift_beats_static(hs, span):
            self.hist_kind = "drift"
            htimes = [t for t, _ in hs]
            self.q_lines = [fit.linfit(htimes, [h[i] for _, h in hs]) for i in range(nb + 1)]
            self.t_latest_hist = htimes[-1]

    def histogram_at(self, clock: Clock, pop_now: float, pop_then: float) -> str | None:
        if self.hist_kind in ("none",):
            return None
        if self.hist_kind == "static":
            return self.latest.histogram
        assert self.codec is not None
        b = self.h_latest
        nb = len(b) - 1
        if self.hist_kind == "drift":
            dt = clock.eff - self.t_latest_hist
            new = [bi + ln.b * dt for bi, ln in zip(b, self.q_lines, strict=True)]
            return self._print(_monotone(new))

        # append. temporal columns follow the calendar, others the row count
        if self.codec.temporal:
            dx = clock.cal - self.t_latest_hist
        else:
            dx = self.rows(clock.eff) - self.rows_latest_hist
        hi_now, lo_now = b[-1], b[0]
        hi_then = hi_now + self.hi_line.b * dx
        lo_then = lo_now + self.lo_line.b * dx if self.lo_moves else lo_now
        lo_then = min(max(lo_then, lo_now), hi_now)
        if hi_then <= hi_now and lo_then <= lo_now:
            return self.latest.histogram

        qs = [i / nb for i in range(nb + 1)]
        f_lo = fit.interp(lo_then, b, qs)
        kept = pop_now * (1.0 - f_lo)
        added = max(0.0, pop_then - kept)
        grid: list[float] = [lo_then] + [x for x in b if x > lo_then]
        mass: list[float] = [pop_now * (fit.interp(x, b, qs) - f_lo) for x in grid]
        if hi_then > hi_now and added > 0:
            seg = self._new_segment_shape(clock, hi_now, hi_then)
            for x, g in seg:
                grid.append(x)
                mass.append(kept + added * g)
        total = mass[-1]
        if total <= 0:
            return self.latest.histogram
        cdf = [m / total for m in mass]
        for i in range(1, len(cdf)):  # float noise
            cdf[i] = max(cdf[i], cdf[i - 1])
        out = [fit.interp(q, cdf, grid) for q in qs]
        out[0], out[-1] = grid[0], grid[-1]
        return self._print(_monotone(out))

    def _new_segment_shape(
        self, clock: Clock, hi_now: float, hi_then: float
    ) -> list[tuple[float, float]]:
        """CDF of the newly added rows over (hi_now, hi_then].

        a value v was the top of the range at time tau(v). the share of new
        rows at or below v is the share of the growth that happened by tau(v),
        so the shape follows the row curve: flat for linear growth, skewed to
        recent values for exponential growth.
        """
        assert self.codec is not None
        steps = 48
        if not self.codec.temporal:
            # one new value per new row, spread evenly
            return [
                (hi_now + (hi_then - hi_now) * j / steps, j / steps) for j in range(1, steps + 1)
            ]
        rate = 1.0
        t_now = self.t_latest_hist
        t_then = clock.cal if self.codec.temporal else clock.eff
        if self.codec.temporal and clock.cal != t_now:
            rate = (clock.eff - t_now) / (clock.cal - t_now)
        r0 = self.rows(t_now)
        r1 = self.rows(t_now + (t_then - t_now) * rate)
        out = []
        prev = 0.0
        for j in range(1, steps + 1):
            frac = j / steps
            v = hi_now + (hi_then - hi_now) * frac
            tau = t_now + (t_then - t_now) * frac * rate
            g = (self.rows(tau) - r0) / (r1 - r0) if r1 > r0 else frac
            g = min(max(g, prev), 1.0)
            prev = g
            out.append((v, g))
        out[-1] = (hi_then, 1.0)
        return out

    def _print(self, xs: list[float]) -> str:
        assert self.codec is not None
        return pgarray.to_literal(self.codec.to_text(x) for x in xs)

    # everything together

    def at(self, clock: Clock, rows_then: float) -> ColumnStats:
        c = self.latest
        null_frac = min(max(self.null_curve(clock.eff), 0.0), 1.0)
        if self.null_kind == "constant":
            null_frac = c.null_frac
        distinct = min(self.distinct(rows_then), max(rows_then * (1 - null_frac), 1.0))
        if self.nd_kind == "proportional":
            n_distinct = -min(self.nd_ratio, 1.0)
        elif rows_then > 0 and distinct > NDISTINCT_RATIO_SWITCH * rows_then:
            n_distinct = -distinct / rows_then
        else:
            n_distinct = distinct

        mcv = self.mcv_at(clock.eff, null_frac, distinct)
        mcv_mass = sum(mcv[1]) if mcv else 0.0
        latest_mass = sum(c.mcv_freqs) if c.mcv_freqs else 0.0
        pop_now = self.rows.last * max(0.0, 1 - c.null_frac - latest_mass)
        pop_then = rows_then * max(0.0, 1 - null_frac - mcv_mass)

        corr = c.correlation
        if c.correlation is not None and self.corr_kind != "constant":
            corr = min(max(self.corr_curve(clock.eff), -1.0), 1.0)

        return replace(
            c,
            null_frac=null_frac,
            n_distinct=n_distinct,
            mcv=pgarray.to_literal(mcv[0])
            if mcv and not self.mcv_array_type
            else (c.mcv if mcv else None),
            mcv_freqs=mcv[1] if mcv else None,
            histogram=self.histogram_at(clock, pop_now, pop_then),
            correlation=corr,
        )

    def describe(self, horizon_days: float) -> list[str]:
        out = []
        t0 = self.rows.last_t
        if self.nd_kind == "power law":
            out.append(f"n_distinct ~ rows^{self.nd_exp:.2f}")
        elif self.nd_kind == "proportional":
            out.append(f"n_distinct {self.nd_ratio:.3g} x rows")
        if self.null_kind != "constant":
            then = min(max(self.null_curve(t0 + horizon_days), 0), 1)
            out.append(f"null_frac {self.latest.null_frac:.3f} -> {then:.3f}")
        moving = []
        for v, k, c in zip(self.mcv_values, self.mcv_kinds, self.mcv_curves, strict=True):
            if k == "constant":
                continue
            a, b = c(t0), c(t0 + horizon_days)
            if abs(b - a) >= 0.01:
                moving.append((abs(b - a), f"{v!s:.20}: {a:.1%} -> {b:.1%}"))
        if moving:
            moving.sort(reverse=True)
            out.append("mcv " + ", ".join(m for _, m in moving[:3]))
        if self.hist_kind == "append":
            assert self.codec is not None
            if self.codec.temporal:
                rate = self.hi_line.b * MONTH_DAYS / self.codec.per_day
                what = f"+{rate:.3g} days/month"
            else:
                what = f"{self.hi_line.b:.3g} per new row"
            out.append(
                f"histogram append, top bound {what}"
                + (", bottom moving too" if self.lo_moves else "")
            )
        elif self.hist_kind == "drift":
            out.append("histogram drifting")
        if self.corr_kind == "linear" and self.latest.correlation is not None:
            then = min(max(self.corr_curve(t0 + horizon_days), -1), 1)
            if abs(then - self.latest.correlation) > 0.05:
                out.append(f"correlation {self.latest.correlation:.2f} -> {then:.2f}")
        return out


def _power_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    ln = fit.linfit([math.log(x) for x in xs], [math.log(y) for y in ys])
    k = min(max(ln.b, 0.0), 1.0)
    # refit the intercept with the clamped slope
    a = sum(math.log(y) - k * math.log(x) for x, y in zip(xs, ys, strict=True)) / len(xs)
    return a, k


def _shift_logit(curve: Curve, delta: float) -> Curve:
    return lambda t: fit.expit(fit.logit(curve(t)) + delta)


def _resample(bounds: list[float], nb: int) -> list[float]:
    if len(bounds) - 1 == nb:
        return list(bounds)
    m = len(bounds) - 1
    qs = [i / m for i in range(m + 1)]
    return [fit.interp(j / nb, qs, bounds) for j in range(nb + 1)]


def _monotone(xs: list[float]) -> list[float]:
    out = list(xs)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return out


def _drift_beats_static(hs: list[tuple[float, list[float]]], span: float) -> bool:
    err_static, err_drift = [], []
    for k in range(2, len(hs)):
        train, (tk, actual) = hs[:k], hs[k]
        times = [t for t, _ in train]
        last = train[-1][1]
        pred = []
        for i in range(len(actual)):
            ln = fit.linfit(times, [h[i] for _, h in train])
            pred.append(last[i] + ln.b * (tk - times[-1]))
        err_static.append(sum(abs(a - b) for a, b in zip(last, actual, strict=True)) / span)
        err_drift.append(sum(abs(a - b) for a, b in zip(pred, actual, strict=True)) / span)
    return sum(err_drift) < 0.9 * sum(err_static)


class TableModel:
    def __init__(self, history: list[tuple[float, TableStats]], default_target: int):
        self.latest = history[-1][1]
        rows = Series()
        pts = [(t, ts) for t, ts in history if ts.reltuples >= 0]
        if not pts:
            pts = [(history[-1][0], replace(self.latest, reltuples=0.0))]
        for t, ts in pts:
            rows.add(t, ts.reltuples)
        self.rows = RowModel(rows)
        rws = [ts.reltuples for _, ts in pts]
        self.pages = RatioModel(rws, [float(ts.relpages) for _, ts in pts])
        lp = max(self.latest.relpages, 1)
        self.vis_frac = min(self.latest.relallvisible / lp, 1.0)
        self.frozen_frac = min(self.latest.relallfrozen / lp, 1.0)

        self.index_pages: dict[str, RatioModel] = {}
        self.index_frac: dict[str, float] = {}
        for ix in self.latest.indexes:
            r, p = [], []
            for _, ts in pts:
                for other in ts.indexes:
                    if other.name == ix.name:
                        r.append(ts.reltuples)
                        p.append(float(other.relpages))
            self.index_pages[ix.name] = RatioModel(
                r or [self.latest.reltuples], p or [float(ix.relpages)]
            )
            base = self.latest.reltuples
            self.index_frac[ix.name] = min(ix.reltuples / base, 1.0) if base > 0 else 1.0

        self.columns: dict[tuple[str, bool], ColumnModel] = {}
        for col in self.latest.columns:
            hist: list[tuple[float, float, ColumnStats]] = []
            for t, ts in pts:
                c = ts.column(col.name, col.inherited)
                if c is not None and c.type_oid == col.type_oid:
                    hist.append((t, max(ts.reltuples, 0.0), c))
            if not hist or hist[-1][2] is not col:
                hist.append((history[-1][0], max(self.latest.reltuples, 0.0), col))
            self.columns[col.key] = ColumnModel(hist, col, self.rows, default_target)

    def at(self, clock: Clock) -> TableStats:
        t = self.latest
        rows = self.rows(clock.eff)
        pages = max(0, round(self.pages(rows)))
        indexes = []
        for ix in t.indexes:
            ip = self.index_pages[ix.name](rows)
            itup = rows * (self.index_frac[ix.name] if ix.partial else 1.0)
            indexes.append(replace(ix, relpages=max(1, round(ip)), reltuples=itup))
        cols = [self.columns[c.key].at(clock, rows) for c in t.columns]
        return replace(
            t,
            reltuples=rows,
            relpages=pages,
            relallvisible=min(pages, round(pages * self.vis_frac)),
            relallfrozen=min(pages, round(pages * self.frozen_frac)),
            n_live_tup=round(rows),
            n_dead_tup=None,
            columns=cols,
            indexes=indexes,
        )

    def describe(self, horizon_days: float) -> list[str]:
        t0 = self.rows.last_t
        head = (
            f"{self.latest.qualname}: {self.rows.describe()}, "
            f"{_fmt_count(self.rows(t0))} now, "
            f"{_fmt_count(self.rows(t0 + horizon_days))} in "
            f"{horizon_days / MONTH_DAYS:.0f} months"
        )
        lines = [head]
        for (name, inh), cm in self.columns.items():
            d = cm.describe(horizon_days)
            if d:
                lines.append(f"  {name}{' (inherited)' if inh else ''}: " + "; ".join(d))
        return lines


class GrowthModel:
    def __init__(self, snaps: list[Snapshot]):
        if not snaps:
            raise ValueError("need at least one snapshot")
        self.snaps = sorted(snaps, key=lambda s: s.taken_at)
        self.latest = self.snaps[-1]
        self.t_latest = days(self.latest.taken_at)
        try:
            default_target = int(self.latest.settings.get("default_statistics_target", "100"))
        except ValueError:
            default_target = 100
        self.tables: dict[str, TableModel] = {}
        for tab in self.latest.tables:
            history = []
            for s in self.snaps:
                other = s.table(tab.qualname)
                if other is not None:
                    history.append((days(s.taken_at), other))
            self.tables[tab.qualname] = TableModel(history, default_target)

    def band(self) -> float:
        """Relative uncertainty in the growth rate, for the slow/fast scenarios.

        Taken from the slope's standard error on the biggest tables, clamped
        so two snapshots don't produce a band of +-500%.
        """
        weighted, weight = 0.0, 0.0
        for tm in self.tables.values():
            if tm.rows.kind == "constant" or math.isinf(tm.rows.rel_se):
                continue
            w = max(tm.rows.last, 1.0)
            weighted += w * tm.rows.rel_se
            weight += w
        if weight == 0:
            return 0.25
        return min(max(1.28 * weighted / weight, 0.10), 0.50)

    def clock(self, at: datetime, rate: float = 1.0) -> Clock:
        cal = days(at)
        return Clock(cal=cal, eff=self.t_latest + (cal - self.t_latest) * rate)

    def project(self, at: datetime, rate: float = 1.0) -> Snapshot:
        clock = self.clock(at, rate)
        tables = [tm.at(clock) for tm in self.tables.values()]
        return replace(self.latest, taken_at=at, tables=tables, plans={}, projected=True)

    def step_times(self, months: int, step_days: float) -> list[datetime]:
        return [self.latest.taken_at + timedelta(days=step_days * m) for m in range(months + 1)]

    def describe(self, horizon_days: float) -> list[str]:
        lines: list[str] = []
        for tm in self.tables.values():
            lines.extend(tm.describe(horizon_days))
        return lines


def rows_of(snap: Snapshot) -> dict[str, float]:
    return {t.qualname: t.reltuples for t in snap.tables}


def index_by_name(t: TableStats, name: str) -> IndexStats | None:
    for ix in t.indexes:
        if ix.name == name:
            return ix
    return None
