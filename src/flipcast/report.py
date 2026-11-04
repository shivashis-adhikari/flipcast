"""turn forecasts and backtests into text, markdown or json."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from . import plans
from .backtest import Result, Score
from .forecast import Forecast, Prediction
from .growth import fmt_count
from .workload import Query

_ARTICLE = {"index": "an", "index-only": "an"}
_FRIENDLY = {"seq scan": "sequential scan", "parallel seq scan": "parallel sequential scan"}


def _a(kind: str) -> str:
    kind = _FRIENDLY.get(kind, kind)
    return f"{_ARTICLE.get(kind.split()[0], 'a')} {kind}"


def _sql_line(q: Query, width: int = 76) -> str:
    one = " ".join(q.sql.split())
    return one if len(one) <= width else one[: width - 3] + "..."


def _month(p: Prediction) -> str:
    return f"{p.month:.0f}"


def _band(p: Prediction, fc: Forecast) -> str:
    if not fc.band:
        return ""

    def m(step: int | None) -> str:
        return f"{step * fc.step_days / 30:.0f}" if step is not None else f">{fc.months}"

    lo, hi = m(p.fast_step), m(p.slow_step)
    return f"{lo}-{hi}" if lo != hi else lo


def summary(p: Prediction) -> str:
    """the one-line version: 'switches from an index scan to a sequential
    scan on events around month 7, at roughly 40.2m rows'."""
    sw = p.switch
    rows = ""
    main = max(p.rows_then.items(), key=lambda kv: kv[1], default=None)
    if main:
        rows = f", at roughly {fmt_count(main[1])} rows in {main[0].split('.')[-1]}"
    head = sw.headline or "changes plan"
    if " to " in head and head.count(" on ") == 1:
        frm, rest = head.split(" to ", 1)
        to, rel = rest.rsplit(" on ", 1)
        head = f"switches from {_a(frm)} to {_a(to)} on {rel}"
    elif " to " in head:
        frm, to = head.split(" to ", 1)
        head = f"switches from {_a(frm)} to {_a(to)}"
    return f"{head} around month {_month(p)}{rows}"


def forecast_text(fc: Forecast) -> str:
    out: list[str] = []
    w = out.append
    w(f"flipcast forecast for {fc.database}")
    w(
        f"trained on {fc.n_snapshots} snapshot{'s' if fc.n_snapshots != 1 else ''}, "
        f"{fc.first_snapshot:%Y-%m-%d} to {fc.last_snapshot:%Y-%m-%d}. "
        f"horizon {fc.months} months in steps of {fc.step_days:g} days."
    )
    if fc.band:
        w(f"month ranges come from re-running with growth {fc.band:.0%} faster and slower.")
    w("")
    w("growth models")
    for line in fc.model_lines:
        w("  " + line)
    w("")
    if not fc.predictions:
        w(f"no plan switches predicted within {fc.months} months.")
    else:
        w("predicted switches")
    for p in fc.predictions:
        sw = p.switch
        band = _band(p, fc)
        w("")
        w(
            f"  month {_month(p):>2}"
            + (f" ({band})" if band and band != _month(p) else "")
            + f"  {p.query.name}"
        )
        w(f"    {_sql_line(p.query)}")
        w(f"    {summary(p)} ({sw.after.at:%Y-%m-%d})")
        for c in sw.changes:
            w(f"      {c}")
        grow = [
            f"{t.split('.')[-1]} {fmt_count(p.rows_now.get(t, 0))} -> {fmt_count(v)}"
            for t, v in sorted(p.rows_then.items())
        ]
        if grow:
            w(f"    rows: {', '.join(grow)}")
        if p.late:
            w(
                f"    estimated cost at the switch: old plan {p.late[0]:,.0f}, "
                f"new plan {p.late[1]:,.0f}"
            )
        else:
            w(
                f"    estimated cost: {sw.before.cost:,.0f} the step before, "
                f"{sw.after.cost:,.0f} after"
            )
        pen = p.early_penalty
        if pen is not None and p.early is not None:
            if pen >= 1.05:
                w(
                    f"    if it flipped today (an unlucky ANALYZE sample can do that): "
                    f"{pen:.1f}x the cost, {p.early[0]:,.0f} -> {p.early[1]:,.0f}"
                )
            elif pen <= 0.95:
                w(
                    f"    the new plan already looks {1 / pen:.1f}x cheaper with today's stats; "
                    "the planner would switch if it saw it"
                )
        if p.ms_now is not None and p.ms_then is not None:
            w(
                f"    latency, scaled by cost from pg_stat_statements: ~{p.ms_now:.2f} ms now, "
                f"~{p.ms_then:.1f} ms at the switch"
            )
        if p.fixes:
            w("    what to do:")
            for f in p.fixes:
                if f.sql:
                    w(f"      {f.sql}")
                    w(f"        {f.why}")
                else:
                    w(f"      - {f.why}")
                if f.verified:
                    w(f"        checked in the shadow: {f.verified}")
    if fc.fragile:
        w("")
        w("plans one ANALYZE away from flipping (sampling noise, no growth needed)")
        for fr in fc.fragile:
            w("")
            w(f"  {fr.query.name}")
            w(f"    {_sql_line(fr.query)}")
            col = f"{fr.table}.{fr.column}"
            w(f"    if ANALYZE comes back with {fr.change} on {col}: {fr.headline}")
            if fr.cost_flipped is None:
                w("    (couldn't make the planner build that plan with the real stats to price it)")
            elif fr.cost_now > 0:
                ratio = fr.cost_flipped / fr.cost_now
                if ratio >= 1.2:
                    w(
                        f"    with the real stats that plan costs {ratio:.1f}x as much "
                        f"({fr.cost_now:,.0f} -> {fr.cost_flipped:,.0f})"
                    )
                else:
                    w(
                        "    both plans cost about the same with the real stats, so this one"
                        " would flip back and forth without much harm"
                    )
    if fc.stable:
        w("")
        w(f"no switch within {fc.months} months")
        width = max(len(q.name) for q, _ in fc.stable)
        for q, shape in fc.stable:
            pts = fc.tracks.get(q.name, [])
            plan = pts[0].plan if pts else None
            desc = _short(plan, shape)
            w(f"  {q.name:<{width}}  {desc}")
    if fc.failed:
        w("")
        w("could not plan (skipped)")
        for q, err in fc.failed:
            w(f"  {q.name}: {err}")
            w(f"    {_sql_line(q, 70)}")
    return "\n".join(out) + "\n"


def _short(plan: dict[str, Any] | None, shape: str) -> str:
    if plan is None:
        return shape
    acc = plans.accesses(plan)
    parts = [f"{rel}: {' + '.join(m)}" for rel, m in sorted(acc.items())]
    j = plans.joins(plan)
    if j:
        parts.append(", ".join(sorted(j)))
    return "; ".join(parts) or shape


def forecast_markdown(fc: Forecast) -> str:
    out: list[str] = []
    w = out.append
    w(f"# Plan forecast: {fc.database}")
    w("")
    w(
        f"Trained on {fc.n_snapshots} snapshots ({fc.first_snapshot:%Y-%m-%d} to "
        f"{fc.last_snapshot:%Y-%m-%d}), {fc.months} months ahead."
    )
    if fc.band:
        w(f"Month ranges are the same forecast with growth {fc.band:.0%} faster and slower.")
    w("")
    if fc.predictions:
        w("| month | query | what changes | rows then |")
        w("|---|---|---|---|")
        for p in fc.predictions:
            band = _band(p, fc)
            m = _month(p) + (f" ({band})" if band and band != _month(p) else "")
            rows = ", ".join(
                f"{t.split('.')[-1]} {fmt_count(v)}" for t, v in sorted(p.rows_then.items())
            )
            w(f"| {m} | `{p.query.name}` | {p.switch.headline} | {rows} |")
        w("")
    else:
        w(f"No switches predicted within {fc.months} months.")
        w("")
    for p in fc.predictions:
        w(f"## Month {_month(p)}: `{p.query.name}`")
        w("")
        w("```sql")
        w(p.query.sql.strip())
        w("```")
        w("")
        w(summary(p) + f" ({p.switch.after.at:%Y-%m-%d}).")
        w("")
        for c in p.switch.changes:
            w(f"- {c}")
        w("")
        if p.late:
            w(
                f"Estimated cost at the switch: old plan {p.late[0]:,.0f}, new plan "
                f"{p.late[1]:,.0f}."
            )
        pen = p.early_penalty
        if pen is not None and p.early is not None and pen >= 1.05:
            w(
                f"If the planner flipped today it would cost {pen:.1f}x more "
                f"({p.early[0]:,.0f} -> {p.early[1]:,.0f})."
            )
        if p.ms_now is not None and p.ms_then is not None:
            w(f"Latency scaled by cost: ~{p.ms_now:.2f} ms now, ~{p.ms_then:.1f} ms at the switch.")
        w("")
        if p.fixes:
            w("Fixes:")
            w("")
            for f in p.fixes:
                line = f"- `{f.sql}` {f.why}." if f.sql else f"- {f.why}."
                if f.verified:
                    line += f" Checked in the shadow: {f.verified}."
                w(line)
            w("")
    if fc.fragile:
        w("## One ANALYZE away from flipping")
        w("")
        w("These plans change under ordinary ANALYZE sampling noise, with no growth at all.")
        w("")
        for fr in fc.fragile:
            line = (
                f"- `{fr.query.name}`: {fr.change} on `{fr.table}.{fr.column}` gives {fr.headline}"
            )
            if fr.cost_flipped is not None and fr.cost_now > 0:
                line += f", {fr.cost_flipped / fr.cost_now:.1f}x the cost with the real stats"
            w(line + ".")
        w("")
    if fc.stable:
        w("## No switch predicted")
        w("")
        for q, shape in fc.stable:
            pts = fc.tracks.get(q.name, [])
            w(f"- `{q.name}`: {_short(pts[0].plan if pts else None, shape)}")
        w("")
    return "\n".join(out)


def _dt(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def forecast_json(fc: Forecast) -> str:
    doc = {
        "database": fc.database,
        "snapshots": fc.n_snapshots,
        "first_snapshot": _dt(fc.first_snapshot),
        "last_snapshot": _dt(fc.last_snapshot),
        "months": fc.months,
        "step_days": fc.step_days,
        "band": fc.band,
        "models": fc.model_lines,
        "switches": [
            {
                "query": p.query.name,
                "sql": p.query.sql,
                "month": round(p.month, 2),
                "at": _dt(p.switch.after.at),
                "month_fast": p.fast_step and p.fast_step * fc.step_days / 30,
                "month_slow": p.slow_step and p.slow_step * fc.step_days / 30,
                "summary": summary(p),
                "changes": p.switch.changes,
                "before": p.switch.before.shape,
                "after": p.switch.after.shape,
                "rows_now": p.rows_now,
                "rows_then": p.rows_then,
                "cost_before_step": p.switch.before.cost,
                "cost_after_step": p.switch.after.cost,
                "cost_if_flipped_today": p.early,
                "cost_old_plan_at_switch": p.late[0] if p.late else None,
                "ms_now": p.ms_now,
                "ms_at_switch": p.ms_then,
                "fixes": [
                    {
                        "kind": f.kind,
                        "sql": f.sql,
                        "why": f.why,
                        "checked": f.verified,
                        "holds": f.holds,
                    }
                    for f in p.fixes
                ],
            }
            for p in fc.predictions
        ],
        "stable": [{"query": q.name, "shape": s} for q, s in fc.stable],
        "fragile": [
            {
                "query": f.query.name,
                "table": f.table,
                "column": f.column,
                "change": f.change,
                "headline": f.headline,
                "shape": f.shape,
                "cost_now": f.cost_now,
                "cost_flipped": f.cost_flipped,
            }
            for f in fc.fragile
        ],
        "failed": [{"query": q.name, "error": e} for q, e in fc.failed],
    }
    return json.dumps(doc, indent=2) + "\n"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def backtest_text(r: Result, database: str) -> str:
    out: list[str] = []
    w = out.append
    m, b = r.model, r.baseline
    w(f"flipcast backtest for {database}")
    w(f"{r.n_snapshots} snapshots, {r.origins} forecast origins. a switch counts as caught when")
    w(f"the predicted date is within {r.tolerance_days:g} days and the new plan is the same one.")
    w("")
    w(f"{'':28}{'flipcast':>14}{'rows only':>14}")
    w(f"{'plan shape right':28}{_pct(m.shape_accuracy):>14}{_pct(b.shape_accuracy):>14}")
    w(f"{'switches caught':28}{_frac(m, 'recall'):>14}{_frac(b, 'recall'):>14}")
    w(f"{'alarms that were right':28}{_frac(m, 'precision'):>14}{_frac(b, 'precision'):>14}")
    mae = m.timing_mae_days
    w(
        f"{'timing error on catches':28}{(f'{mae:.0f} days' if mae is not None else '-'):>14}"
        f"{'-':>14}"
    )
    w("")
    if m.row_err:
        w("row count error by horizon (mean |predicted - actual| / actual)")
        for h in sorted(m.row_err):
            v = m.row_err[h]
            if h in (1, 2, 3, 6, 9, 12) or h == max(m.row_err):
                label = f"{h} month" + ("s" if h != 1 else "")
                w(f"  {label:>9}  {sum(v) / len(v):6.1%}  ({len(v)} points)")
        w("")
    if r.fidelity_total:
        w(
            f"shadow fidelity: {r.fidelity_agree}/{r.fidelity_total} replayed plans match the plan "
            f"the source database chose ({r.fidelity_agree / r.fidelity_total:.0%})"
        )
    if r.skipped_schema_change:
        w(
            f"{r.skipped_schema_change} origins had a schema change inside the horizon; "
            "comparisons across it were skipped"
        )
    w("")
    names = sorted({c.query for c in m.cases if c.outcome != "quiet"})
    quiet = len({c.query for c in m.cases}) - len(names)
    if names:
        w("queries with a real or predicted switch")
        width = max(len(n) for n in names)
        w(f"  {'query':<{width}}  real  caught  missed  false  off")
        for n in names:
            cs = [c for c in m.cases if c.query == n]
            real = sum(1 for c in cs if c.outcome in ("hit", "miss", "mismatch"))
            caught = sum(1 for c in cs if c.outcome == "hit")
            miss = sum(1 for c in cs if c.outcome == "miss")
            false = sum(1 for c in cs if c.outcome == "false_alarm")
            off = sum(1 for c in cs if c.outcome == "mismatch")
            w(f"  {n:<{width}}  {real:>4}  {caught:>6}  {miss:>6}  {false:>5}  {off:>3}")
    if quiet:
        w(f"{quiet} other quer{'y' if quiet == 1 else 'ies'}: no switch in history, none predicted")
    return "\n".join(out) + "\n"


def _frac(s: Score, which: str) -> str:
    hits = s.count("hit")
    if which == "recall":
        den = hits + s.count("miss") + s.count("mismatch")
    else:
        den = hits + s.count("false_alarm") + s.count("mismatch")
    if not den:
        return "-"
    return f"{hits}/{den} {hits / den:.0%}"


def backtest_json(r: Result, database: str) -> str:
    def score(s: Score) -> dict[str, Any]:
        return {
            "shape_accuracy": s.shape_accuracy,
            "precision": s.precision,
            "recall": s.recall,
            "timing_mae_days": s.timing_mae_days,
            "outcomes": {
                o: s.count(o) for o in ("hit", "miss", "false_alarm", "mismatch", "quiet")
            },
            "row_error_by_month": {str(h): sum(v) / len(v) for h, v in sorted(s.row_err.items())},
            "cases": [
                {
                    "query": c.query,
                    "origin": _dt(c.origin),
                    "truth_at": _dt(c.truth_at),
                    "pred_at": _dt(c.pred_at),
                    "outcome": c.outcome,
                    "error_days": c.error_days,
                }
                for c in s.cases
                if c.outcome != "quiet"
            ],
        }

    return (
        json.dumps(
            {
                "database": database,
                "snapshots": r.n_snapshots,
                "origins": r.origins,
                "tolerance_days": r.tolerance_days,
                "fidelity": {"agree": r.fidelity_agree, "total": r.fidelity_total},
                "flipcast": score(r.model),
                "rows_only": score(r.baseline),
            },
            indent=2,
        )
        + "\n"
    )
