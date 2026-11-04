"""how the share of one value (one mcv entry) changes as a table grows.

modeling the share directly in time goes wrong for append-mostly tables. if
new rows are 18% 'export' and old ones were 3%, the overall share climbs,
but it can never pass 18% however long you wait. a logistic in time doesn't
know that and runs off toward 1.

so the models here work on counts instead. c = share * rows is the number of
rows holding the value, and m = dc/drows is its share among *new* rows (the
marginal share). projection integrates m over the projected row growth:

  constant       share stays where it is
  steady mix     m fixed at its recent value
  shifting mix   m keeps trending, damped so the trend fades out (holt-style
                 damped trend, in logit space so m stays inside (0, 1))
  logistic       share trends in time; only for tables that aren't growing,
                 where there are no new rows to take a share of
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

from . import fit

Path = Callable[[float], float]  # rows at time t
Pred = Callable[[float, Path], float]  # share at time t
Builder = Callable[[list[tuple[float, float, float]]], Pred]

# monthly damping for the shifting-mix trend. after h months the trend has
# added b * (phi + phi^2 + ... + phi^h), so the total change is capped at
# about 5.7 months of the current trend
PHI = 0.85
MONTH = 30.0
SUBSTEPS = 12


def _constant(pts: list[tuple[float, float, float]]) -> Pred:
    f = pts[-1][2]
    return lambda _t, _path: f


def _logistic_time(pts: list[tuple[float, float, float]]) -> Pred:
    t_last, _, f_last = pts[-1]
    ln = fit.linfit([p[0] for p in pts], [fit.logit(p[2]) for p in pts])
    base = fit.logit(f_last)
    return lambda t, _path: fit.expit(base + ln.b * (t - t_last))


def _window(pts: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    k = max(2, math.ceil(len(pts) / 2))
    return pts[-k:]


def _marginals(pts: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    out = []
    for (t0, r0, f0), (t1, r1, f1) in itertools.pairwise(pts):
        dr = r1 - r0
        if dr <= 0.005 * max(r1, 1.0):
            continue
        m = (f1 * r1 - f0 * r0) / dr
        out.append(((t0 + t1) / 2, min(max(m, 0.0), 1.0)))
    return out


def _integrate(
    t_last: float,
    r_last: float,
    c_last: float,
    t: float,
    path: Path,
    m_at: Callable[[float], float],
) -> float:
    if t <= t_last:
        return c_last / r_last if r_last > 0 else 0.0
    c, prev_t, prev_r = c_last, t_last, r_last
    for j in range(1, SUBSTEPS + 1):
        tj = t_last + (t - t_last) * j / SUBSTEPS
        rj = path(tj)
        c += m_at((prev_t + tj) / 2) * max(rj - prev_r, 0.0)
        prev_t, prev_r = tj, rj
    r = path(t)
    return min(max(c / r, 0.0), 1.0) if r > 0 else 0.0


def _steady_mix(pts: list[tuple[float, float, float]]) -> Pred:
    w = _window(pts)
    ln = fit.linfit([p[1] for p in w], [p[1] * p[2] for p in w])
    m = min(max(ln.b, 0.0), 1.0)
    t_last, r_last, f_last = pts[-1]
    return lambda t, path: _integrate(t_last, r_last, f_last * r_last, t, path, lambda _t: m)


def _shifting_mix(pts: list[tuple[float, float, float]]) -> Pred:
    ms = _marginals(pts)
    t_last, r_last, f_last = pts[-1]
    if len(ms) < 2:
        return _steady_mix(pts)
    recent = ms[-max(2, math.ceil(len(ms) / 2)) :]
    ln = fit.linfit([m[0] for m in recent], [fit.logit(m[1]) for m in recent])
    b = ln.b * MONTH  # logit units per month
    t_m, _ = recent[-1]
    base = fit.logit(ln(t_m))

    def m_at(t: float) -> float:
        h = max(0.0, (t - t_m) / MONTH)
        return fit.expit(base + b * PHI * (1 - PHI**h) / (1 - PHI))

    return lambda t, path: _integrate(t_last, r_last, f_last * r_last, t, path, m_at)


class ShareModel:
    def __init__(self, pts: list[tuple[float, float, float]], growing: bool):
        # pts: (t, rows, share), oldest first
        self.pts = pts
        cands: list[tuple[str, Builder]] = [("constant", _constant)]
        if growing:
            cands += [("steady mix", _steady_mix), ("shifting mix", _shifting_mix)]
        else:
            cands += [("logistic", _logistic_time)]
        self.kind, self.pred = _choose(pts, cands)

    def __call__(self, t: float, path: Path) -> float:
        return min(max(self.pred(t, path), 0.0), 1.0)


def _choose(
    pts: list[tuple[float, float, float]], cands: list[tuple[str, Builder]]
) -> tuple[str, Pred]:
    n = len(pts)
    if n < 3:
        name, b = cands[0] if n < 2 else cands[1]
        return name, b(pts)
    min_train = 2 if n < 6 else 3
    best: tuple[str, Builder] | None = None
    best_err = math.inf
    for name, b in cands:
        num = den = 0.0
        for k in range(min_train, n):
            (t0, r0, _), (t1, r1, f1) = pts[k - 1], pts[k]

            def path(
                t: float, t0: float = t0, r0: float = r0, t1: float = t1, r1: float = r1
            ) -> float:
                return r0 + (r1 - r0) * (t - t0) / (t1 - t0) if t1 != t0 else r1

            pred = b(pts[:k])(t1, path)
            w = 0.5 ** ((n - 1 - k) / fit.HALF_LIFE)
            num += w * abs(pred - f1) / max(f1, 0.005)
            den += w
        err = num / den
        if best is None or err < best_err * (1 - 0.1) - 1e-12:
            best, best_err = (name, b), err
    assert best is not None
    return best[0], best[1](pts)
