"""Least squares on short series, and picking between candidate curves.

The series are tiny (one point per snapshot), so everything here is plain
Python. Candidates are compared by rolling-origin forecast error: fit on the
first k points, predict point k+1, repeat. In-sample fit would always favor
the more flexible curve, which is exactly wrong for extrapolation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Curve = Callable[[float], float]
Fitter = Callable[[Sequence[float], Sequence[float]], Curve]


@dataclass(frozen=True)
class Line:
    a: float
    b: float
    se_b: float  # standard error of the slope, inf when unknown
    n: int

    def __call__(self, x: float) -> float:
        return self.a + self.b * x


def linfit(xs: Sequence[float], ys: Sequence[float]) -> Line:
    n = len(xs)
    if n == 0:
        raise ValueError("no points")
    my = sum(ys) / n
    if n == 1:
        return Line(my, 0.0, math.inf, 1)
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return Line(my, 0.0, math.inf, n)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    a = my - b * mx
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
        se = math.sqrt(rss / (n - 2) / sxx)
    else:
        se = math.inf
    return Line(a, b, se, n)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def constant(xs: Sequence[float], ys: Sequence[float]) -> Curve:
    # the latest value, not the mean: if nothing is trending, the newest
    # observation is the best guess for tomorrow
    v = ys[-1]
    return lambda _x: v


def linear(xs: Sequence[float], ys: Sequence[float]) -> Curve:
    return linfit(xs, ys)


def exponential(xs: Sequence[float], ys: Sequence[float]) -> Curve:
    if any(y <= 0 for y in ys):
        return linear(xs, ys)
    ln = linfit(xs, [math.log(y) for y in ys])
    return lambda x: math.exp(min(ln(x), 700.0))


def accelerating(xs: Sequence[float], ys: Sequence[float]) -> Curve:
    """The insert rate grows exponentially and the table is its running sum.

    dy/dx = exp(a + b x), so y = y_last + (exp(a + b x) - exp(a + b x_last)) / b.
    b = 0 is plain linear growth. this is what a table looks like when the
    business behind it grows: more rows per month every month, on top of
    everything inserted before.
    """
    rates, mids = [], []
    for (x0, y0), (x1, y1) in zip(
        zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False
    ):
        if x1 > x0 and y1 > y0:
            rates.append((y1 - y0) / (x1 - x0))
            mids.append((x0 + x1) / 2)
    if len(rates) < 2 or len(rates) < len(xs) - 1:
        return linear(xs, ys)
    ln = linfit(mids, [math.log(r) for r in rates])
    x_last, y_last = xs[-1], ys[-1]
    a, b = ln.a, ln.b
    if abs(b) < 1e-9:
        rate = math.exp(a + b * x_last)
        return lambda x: y_last + rate * (x - x_last)

    def f(x: float) -> float:
        return y_last + (math.exp(min(a + b * x, 700.0)) - math.exp(a + b * x_last)) / b

    return f


def logistic(xs: Sequence[float], ys: Sequence[float]) -> Curve:
    # straight line in logit space: a share that grows but can never pass 1
    ln = linfit(xs, [logit(y) for y in ys])
    return lambda x: expit(ln(x))


# recent forecast origins count more. a curve that was wrong when it only had
# three points to go on says little about how it does with twelve.
HALF_LIFE = 3.0


def rolling_error(
    xs: Sequence[float],
    ys: Sequence[float],
    fitter: Fitter,
    *,
    min_train: int | None = None,
    scale: float | None = None,
) -> float:
    """Weighted mean relative one-step-ahead error."""
    n = len(xs)
    if min_train is None:
        min_train = 2 if n < 6 else 3
    num = den = 0.0
    for k in range(min_train, n):
        f = fitter(xs[:k], ys[:k])
        pred, actual = f(xs[k]), ys[k]
        denom = scale if scale is not None else max(abs(actual), 1e-9)
        w = 0.5 ** ((n - 1 - k) / HALF_LIFE)
        num += w * abs(pred - actual) / denom
        den += w
    return num / den if den else math.inf


def choose(
    xs: Sequence[float],
    ys: Sequence[float],
    candidates: list[tuple[str, Fitter]],
    *,
    scale: float | None = None,
    slack: float = 0.1,
) -> tuple[str, Curve]:
    """Pick the candidate with the lowest rolling error.

    candidates go simplest first, and the first one must be the no-trend
    model. a more complex one has to beat every simpler one by `slack`
    (relative) to be picked. with 4+ points a trend also has to be there at
    all: a plain line through the points, or through the newer half, needs a
    slope of at least two standard errors. without that check, noise around a flat value wins
    every so often, and twelve months of extrapolated noise is a fake switch.
    """
    if len(xs) < 3:
        name, fitter = candidates[0] if len(xs) < 2 else candidates[min(1, len(candidates) - 1)]
        return name, fitter(xs, ys)
    half = max(4, math.ceil(len(xs) / 2))
    if len(xs) >= 4 and not (has_trend(xs, ys) or has_trend(xs[-half:], ys[-half:])):
        return candidates[0][0], candidates[0][1](xs, ys)
    best_name, best_fit = candidates[0]
    best_err = rolling_error(xs, ys, best_fit, scale=scale)
    for name, fitter in candidates[1:]:
        err = rolling_error(xs, ys, fitter, scale=scale)
        if err < best_err * (1 - slack) - 1e-12:
            best_name, best_fit, best_err = name, fitter, err
    return best_name, best_fit(xs, ys)


def has_trend(xs: Sequence[float], ys: Sequence[float], t: float = 2.0) -> bool:
    ln = linfit(xs, ys)
    if ln.b == 0:
        return False
    if math.isinf(ln.se_b) or ln.se_b == 0:
        return ln.se_b == 0  # a perfect line is a trend
    return abs(ln.b) >= t * ln.se_b


def interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise linear interpolation, xs ascending. Clamps at the ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span == 0:
        return ys[hi]
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / span
