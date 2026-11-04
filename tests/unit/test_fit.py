import math
import random

import pytest

from flipcast import fit, share


def test_linfit_exact_line():
    ln = fit.linfit([0, 1, 2, 3], [1, 3, 5, 7])
    assert ln.a == pytest.approx(1)
    assert ln.b == pytest.approx(2)
    assert ln.se_b == pytest.approx(0)


def test_linfit_degenerate():
    assert fit.linfit([5], [3]).b == 0
    assert fit.linfit([1, 1, 1], [1, 2, 3]).b == 0


def test_choose_prefers_constant_for_noise():
    rng = random.Random(7)
    xs = list(range(12))
    ys = [100 + rng.uniform(-1, 1) for _ in xs]
    name, _ = fit.choose(xs, ys, [("constant", fit.constant), ("linear", fit.linear)])
    assert name == "constant"


def test_choose_picks_linear_for_a_line():
    xs = [float(x) for x in range(10)]
    ys = [5 + 3 * x for x in xs]
    name, f = fit.choose(xs, ys, [("constant", fit.constant), ("linear", fit.linear)])
    assert name == "linear"
    assert f(20) == pytest.approx(65)


def test_accelerating_recovers_geometric_inserts():
    # 1000 rows in month 0, 20% more inserts every month after that
    xs, ys, total = [], [], 0.0
    for m in range(10):
        total += 1000 * 1.2**m
        xs.append(30.0 * m)
        ys.append(total)
    f = fit.accelerating(xs, ys)
    truth = ys[-1] + sum(1000 * 1.2**m for m in range(10, 13))
    assert f(30.0 * 12) == pytest.approx(truth, rel=0.03)
    assert f(xs[-1]) == pytest.approx(ys[-1])


def test_accelerating_falls_back_when_rows_shrink():
    f = fit.accelerating([0, 1, 2, 3], [10, 12, 11, 13])
    assert math.isfinite(f(10))


def test_interp():
    xs, ys = [0, 10, 20], [0, 100, 150]
    assert fit.interp(5, xs, ys) == 50
    assert fit.interp(15, xs, ys) == 125
    assert fit.interp(-1, xs, ys) == 0
    assert fit.interp(99, xs, ys) == 150


def test_logit_expit_inverse():
    for p in (0.001, 0.2, 0.5, 0.97):
        assert fit.expit(fit.logit(p)) == pytest.approx(p)


# share model


def _mix_history(marginal, months=10, start_rows=1000.0, monthly=1000.0, start_share=0.02):
    pts, rows, count = [], start_rows, start_rows * start_share
    for m in range(months):
        pts.append((30.0 * m, rows, count / rows))
        rows += monthly
        count += monthly * marginal(m)
    return pts


def test_share_saturates_at_the_marginal_share():
    # new rows are 30% of the value, old ones were 2%. a time trend would run off to 1
    pts = _mix_history(lambda m: 0.30)
    m = share.ShareModel(pts, growing=True)
    assert m.kind in ("steady mix", "shifting mix")
    far = m(pts[-1][0] + 3000, lambda t: pts[-1][1] + 1000 * (t - pts[-1][0]) / 30)
    assert 0.25 < far <= 0.31


def test_share_constant_when_nothing_moves():
    pts = [(30.0 * i, 1000.0 * (i + 1), 0.1) for i in range(8)]
    m = share.ShareModel(pts, growing=True)
    assert m(400, lambda t: 9000) == pytest.approx(0.1, abs=0.005)


def test_share_on_static_table_uses_time_trend():
    pts = [(30.0 * i, 1000.0, 0.1 + 0.02 * i) for i in range(8)]
    m = share.ShareModel(pts, growing=False)
    assert m.kind == "logistic"
    assert m(pts[-1][0] + 60, lambda t: 1000.0) > pts[-1][2]
