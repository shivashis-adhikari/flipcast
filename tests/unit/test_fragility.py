import pytest

from flipcast.fragility import nudges, sample_size
from flipcast.snapshot import ColumnStats, TableStats


def _table(rows=1_000_000, target=None):
    c = ColumnStats(
        name="deleted_at",
        type_name="timestamptz",
        type_oid=1184,
        null_frac=0.95,
        n_distinct=-0.05,
        stattarget=target,
        correlation=0.4,
    )
    k = ColumnStats(
        name="kind",
        type_name="text",
        type_oid=25,
        null_frac=0.0,
        n_distinct=4,
        mcv='{"a","b","c","d"}',
        mcv_freqs=[0.5, 0.3, 0.15, 0.05],
    )
    return TableStats(
        schema="public", name="t", relkind="r", relpages=10000, reltuples=rows, columns=[c, k]
    )


def test_sample_size_follows_statistics_target():
    assert sample_size(_table(), 100) == 30_000
    assert sample_size(_table(target=1000), 100) == 300_000
    assert sample_size(_table(rows=500), 100) == 500


def test_null_frac_nudges_include_the_almost_all_null_case():
    t = _table()
    col = t.column("deleted_at")
    changes = dict(nudges(t, col, 30_000))
    fracs = sorted(c.null_frac for c in changes.values() if c.null_frac != col.null_frac)
    # two standard errors of a 30k sample around 0.95 is about +-0.0025
    assert fracs[0] == pytest.approx(0.95 - 2 * (0.95 * 0.05 / 30_000) ** 0.5)
    assert fracs[-1] == pytest.approx(1 - 1 / 30_000)


def test_mcv_nudges_stay_valid():
    t = _table()
    col = t.column("kind")
    for what, c in nudges(t, col, 30_000):
        if c.mcv_freqs != col.mcv_freqs:
            assert all(0 <= f <= 1 for f in c.mcv_freqs)
            assert sum(c.mcv_freqs) <= 1 - c.null_frac + 1e-9
            assert "common value" in what


def test_ndistinct_nudges_halve_and_double():
    t = _table()
    nds = sorted(c.n_distinct for _, c in nudges(t, t.column("kind"), 30_000) if c.n_distinct != 4)
    assert nds == [2, 8]
    ratios = sorted(
        c.n_distinct for _, c in nudges(t, t.column("deleted_at"), 30_000) if c.n_distinct != -0.05
    )
    assert ratios == pytest.approx([-0.1, -0.025])
