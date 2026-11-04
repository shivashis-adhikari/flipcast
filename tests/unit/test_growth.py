import itertools
from datetime import timedelta

import pytest

from flipcast import pgarray
from flipcast.growth import GrowthModel
from flipcast.values import codec_for
from tests.conftest import events_history


def _events(snap):
    return snap.table("public.events")


def test_projection_at_latest_time_is_the_latest_snapshot():
    snaps = events_history()
    gm = GrowthModel(snaps)
    p = _events(gm.project(snaps[-1].taken_at))
    real = _events(snaps[-1])
    assert p.reltuples == pytest.approx(real.reltuples)
    assert p.relpages == pytest.approx(real.relpages, rel=0.01)
    assert p.column("created_at").histogram == real.column("created_at").histogram
    pf = dict(zip(pgarray.parse(p.column("kind").mcv), p.column("kind").mcv_freqs, strict=True))
    rf = dict(
        zip(pgarray.parse(real.column("kind").mcv), real.column("kind").mcv_freqs, strict=True)
    )
    for k, v in rf.items():
        assert pf[k] == pytest.approx(v, abs=1e-6)


def test_rows_follow_accelerating_inserts():
    snaps = events_history(months=10)
    truth = events_history(months=13)
    gm = GrowthModel(snaps)
    p = _events(gm.project(truth[-1].taken_at))
    assert p.reltuples == pytest.approx(_events(truth[-1]).reltuples, rel=0.05)
    assert gm.tables["public.events"].rows.kind == "accelerating"


def test_timestamp_histogram_extends_with_the_calendar():
    snaps = events_history()
    gm = GrowthModel(snaps)
    later = snaps[-1].taken_at + timedelta(days=180)
    h = pgarray.parse(_events(gm.project(later)).column("created_at").histogram)
    c = codec_for(1184)
    top = c.to_float(h[-1])
    assert top == pytest.approx(c.to_float(later.isoformat()), abs=86400 * 2)
    # the bottom stays put, nothing gets deleted
    first = pgarray.parse(_events(snaps[-1]).column("created_at").histogram)[0]
    assert h[0] == first
    vals = [c.to_float(x) for x in h]
    assert vals == sorted(vals)


def test_recent_values_get_more_histogram_weight_under_growth():
    # with accelerating inserts, the last month of the range holds more rows
    # than the first month did, so it gets more buckets
    snaps = events_history(growth=1.3)
    gm = GrowthModel(snaps)
    later = snaps[-1].taken_at + timedelta(days=120)
    c = codec_for(1184)
    h = [
        c.to_float(x)
        for x in pgarray.parse(_events(gm.project(later)).column("created_at").histogram)
    ]
    widths = [b - a for a, b in itertools.pairwise(h)]
    assert widths[-1] < widths[0]


def test_id_histogram_tracks_row_count():
    snaps = events_history()
    gm = GrowthModel(snaps)
    p = _events(gm.project(snaps[-1].taken_at + timedelta(days=90)))
    h = pgarray.parse(p.column("id").histogram)
    assert float(h[-1]) == pytest.approx(p.reltuples, rel=0.02)
    # nothing is deleted, so the smallest id stays where it was
    assert h[0] == "1"
    assert not gm.tables["public.events"].columns[("id", False)].lo_moves


def test_retention_moves_the_bottom_of_the_range():
    # keep only the newest 3 months: the lowest id climbs as old rows go
    snaps = events_history()
    for i, s in enumerate(snaps):
        col = s.table("events").column("id")
        h = [float(x) for x in pgarray.parse(col.histogram)]
        lo = max(1.0, (i - 3) * 20000.0)
        col.histogram = pgarray.to_literal(
            str(round(lo + (h[-1] - lo) * k / 100)) for k in range(101)
        )
    gm = GrowthModel(snaps)
    assert gm.tables["public.events"].columns[("id", False)].lo_moves


def test_mcv_share_moves_toward_new_rows_share():
    snaps = events_history(months=10, export_new=0.25)
    gm = GrowthModel(snaps)
    p = _events(gm.project(snaps[-1].taken_at + timedelta(days=360)))
    freqs = dict(zip(pgarray.parse(p.column("kind").mcv), p.column("kind").mcv_freqs, strict=True))
    now = dict(
        zip(
            pgarray.parse(_events(snaps[-1]).column("kind").mcv),
            _events(snaps[-1]).column("kind").mcv_freqs,
            strict=True,
        )
    )
    assert now["export"] < freqs["export"] <= 0.26
    # closed set: all three values still sum to everything
    assert sum(freqs.values()) == pytest.approx(1.0, abs=1e-6)
    # the list stays sorted by frequency, like ANALYZE writes it
    assert p.column("kind").mcv_freqs == sorted(p.column("kind").mcv_freqs, reverse=True)


def test_negative_ndistinct_is_kept_as_a_ratio():
    gm = GrowthModel(events_history())
    p = _events(gm.project(gm.latest.taken_at + timedelta(days=200)))
    assert p.column("id").n_distinct == -1.0


def test_rows_only_baseline_freezes_columns():
    snaps = events_history()
    gm = GrowthModel(snaps, rows_only=True)
    p = _events(gm.project(snaps[-1].taken_at + timedelta(days=200)))
    assert p.reltuples > _events(snaps[-1]).reltuples
    assert p.column("created_at").histogram == _events(snaps[-1]).column("created_at").histogram


def test_scenarios_order_row_counts():
    snaps = events_history()
    gm = GrowthModel(snaps)
    at = snaps[-1].taken_at + timedelta(days=180)
    band = gm.band()
    slow = _events(gm.project(at, 1 - band)).reltuples
    base = _events(gm.project(at)).reltuples
    fast = _events(gm.project(at, 1 + band)).reltuples
    assert slow < base < fast
    # the calendar doesn't move with the scenario: created_at still ends at `at`
    c = codec_for(1184)
    h = pgarray.parse(_events(gm.project(at, 1 + band)).column("created_at").histogram)
    assert c.to_float(h[-1]) == pytest.approx(c.to_float(at.isoformat()), abs=86400 * 2)


def test_single_snapshot_projects_flat():
    snaps = events_history(months=1)
    gm = GrowthModel(snaps)
    p = _events(gm.project(snaps[0].taken_at + timedelta(days=90)))
    assert p.reltuples == _events(snaps[0]).reltuples


def test_describe_mentions_each_model():
    gm = GrowthModel(events_history())
    text = "\n".join(gm.describe(365))
    assert "public.events" in text
    assert "histogram append" in text
    assert "accelerating" in text
