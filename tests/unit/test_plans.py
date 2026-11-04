from datetime import UTC, datetime

from flipcast import advise, plans
from flipcast.snapshot import ColumnStats, IndexStats
from flipcast.sweep import Point, Switch
from flipcast.workload import Query
from tests.conftest import events_history


def test_shape_ignores_costs_but_not_direction(plan):
    p = plan("plan_limit_backward")
    s = plans.shape(p)
    assert s.startswith("Limit(Index Scan<Backward>[events using events_created_at_idx]")
    assert "cost" not in s.lower()
    p2 = dict(p, **{"Total Cost": 1.0})
    assert plans.shape(p2) == s


def test_accesses_and_headline(plan):
    old, new = plan("plan_bitmap_kind"), plan("plan_seq_kind")
    assert plans.accesses(old) == {"events": ["bitmap scan (events_kind_idx)"]}
    assert plans.accesses(new) == {"events": ["parallel seq scan"]}
    assert plans.headline(old, new) == "bitmap scan to parallel seq scan on events"
    lines = plans.describe_change(old, new)
    assert lines[0] == "events: bitmap scan (events_kind_idx) -> parallel seq scan"
    assert "adds gather" in lines


def test_join_change_is_described(plan):
    lines = plans.describe_change(plan("plan_join_hash"), plan("plan_join_nestloop"))
    assert any(line.startswith("joins:") for line in lines)


def test_toggles_against(plan):
    # to get the bitmap plan back from the seq scan one, seqscan and parallelism go off
    t = plans.toggles_against(plan("plan_bitmap_kind"), plan("plan_seq_kind"))
    assert t == ["enable_seqscan", "max_parallel_workers_per_gather"]


def _switch(before, after, q=None):
    q = q or Query("q", "select 1")
    at = datetime(2027, 1, 1, tzinfo=UTC)
    return Switch(
        query=q,
        step=3,
        before=Point(2, at, plans.shape(before), 1, 1, before),
        after=Point(3, at, plans.shape(after), 1, 1, after),
    )


def test_limit_trap_gets_a_composite_index(plan):
    stats = _demo_stats()
    sw = _switch(plan("plan_limit_sort_bitmap"), plan("plan_limit_backward"))
    fixes = advise.suggest(sw, stats, {"random_page_cost": "1.1"})
    idx = [f for f in fixes if f.kind == "index"]
    assert idx, fixes
    assert idx[0].index == ("public", "events", ["account_id", "created_at"], [])
    assert idx[0].sql == "CREATE INDEX CONCURRENTLY ON public.events (account_id, created_at);"


def test_lost_index_access_suggests_covering_index_and_cost_setting(plan):
    stats = _demo_stats()
    sw = _switch(plan("plan_bitmap_kind"), plan("plan_seq_kind"))
    fixes = advise.suggest(sw, stats, {"random_page_cost": "4"})
    kinds = [f.kind for f in fixes]
    assert "setting" in kinds
    cover = next(f for f in fixes if f.kind == "index")
    assert cover.index[2] == ["kind"]
    assert set(cover.index[3]) == {"id", "account_id", "created_at"}
    assert any("parallel" in f.why for f in fixes if f.kind == "note")


def test_generic_plans_get_the_plan_cache_note(plan):
    q = Query("g", "select * from events where kind = $1")
    fixes = advise.suggest(
        _switch(plan("plan_bitmap_kind"), plan("plan_seq_kind"), q), _demo_stats(), {}
    )
    assert any(f.kind == "plan_cache" for f in fixes)


def test_index_size_is_in_the_right_ballpark():
    t = _demo_stats().table("events")
    pages, tuples = advise.index_size(t, ["id"], [], 40_000_000)
    # a bigint btree holds roughly 360 entries per leaf page
    assert 100_000 < pages < 130_000
    assert tuples == 40_000_000


def _demo_stats():
    snap = events_history()[-1]
    t = snap.table("events")
    t.columns.append(
        ColumnStats(name="account_id", type_name="bigint", type_oid=20, n_distinct=500, avg_width=8)
    )
    for name, col in (
        ("events_created_at_idx", "created_at"),
        ("events_kind_idx", "kind"),
        ("events_account_id_idx", "account_id"),
    ):
        t.indexes.append(
            IndexStats(name=name, relpages=100, reltuples=t.reltuples, definition="", columns=[col])
        )
    return snap
