"""the shadow has to plan exactly like the real database, or nothing else matters."""

from datetime import timedelta

import psycopg
import pytest

from flipcast import collect, config, db, pgarray, plans, shadow, workload
from flipcast.growth import GrowthModel

SCHEMA = """
create table ev (
    id bigserial primary key,
    user_id int not null,
    kind text not null,
    created_at timestamptz not null
);
insert into ev (user_id, kind, created_at)
select (random() * 20000)::int,
       (array['click', 'view', 'export', 'login'])[1 + floor(random() ^ 3 * 4)::int],
       timestamptz '2026-01-01' + g * interval '1 second'
from generate_series(1, 300000) g;
create index on ev (user_id);
create index on ev (created_at);
create index on ev (kind);
"""

QUERIES = {
    "by_user": ("select * from ev where user_id = $1 order by created_at desc limit 20", [42]),
    "by_kind": ("select * from ev where kind = $1", ["export"]),
    "common_kind": ("select * from ev where kind = 'click'", None),
    "range": (
        "select count(*) from ev where created_at between $1 and $2",
        ["2026-01-02", "2026-01-03"],
    ),
    "first_after": (
        "select * from ev where created_at > '2026-01-03' order by created_at limit 5",
        None,
    ),
    "group": ("select kind, count(*) from ev group by kind", None),
    "self_join": (
        "select * from ev a join ev b on a.id = b.user_id where a.kind = 'export'"
        " and b.created_at > '2026-01-03'",
        None,
    ),
}


@pytest.fixture
def snapshot(scratch_db, admin_dsn, pg_dump):
    _, dsn = scratch_db
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(SCHEMA)
        c.execute("vacuum analyze ev")
    cfg = config.Config(source_dsn=dsn, shadow_dsn=admin_dsn, pg_dump=pg_dump, top_queries=0)
    cfg.queries = [
        workload.Query(name, q, [workload.parse_param(p) for p in params] if params else None)
        for name, (q, params) in QUERIES.items()
    ]
    return cfg, collect.collect(cfg)


@pytest.fixture
def shadow_db(admin_dsn):
    sh = shadow.Shadow(admin_dsn, "flipcast_t_shadow")
    yield sh
    sh.drop()


def test_replay_matches_source_plans_and_costs(snapshot, shadow_db):
    cfg, snap = snapshot
    shadow_db.build(snap)
    conn = shadow_db.session(snap.settings)
    shadow_db.load(conn, snap)
    for q in collect.tracked(cfg.queries, snap):
        observed = snap.plans[q.name]
        assert observed.error is None, observed.error
        p = plans.explain(conn, workload.render(q, snap.taken_at, snap))
        assert plans.shape(p) == observed.shape, q.name
        assert plans.cost(p) == pytest.approx(observed.total_cost, rel=1e-6), q.name
    conn.close()


def test_without_the_module_injected_stats_are_ignored(snapshot, shadow_db):
    # this is the planner behavior flipcast_sizes exists to work around
    _, snap = snapshot
    shadow_db.build(snap)
    conn = shadow_db.session(snap.settings)
    shadow_db.load(conn, snap)
    conn.execute("set flipcast_sizes.enabled = off")
    p = plans.explain(conn, workload.Rendered("select * from ev", None))
    assert p["Plan Rows"] < 10
    conn.execute("set flipcast_sizes.enabled = on")
    p = plans.explain(conn, workload.Rendered("select * from ev", None))
    assert p["Plan Rows"] == pytest.approx(300000, rel=0.01)
    conn.close()


def test_autovacuum_is_off_on_every_table(snapshot, shadow_db):
    _, snap = snapshot
    shadow_db.build(snap)
    with db.connect(shadow_db.dsn) as c:
        opts = c.execute("select reloptions from pg_class where relname = 'ev'").fetchone()[0]
    assert "autovacuum_enabled=off" in opts


def test_projected_stats_land_in_pg_stats(snapshot, shadow_db):
    _, snap = snapshot
    later = snap.taken_at + timedelta(days=90)
    # two snapshots, the second with 50% more rows and a later created_at range
    grown = GrowthModel([snap]).project(later)
    t = grown.table("ev")
    t.reltuples *= 1.5
    t.relpages = round(t.relpages * 1.5)
    proj = GrowthModel([snap, grown]).project(later + timedelta(days=90))
    shadow_db.build(snap)
    conn = shadow_db.session(snap.settings)
    shadow_db.load(conn, proj)
    row = conn.execute("select reltuples from pg_class where relname = 'ev'").fetchone()
    assert row[0] == pytest.approx(proj.table("ev").reltuples, rel=1e-5)
    hist = conn.execute(
        "select histogram_bounds::text, most_common_vals::text from pg_stats"
        " where tablename = 'ev' and attname = %s",
        ("kind",),
    ).fetchone()
    assert pgarray.parse(hist[1]) == pgarray.parse(proj.table("ev").column("kind").mcv)
    assert db.notices(conn).take() == []
    conn.close()


def test_bad_stats_are_reported_not_swallowed(snapshot, shadow_db):
    _, snap = snapshot
    shadow_db.build(snap)
    conn = shadow_db.session(snap.settings)
    snap.table("ev").column("kind").mcv_freqs = [0.5]  # length no longer matches the values
    with pytest.raises(shadow.ShadowError):
        shadow_db.load(conn, snap)
    conn.close()
