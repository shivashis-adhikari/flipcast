"""a mostly-null column, the shape of clerk's february 2026 plan flip."""

import psycopg
import pytest

from flipcast import collect, config, fragility, shadow, workload
from flipcast.sweep import explain_point

SETUP = """
create table accounts (id int primary key, plan text not null);
insert into accounts select g, (array['free', 'pro', 'team'])[1 + g % 3]
from generate_series(0, 5000) g;
create table jobs (
    id bigserial primary key,
    account_id int not null references accounts (id),
    finished_at timestamptz,
    payload text
);
-- 3% of jobs ever finish, at random times
insert into jobs (account_id, finished_at, payload)
select (random() * 5000)::int,
       case when random() < 0.03 then timestamptz '2026-01-01' + random() * interval '90 days' end,
       repeat('x', 80)
from generate_series(1, 200000) g;
create index on jobs (finished_at);
"""


def test_almost_all_null_sample_flips_the_plan(scratch_db, admin_dsn, pg_dump):
    _, dsn = scratch_db
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(SETUP)
        c.execute("vacuum analyze jobs")
        c.execute("vacuum analyze accounts")
    q = workload.Query(
        "finished",
        "select a.plan, count(*) from jobs j join accounts a on a.id = j.account_id"
        " where j.finished_at is not null group by a.plan",
    )
    cfg = config.Config(
        source_dsn=dsn, shadow_dsn=admin_dsn, pg_dump=pg_dump, top_queries=0, queries=[q]
    )
    snap = collect.collect(cfg)
    sh = shadow.Shadow(admin_dsn, "flipcast_t_fragile")
    try:
        sh.build(snap)
        conn = sh.session(snap.settings)
        sh.load(conn, snap)
        base = explain_point(conn, q, snap, 0)
        found = fragility.check(sh, conn, [q], snap, {q.name: base})
        conn.close()
    finally:
        sh.drop()
    assert len(found) == 1
    f = found[0]
    assert f.column == "finished_at"
    assert f.change.startswith("null_frac")
    # the plan picked for "almost nothing matches" is worse with the real stats
    assert f.cost_flipped > f.cost_now
    assert f.shape != base.shape
    assert base.cost == pytest.approx(f.cost_now)
