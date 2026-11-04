"""Grow a toy SaaS database month by month and snapshot it after each month.

This exists so there is real snapshot history to fit and backtest against.
Each simulated month inserts that month's accounts and events (with
created_at inside the month), runs ANALYZE, runs the workload a few times so
pg_stat_statements has timings, and then calls the collector with the clock
set to the end of the month.

The data has the kinds of drift that move real plans:
  - events per month compound (x1.22 by default), accounts grow linearly
  - a few old accounts produce most of the traffic
  - 'export' events go from ~2% to ~20% of new traffic
"""

from __future__ import annotations

import logging
import math
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from . import collect, db, sqltext
from .config import Config
from .config import load as load_config
from .snapshot import Snapshot, Store
from .workload import ParamError, Query, render

log = logging.getLogger(__name__)

SCHEMA = """
create table accounts (
    id bigserial primary key,
    plan text not null,
    region text not null,
    created_at timestamptz not null
);

create table events (
    id bigserial primary key,
    account_id bigint not null references accounts (id),
    kind text not null,
    status text not null,
    created_at timestamptz not null,
    payload text
);

create index events_account_id_idx on events (account_id);
create index events_created_at_idx on events (created_at);
create index events_kind_idx on events (kind);
create index accounts_created_at_idx on accounts (created_at);
"""

# one month of accounts. %(n)s rows spread over [%(lo)s, %(hi)s)
INSERT_ACCOUNTS = """
insert into accounts (plan, region, created_at)
select case when r < 0.80 then 'free' when r < 0.97 then 'pro' else 'enterprise' end,
       (array['us', 'eu', 'ap', 'sa'])[1 + floor(random() * 4)::int],
       %(lo)s::timestamptz + (%(hi)s::timestamptz - %(lo)s::timestamptz) * (g::float8 / %(n)s)
-- random() has to sit in a per-row subquery. in a LATERAL that doesn't
-- reference g it is evaluated once and every row gets the same value.
from (select g, random() as r from generate_series(0, %(n)s - 1) g) x
order by g
"""

# one month of events, in time order. account choice is skewed toward low
# (old) ids: random()^3 puts ~46%% of traffic on the first 10%% of accounts.
INSERT_EVENTS = """
with acc as (select max(id) as n from accounts)
insert into events (account_id, kind, status, created_at, payload)
select 1 + floor(power(random(), 3) * acc.n)::bigint,
       case when k < %(export)s then 'export'
            when k < %(export)s + 0.05 then 'login'
            when k < %(export)s + 0.30 then 'api_call'
            when k < %(export)s + 0.55 then 'click'
            else 'page_view' end,
       case when random() < 0.01 then 'failed' else 'ok' end,
       %(lo)s::timestamptz + (%(hi)s::timestamptz - %(lo)s::timestamptz) * (g::float8 / %(n)s),
       repeat('x', 40 + floor(random() * 60)::int)
from acc, (select g, random() as k from generate_series(0, %(n)s - 1) g) x
order by g
"""

WORKLOAD_TOML = """
[[query]]
name = "account_feed"
sql = '''
select id, kind, created_at from events
where account_id = $1 order by created_at desc limit 20
'''
params = [3]

[[query]]
name = "events_of_kind"
sql = "select id, account_id, created_at from events where kind = $1"
params = ["export"]

[[query]]
name = "recent_exports"
sql = '''
select count(*) from events
where kind = 'export' and created_at > now() - interval '30 days'
'''

[[query]]
name = "plan_breakdown"
sql = '''
select a.plan, count(*)
from events e join accounts a on a.id = e.account_id
where e.created_at > now() - interval '2 days'
group by a.plan
'''

[[query]]
name = "new_accounts"
sql = "select * from accounts where created_at > now() - interval '7 days' order by created_at"
"""


def _months(start: datetime, n: int) -> list[tuple[datetime, datetime]]:
    out = []
    y, m = start.year, start.month
    for _ in range(n):
        lo = datetime(y, m, 1, tzinfo=UTC)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((lo, datetime(y, m, 1, tzinfo=UTC)))
    return out


def export_share(month: int, months: int) -> float:
    # logistic ramp from ~2% to ~20%
    x = (month - months * 0.6) / (months * 0.18)
    return 0.02 + 0.18 / (1 + math.exp(-x))


def write_config(out_dir: Path, source_dsn: str, shadow_dsn: str, pg_dump: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "flipcast.toml"
    dump = " ".join(pg_dump)
    path.write_text(
        textwrap.dedent(f"""\
        # written by `flipcast demo`
        [source]
        dsn = "{source_dsn}"
        pg_dump = "{dump}"
        top_queries = 10

        [shadow]
        dsn = "{shadow_dsn}"
        database = "flipcast_demo_shadow"

        [store]
        path = "snapshots"

        [forecast]
        months = 12
        """)
        + WORKLOAD_TOML
    )
    return path


def run(
    admin_dsn: str,
    out_dir: Path,
    *,
    months: int = 14,
    base_events: int = 20000,
    growth: float = 1.22,
    base_accounts: int = 150,
    start: datetime | None = None,
    pg_dump: list[str] | None = None,
    shadow_dsn: str | None = None,
    database: str = "flipcast_demo",
) -> Config:
    start = start or datetime(2025, 9, 1, tzinfo=UTC)
    source_dsn = make_conninfo(admin_dsn, dbname=database)
    with db.connect(admin_dsn, application_name="flipcast-demo") as admin:
        db.require_pg18(admin, "the demo server")
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database))
        )
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))

    cfg_path = write_config(out_dir, source_dsn, shadow_dsn or admin_dsn, pg_dump or ["pg_dump"])
    cfg = load_config(cfg_path)
    store = Store(cfg.store)
    for old in store.files():
        old.unlink()

    with db.connect(source_dsn, application_name="flipcast-demo") as conn:
        conn.execute(SCHEMA)
        has_pss = _try_pss(conn)
        for m, (lo, hi) in enumerate(_months(start, months)):
            n_acc = base_accounts + 40 * m
            n_ev = round(base_events * growth**m)
            # the loader is not workload, keep it out of pg_stat_statements
            if has_pss:
                conn.execute("set pg_stat_statements.track = 'none'")
            conn.execute(INSERT_ACCOUNTS, {"n": n_acc, "lo": lo, "hi": hi})
            conn.execute(
                INSERT_EVENTS, {"n": n_ev, "lo": lo, "hi": hi, "export": export_share(m, months)}
            )
            if has_pss:
                conn.execute("reset pg_stat_statements.track")
            conn.execute("analyze accounts")
            conn.execute("analyze events")
            if has_pss:
                _exercise(conn, cfg.queries, hi)
            snap = collect.collect(cfg, at=hi)
            store.save(snap)
            ev = snap.table("public.events")
            log.info(
                "month %2d %s  events %9.0f  export share %.1f%%",
                m,
                lo.date(),
                ev.reltuples if ev else 0,
                100 * export_share(m, months),
            )
    return cfg


def _try_pss(conn: psycopg.Connection[Any]) -> bool:
    try:
        conn.execute("create extension if not exists pg_stat_statements")
        conn.execute("select pg_stat_statements_reset()")
    except psycopg.Error as e:
        log.warning(
            "no pg_stat_statements on the demo server (%s); skipping timings",
            str(e).strip().splitlines()[0],
        )
        return False
    return True


def _exercise(conn: psycopg.Connection[Any], queries: list[Query], at: datetime) -> None:
    """Run the parameterized queries a few times so pg_stat_statements has
    timings. Queries that call now() are skipped: run for real they'd use
    today's date instead of the simulated one, and the timings would be
    for an empty range."""
    stub = Snapshot(taken_at=at, database="", server_version_num=0)
    raw = psycopg.RawCursor(conn)
    try:
        for q in queries:
            if sqltext.mentions_clock(q.sql) or q.params is None:
                continue
            try:
                r = render(q, at, stub)
            except ParamError:
                continue  # mcv/quantile params need stats
            for _ in range(3):
                raw.execute(r.sql, r.values)
                raw.fetchall()
    finally:
        raw.close()
