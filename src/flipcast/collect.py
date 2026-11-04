"""Take a snapshot of everything the planner reads, plus the queries that matter.

Run this on a schedule (daily is plenty). Every snapshot holds:
  - pg_class sizes for each table and index
  - pg_stats for each column, arrays kept as text
  - the planner settings in effect
  - the schema, as pg_dump --schema-only prints it
  - the top statements from pg_stat_statements
  - the plan each tracked query gets right now, as observed ground truth
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from typing import Any

import psycopg

from . import db, plans, sqltext
from .config import Config
from .snapshot import ColumnStats, IndexStats, PlanRecord, QueryStat, Snapshot, TableStats
from .workload import ParamError, Query, render

log = logging.getLogger(__name__)

_TABLES = """
select c.oid, n.nspname, c.relname, c.relkind, c.relpages, c.reltuples,
       c.relallvisible, c.relallfrozen, s.n_live_tup, s.n_dead_tup
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
left join pg_stat_all_tables s on s.relid = c.oid
where c.relkind in ('r', 'm', 'p')
  and n.nspname not in ('pg_catalog', 'information_schema')
  and n.nspname !~ '^pg_(toast|temp)'
  and (cardinality(%(schemas)s::text[]) = 0 or n.nspname = any(%(schemas)s::text[]))
  -- tables that belong to an extension come back with CREATE EXTENSION
  and not exists (select 1 from pg_depend d
                  where d.classid = 'pg_class'::regclass and d.objid = c.oid and d.deptype = 'e')
order by n.nspname, c.relname
"""

_COLUMNS = """
select a.attrelid, a.attname, format_type(a.atttypid, a.atttypmod), a.atttypid,
       a.attstattarget, s.inherited, s.null_frac, s.avg_width, s.n_distinct,
       s.most_common_vals::text, s.most_common_freqs, s.histogram_bounds::text,
       s.correlation, s.most_common_elems::text, s.most_common_elem_freqs,
       s.elem_count_histogram, s.range_length_histogram::text, s.range_empty_frac,
       s.range_bounds_histogram::text
from pg_attribute a
join pg_class c on c.oid = a.attrelid
join pg_namespace n on n.oid = c.relnamespace
join pg_stats s on s.schemaname = n.nspname and s.tablename = c.relname
               and s.attname = a.attname
where a.attrelid = any(%(oids)s) and a.attnum > 0 and not a.attisdropped
order by a.attrelid, a.attnum, s.inherited
"""

_INDEXES = """
select i.indrelid, ic.oid, ic.relname, ic.relpages, ic.reltuples,
       pg_get_indexdef(i.indexrelid), am.amname, i.indisunique, i.indpred is not null,
       array(select a.attname
             from unnest(i.indkey::int2[]) with ordinality u(k, ord)
             left join pg_attribute a on a.attrelid = i.indrelid and a.attnum = u.k
             where u.ord <= i.indnkeyatts
             order by u.ord)
from pg_index i
join pg_class ic on ic.oid = i.indexrelid
join pg_am am on am.oid = ic.relam
where i.indrelid = any(%(oids)s) and ic.relkind = 'i'
order by i.indrelid, ic.relname
"""

_SETTINGS = """
select name, setting from pg_settings
where category like 'Query Tuning%%'
   or name in ('work_mem', 'hash_mem_multiplier', 'max_parallel_workers_per_gather',
               'default_statistics_target', 'effective_io_concurrency', 'plan_cache_mode')
order by name
"""

_PSS = """
select s.queryid::text, s.query, s.calls, s.total_exec_time, s.mean_exec_time, s.rows
from {schema}.pg_stat_statements s
where s.dbid = (select oid from pg_database where datname = current_database())
  and s.toplevel
  and s.calls >= %(min_calls)s
  and s.query ~* '^\\s*(with|select|update|delete|insert|table)\\y'
  -- catalog queries: monitoring, pg_dump, and flipcast itself
  and s.query !~* '\\ypg_[a-z_]+\\y'
order by s.total_exec_time desc
limit %(limit)s
"""


def collect(cfg: Config, *, at: datetime | None = None, dump_schema: bool = True) -> Snapshot:
    if not cfg.source_dsn:
        raise db.FlipcastError("no source dsn (set [source] dsn or FLIPCAST_SOURCE_DSN)")
    conn = db.connect(cfg.source_dsn, application_name="flipcast-collect")
    try:
        version = db.require_pg18(conn, "the source database")
        taken = (at or datetime.now(UTC)).astimezone(UTC)
        snap = Snapshot(
            taken_at=taken,
            database=conn.info.dbname,
            server_version_num=version,
            settings=_settings(conn),
            tables=_tables(conn, cfg.schemas),
            queries=_top_queries(conn, cfg.top_queries, cfg.min_calls),
        )
        if dump_schema:
            snap.schema_sql = dump_schema_sql(cfg.pg_dump, cfg.source_dsn)
        snap.plans = observe_plans(conn, tracked(cfg.queries, snap), snap)
    finally:
        conn.close()
    return snap


def _settings(conn: psycopg.Connection[Any]) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(_SETTINGS)
        return dict(cur.fetchall())


def _base_types(conn: psycopg.Connection[Any]) -> dict[int, int]:
    with conn.cursor() as cur:
        cur.execute("select oid, typbasetype from pg_type where typtype = 'd'")
        dom = {int(o): int(b) for o, b in cur.fetchall()}
    out: dict[int, int] = {}
    for d in dom:
        b = d
        for _ in range(32):  # domains over domains, rare but legal
            if b not in dom:
                break
            b = dom[b]
        out[d] = b
    return out


def _tables(conn: psycopg.Connection[Any], schemas: list[str]) -> list[TableStats]:
    base = _base_types(conn)
    by_oid: dict[int, TableStats] = {}
    with conn.cursor() as cur:
        cur.execute(_TABLES, {"schemas": schemas})
        for oid, nsp, rel, kind, pages, tuples, allvis, allfrozen, live, dead in cur.fetchall():
            by_oid[oid] = TableStats(
                schema=nsp,
                name=rel,
                relkind=kind,
                relpages=pages,
                reltuples=float(tuples),
                relallvisible=allvis,
                relallfrozen=allfrozen or 0,
                n_live_tup=live,
                n_dead_tup=dead,
            )
        if not by_oid:
            return []

        idx_owner: dict[int, IndexStats] = {}
        cur.execute(_INDEXES, {"oids": list(by_oid)})
        for relid, ioid, name, pages, tuples, ddl, am, uniq, partial, cols in cur.fetchall():
            ix = IndexStats(
                name=name,
                relpages=pages,
                reltuples=float(tuples),
                definition=ddl,
                am=am,
                unique=uniq,
                partial=partial,
                columns=list(cols),
            )
            by_oid[relid].indexes.append(ix)
            idx_owner[ioid] = ix

        cur.execute(_COLUMNS, {"oids": list(by_oid) + list(idx_owner)})
        for row in cur.fetchall():
            relid = row[0]
            col = _column(row, base)
            if relid in by_oid:
                by_oid[relid].columns.append(col)
            else:
                idx_owner[relid].expr_stats.append(col)
    return list(by_oid.values())


def _column(row: tuple[Any, ...], base: dict[int, int]) -> ColumnStats:
    (
        _,
        name,
        tname,
        typoid,
        target,
        inh,
        nullf,
        width,
        nd,
        mcv,
        mcf,
        hist,
        corr,
        mce,
        mcef,
        ech,
        rlh,
        ref,
        rbh,
    ) = row
    return ColumnStats(
        name=name,
        type_name=tname,
        type_oid=base.get(int(typoid), int(typoid)),
        inherited=bool(inh),
        stattarget=target,
        null_frac=float(nullf),
        avg_width=int(width),
        n_distinct=float(nd),
        mcv=mcv,
        mcv_freqs=[float(x) for x in mcf] if mcf is not None else None,
        histogram=hist,
        correlation=float(corr) if corr is not None else None,
        most_common_elems=mce,
        most_common_elem_freqs=[float(x) for x in mcef] if mcef is not None else None,
        elem_count_histogram=[float(x) for x in ech] if ech is not None else None,
        range_length_histogram=rlh,
        range_empty_frac=float(ref) if ref is not None else None,
        range_bounds_histogram=rbh,
    )


def _top_queries(conn: psycopg.Connection[Any], limit: int, min_calls: int) -> list[QueryStat]:
    if limit <= 0:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "select n.nspname from pg_extension e join pg_namespace n on n.oid = e.extnamespace"
            " where e.extname = 'pg_stat_statements'"
        )
        row = cur.fetchone()
        if row is None:
            log.warning(
                "pg_stat_statements is not installed in this database; "
                "only queries from the config file will be tracked"
            )
            return []
        q = _PSS.replace("{schema}", '"' + row[0].replace('"', '""') + '"')
        try:
            cur.execute(q, {"limit": limit, "min_calls": min_calls})
        except psycopg.errors.ObjectNotInPrerequisiteState:
            log.warning("pg_stat_statements is installed but not in shared_preload_libraries")
            return []
        return [
            QueryStat(
                queryid=qid,
                query=text,
                calls=int(calls),
                total_ms=float(total),
                mean_ms=float(mean),
                rows=int(rows),
            )
            for qid, text, calls, total, mean, rows in cur.fetchall()
        ]


_DUMP_FLAGS = [
    "--schema-only",
    "--no-owner",
    "--no-privileges",
    "--no-comments",
    "--no-publications",
    "--no-subscriptions",
    "--no-security-labels",
    "--no-tablespaces",
]


def dump_schema_sql(pg_dump: list[str], dsn: str) -> str:
    cmd = [*pg_dump, *_DUMP_FLAGS, f"--dbname={dsn}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    except FileNotFoundError:
        raise db.FlipcastError(
            f"could not run {pg_dump[0]!r}; set [source] pg_dump to a PostgreSQL 18 pg_dump"
        ) from None
    if res.returncode != 0:
        msg = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "no output"
        raise db.FlipcastError(f"pg_dump failed: {msg}")
    return res.stdout


def tracked(configured: list[Query], snap: Snapshot) -> list[Query]:
    """Config queries first, then pg_stat_statements entries not already covered.

    A statement whose text matches a config query is merged into it (its
    timings attach to the config entry). If the config entry plans with its
    own parameter values, the statement is also kept as "<name> (generic)",
    because that is the plan a prepared statement ends up running.
    """
    by_id = {q.queryid: q for q in snap.queries}
    by_text = {sqltext.fingerprint(q.query): q for q in snap.queries}
    out: list[Query] = []
    claimed: set[str] = set()
    generic: list[Query] = []
    for q in configured:
        st = None
        if q.queryid is not None:
            st = by_id.get(q.queryid)
            if st is None:
                log.info(
                    "query %s: queryid %s not in pg_stat_statements this time", q.name, q.queryid
                )
                if not q.sql:
                    continue
        elif q.sql:
            st = by_text.get(sqltext.fingerprint(q.sql))
        if st is not None:
            claimed.add(st.queryid)
            q = Query(
                name=q.name,
                sql=q.sql or st.query,
                params=q.params,
                source="config",
                queryid=st.queryid,
                calls=st.calls,
                mean_ms=st.mean_ms,
                total_ms=st.total_ms,
            )
            if q.params is not None and sqltext.max_param(st.query):
                generic.append(
                    Query(
                        name=f"{q.name} (generic)",
                        sql=sqltext.fix_typed_params(st.query),
                        params=None,
                        source="pg_stat_statements",
                        queryid=st.queryid,
                        calls=st.calls,
                        mean_ms=st.mean_ms,
                        total_ms=st.total_ms,
                    )
                )
        out.append(q)
    out.extend(generic)
    for st in snap.queries:
        if st.queryid in claimed:
            continue
        out.append(
            Query(
                name=f"pss:{st.queryid}",
                sql=sqltext.fix_typed_params(st.query),
                params=None,
                source="pg_stat_statements",
                queryid=st.queryid,
                calls=st.calls,
                mean_ms=st.mean_ms,
                total_ms=st.total_ms,
            )
        )
    return out


def observe_plans(
    conn: psycopg.Connection[Any], queries: list[Query], snap: Snapshot
) -> dict[str, PlanRecord]:
    out: dict[str, PlanRecord] = {}
    for q in queries:
        try:
            r = render(q, snap.taken_at, snap)
            p = plans.explain(conn, r)
        except (ParamError, plans.ExplainError) as e:
            out[q.name] = PlanRecord(shape="", total_cost=0.0, plan_rows=0.0, error=str(e))
            continue
        out[q.name] = PlanRecord(
            shape=plans.shape(p), total_cost=plans.cost(p), plan_rows=float(p["Plan Rows"])
        )
    return out


def clean_schema_sql(text: str) -> list[str]:
    """pg_dump output as a list of statements we can run over a plain connection."""
    # psql meta-commands: \restrict and \unrestrict since 18.0, \connect sometimes
    keep = [line for line in text.splitlines() if not line.startswith("\\")]
    return sqltext.split_statements("\n".join(keep))
