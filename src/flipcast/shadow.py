"""The shadow database: an empty copy of the schema with made-up statistics.

Building one:
  1. CREATE DATABASE on a scratch PostgreSQL 18 server
  2. replay pg_dump --schema-only from the snapshot
  3. turn autovacuum off on every table, so nothing ever re-analyzes the
     (empty) tables and wipes out the injected stats
  4. per step: pg_restore_relation_stats() for every table and index,
     pg_restore_attribute_stats() for every column

The planner still won't believe relpages for a table with no blocks on disk,
which is what the flipcast_sizes module is for. Each session LOADs it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from . import db
from .collect import clean_schema_sql
from .snapshot import ColumnStats, Snapshot, TableStats

log = logging.getLogger(__name__)

MODULE = "flipcast_sizes"


@dataclass
class ShadowTable:
    schema: str
    name: str
    relkind: str
    columns: list[str] = field(default_factory=list)
    indexes: list[str] = field(default_factory=list)


class ShadowError(db.FlipcastError):
    pass


class Shadow:
    def __init__(self, admin_dsn: str, database: str = "flipcast_shadow"):
        if not admin_dsn:
            raise ShadowError("no shadow dsn (set [shadow] dsn or FLIPCAST_SHADOW_DSN)")
        self.admin_dsn = admin_dsn
        self.database = database
        self.dsn = make_conninfo(admin_dsn, dbname=database)
        self.catalog: dict[str, ShadowTable] = {}
        self.schema_hash: str | None = None

    def build(self, snap: Snapshot) -> None:
        if not snap.schema_sql.strip():
            raise ShadowError("snapshot has no schema dump; re-run collect with pg_dump available")
        with db.connect(self.admin_dsn, application_name="flipcast-admin") as admin:
            db.require_pg18(admin, "the shadow server")
            name = sql.Identifier(self.database)
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(name))
            admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(name))

        failed: list[str] = []
        with db.connect(self.dsn, application_name="flipcast-build") as conn:
            for stmt in clean_schema_sql(snap.schema_sql):
                try:
                    conn.execute(stmt)
                except psycopg.Error as e:
                    head = " ".join(stmt.split()[:3]).upper()
                    msg = str(e).strip().splitlines()[0]
                    if head.startswith(("SET ", "SELECT PG_CATALOG.SET_CONFIG", "COMMENT ")):
                        log.debug("ignored: %s (%s)", head, msg)
                    elif head.startswith("CREATE EXTENSION"):
                        log.warning("shadow: %s failed: %s", " ".join(stmt.split()[:6]), msg)
                    else:
                        failed.append(f"{' '.join(stmt.split()[:8])}...: {msg}")
            if failed:
                raise ShadowError(
                    f"{len(failed)} schema statements failed in the shadow, first: {failed[0]}"
                )
            self._disable_autovacuum(conn)
            self.catalog = _read_catalog(conn)
        self.schema_hash = snap.schema_hash
        log.info("shadow %s: %d tables", self.database, len(self.catalog))

    def _disable_autovacuum(self, conn: psycopg.Connection[Any]) -> None:
        rows = conn.execute(
            "select n.nspname, c.relname from pg_class c"
            " join pg_namespace n on n.oid = c.relnamespace"
            " where c.relkind in ('r', 'm')"
            " and n.nspname not in ('pg_catalog', 'information_schema')"
            " and n.nspname !~ '^pg_toast'"
        ).fetchall()
        for nsp, rel in rows:
            conn.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} SET (autovacuum_enabled = off,"
                    " toast.autovacuum_enabled = off)"
                ).format(sql.Identifier(nsp), sql.Identifier(rel))
            )
        if conn.execute("show autovacuum").fetchone() == ("on",):
            log.info("autovacuum is on for the shadow server; disabled per table instead")

    def session(self, settings: dict[str, str] | None = None) -> psycopg.Connection[Any]:
        conn = db.connect(self.dsn, application_name="flipcast-sweep")
        try:
            conn.execute(f"LOAD '{MODULE}'")
        except psycopg.Error as e:
            conn.close()
            raise ShadowError(
                f"could not LOAD '{MODULE}' on the shadow server ({str(e).strip()}). "
                "build it from ext/flipcast_sizes and install it there, or use the docker image"
            ) from None
        row = conn.execute(f"select current_setting('{MODULE}.enabled', true)").fetchone()
        if row is None or row[0] != "on":
            conn.close()
            raise ShadowError(f"{MODULE} loaded but {MODULE}.enabled is not on")
        if not self.catalog:
            self.catalog = _read_catalog(conn)
        if settings:
            apply_settings(conn, settings)
        return conn

    def load(self, conn: psycopg.Connection[Any], snap: Snapshot) -> None:
        """Make the shadow's catalog stats match snap, exactly."""
        notes = db.notices(conn)
        notes.take()
        by_name = {t.qualname: t for t in snap.tables}
        with conn.transaction(), conn.cursor() as cur, conn.pipeline():
            for qual, st in self.catalog.items():
                t = by_name.get(qual)
                if t is None:
                    _clear_table(cur, st)
                    continue
                _restore_table(cur, st, t, snap.server_version_num)
        warnings = notes.take()
        if warnings:
            raise ShadowError(
                f"stats restore complained {len(warnings)} times, first: {warnings[0]}"
            )

    def drop(self) -> None:
        with db.connect(self.admin_dsn, application_name="flipcast-admin") as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(self.database)
                )
            )


def apply_settings(conn: psycopg.Connection[Any], settings: dict[str, str]) -> None:
    for name, value in settings.items():
        try:
            conn.execute("select set_config(%s, %s, false)", (name, value))
        except psycopg.Error as e:
            log.debug("setting %s=%s not applied: %s", name, value, e)


def _read_catalog(conn: psycopg.Connection[Any]) -> dict[str, ShadowTable]:
    out: dict[str, ShadowTable] = {}
    for oid, nsp, rel, kind in conn.execute(
        "select c.oid, n.nspname, c.relname, c.relkind from pg_class c"
        " join pg_namespace n on n.oid = c.relnamespace"
        " where c.relkind in ('r', 'm', 'p')"
        " and n.nspname not in ('pg_catalog', 'information_schema')"
        " and n.nspname !~ '^pg_(toast|temp)'"
    ).fetchall():
        st = ShadowTable(nsp, rel, kind)
        st.columns = [
            r[0]
            for r in conn.execute(
                "select attname from pg_attribute where attrelid = %s and attnum > 0"
                " and not attisdropped order by attnum",
                (oid,),
            ).fetchall()
        ]
        st.indexes = [
            r[0]
            for r in conn.execute(
                "select c.relname from pg_index i join pg_class c on c.oid = i.indexrelid"
                " where i.indrelid = %s and c.relkind = 'i'",
                (oid,),
            ).fetchall()
        ]
        out[f"{nsp}.{rel}"] = st
    return out


def _call(cur: psycopg.Cursor[Any], fn: str, args: list[tuple[str, Any, str]]) -> None:
    parts, params = [], []
    for key, value, typ in args:
        if value is None:
            continue
        parts.append(f"'{key}', %s::{typ}")
        params.append(value)
    cur.execute(f"select pg_catalog.{fn}({', '.join(parts)})", params)


def _clear_table(cur: psycopg.Cursor[Any], st: ShadowTable) -> None:
    cur.execute("select pg_catalog.pg_clear_relation_stats(%s, %s)", (st.schema, st.name))
    for col in st.columns:
        for inh in (False, True):
            cur.execute(
                "select pg_catalog.pg_clear_attribute_stats(%s, %s, %s, %s)",
                (st.schema, st.name, col, inh),
            )


def _restore_table(cur: psycopg.Cursor[Any], st: ShadowTable, t: TableStats, version: int) -> None:
    rel_args: list[tuple[str, Any, str]] = [
        ("version", version, "integer"),
        ("schemaname", st.schema, "text"),
        ("relname", st.name, "text"),
        ("relpages", int(t.relpages), "integer"),
        ("reltuples", float(t.reltuples), "real"),
    ]
    if st.relkind != "p":
        rel_args += [
            ("relallvisible", int(t.relallvisible), "integer"),
            ("relallfrozen", int(t.relallfrozen), "integer"),
        ]
    _call(cur, "pg_restore_relation_stats", rel_args)

    indexes = {ix.name: ix for ix in t.indexes}
    for name in st.indexes:
        ix = indexes.get(name)
        if ix is None:
            # an index that only exists in the shadow (a suggested fix); size
            # it like the table so it isn't free
            continue
        _call(
            cur,
            "pg_restore_relation_stats",
            [
                ("version", version, "integer"),
                ("schemaname", st.schema, "text"),
                ("relname", name, "text"),
                ("relpages", int(ix.relpages), "integer"),
                ("reltuples", float(ix.reltuples), "real"),
            ],
        )
        for c in ix.expr_stats:
            _restore_column(cur, st.schema, name, c, version)

    have = {c.key: c for c in t.columns}
    for col in st.columns:
        for inh in (False, True):
            cur.execute(
                "select pg_catalog.pg_clear_attribute_stats(%s, %s, %s, %s)",
                (st.schema, st.name, col, inh),
            )
            found = have.get((col, inh))
            if found is not None:
                _restore_column(cur, st.schema, st.name, found, version)


def _restore_column(
    cur: psycopg.Cursor[Any], schema: str, rel: str, c: ColumnStats, version: int
) -> None:
    args: list[tuple[str, Any, str]] = [
        ("version", version, "integer"),
        ("schemaname", schema, "text"),
        ("relname", rel, "text"),
        ("attname", c.name, "text"),
        ("inherited", c.inherited, "boolean"),
        ("null_frac", c.null_frac, "real"),
        ("avg_width", c.avg_width, "integer"),
        ("n_distinct", c.n_distinct, "real"),
    ]
    if c.mcv is not None and c.mcv_freqs:
        args += [("most_common_vals", c.mcv, "text"), ("most_common_freqs", c.mcv_freqs, "real[]")]
    args += [
        ("histogram_bounds", c.histogram, "text"),
        ("correlation", c.correlation, "real"),
    ]
    if c.most_common_elems is not None and c.most_common_elem_freqs:
        args += [
            ("most_common_elems", c.most_common_elems, "text"),
            ("most_common_elem_freqs", c.most_common_elem_freqs, "real[]"),
        ]
    args += [
        ("elem_count_histogram", c.elem_count_histogram, "real[]"),
        ("range_length_histogram", c.range_length_histogram, "text"),
        ("range_empty_frac", c.range_empty_frac, "real"),
        ("range_bounds_histogram", c.range_bounds_histogram, "text"),
    ]
    _call(cur, "pg_restore_attribute_stats", args)
