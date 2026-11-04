from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import sql

log = logging.getLogger(__name__)

MIN_SERVER_VERSION = 180000


class FlipcastError(RuntimeError):
    pass


class NoticeLog:
    """Collects server WARNINGs. The stats restore functions report bad input
    as a warning plus a false return, never as an error, so we have to look."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def __call__(self, diag: psycopg.errors.Diagnostic) -> None:
        if diag.severity_nonlocalized in ("WARNING", "ERROR"):
            self.warnings.append(diag.message_primary or "")
        else:
            log.debug("server notice: %s", diag.message_primary)

    def take(self) -> list[str]:
        out, self.warnings = self.warnings, []
        return out


def connect(dsn: str, *, application_name: str = "flipcast") -> psycopg.Connection[Any]:
    conn = psycopg.connect(dsn, autocommit=True, application_name=application_name)
    notices = NoticeLog()
    conn.add_notice_handler(notices)
    conn.info  # noqa: B018 - fail fast if the connection is unusable
    with conn.cursor() as cur:
        # array and timestamp text has to be parseable by values.py
        cur.execute("SET DateStyle = 'ISO, YMD'")
        cur.execute("SET TimeZone = 'UTC'")
        cur.execute("SET IntervalStyle = 'postgres'")
        cur.execute("SET extra_float_digits = 3")
    _notices[conn] = notices
    return conn


_notices: weakref.WeakKeyDictionary[psycopg.Connection[Any], NoticeLog] = (
    weakref.WeakKeyDictionary()
)


def notices(conn: psycopg.Connection[Any]) -> NoticeLog:
    return _notices[conn]


def require_pg18(conn: psycopg.Connection[Any], what: str) -> int:
    v = conn.info.server_version
    if v < MIN_SERVER_VERSION:
        raise FlipcastError(
            f"{what} runs PostgreSQL {v // 10000}; flipcast needs 18 or later "
            "(pg_restore_relation_stats and pg_restore_attribute_stats are new in 18)"
        )
    return v


@contextmanager
def transaction(conn: psycopg.Connection[Any]) -> Iterator[psycopg.Cursor[Any]]:
    with conn.transaction(), conn.cursor() as cur:
        yield cur


def quote_ident(name: str) -> sql.Identifier:
    return sql.Identifier(name)
