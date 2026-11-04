import os
import shlex
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

DSN = os.environ.get("FLIPCAST_TEST_DSN")
PG_DUMP = shlex.split(os.environ.get("FLIPCAST_TEST_PG_DUMP", "pg_dump"))

pytestmark = pytest.mark.integration


def pytest_collection_modifyitems(items):
    if DSN:
        return
    skip = pytest.mark.skip(reason="set FLIPCAST_TEST_DSN to a PostgreSQL 18 server")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def admin_dsn():
    return DSN


@pytest.fixture(scope="session")
def pg_dump():
    return PG_DUMP


@pytest.fixture
def scratch_db(admin_dsn):
    """A throwaway database, dropped afterwards."""
    name = f"flipcast_t_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_dsn, autocommit=True) as c:
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield name, make_conninfo(admin_dsn, dbname=name)
    with psycopg.connect(admin_dsn, autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
