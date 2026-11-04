import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from flipcast import pgarray
from flipcast.snapshot import ColumnStats, IndexStats, Snapshot, TableStats
from flipcast.values import codec_for

FIXTURES = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def load_plan(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())[0]["Plan"]


@pytest.fixture
def plan():
    return load_plan


def ts_hist(lo: datetime, hi: datetime, buckets: int = 100) -> str:
    c = codec_for(1184)
    a, b = c.to_float(lo.isoformat()), c.to_float(hi.isoformat())
    return pgarray.to_literal(c.to_text(a + (b - a) * i / buckets) for i in range(buckets + 1))


def events_history(
    months: int = 8, *, growth: float = 1.2, base: float = 10000.0, export_new: float = 0.25
) -> list[Snapshot]:
    """snapshots of an append-only events table, built by hand.

    inserts per month grow by `growth`. created_at covers the data so far.
    'export' is 2% of the first month and export_new of every month after.
    """
    snaps = []
    rows = exports = 0.0
    for m in range(months):
        n = base * growth**m
        rows += n
        exports += n * (0.02 if m == 0 else export_new)
        at = T0 + timedelta(days=30 * (m + 1))
        ex = exports / rows
        kinds = sorted(
            [("export", ex), ("click", (1 - ex) * 0.6), ("view", (1 - ex) * 0.4)],
            key=lambda kv: -kv[1],
        )
        cols = [
            ColumnStats(
                name="id",
                type_name="bigint",
                type_oid=20,
                n_distinct=-1.0,
                avg_width=8,
                correlation=1.0,
                histogram=pgarray.to_literal(
                    str(round(1 + (rows - 1) * i / 100)) for i in range(101)
                ),
            ),
            ColumnStats(
                name="kind",
                type_name="text",
                type_oid=25,
                n_distinct=3,
                avg_width=6,
                correlation=0.3,
                mcv=pgarray.to_literal(k for k, _ in kinds),
                mcv_freqs=[f for _, f in kinds],
            ),
            ColumnStats(
                name="created_at",
                type_name="timestamp with time zone",
                type_oid=1184,
                n_distinct=-1.0,
                avg_width=8,
                correlation=1.0,
                histogram=ts_hist(T0, at),
            ),
        ]
        t = TableStats(
            schema="public",
            name="events",
            relkind="r",
            relpages=round(rows / 80),
            reltuples=rows,
            relallvisible=round(rows / 80),
            columns=cols,
            indexes=[
                IndexStats(
                    name="events_pkey",
                    relpages=round(rows / 300),
                    reltuples=rows,
                    definition="",
                    columns=["id"],
                )
            ],
        )
        snaps.append(
            Snapshot(
                taken_at=at,
                database="app",
                server_version_num=180006,
                settings={"default_statistics_target": "100"},
                tables=[t],
            )
        )
    return snaps
