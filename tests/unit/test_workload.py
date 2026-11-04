from datetime import UTC, datetime, timedelta

import pytest

from flipcast import config, workload
from flipcast.collect import tracked
from flipcast.snapshot import QueryStat
from flipcast.values import codec_for
from tests.conftest import events_history

AT = datetime(2027, 3, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("now", "now"),
        ("now - 7 days", "now"),
        ("NOW + 2 hours", "now"),
        ("mcv:events.kind", "mcv"),
        ("mcv:public.events.kind:2", "mcv"),
        ("quantile:events.created_at:0.9", "quantile"),
        ("max:events.id", "quantile"),
        ("hello", "literal"),
        (42, "literal"),
        ({"literal": "now"}, "literal"),
    ],
)
def test_parse_param_kinds(raw, kind):
    assert workload.parse_param(raw).kind == kind


def test_parse_param_errors():
    with pytest.raises(workload.ParamError):
        workload.parse_param("now - 3 fortnights")
    with pytest.raises(workload.ParamError):
        workload.parse_param("mcv:events.kind:0")
    with pytest.raises(workload.ParamError):
        workload.parse_param("quantile:events.id:1.5")
    with pytest.raises(workload.ParamError):
        workload.parse_param({"value": 1})


def test_render_moves_the_clock():
    q = workload.Query(
        "q",
        "select * from events where created_at > now() - interval '1 day' and created_at > $1",
        [workload.parse_param("now - 7 days")],
    )
    r = workload.render(q, AT, events_history()[-1])
    assert "now()" not in r.sql
    assert "'2027-03-01 00:00:00+00:00'::timestamptz" in r.sql
    assert r.values == [AT - timedelta(days=7)]


def test_render_resolves_stats_params():
    snap = events_history()[-1]
    q = workload.Query(
        "q",
        "select $1, $2, $3",
        [
            workload.parse_param("mcv:events.kind"),
            workload.parse_param("max:events.created_at"),
            workload.parse_param("median:events.id"),
        ],
    )
    mcv, top, mid = workload.render(q, AT, snap).values
    assert mcv in ("click", "export", "view")
    c = codec_for(1184)
    assert c.to_float(top) == pytest.approx(c.to_float(snap.taken_at.isoformat()), abs=1)
    assert float(mid) == pytest.approx(snap.table("events").reltuples / 2, rel=0.02)


def test_render_checks_param_count():
    q = workload.Query("q", "select $1, $2", [workload.parse_param(1)])
    with pytest.raises(workload.ParamError):
        workload.render(q, AT, events_history()[-1])


def test_config_and_env(tmp_path, monkeypatch):
    p = tmp_path / "flipcast.toml"
    p.write_text("""
[source]
dsn = "postgresql://file"
pg_dump = "docker exec -i pg pg_dump"
[store]
path = "snaps"
[[query]]
name = "feed"
sql = "select * from events where account_id = $1"
params = ["mcv:events.account_id:3"]
[[query]]
queryid = "123"
""")
    monkeypatch.setenv("FLIPCAST_SOURCE_DSN", "postgresql://env")
    cfg = config.load(p)
    assert cfg.source_dsn == "postgresql://env"
    assert cfg.pg_dump == ["docker", "exec", "-i", "pg", "pg_dump"]
    assert cfg.store == tmp_path / "snaps"
    assert [q.name for q in cfg.queries] == ["feed", "pss:123"]
    assert cfg.queries[0].params[0].arg == 3


def test_config_rejects_duplicate_names(tmp_path):
    p = tmp_path / "f.toml"
    p.write_text('[[query]]\nname = "a"\nsql = "select 1"\n[[query]]\nname = "a"\nsql = "x"\n')
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_tracked_merges_pg_stat_statements_twins():
    snap = events_history()[-1]
    snap.queries = [
        QueryStat("11", "select id from events where kind = $1 limit $2", 50, 100.0, 2.0, 5),
        QueryStat(
            "22",
            "select count(*) from events where created_at > now() - interval $1",
            9,
            9.0,
            1.0,
            9,
        ),
    ]
    cfg_q = [
        workload.Query(
            "by_kind",
            "SELECT id FROM events WHERE kind = $1 LIMIT 10",
            [workload.parse_param("export")],
        )
    ]
    out = {q.name: q for q in tracked(cfg_q, snap)}
    assert set(out) == {"by_kind", "by_kind (generic)", "pss:22"}
    assert out["by_kind"].mean_ms == 2.0
    assert out["by_kind (generic)"].generic
    # interval $1 would not parse, it gets turned around
    assert "$1::interval" in out["pss:22"].sql
