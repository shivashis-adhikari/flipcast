"""Demo history -> backtest -> forecast, through the CLI."""

import json

import pytest

from flipcast import cli


def flipcast(*args: str) -> int:
    return cli.main(list(args))


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory, admin_dsn, pg_dump):
    out = tmp_path_factory.mktemp("demo")
    rc = flipcast(
        "demo",
        "--dsn",
        admin_dsn,
        "--dir",
        str(out),
        "--months",
        "9",
        "--pg-dump",
        " ".join(pg_dump),
    )
    assert rc == 0
    return out


def test_demo_makes_one_snapshot_per_month(demo_dir):
    assert len(list((demo_dir / "snapshots").glob("*.json.gz"))) == 9


def test_check_passes(demo_dir, capsys):
    assert flipcast("-c", str(demo_dir / "flipcast.toml"), "check") == 0
    assert "FAIL" not in capsys.readouterr().out


def test_backtest_runs_and_shadow_is_faithful(demo_dir, capsys):
    assert flipcast("-c", str(demo_dir / "flipcast.toml"), "backtest", "--format", "json") == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["origins"] == 5
    f = doc["fidelity"]
    assert f["total"] > 0
    assert f["agree"] == f["total"]
    assert doc["flipcast"]["shape_accuracy"] >= doc["rows_only"]["shape_accuracy"]


def test_forecast_text_and_json(demo_dir, capsys):
    cfg = str(demo_dir / "flipcast.toml")
    assert flipcast("-c", cfg, "forecast", "--months", "6") == 0
    text = capsys.readouterr().out
    assert "growth models" in text
    assert "public.events" in text
    assert flipcast("-c", cfg, "forecast", "--months", "6", "--format", "json", "--no-verify") == 0
    doc = json.loads(capsys.readouterr().out)
    names = {s["query"] for s in doc["stable"]} | {s["query"] for s in doc["switches"]}
    assert {"account_feed", "events_of_kind", "recent_exports"} <= names


def test_fail_within_exit_code(demo_dir, capsys):
    cfg = str(demo_dir / "flipcast.toml")
    rc = flipcast(
        "-c",
        cfg,
        "forecast",
        "--no-verify",
        "--no-scenarios",
        "--format",
        "json",
        "--fail-within",
        "12",
    )
    doc = json.loads(capsys.readouterr().out)
    assert rc == (cli.EXIT_SWITCH_SOON if doc["switches"] else 0)
