from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from . import __version__, backtest, collect, config, db, demo, forecast, report
from .growth import MONTH_DAYS, GrowthModel
from .shadow import MODULE, Shadow
from .snapshot import Snapshot, Store

log = logging.getLogger("flipcast")

EXIT_SWITCH_SOON = 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="flipcast",
        description="Forecast when Postgres query plans will switch as tables grow.",
    )
    ap.add_argument(
        "-c", "--config", default=None, help="config file (default: ./flipcast.toml if it exists)"
    )
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--version", action="version", version=f"flipcast {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect", help="take a snapshot of the source database")
    p.add_argument("--at", help="record the snapshot as taken at this ISO time (backfills, demos)")
    p.add_argument(
        "--no-schema",
        action="store_true",
        help="skip pg_dump and reuse the schema from the previous snapshot",
    )

    p = sub.add_parser("models", help="show the fitted growth models")
    p.add_argument("--months", type=int, default=None)

    p = sub.add_parser("forecast", help="predict plan switches")
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--step-days", type=float, default=None)
    p.add_argument(
        "--no-scenarios",
        action="store_true",
        help="skip the faster/slower growth runs (no month ranges)",
    )
    p.add_argument("--no-verify", action="store_true", help="don't test suggested fixes")
    p.add_argument(
        "--no-noise",
        action="store_true",
        help="skip the check for plans that ANALYZE sampling noise can flip today",
    )
    p.add_argument("--keep-shadow", action="store_true", help="leave the shadow database")
    p.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    p.add_argument("-o", "--output", help="write the report here instead of stdout")
    p.add_argument(
        "--fail-within",
        type=float,
        metavar="MONTHS",
        default=None,
        help=f"exit {EXIT_SWITCH_SOON} if any switch is predicted within MONTHS",
    )

    p = sub.add_parser("backtest", help="score forecasts against history")
    p.add_argument("--min-train", type=int, default=4)
    p.add_argument("--horizon", type=float, default=12, help="months")
    p.add_argument("--tolerance-days", type=float, default=45)
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("-o", "--output")

    sub.add_parser("check", help="check the source and shadow setup")

    p = sub.add_parser("demo", help="build a growing demo database with snapshot history")
    p.add_argument("--dsn", required=True, help="admin connection to a PostgreSQL 18 server")
    p.add_argument("--dir", default="flipcast-demo", help="where to put config and snapshots")
    p.add_argument("--months", type=int, default=14)
    p.add_argument("--growth", type=float, default=1.22, help="monthly growth of new events")
    p.add_argument("--pg-dump", default="pg_dump", help="pg_dump command (PostgreSQL 18)")
    p.add_argument("--shadow-dsn", default=None, help="shadow server, default same as --dsn")

    args = ap.parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG), format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        return _run(args)
    except (db.FlipcastError, config.ConfigError, ValueError) as e:
        print(f"flipcast: {e}", file=sys.stderr)
        return 1
    except psycopg.OperationalError as e:
        print(f"flipcast: could not connect: {str(e).strip()}", file=sys.stderr)
        return 1
    except psycopg.Error as e:
        print(f"flipcast: database error: {str(e).strip()}", file=sys.stderr)
        if args.verbose:
            raise
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # `flipcast forecast | head`: stop quietly, like other unix tools
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


def _config(args: argparse.Namespace) -> config.Config:
    path = args.config
    if path is None and Path("flipcast.toml").exists():
        path = "flipcast.toml"
    return config.load(path)


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "demo":
        out = Path(args.dir)
        cfg = demo.run(
            args.dsn,
            out,
            months=args.months,
            growth=args.growth,
            pg_dump=shlex.split(args.pg_dump),
            shadow_dsn=args.shadow_dsn,
        )
        n = len(Store(cfg.store).files())
        cfg_path = out / "flipcast.toml"
        print(f"demo database ready, {n} monthly snapshots in {cfg.store}")
        print(f"next: flipcast -c {cfg_path} backtest")
        print(f"      flipcast -c {cfg_path} forecast")
        return 0

    cfg = _config(args)
    store = Store(cfg.store)

    if args.cmd == "collect":
        at = datetime.fromisoformat(args.at) if args.at else None
        if at is not None and at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        snap = collect.collect(cfg, at=at, dump_schema=not args.no_schema)
        if args.no_schema:
            prev = store.load_all(snap.database)
            if not prev:
                raise db.FlipcastError("--no-schema needs an earlier snapshot to copy from")
            snap.schema_sql = prev[-1].schema_sql
        path = store.save(snap)
        bad = sum(1 for p in snap.plans.values() if p.error)
        print(
            f"{path}: {len(snap.tables)} tables, {len(snap.queries)} top statements, "
            f"{len(snap.plans) - bad} plans recorded" + (f" ({bad} failed)" if bad else "")
        )
        return 0

    snaps = store.load_all()
    if args.cmd == "check":
        return _check(cfg, snaps)
    if not snaps:
        raise db.FlipcastError(f"no snapshots in {cfg.store}; run `flipcast collect` first")
    _one_database(snaps)

    if args.cmd == "models":
        months = args.months or cfg.months
        for line in GrowthModel(snaps).describe(months * MONTH_DAYS):
            print(line)
        return 0

    if args.cmd == "forecast":
        fc = forecast.run(
            cfg,
            snaps,
            months=args.months,
            step_days=args.step_days,
            scenarios=False if args.no_scenarios else None,
            verify=not args.no_verify,
            keep_shadow=args.keep_shadow,
            noise=not args.no_noise,
        )
        render = {
            "text": report.forecast_text,
            "markdown": report.forecast_markdown,
            "json": report.forecast_json,
        }[args.format]
        _emit(render(fc), args.output)
        if args.fail_within is not None and any(
            p.month <= args.fail_within for p in fc.predictions
        ):
            return EXIT_SWITCH_SOON
        return 0

    if args.cmd == "backtest":
        r = backtest.run(
            cfg,
            snaps,
            min_train=args.min_train,
            horizon_months=args.horizon,
            tolerance_days=args.tolerance_days,
        )
        db_name = snaps[-1].database
        text = (
            report.backtest_text(r, db_name)
            if args.format == "text"
            else report.backtest_json(r, db_name)
        )
        _emit(text, args.output)
        return 0

    raise AssertionError(args.cmd)


def _one_database(snaps: list[Snapshot]) -> None:
    names = {s.database for s in snaps}
    if len(names) > 1:
        raise db.FlipcastError(
            f"the snapshot store mixes databases ({', '.join(sorted(names))}); "
            "give each database its own [store] path"
        )


def _check(cfg: config.Config, snaps: list[Snapshot]) -> int:
    ok = True

    def say(good: bool, msg: str) -> None:
        nonlocal ok
        ok &= good
        print(("ok    " if good else "FAIL  ") + msg)

    if cfg.source_dsn:
        try:
            with db.connect(cfg.source_dsn) as c:
                v = c.info.server_version
                name = f"source: PostgreSQL {v // 10000}.{v % 100}"
                if collect.MIN_SOURCE_VERSION <= v < db.MIN_SERVER_VERSION:
                    print(f"warn  {name}; plans come from the 18 planner in the shadow")
                else:
                    say(v >= db.MIN_SERVER_VERSION, name)
                row = c.execute(
                    "select 1 from pg_extension where extname = 'pg_stat_statements'"
                ).fetchone()
                say(row is not None, "source: pg_stat_statements installed")
        except psycopg.Error as e:
            say(False, f"source: {str(e).strip().splitlines()[0]}")
        try:
            collect.dump_schema_sql(cfg.pg_dump, cfg.source_dsn)
            say(True, f"pg_dump works ({' '.join(cfg.pg_dump)})")
        except db.FlipcastError as e:
            say(False, str(e))
    else:
        say(False, "no source dsn")

    if cfg.shadow_dsn:
        try:
            with db.connect(cfg.shadow_dsn) as c:
                v = c.info.server_version
                say(v >= db.MIN_SERVER_VERSION, f"shadow: PostgreSQL {v // 10000}.{v % 100}")
                try:
                    c.execute(f"LOAD '{MODULE}'")
                    say(True, f"shadow: {MODULE} loads")
                except psycopg.Error as e:
                    say(False, f"shadow: LOAD '{MODULE}' failed: {str(e).strip()}")
                av = c.execute("show autovacuum").fetchone()
                print(
                    f"info  shadow: autovacuum={av[0] if av else '?'} "
                    "(flipcast turns it off per table either way)"
                )
        except psycopg.Error as e:
            say(False, f"shadow: {str(e).strip().splitlines()[0]}")
        if snaps and snaps[-1].schema_sql:
            sh = Shadow(cfg.shadow_dsn, cfg.shadow_database + "_check")
            try:
                sh.build(snaps[-1])
                conn = sh.session(snaps[-1].settings)
                sh.load(conn, snaps[-1])
                conn.close()
                say(
                    True,
                    f"shadow: schema and stats from the latest snapshot load "
                    f"({len(sh.catalog)} tables)",
                )
            except db.FlipcastError as e:
                say(False, f"shadow: {e}")
            finally:
                sh.drop()
    else:
        say(False, "no shadow dsn")

    print(f"info  {len(snaps)} snapshots in {cfg.store}")
    return 0 if ok else 1
