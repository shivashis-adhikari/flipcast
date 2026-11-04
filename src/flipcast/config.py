from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .workload import Param, ParamError, Query, parse_param


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    source_dsn: str = ""
    shadow_dsn: str = ""
    shadow_database: str = "flipcast_shadow"
    schemas: list[str] = field(default_factory=list)  # empty: every non-system schema
    top_queries: int = 20
    min_calls: int = 5
    pg_dump: list[str] = field(default_factory=lambda: ["pg_dump"])
    store: Path = Path(".flipcast/snapshots")
    months: int = 12
    step_days: float = 30.0
    scenarios: bool = True
    queries: list[Query] = field(default_factory=list)
    base_dir: Path = Path(".")


def load(path: str | os.PathLike[str] | None) -> Config:
    cfg = Config()
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        try:
            data = tomllib.loads(p.read_text())
        except FileNotFoundError:
            raise ConfigError(f"config file {p} not found") from None
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{p}: {e}") from None
        cfg.base_dir = p.parent

    src = data.get("source", {})
    cfg.source_dsn = src.get("dsn", "")
    cfg.schemas = list(src.get("schemas", []))
    cfg.top_queries = int(src.get("top_queries", cfg.top_queries))
    cfg.min_calls = int(src.get("min_calls", cfg.min_calls))
    dump = src.get("pg_dump", "pg_dump")
    cfg.pg_dump = shlex.split(dump) if isinstance(dump, str) else list(dump)

    sh = data.get("shadow", {})
    cfg.shadow_dsn = sh.get("dsn", "")
    cfg.shadow_database = sh.get("database", cfg.shadow_database)

    st = data.get("store", {})
    cfg.store = cfg.base_dir / st.get("path", str(cfg.store))

    fc = data.get("forecast", {})
    cfg.months = int(fc.get("months", cfg.months))
    cfg.step_days = float(fc.get("step_days", cfg.step_days))
    cfg.scenarios = bool(fc.get("scenarios", cfg.scenarios))

    # env wins over the file so secrets can stay out of it
    cfg.source_dsn = os.environ.get("FLIPCAST_SOURCE_DSN", cfg.source_dsn)
    cfg.shadow_dsn = os.environ.get("FLIPCAST_SHADOW_DSN", cfg.shadow_dsn)

    seen: set[str] = set()
    for i, q in enumerate(data.get("query", [])):
        query = _query(q, i)
        if query.name in seen:
            raise ConfigError(f"duplicate query name {query.name!r}")
        seen.add(query.name)
        cfg.queries.append(query)
    return cfg


def _query(q: dict[str, Any], i: int) -> Query:
    name = q.get("name") or (f"pss:{q['queryid']}" if "queryid" in q else f"query{i + 1}")
    if "sql" not in q and "queryid" not in q:
        raise ConfigError(f"query {name!r} needs sql or queryid")
    params: list[Param] | None = None
    if "params" in q:
        try:
            params = [parse_param(p) for p in q["params"]]
        except ParamError as e:
            raise ConfigError(f"query {name!r}: {e}") from None
    return Query(
        name=str(name),
        sql=str(q.get("sql", "")),
        params=params,
        source="config",
        queryid=str(q["queryid"]) if "queryid" in q else None,
    )
