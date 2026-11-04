"""queries to track, and how to fill in their parameters at a given step.

a plan depends on the parameter values as much as on the stats. a query like
"events from the last 7 days" means something different in month 9 than it
does today, so parameters are written relative to the step being planned:

    params = ["now - 7 days", "mcv:events.account_id", 50]

    now [+|-] n unit        the clock at that step (unit: second..year)
    mcv:table.col[:rank]    rank-th most common value at that step (default 1)
    quantile:table.col:q    value at quantile q of the histogram at that step
    min: / max: / median:   shorthands for quantile 0, 1, 0.5

anything else is a literal. a toml table {literal = "now"} forces a literal.
calls to now(), current_timestamp and friends inside the sql text get the
same treatment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from . import pgarray, sqltext
from .snapshot import Snapshot
from .values import codec_for

_UNITS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 7 * 86400,
    "month": 30 * 86400,
    "year": 365 * 86400,
}
_NOW = re.compile(r"^\s*now\s*(?:([+-])\s*(\d+(?:\.\d+)?)\s*([a-z]+?)s?\s*)?$", re.I)
_STAT = re.compile(r"^(mcv|quantile|min|max|median):([\w$.\"]+?)(?::([\d.]+))?$", re.I)


class ParamError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    kind: str  # literal, now, mcv, quantile
    value: Any = None
    offset_s: float = 0.0
    table: str = ""
    column: str = ""
    arg: float = 0.0

    def describe(self) -> str:
        if self.kind == "literal":
            return repr(self.value)
        if self.kind == "now":
            if not self.offset_s:
                return "now"
            sign = "-" if self.offset_s < 0 else "+"
            return f"now {sign} {abs(self.offset_s) / 86400:g} days"
        if self.kind == "mcv":
            return f"mcv:{self.table}.{self.column}:{int(self.arg)}"
        return f"quantile:{self.table}.{self.column}:{self.arg:g}"


def parse_param(raw: Any) -> Param:
    if isinstance(raw, dict):
        if set(raw) != {"literal"}:
            raise ParamError(f"param tables only take 'literal', got {sorted(raw)}")
        return Param("literal", raw["literal"])
    if not isinstance(raw, str):
        return Param("literal", raw)

    m = _NOW.match(raw)
    if m:
        if not m.group(1):
            return Param("now")
        unit = m.group(3).lower()
        if unit not in _UNITS:
            raise ParamError(f"unknown time unit {unit!r} in {raw!r}")
        secs = float(m.group(2)) * _UNITS[unit]
        return Param("now", offset_s=-secs if m.group(1) == "-" else secs)

    m = _STAT.match(raw.strip())
    if m:
        kind, target, arg = m.group(1).lower(), m.group(2), m.group(3)
        if target.count(".") == 0:
            raise ParamError(f"{raw!r}: write table.column")
        table, column = target.rsplit(".", 1)
        if kind == "mcv":
            rank = float(arg) if arg else 1.0
            if rank < 1 or rank != int(rank):
                raise ParamError(f"{raw!r}: mcv rank starts at 1")
            return Param("mcv", table=table, column=column, arg=rank)
        q = {"min": 0.0, "max": 1.0, "median": 0.5}.get(kind)
        if q is None:
            if arg is None:
                raise ParamError(f"{raw!r}: quantile needs a value, like quantile:t.c:0.9")
            q = float(arg)
        if not 0.0 <= q <= 1.0:
            raise ParamError(f"{raw!r}: quantile must be in [0, 1]")
        return Param("quantile", table=table, column=column, arg=q)

    return Param("literal", raw)


@dataclass
class Query:
    name: str
    sql: str
    params: list[Param] | None = None  # none: generic plan when the sql has $n
    source: str = "config"
    queryid: str | None = None
    calls: int | None = None
    mean_ms: float | None = None
    total_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def generic(self) -> bool:
        return self.params is None and sqltext.max_param(self.sql) > 0


@dataclass(frozen=True)
class Rendered:
    sql: str
    values: list[Any] | None  # none means plan it generic


def render(q: Query, at: datetime, stats: Snapshot) -> Rendered:
    text = q.sql
    if sqltext.mentions_clock(text):
        at_utc = at.astimezone(UTC)
        text = sqltext.rewrite_clock(
            text,
            tstz=at_utc.isoformat(sep=" "),
            ts=at_utc.replace(tzinfo=None).isoformat(sep=" "),
            d=at_utc.date().isoformat(),
        )
    if q.params is None:
        return Rendered(text, None)
    need = sqltext.max_param(text)
    if need != len(q.params):
        raise ParamError(f"{q.name}: sql uses {need} parameters, config gives {len(q.params)}")
    return Rendered(text, [resolve(p, at, stats) for p in q.params])


def resolve(p: Param, at: datetime, stats: Snapshot) -> Any:
    if p.kind == "literal":
        return p.value
    if p.kind == "now":
        return at.astimezone(UTC) + timedelta(seconds=p.offset_s)

    table = stats.table(p.table)
    if table is None:
        raise ParamError(f"{p.describe()}: no table {p.table} in the stats")
    col = table.column(p.column)
    if col is None:
        raise ParamError(f"{p.describe()}: no stats for column {p.column}")

    if p.kind == "mcv":
        vals = pgarray.parse(col.mcv) if col.mcv else []
        rank = int(p.arg)
        if rank > len(vals):
            raise ParamError(f"{p.describe()}: only {len(vals)} common values at this step")
        return vals[rank - 1]

    # quantile: walk the equi-depth histogram, interpolate inside a bucket when
    # the type has a number line
    if not col.histogram:
        raise ParamError(f"{p.describe()}: column has no histogram at this step")
    bounds = pgarray.parse(col.histogram)
    pos = p.arg * (len(bounds) - 1)
    lo = int(pos)
    if lo >= len(bounds) - 1:
        return bounds[-1]
    codec = codec_for(col.type_oid)
    a, b = bounds[lo], bounds[lo + 1]
    if codec is None or a is None or b is None:
        return a
    fa, fb = codec.to_float(a), codec.to_float(b)
    return codec.to_text(fa + (fb - fa) * (pos - lo))


def to_python_literal(v: Any) -> Any:
    # toml gives us dates and datetimes as real objects already; keep them
    if isinstance(v, (datetime, date, int, float, bool)) or v is None:
        return v
    return str(v)
