"""What one snapshot of a database's planner inputs looks like.

The same classes describe a real snapshot taken by the collector and a
projected one made up by the growth models. That symmetry is on purpose: the
shadow loader and plan sweeper don't care where the numbers came from, which
is what makes backtesting honest.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


@dataclass
class ColumnStats:
    name: str
    type_name: str
    type_oid: int  # base type, domains resolved
    inherited: bool = False
    stattarget: int | None = None  # None: default_statistics_target
    null_frac: float = 0.0
    avg_width: int = 0
    n_distinct: float = 0.0
    # arrays are kept exactly as pg_stats printed them. models parse on demand.
    mcv: str | None = None
    mcv_freqs: list[float] | None = None
    histogram: str | None = None
    correlation: float | None = None
    # array and range column stats, restored as they were
    most_common_elems: str | None = None
    most_common_elem_freqs: list[float] | None = None
    elem_count_histogram: list[float] | None = None
    range_length_histogram: str | None = None
    range_empty_frac: float | None = None
    range_bounds_histogram: str | None = None

    @property
    def key(self) -> tuple[str, bool]:
        return (self.name, self.inherited)


@dataclass
class IndexStats:
    name: str
    relpages: int
    reltuples: float
    definition: str
    am: str = "btree"
    unique: bool = False
    partial: bool = False
    columns: list[str] = field(default_factory=list)  # key columns, expressions as None
    # stats on index expressions live in pg_statistic under the index
    expr_stats: list[ColumnStats] = field(default_factory=list)


@dataclass
class TableStats:
    schema: str
    name: str
    relkind: str
    relpages: int
    reltuples: float
    relallvisible: int = 0
    relallfrozen: int = 0
    n_live_tup: int | None = None
    n_dead_tup: int | None = None
    columns: list[ColumnStats] = field(default_factory=list)
    indexes: list[IndexStats] = field(default_factory=list)

    @property
    def qualname(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, name: str, inherited: bool = False) -> ColumnStats | None:
        for c in self.columns:
            if c.name == name and c.inherited == inherited:
                return c
        for c in self.columns:
            if c.name == name:
                return c
        return None


@dataclass
class QueryStat:
    queryid: str
    query: str
    calls: int
    total_ms: float
    mean_ms: float
    rows: int


@dataclass
class PlanRecord:
    shape: str
    total_cost: float
    plan_rows: float
    error: str | None = None


@dataclass
class Snapshot:
    taken_at: datetime
    database: str
    server_version_num: int
    settings: dict[str, str] = field(default_factory=dict)
    schema_sql: str = ""
    tables: list[TableStats] = field(default_factory=list)
    queries: list[QueryStat] = field(default_factory=list)
    plans: dict[str, PlanRecord] = field(default_factory=dict)  # observed on the source
    projected: bool = False

    def table(self, qualname: str) -> TableStats | None:
        if "." not in qualname:
            qualname = "public." + qualname
        for t in self.tables:
            if t.qualname == qualname:
                return t
        return None

    @property
    def schema_hash(self) -> str:
        # pg_dump 18 writes a random key on its \restrict line, and version
        # comments change with minor upgrades. neither is schema.
        keep = [
            line
            for line in self.schema_sql.splitlines()
            if line.strip() and not line.startswith(("\\", "--"))
        ]
        return hashlib.sha256("\n".join(keep).encode()).hexdigest()[:12]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["taken_at"] = self.taken_at.astimezone(UTC).isoformat()
        d["format"] = FORMAT_VERSION
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Snapshot:
        if d.get("format", 1) != FORMAT_VERSION:
            raise ValueError(f"unsupported snapshot format {d.get('format')}")
        tables = [_table_from(t) for t in d.get("tables", [])]
        return cls(
            taken_at=datetime.fromisoformat(d["taken_at"]),
            database=d["database"],
            server_version_num=d["server_version_num"],
            settings=d.get("settings", {}),
            schema_sql=d.get("schema_sql", ""),
            tables=tables,
            queries=[_build(QueryStat, q) for q in d.get("queries", [])],
            plans={k: _build(PlanRecord, v) for k, v in d.get("plans", {}).items()},
            projected=d.get("projected", False),
        )

    def without_plans(self) -> Snapshot:
        return replace(self, plans={})


def _build(cls: Any, d: dict[str, Any]) -> Any:
    # ignore keys from newer versions instead of blowing up
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def _table_from(d: dict[str, Any]) -> TableStats:
    t = _build(TableStats, d)
    t.columns = [_build(ColumnStats, c) for c in d.get("columns", [])]
    idx = []
    for i in d.get("indexes", []):
        ix = _build(IndexStats, i)
        ix.expr_stats = [_build(ColumnStats, c) for c in i.get("expr_stats", [])]
        idx.append(ix)
    t.indexes = idx
    return t  # type: ignore[no-any-return]


class Store:
    """A directory of gzipped JSON snapshots, one file each, sorted by name."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def save(self, snap: Snapshot) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        stamp = snap.taken_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = self.path / f"{snap.database}-{stamp}.json.gz"
        tmp = out.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(snap.to_json(), f, separators=(",", ":"))
        tmp.replace(out)
        return out

    def files(self) -> list[Path]:
        if not self.path.exists():
            return []
        return sorted(self.path.glob("*.json.gz"))

    def load_all(self, database: str | None = None) -> list[Snapshot]:
        snaps = []
        for p in self.files():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                s = Snapshot.from_json(json.load(f))
            if database is None or s.database == database:
                snaps.append(s)
        snaps.sort(key=lambda s: s.taken_at)
        return snaps
