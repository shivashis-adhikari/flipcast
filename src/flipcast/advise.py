"""Suggest fixes for a predicted switch.

These are rules of thumb read off the two plans, not an optimizer. Index and
setting suggestions get checked afterwards: the forecast re-runs the sweep
with the fix applied in the shadow and says whether the switch still happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import pgarray, plans, sqltext
from .snapshot import Snapshot, TableStats
from .sweep import Switch


@dataclass
class Fix:
    kind: str  # index, statistics, setting, plan_cache, note
    why: str
    sql: str | None = None
    # for kind=index: (schema, table, key columns, include columns)
    index: tuple[str, str, list[str], list[str]] | None = None
    setting: tuple[str, str] | None = None
    verified: str | None = None
    holds: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scan:
    node: dict[str, Any]
    schema: str
    table: str
    index: str | None
    eq_cols: list[str]
    range_cols: list[str]
    filter_cols: list[str]
    outputs: list[str]
    under_limit: bool
    sorted_above: bool


def _cols_in(text: str | None, columns: set[str]) -> tuple[list[str], list[str]]:
    """Columns used with = and with a range operator, in order of appearance."""
    if not text:
        return [], []
    eq: list[str] = []
    rng: list[str] = []
    toks = [t for t in sqltext.tokens(text) if t.kind not in ("space", "comment")]
    for i, t in enumerate(toks):
        if t.kind not in ("ident", "qident"):
            continue
        name = t.text.strip('"')
        if name not in columns:
            continue
        # "e.account_id" lexes as e . account_id, which is fine: we only look at
        # the column token and what follows it
        op = toks[i + 1].text if i + 1 < len(toks) else ""
        if i + 2 < len(toks) and toks[i + 1].kind == "op" and toks[i + 2].kind == "op":
            op += toks[i + 2].text
        if op == "=" and name not in eq:
            eq.append(name)
        elif (op in ("<", ">", "<=", ">=") or op.upper() == "BETWEEN") and name not in rng:
            rng.append(name)
    return eq, rng


def scans(plan: dict[str, Any], stats: Snapshot) -> list[Scan]:
    out: list[Scan] = []

    def visit(n: dict[str, Any], under_limit: bool, sorted_above: bool) -> None:
        nt = n["Node Type"]
        if nt in ("Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan"):
            schema = n.get("Schema", "public")
            table = n.get("Relation Name", "")
            ts = stats.table(f"{schema}.{table}")
            cols = {c.name for c in ts.columns} if ts else set()
            texts = [n.get("Filter"), n.get("Index Cond"), n.get("Recheck Cond")]
            eq: list[str] = []
            rng: list[str] = []
            for txt in texts:
                e, r = _cols_in(txt, cols)
                eq += [c for c in e if c not in eq]
                rng += [c for c in r if c not in rng]
            fe, fr = _cols_in(n.get("Filter"), cols)
            outputs = []
            for o in n.get("Output", []):
                name = o.split(".")[-1].strip('"')
                if name in cols:
                    outputs.append(name)
            idx = n.get("Index Name")
            if nt == "Bitmap Heap Scan":
                kids = [k for k in plans.walk(n)[1:] if k["Node Type"] == "Bitmap Index Scan"]
                idx = kids[0]["Index Name"] if kids else None
            out.append(
                Scan(
                    n,
                    schema,
                    table,
                    idx,
                    eq,
                    rng,
                    fe + [c for c in fr if c not in fe],
                    outputs,
                    under_limit,
                    sorted_above,
                )
            )
        limit = under_limit or nt == "Limit"
        srt = sorted_above or nt in ("Sort", "Incremental Sort")
        for k in n.get("Plans", []):
            visit(k, limit, srt)

    visit(plan, False, False)
    return out


def _index_cols(ts: TableStats | None, name: str | None) -> list[str]:
    if ts is None or name is None:
        return []
    for ix in ts.indexes:
        if ix.name == name:
            return [c for c in ix.columns if c]
    return []


def _has_index(ts: TableStats | None, cols: list[str]) -> bool:
    if ts is None:
        return False
    for ix in ts.indexes:
        key = [c for c in ix.columns if c]
        if key[: len(cols)] == cols:
            return True
    return False


def _create_index(schema: str, table: str, cols: list[str], include: list[str]) -> str:
    def q(x: str) -> str:
        return x if x.isidentifier() and x.islower() else '"' + x.replace('"', '""') + '"'

    s = f"CREATE INDEX CONCURRENTLY ON {q(schema)}.{q(table)} ({', '.join(q(c) for c in cols)})"
    if include:
        s += f" INCLUDE ({', '.join(q(c) for c in include)})"
    return s + ";"


def suggest(sw: Switch, at_switch: Snapshot, settings: dict[str, str]) -> list[Fix]:
    old, new = sw.before.plan, sw.after.plan
    if old is None or new is None:
        return []
    fixes: list[Fix] = []
    old_scans = {(s.schema, s.table): s for s in scans(old, at_switch)}
    new_scans = scans(new, at_switch)

    for s in new_scans:
        ts = at_switch.table(f"{s.schema}.{s.table}")
        before = old_scans.get((s.schema, s.table))
        nt = s.node["Node Type"]

        # the LIMIT trap: walking an index in ORDER BY order and filtering
        if (
            nt in ("Index Scan", "Index Only Scan")
            and s.under_limit
            and not s.sorted_above
            and s.filter_cols
        ):
            order_cols = _index_cols(ts, s.index)[:1]
            key = [c for c in s.filter_cols if c not in order_cols] + order_cols
            if order_cols and not _has_index(ts, key):
                fixes.append(
                    Fix(
                        kind="index",
                        index=(s.schema, s.table, key, []),
                        sql=_create_index(s.schema, s.table, key, []),
                        why=(
                            f"the new plan reads {s.index} in {order_cols[0]} order and"
                            " discards rows until the LIMIT is met. that is fast only while"
                            f" matches on {', '.join(s.filter_cols)} are dense near the top"
                            " of the index; one index on both filter and order serves it"
                            " either way"
                        ),
                    )
                )

        # lost index access on a table
        lost = before is not None and nt == "Seq Scan" and before.node["Node Type"] != "Seq Scan"
        if lost and before is not None and ts is not None and ts.reltuples > 0:
            sel = float(s.node.get("Plan Rows", 0)) / ts.reltuples
            cols = before.eq_cols + [c for c in before.range_cols if c not in before.eq_cols]
            include = [c for c in s.outputs if c not in cols][:4]
            if cols and len(s.outputs) <= 6 and not _has_index(ts, cols + include):
                fixes.append(
                    Fix(
                        kind="index",
                        index=(s.schema, s.table, cols, include),
                        sql=_create_index(s.schema, s.table, cols, include),
                        why=(
                            f"about {sel:.1%} of {s.table} matches by then, past the point where"
                            " fetching heap pages one by one beats reading the table. a covering"
                            " index allows an index-only scan, which skips the heap for"
                            " all-visible pages"
                        ),
                    )
                )
            if settings.get("random_page_cost") in ("4", "4.0"):
                fixes.append(
                    Fix(
                        kind="setting",
                        setting=("random_page_cost", "1.1"),
                        sql="ALTER SYSTEM SET random_page_cost = 1.1;  -- SSD/NVMe storage only",
                        why=(
                            "random_page_cost is still the spinning-disk default of 4. on SSDs"
                            " random reads cost about the same as sequential ones"
                        ),
                    )
                )

        # columns whose MCV list is full: values that fall off it get an average guess
        if ts is not None:
            target = int(settings.get("default_statistics_target", "100") or 100)
            for c in s.eq_cols:
                col = ts.column(c)
                if col is None or not col.mcv:
                    continue
                t = col.stattarget if col.stattarget is not None and col.stattarget >= 0 else target
                if len(pgarray.parse(col.mcv)) >= t:
                    new_t = min(t * 5, 10000)
                    fixes.append(
                        Fix(
                            kind="statistics",
                            sql=(
                                f"ALTER TABLE {s.schema}.{s.table} ALTER COLUMN {c}"
                                f" SET STATISTICS {new_t};"
                            ),
                            why=(
                                f"{s.table}.{c} has a full MCV list ({t} values) at that point."
                                " any value not on it is estimated as an average, which is where"
                                " the row estimate for a skewed key goes wrong"
                            ),
                        )
                    )

    j_old, j_new = plans.joins(old), plans.joins(new)
    if j_new["nested loop"] > j_old["nested loop"]:
        fixes.append(
            Fix(
                kind="note",
                why=(
                    "switches to a nested loop. that is cheap only while the outer side really"
                    " returns few rows; if two filter columns are correlated the estimate is low"
                    " and the loop blows up. CREATE STATISTICS (dependencies) on them helps, but"
                    " extended statistics can't be injected before PostgreSQL 19 so flipcast"
                    " can't check that fix"
                ),
            )
        )
    elif j_old["nested loop"] > j_new["nested loop"] and j_new:
        fixes.append(
            Fix(
                kind="note",
                why=(
                    "moves off a nested loop because more rows are expected. usually the right"
                    " call at that size; confirm with EXPLAIN ANALYZE when you get there"
                ),
            )
        )

    if plans.node_types(new)["Gather"] + plans.node_types(new)["Gather Merge"] > (
        plans.node_types(old)["Gather"] + plans.node_types(old)["Gather Merge"]
    ):
        fixes.append(
            Fix(
                kind="note",
                why=(
                    "goes parallel. latency usually drops, but each run takes worker slots; check"
                    " max_parallel_workers if this query runs often"
                ),
            )
        )

    if sw.query.generic:
        fixes.append(
            Fix(
                kind="plan_cache",
                sql="ALTER ROLE <app_role> SET plan_cache_mode = force_custom_plan;",
                why=(
                    "this is the generic plan, which is what a prepared statement gets after five"
                    " runs. a generic plan can't use the MCV list for the parameter value"
                ),
            )
        )

    # same index suggested twice from two scans of one table
    seen: set[str] = set()
    out = []
    for f in fixes:
        k = f.sql or f.why
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def index_size(
    ts: TableStats, cols: list[str], include: list[str], rows: float
) -> tuple[int, float]:
    """Rough btree size for a hypothetical index: (pages, tuples).

    leaf tuple = 8 byte header + aligned key widths, plus a 4 byte line pointer,
    pages filled to 90%. deduplication is ignored, so this overestimates on
    low-cardinality keys, which makes the check conservative.
    """
    width = 0
    for c in cols + include:
        col = ts.column(c)
        width += max(col.avg_width if col is not None else 8, 1)
    tup = (8 + width + 7) // 8 * 8 + 4
    per_page = max(1, int((8192 - 24) * 0.9 // tup))
    leaves = math.ceil(rows / per_page)
    inner = math.ceil(leaves / 300)
    return max(1, leaves + inner + 1), rows
