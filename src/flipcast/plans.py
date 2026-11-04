"""EXPLAIN a query and reduce the plan to its shape.

Two plans have the same shape when they use the same node types, on the same
relations and indexes, in the same tree. Costs, row counts and worker counts
are left out: those drift every step and are not what breaks a production
database. A switch is a change of shape.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import psycopg
from psycopg import sql

from . import sqltext
from .workload import Rendered

_STMT = "flipcast_q"


class ExplainError(RuntimeError):
    pass


def explain(conn: psycopg.Connection[Any], r: Rendered) -> dict[str, Any]:
    """Plan-only EXPLAIN. Returns the top "Plan" node."""
    try:
        with conn.transaction(), conn.cursor() as cur:
            # a schema copy has no data, but a source database does: keep the
            # planner from ever sitting on a lock for long
            cur.execute("SET LOCAL lock_timeout = '2s'")
            cur.execute("SET LOCAL statement_timeout = '30s'")
            if r.values is None:
                opts = "GENERIC_PLAN, FORMAT JSON" if sqltext.max_param(r.sql) else "FORMAT JSON"
                cur.execute(f"EXPLAIN ({opts}) {r.sql}")
            else:
                cur.execute(f"PREPARE {_STMT} AS {r.sql}")
                cur.execute("SET LOCAL plan_cache_mode = force_custom_plan")
                args = sql.SQL(", ").join(sql.Literal(v) for v in r.values)
                cur.execute(
                    sql.SQL("EXPLAIN (FORMAT JSON) EXECUTE {} ({})").format(
                        sql.Identifier(_STMT), args
                    )
                    if r.values
                    else sql.SQL("EXPLAIN (FORMAT JSON) EXECUTE {}").format(sql.Identifier(_STMT))
                )
            row = cur.fetchone()
            if r.values is not None:
                cur.execute(f"DEALLOCATE {_STMT}")
    except psycopg.Error as e:
        raise ExplainError(str(e).strip().splitlines()[0]) from e
    assert row is not None
    doc = row[0]
    if isinstance(doc, str):  # json arrives parsed, but be lenient
        doc = json.loads(doc)
    plan: dict[str, Any] = doc[0]["Plan"]
    return plan


# node attributes that are part of the shape
_SHAPE_KEYS = (
    "Join Type",
    "Strategy",
    "Partial Mode",
    "Scan Direction",
    "Command",
    "Operation",
    "Subplan Name",
    "CTE Name",
)


def _label(node: dict[str, Any]) -> str:
    parts = [node["Node Type"]]
    if node.get("Parallel Aware"):
        parts[0] = "Parallel " + parts[0]
    extras = [
        f"{node[k]}"
        for k in _SHAPE_KEYS
        if k in node and not (k == "Scan Direction" and node[k] != "Backward")
    ]
    where = []
    if "Relation Name" in node:
        rel = node["Relation Name"]
        alias = node.get("Alias")
        where.append(rel if not alias or alias == rel else f"{rel} {alias}")
    if "Index Name" in node:
        where.append("using " + node["Index Name"])
    s = str(parts[0])
    if extras:
        s += "<" + ",".join(extras) + ">"
    if where:
        s += "[" + " ".join(where) + "]"
    return s


def shape(plan: dict[str, Any]) -> str:
    kids = plan.get("Plans", [])
    lab = _label(plan)
    if not kids:
        return lab
    return lab + "(" + ", ".join(shape(k) for k in kids) + ")"


def shape_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def walk(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = [plan]
    for k in plan.get("Plans", []):
        out.extend(walk(k))
    return out


_SCAN_WORDS = {
    "Seq Scan": "seq scan",
    "Index Scan": "index scan",
    "Index Only Scan": "index-only scan",
    "Bitmap Heap Scan": "bitmap scan",
    "Tid Scan": "tid scan",
    "Tid Range Scan": "tid range scan",
}
_JOINS = {"Nested Loop": "nested loop", "Hash Join": "hash join", "Merge Join": "merge join"}


def accesses(plan: dict[str, Any]) -> dict[str, list[str]]:
    """How each relation (by alias) is read, in plan order."""
    out: dict[str, list[str]] = {}
    for n in walk(plan):
        nt = n["Node Type"]
        if nt not in _SCAN_WORDS:
            continue
        rel = n.get("Relation Name", "?")
        alias = n.get("Alias") or rel
        key = rel if alias == rel else f"{rel} {alias}"
        m = _SCAN_WORDS[nt]
        if n.get("Parallel Aware"):
            m = "parallel " + m
        idx = n.get("Index Name")
        if nt == "Bitmap Heap Scan":
            idx = (
                ", ".join(
                    k["Index Name"] for k in walk(n)[1:] if k["Node Type"] == "Bitmap Index Scan"
                )
                or None
            )
        if idx:
            m += f" ({idx})"
        if n.get("Scan Direction") == "Backward":
            m += " backward"
        out.setdefault(key, []).append(m)
    return out


def joins(plan: dict[str, Any]) -> Counter[str]:
    return Counter(_JOINS[n["Node Type"]] for n in walk(plan) if n["Node Type"] in _JOINS)


def node_types(plan: dict[str, Any]) -> Counter[str]:
    c: Counter[str] = Counter()
    for n in walk(plan):
        nt = n["Node Type"]
        if nt == "Aggregate" and n.get("Strategy") == "Hashed":
            nt = "HashAggregate"
        c[nt] += 1
        if n.get("Parallel Aware"):
            c["Parallel"] += 1
    return c


def describe_change(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Plain-English lines saying what changed between two plans."""
    lines: list[str] = []
    a_old, a_new = accesses(old), accesses(new)
    for rel in sorted(set(a_old) | set(a_new)):
        before, after = a_old.get(rel, []), a_new.get(rel, [])
        if before != after:
            b = " + ".join(before) or "not scanned"
            f = " + ".join(after) or "not scanned"
            lines.append(f"{rel}: {b} -> {f}")
    j_old, j_new = joins(old), joins(new)
    if j_old != j_new:
        lines.append(f"joins: {_counts(j_old) or 'none'} -> {_counts(j_new) or 'none'}")
    t_old, t_new = node_types(old), node_types(new)
    for nt, word in (
        ("Sort", "sort"),
        ("Incremental Sort", "incremental sort"),
        ("Materialize", "materialize"),
        ("Memoize", "memoize"),
        ("HashAggregate", "hash aggregate"),
        ("Gather", "gather"),
        ("Gather Merge", "gather merge"),
    ):
        if t_old[nt] and not t_new[nt]:
            lines.append(f"drops {word}")
        elif t_new[nt] and not t_old[nt]:
            lines.append(f"adds {word}")
    if not lines:
        lines.append("plan tree reshaped (same scans and joins, different order)")
    return lines


def _counts(c: Counter[str]) -> str:
    return ", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(c.items()))


def headline(old: dict[str, Any], new: dict[str, Any]) -> str:
    """One phrase for the report, e.g. 'index scan to seq scan on events'."""
    a_old, a_new = accesses(old), accesses(new)
    for rel in sorted(set(a_old) & set(a_new)):
        b, f = a_old[rel], a_new[rel]
        if b != f:
            return f"{_kind(b[0])} to {_kind(f[0])} on {rel}"
    j_old, j_new = joins(old), joins(new)
    if j_old != j_new:
        gone = j_old - j_new
        came = j_new - j_old
        if gone and came:
            return f"{next(iter(gone))} to {next(iter(came))}"
    return describe_change(old, new)[0]


def _kind(method: str) -> str:
    # "index scan (events_kind_idx)" -> "index scan"
    return method.split(" (", maxsplit=1)[0]


# which enable_* setting turns off each node type, for forcing a plan
_DISABLE = {
    "Seq Scan": "enable_seqscan",
    "Index Scan": "enable_indexscan",
    "Index Only Scan": "enable_indexonlyscan",
    "Bitmap Heap Scan": "enable_bitmapscan",
    "Nested Loop": "enable_nestloop",
    "Hash Join": "enable_hashjoin",
    "Merge Join": "enable_mergejoin",
    "Sort": "enable_sort",
    "Incremental Sort": "enable_incremental_sort",
    "HashAggregate": "enable_hashagg",
    "Materialize": "enable_material",
    "Memoize": "enable_memoize",
    "Tid Scan": "enable_tidscan",
    "Gather Merge": "enable_gathermerge",
    "Parallel Append": "enable_parallel_append",
}


def toggles_against(target: dict[str, Any], other: dict[str, Any]) -> list[str]:
    """enable_* settings to turn off so the planner can't pick `other`'s nodes."""
    t, o = node_types(target), node_types(other)
    off = [g for nt, g in _DISABLE.items() if o[nt] and not t[nt]]
    if (o["Gather"] or o["Gather Merge"] or o["Parallel"]) and not (t["Gather"] or t["Parallel"]):
        off.append("max_parallel_workers_per_gather")
    return sorted(set(off))


def cost(plan: dict[str, Any]) -> float:
    return float(plan["Total Cost"])
