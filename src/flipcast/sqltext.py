"""A small SQL lexer, just enough to split scripts and rewrite clock calls.

It knows about the things that make naive string handling wrong: quoted
strings (including E'' and doubled quotes), quoted identifiers, dollar quoting,
and both comment styles. It does not try to understand the grammar.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

_IDENT = re.compile(r"[A-Za-z_\x80-￿][A-Za-z0-9_$\x80-￿]*")
_PARAM = re.compile(r"\$(\d+)")
_DOLLAR_TAG = re.compile(r"\$([A-Za-z_\x80-￿][A-Za-z0-9_\x80-￿]*)?\$")
_NUMBER = re.compile(r"\d+(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?")


@dataclass(frozen=True)
class Token:
    kind: str  # ident, string, qident, param, number, op, comment, space
    text: str
    start: int


def tokens(sql: str) -> Iterator[Token]:
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        start = i
        if c.isspace():
            while i < n and sql[i].isspace():
                i += 1
            yield Token("space", sql[start:i], start)
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            yield Token("comment", sql[start:i], start)
        elif sql.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            yield Token("comment", sql[start:i], start)
        elif c in "eE" and i + 1 < n and sql[i + 1] == "'":
            i = _skip_string(sql, i + 1, backslash=True)
            yield Token("string", sql[start:i], start)
        elif c == "'":
            i = _skip_string(sql, i, backslash=False)
            yield Token("string", sql[start:i], start)
        elif c == '"':
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            yield Token("qident", sql[start:i], start)
        elif c == "$":
            m = _PARAM.match(sql, i)
            if m:
                i = m.end()
                yield Token("param", m.group(0), start)
                continue
            m = _DOLLAR_TAG.match(sql, i)
            if m:
                tag = m.group(0)
                j = sql.find(tag, m.end())
                i = n if j < 0 else j + len(tag)
                yield Token("string", sql[start:i], start)
                continue
            i += 1
            yield Token("op", c, start)
        elif c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            m = _NUMBER.match(sql, i)
            assert m is not None
            i = m.end()
            yield Token("number", sql[start:i], start)
        else:
            m = _IDENT.match(sql, i)
            if m:
                i = m.end()
                yield Token("ident", m.group(0), start)
            else:
                i += 1
                yield Token("op", c, start)


def _skip_string(sql: str, i: int, *, backslash: bool) -> int:
    # i points at the opening quote
    n = len(sql)
    i += 1
    while i < n:
        c = sql[i]
        if backslash and c == "\\":
            i += 2
            continue
        if c == "'":
            if i + 1 < n and sql[i + 1] == "'":
                i += 2
                continue
            return i + 1
        i += 1
    return n


def split_statements(sql: str) -> list[str]:
    """Split a script on top-level semicolons. Empty statements are dropped."""
    out: list[str] = []
    begin = 0
    for t in tokens(sql):
        if t.kind == "op" and t.text == ";":
            stmt = sql[begin : t.start].strip()
            if _has_code(stmt):
                out.append(stmt)
            begin = t.start + 1
    tail = sql[begin:].strip()
    if _has_code(tail):
        out.append(tail)
    return out


def _has_code(stmt: str) -> bool:
    return any(t.kind not in ("space", "comment") for t in tokens(stmt))


def max_param(sql: str) -> int:
    best = 0
    for t in tokens(sql):
        if t.kind == "param":
            best = max(best, int(t.text[1:]))
    return best


# functions whose value the planner folds in at plan time. rewriting them to
# literals is how a query is planned "as of" a future date.
_CLOCK_FUNCS = {"now", "transaction_timestamp", "statement_timestamp", "clock_timestamp"}
_CLOCK_KEYWORDS = {
    "current_timestamp": "timestamptz",
    "localtimestamp": "timestamp",
    "current_date": "date",
}


def rewrite_clock(sql: str, tstz: str, ts: str, d: str) -> str:
    """Replace now() and friends with literals.

    tstz, ts and d are the replacement values already formatted for
    timestamptz, timestamp and date.
    """
    toks = list(tokens(sql))
    literal = {
        "timestamptz": f"'{tstz}'::timestamptz",
        "timestamp": f"'{ts}'::timestamp",
        "date": f"'{d}'::date",
    }
    out: list[str] = []
    k = 0
    while k < len(toks):
        t = toks[k]
        if t.kind == "ident":
            low = t.text.lower()
            nxt = _next_code(toks, k + 1)
            if low in _CLOCK_FUNCS and nxt is not None and toks[nxt].text == "(":
                close = _next_code(toks, nxt + 1)
                if close is not None and toks[close].text == ")":
                    _drop_schema_prefix(out)
                    out.append("(" + literal["timestamptz"] + ")")
                    k = close + 1
                    continue
            if low in _CLOCK_KEYWORDS:
                end = k + 1
                # current_timestamp(3), localtimestamp(0)
                if nxt is not None and toks[nxt].text == "(" and low != "current_date":
                    num = _next_code(toks, nxt + 1)
                    close = _next_code(toks, num + 1) if num is not None else None
                    if (
                        num is not None
                        and toks[num].kind == "number"
                        and close is not None
                        and toks[close].text == ")"
                    ):
                        end = close + 1
                out.append("(" + literal[_CLOCK_KEYWORDS[low]] + ")")
                k = end
                continue
        out.append(t.text)
        k += 1
    return "".join(out)


def _next_code(toks: list[Token], k: int) -> int | None:
    while k < len(toks):
        if toks[k].kind not in ("space", "comment"):
            return k
        k += 1
    return None


def _drop_schema_prefix(out: list[str]) -> None:
    # pg_catalog.now() -> drop the "pg_catalog." we already emitted
    if len(out) >= 2 and out[-1] == "." and out[-2].lower() in ("pg_catalog", '"pg_catalog"'):
        del out[-2:]


def mentions_clock(sql: str) -> bool:
    for t in tokens(sql):
        if t.kind == "ident":
            low = t.text.lower()
            if low in _CLOCK_FUNCS or low in _CLOCK_KEYWORDS:
                return True
    return False
