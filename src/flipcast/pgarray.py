"""parse and print one-dimensional postgres array literals.

pg_stats hands back anyarray columns, which only survive the trip to the
client as text. everything flipcast does with mcv lists and histograms starts
by splitting that text into elements, and ends by printing a literal that
pg_restore_attribute_stats() can feed to the column type's input function.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class ArrayParseError(ValueError):
    pass


def parse(text: str) -> list[str | None]:
    s = text.strip()
    # arrays with non-default lower bounds print as [2:3]={...}; pg_stats never
    # produces those but a hand-edited snapshot might
    if s.startswith("["):
        eq = s.find("=")
        if eq < 0:
            raise ArrayParseError(f"bad array dimensions in {text!r}")
        s = s[eq + 1 :]
    if len(s) < 2 or s[0] != "{" or s[-1] != "}":
        raise ArrayParseError(f"not an array literal: {text!r}")

    out: list[str | None] = []
    i, n = 1, len(s) - 1
    if i == n:
        return out

    while i <= n:
        if s[i] == '"':
            i += 1
            buf: list[str] = []
            while True:
                if i >= n:
                    raise ArrayParseError(f"unterminated quote in {text!r}")
                c = s[i]
                if c == "\\":
                    buf.append(s[i + 1])
                    i += 2
                elif c == '"':
                    i += 1
                    break
                else:
                    buf.append(c)
                    i += 1
            out.append("".join(buf))
        elif s[i] == "{":
            # nested element, e.g. stats on an array column. keep it verbatim.
            depth, j = 0, i
            in_quote = False
            while j < n:
                c = s[j]
                if in_quote:
                    if c == "\\":
                        j += 1
                    elif c == '"':
                        in_quote = False
                elif c == '"':
                    in_quote = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(s[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and s[j] != ",":
                j += 1
            raw = s[i:j].strip()
            out.append(None if raw.upper() == "NULL" else raw)
            i = j

        if i < n:
            if s[i] != ",":
                raise ArrayParseError(f"expected ',' at offset {i} in {text!r}")
            i += 1
        else:
            break
    return out


def _quote(elem: str) -> str:
    return '"' + elem.replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_literal(elems: Iterable[str | None]) -> str:
    # quoting every element is always legal and saves us from knowing which
    # characters are special for the element type
    parts = ["NULL" if e is None else _quote(e) for e in elems]
    return "{" + ",".join(parts) + "}"


def parse_floats(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(x) for x in parse(text) if x is not None]


def floats_literal(values: Sequence[float]) -> str:
    return "{" + ",".join(repr(float(v)) for v in values) + "}"
