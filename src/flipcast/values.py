"""map column values to floats and back, for the types where that makes sense.

growth models do arithmetic on histogram bounds (a timestamp range extends,
an id range grows). that needs a number line, so each supported type gets a
codec: text as postgres prints it -> float, and float -> text postgres will
accept. types without a codec are carried forward unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

EPOCH_DATE = date(1970, 1, 1)
EPOCH_TS = datetime(1970, 1, 1)
EPOCH_TSTZ = datetime(1970, 1, 1, tzinfo=UTC)


class Codec:
    name = "?"
    integral = False
    # true when values follow the wall clock (a created_at column). those keep
    # their calendar position even in the faster/slower growth scenarios.
    temporal = False

    def to_float(self, text: str) -> float:
        raise NotImplementedError

    def to_text(self, x: float) -> str:
        raise NotImplementedError

    # how many float units one day is, for temporal types
    per_day = 1.0


@dataclass(frozen=True)
class _IntCodec(Codec):
    lo: int
    hi: int
    name: str = "int"
    integral: bool = True

    def to_float(self, text: str) -> float:
        return float(int(text))

    def to_text(self, x: float) -> str:
        v = round(x)
        return str(min(max(v, self.lo), self.hi))


class _FloatCodec(Codec):
    name = "float"

    def to_float(self, text: str) -> float:
        v = float(text)
        if math.isinf(v) or math.isnan(v):
            raise ValueError("non-finite")
        return v

    def to_text(self, x: float) -> str:
        return repr(float(x))


class _NumericCodec(_FloatCodec):
    name = "numeric"

    def to_text(self, x: float) -> str:
        # numeric_in takes exponent notation but people reading snapshots don't
        s = f"{x:.12g}"
        if "e" in s:
            s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s


class _DateCodec(Codec):
    name = "date"
    integral = True
    temporal = True
    per_day = 1.0

    def to_float(self, text: str) -> float:
        return float((date.fromisoformat(text) - EPOCH_DATE).days)

    def to_text(self, x: float) -> str:
        return (EPOCH_DATE + timedelta(days=round(x))).isoformat()


class _TimestampCodec(Codec):
    name = "timestamp"
    temporal = True
    per_day = 86400.0

    def to_float(self, text: str) -> float:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return (dt - EPOCH_TS).total_seconds()

    def to_text(self, x: float) -> str:
        return (EPOCH_TS + timedelta(seconds=x)).isoformat(sep=" ")


class _TimestamptzCodec(Codec):
    name = "timestamptz"
    temporal = True
    per_day = 86400.0

    def to_float(self, text: str) -> float:
        # the collector pins TimeZone to utc, so this is "...+00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (dt - EPOCH_TSTZ).total_seconds()

    def to_text(self, x: float) -> str:
        return (EPOCH_TSTZ + timedelta(seconds=x)).isoformat(sep=" ")


# keyed by base type oid, see pg_type.dat
_CODECS: dict[int, Codec] = {
    21: _IntCodec(-(2**15), 2**15 - 1, "int2"),
    23: _IntCodec(-(2**31), 2**31 - 1, "int4"),
    20: _IntCodec(-(2**63), 2**63 - 1, "int8"),
    26: _IntCodec(0, 2**32 - 1, "oid"),
    700: _FloatCodec(),
    701: _FloatCodec(),
    1700: _NumericCodec(),
    1082: _DateCodec(),
    1114: _TimestampCodec(),
    1184: _TimestamptzCodec(),
}


def codec_for(type_oid: int) -> Codec | None:
    return _CODECS.get(type_oid)


def decode_all(codec: Codec, elems: list[str | None]) -> list[float] | None:
    """floats for every element, or none if any element won't convert.

    'infinity' in a timestamp histogram is the usual reason.
    """
    out: list[float] = []
    for e in elems:
        if e is None:
            return None
        try:
            out.append(codec.to_float(e))
        except ValueError:
            return None
    return out
