"""Reject price series that a corporate action or a bad vendor tick broke.

The EGX applies a daily price limit (±10% for most names, ±20% for the
small-cap board). A single session outside that band is therefore not a market
move — it is a split, a capital reduction, a bonus issue, or a vendor error.
Grading such a window measures the corporate action, not the model.

This bit the live evidence base directly: HDBK graded 161.92 -> 82.70 (-49% in
one session) across six June rows, and EGBE graded with entry_price -0.3392,
producing a -242% "return". Those rows sat inside the 20-call sample that the
reliability gate reads, so the model's measured edge was partly an artifact.

Used by the grader (quarantine the row), the panel builder (drop the name at
that date) and sizing (an ATR computed across a split is meaningless).
"""
from __future__ import annotations

import pandas as pd

# Widest EGX daily band (small-cap board) plus headroom for a stale-quote
# rebound the day after a halt. Anything past this is a data break.
MAX_SESSION_MOVE_PCT = 25.0


def is_valid_price(value: object) -> bool:
    """A price must be a finite, strictly positive number."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == value          # not NaN
            and value not in (float("inf"), float("-inf"))
            and value > 0)


def clean_series(series: pd.Series) -> pd.Series:
    """Drop non-positive / non-finite observations, keep the order."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s[(s > 0) & (s != float("inf"))]


def find_break(
    series: pd.Series,
    start: object = None,
    end: object = None,
    limit_pct: float = MAX_SESSION_MOVE_PCT,
) -> dict | None:
    """First session in [start, end] whose move exceeds the daily limit.

    Returns ``{"date", "pct", "from", "to"}`` for the offending session, or
    ``None`` when the window is clean. Bounds are inclusive and optional.
    """
    s = clean_series(series)
    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[:pd.Timestamp(end)]
    if len(s) < 2:
        return None
    moves = s.pct_change().dropna() * 100
    hit = moves[moves.abs() > limit_pct]
    if hit.empty:
        return None
    when = hit.index[0]
    pos = s.index.get_loc(when)
    return {
        "date": when.strftime("%Y-%m-%d") if hasattr(when, "strftime") else str(when),
        "pct": round(float(hit.iloc[0]), 2),
        "from": round(float(s.iloc[pos - 1]), 4),
        "to": round(float(s.iloc[pos]), 4),
    }
