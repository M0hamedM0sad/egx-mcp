"""US Federal Reserve policy rate, from the Fed's own FRED series.

DFEDTARU / DFEDTARL are the daily upper / lower bounds of the federal funds
target range (St. Louis Fed). A change in DFEDTARU marks an FOMC move; the
series is dated by EFFECTIVE date, normally the day after the Wednesday
announcement. The announcement lands at ~21:00 Cairo, after the EGX close,
so the effective date is the first session that can react.

Holds are not visible in these series (no change); they need the FOMC
calendar, which this module does not scrape.
"""
from __future__ import annotations

import csv
import io
import logging
import time
from datetime import date
from typing import Any

log = logging.getLogger("egx-mcp.fed")

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_CACHE: dict[str, tuple[float, list[tuple[date, float]]]] = {}
_TTL_S = 6 * 3600


def fetch_series(series_id: str) -> list[tuple[date, float]]:
    """[(date, value)] ascending; missing values ('.') dropped. Cached 6h."""
    hit = _CACHE.get(series_id)
    if hit and time.time() - hit[0] < _TTL_S:
        return hit[1]
    import httpx
    from ._certs import ensure_ca_bundle

    ensure_ca_bundle()
    r = httpx.get(_FRED_CSV.format(sid=series_id), timeout=30, follow_redirects=True,
                  headers={"User-Agent": "egx-mcp (research)"})
    r.raise_for_status()
    out = parse_csv(r.text)
    _CACHE[series_id] = (time.time(), out)
    return out


def parse_csv(text: str) -> list[tuple[date, float]]:
    rows = csv.reader(io.StringIO(text))
    next(rows, None)                       # header: observation_date,<SERIES>
    out = []
    for row in rows:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except ValueError:
            continue
    return sorted(out)


def decisions(upper: list[tuple[date, float]]) -> list[dict[str, Any]]:
    """FOMC moves: every change in the upper bound, with its size in bp."""
    out = []
    for (d0, v0), (d1, v1) in zip(upper, upper[1:]):
        if v1 != v0:
            bp = round((v1 - v0) * 100)
            out.append({"effective_date": d1.isoformat(), "upper_pct": v1,
                        "change_bp": bp, "action": "hike" if bp > 0 else "cut"})
    return out


def current() -> dict[str, Any]:
    """Current target range and the last move, for the macro context."""
    try:
        upper, lower = fetch_series("DFEDTARU"), fetch_series("DFEDTARL")
    except Exception as e:  # noqa: BLE001
        log.warning(f"FRED fetch failed: {e}")
        return {"upper_pct": None, "lower_pct": None, "error": f"FRED unreachable: {e}"}
    moves = decisions(upper)
    last = moves[-1] if moves else None
    return {
        "upper_pct": upper[-1][1] if upper else None,
        "lower_pct": lower[-1][1] if lower else None,
        "as_of": upper[-1][0].isoformat() if upper else None,
        "last_move": last,
        "days_since_last_move": ((date.today() - date.fromisoformat(last["effective_date"])).days
                                 if last else None),
        "source": "FRED DFEDTARU/DFEDTARL (Federal Reserve target range)",
    }
