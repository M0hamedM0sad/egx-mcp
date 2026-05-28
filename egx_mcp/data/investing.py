"""Investing.com data adapter — daily OHLCV for EGX names + macro factors.

Why this exists: Yahoo Finance now 401s EGX `.CA` symbols from many
environments, leaving the price-dependent tools (factor betas, volatility,
behavior) blind. Investing.com's public financial-data API still serves
clean daily history — the only trick is the `domain-id: www` request
header, without which it 403s.

Two endpoints:
  search     https://api.investing.com/api/search/v2/search?q=...
             resolves a ticker/name to a numeric pairId
  historical https://api.investing.com/api/financialdata/historical/{pairId}
             ?start-date=YYYY-MM-DD&end-date=YYYY-MM-DD&time-frame=Daily

pairId resolution is cached to disk (investing_pairids.json) so we only
hit the search endpoint once per symbol. Be gentle — investing.com will
rate-limit a client that hammers it.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("egx-mcp.investing")

# Verify TLS by default. Set EGX_INSECURE_SSL=1 only when running behind a
# TLS-inspecting proxy whose CA isn't in the trust store (corporate networks,
# some sandboxes) — it disables certificate verification for these calls.
_VERIFY_SSL = os.environ.get("EGX_INSECURE_SSL", "").strip() not in ("1", "true", "True")

_BASE = "https://api.investing.com/api"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "domain-id": "www",
    "Accept": "application/json",
    "Referer": "https://www.investing.com/",
}

_PAIRID_CACHE_PATH = Path(__file__).parent / "investing_pairids.json"

# Known pair IDs — factors plus a few EGX names verified by hand. Anything
# not listed is resolved via the search API and persisted to disk.
KNOWN_PAIR_IDS: dict[str, int] = {
    # Macro factors (keys match factors._FACTORS)
    "EGX30": 12860,   # EGX 30 index
    "EGP": 2122,      # USD/EGP — positive move = EGP weakening
    "BRENT": 8833,    # Brent crude (LCO)
    "GOLD": 8830,     # Gold futures (GC)
    "EM": 505,        # iShares MSCI EM ETF (EEM)
    # Verified EGX equities
    "TMGH": 12889,
}


def _load_pairid_cache() -> dict[str, int]:
    if _PAIRID_CACHE_PATH.exists():
        try:
            return json.loads(_PAIRID_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_pairid_cache(cache: dict[str, int]) -> None:
    try:
        _PAIRID_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"pairid cache write failed: {e}")


_PAIRIDS: dict[str, int] = {**KNOWN_PAIR_IDS, **_load_pairid_cache()}


def _get(url: str, params: dict | None = None) -> Any:
    with httpx.Client(timeout=25.0, headers=_HEADERS, follow_redirects=True, verify=_VERIFY_SSL) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.json()


def resolve_pair_id(ticker: str) -> int | None:
    """Resolve an EGX ticker (or factor key) to an investing.com pairId.

    Prefers Egypt-listed equity hits when searching a bare ticker.
    Caches every resolution to disk.
    """
    key = ticker.strip().upper()
    if key in _PAIRIDS:
        return _PAIRIDS[key]

    try:
        data = _get(f"{_BASE}/search/v2/search", params={"q": key})
    except Exception as e:
        log.warning(f"search failed for {key}: {e}")
        return None

    quotes = data.get("quotes", []) if isinstance(data, dict) else []
    # Prefer an exact-symbol Egypt-listed equity, then any exact symbol.
    chosen = None
    for q in quotes:
        if (q.get("symbol") or "").upper() == key and (q.get("flag") == "Egypt" or q.get("exchange") == "Egypt"):
            chosen = q
            break
    if chosen is None:
        for q in quotes:
            if (q.get("symbol") or "").upper() == key:
                chosen = q
                break
    if chosen is None and quotes:
        chosen = quotes[0]
    if not chosen:
        return None

    try:
        pid = int(chosen["id"])
    except (KeyError, ValueError, TypeError):
        return None

    _PAIRIDS[key] = pid
    _save_pairid_cache({k: v for k, v in _PAIRIDS.items() if k not in KNOWN_PAIR_IDS})
    return pid


def fetch_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 400,
) -> list[dict[str, Any]]:
    """Daily OHLCV rows ascending by date.

    Each row: {date 'YYYY-MM-DD', open, high, low, close, volume, change_pct}.
    Returns [] on any failure (resolution miss, HTTP error, empty payload).
    """
    pid = resolve_pair_id(ticker)
    if pid is None:
        return []
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.today() - timedelta(days=lookback_days)).isoformat()

    try:
        data = _get(
            f"{_BASE}/financialdata/historical/{pid}",
            params={
                "start-date": start,
                "end-date": end,
                "time-frame": "Daily",
                "add-missing-rows": "false",
            },
        )
    except Exception as e:
        log.warning(f"history fetch failed for {ticker} (pid {pid}): {e}")
        return []

    rows = data.get("data", []) if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            out.append({
                "date": r["rowDateTimestamp"][:10],
                "open": float(r["last_openRaw"]),
                "high": float(r["last_maxRaw"]),
                "low": float(r["last_minRaw"]),
                "close": float(r["last_closeRaw"]),
                "volume": int(r.get("volumeRaw") or 0),
                "change_pct": float(r["change_precent"]) if r.get("change_precent") not in (None, "") else None,
            })
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda x: x["date"])
    return out


def fetch_quote(ticker: str) -> dict[str, Any]:
    """Latest available daily bar as a quote-like dict."""
    rows = fetch_history(ticker, lookback_days=14)
    if not rows:
        return {"ticker": ticker.upper(), "error": "no data from investing.com"}
    last = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    return {
        "ticker": ticker.upper(),
        "price": last["close"],
        "previous_close": prev["close"] if prev else None,
        "change_pct": last.get("change_pct"),
        "date": last["date"],
        "volume": last["volume"],
        "source": "investing.com",
    }
