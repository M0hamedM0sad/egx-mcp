"""EGP risk-free rate — the opportunity cost of every long.

Egyptian T-bills currently yield 25-28%. That's the bar every equity
expected return must clear. This module returns the risk-free rate
used everywhere else for excess-return calculations.

Source priority:
  1. EGX_TBILL_RATE_PCT env var (operator override, e.g. "27.5")
  2. CBE T-bill auction page (best-effort scrape)
  3. CBE policy rate midpoint (from macro.py) as a proxy
  4. Hardcoded fallback of 25.0% — explicitly flagged as stale

Outputs both annual and per-period rates so downstream code can
compute excess returns over any horizon cleanly.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any

import httpx

from . import macro

log = logging.getLogger("egx-mcp.risk_free")

_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 3600  # 1 hour

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}


def _from_env() -> float | None:
    val = os.environ.get("EGX_TBILL_RATE_PCT")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        log.warning(f"EGX_TBILL_RATE_PCT={val!r} is not a number — ignoring")
        return None


def _from_cbe() -> float | None:
    """Best-effort scrape of CBE auction results.

    The /en/auctions/treasury-bills URL went 404 in 2026; /en/auctions
    is the live parent that lists T-bill, T-bond, and corridor data.
    """
    candidates = (
        "https://www.cbe.org.eg/en/auctions",
        "https://www.cbe.org.eg/en/economic-research/Statistics/government-securities",
    )
    for url in candidates:
        try:
            with httpx.Client(timeout=10, headers=_HEADERS, follow_redirects=True) as c:
                r = c.get(url)
                if r.status_code != 200:
                    continue
                # Look for any percentage figure on the page
                matches = re.findall(r"(\d{2}\.\d{1,3})\s*%", r.text)
                # Filter to plausible T-bill range (10-35%)
                rates = [float(m) for m in matches if 10 <= float(m) <= 35]
                if rates:
                    rates.sort()
                    return rates[len(rates) // 2]
        except Exception as e:
            log.warning(f"CBE scrape failed at {url}: {e}")
    return None


def _from_policy_proxy() -> float | None:
    """Use CBE corridor midpoint as a T-bill proxy (typically T-bill ≈ corridor + 50bps)."""
    ctx = macro.get_context()
    midpoint = (ctx.get("cbe_rates") or {}).get("midpoint_pct")
    if midpoint:
        return float(midpoint) + 0.5
    return None


def get_rate(annual_pct: bool = True) -> dict[str, Any]:
    """Return the EGP risk-free rate with source provenance.

    Returns:
        Dict with: rate_pct (annual), source, daily_rate_pct,
        as_of, is_stale (True if hardcoded fallback used).
    """
    now = time.time()
    if "rate" in _CACHE:
        ts, val = _CACHE["rate"]
        if now - ts < _TTL:
            return val

    source = None
    rate = _from_env()
    if rate is not None:
        source = "env:EGX_TBILL_RATE_PCT"
    if rate is None:
        rate = _from_cbe()
        if rate is not None:
            source = "cbe:treasury-bills"
    if rate is None:
        rate = _from_policy_proxy()
        if rate is not None:
            source = "macro:cbe_midpoint+50bps"
    if rate is None:
        rate = 25.0
        source = "fallback:hardcoded_stale"

    daily = (1 + rate / 100) ** (1 / 252) - 1

    payload = {
        "rate_pct": round(rate, 3),
        "source": source,
        "is_stale": source.startswith("fallback"),
        "daily_rate_pct": round(daily * 100, 5),
        "as_of": datetime.utcnow().isoformat() + "Z",
        "note": (
            "Used as the baseline for excess-return calculations across the "
            "scoring engine, decision tool, and backtest harness. Override "
            "with EGX_TBILL_RATE_PCT env var when CBE source is stale."
        ),
    }
    _CACHE["rate"] = (now, payload)
    return payload


def excess_return_pct(period_return_pct: float, horizon_days: int) -> float:
    """Convert a period return to an excess return over T-bills."""
    rf = get_rate()
    daily_rf = rf["daily_rate_pct"] / 100
    rf_period = ((1 + daily_rf) ** horizon_days - 1) * 100
    return round(period_return_pct - rf_period, 4)
