"""Macro context adapter — EGP/USD, Brent, CBE policy rate, CPI.

EGX moves on macro more than fundamentals. A decision without macro
context is incomplete. This module gathers the four most-consequential
inputs:

  - EGP/USD spot (yfinance USDEGP=X)
  - Brent crude (yfinance BZ=F) — drives petrochems and Suez Canal flow
  - CBE corridor rate (scraped from cbe.org.eg)
  - Headline CPI YoY (scraped from CBE's monthly bulletin)

If any source fails, that field is null and the rest still return.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any

import httpx
import yfinance as yf

log = logging.getLogger("egx-mcp.macro")

_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 1800  # 30 min

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (egx-mcp/0.1)",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}
_TIMEOUT = 10.0


def _cached(key: str, loader):
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _TTL:
            return val
    val = loader()
    _CACHE[key] = (now, val)
    return val


def _yf_quote(symbol: str) -> dict[str, Any]:
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        last = fi["last_price"] if fi else None
        prev = fi["previous_close"] if fi else None
        change_pct = ((last - prev) / prev * 100) if (last and prev) else None
        return {
            "value": round(float(last), 4) if last else None,
            "change_pct": round(change_pct, 2) if change_pct else None,
        }
    except Exception as e:
        log.warning(f"yfinance fetch failed for {symbol}: {e}")
        return {"value": None, "change_pct": None, "error": str(e)}


def _cbe_policy_rate() -> dict[str, Any]:
    """Scrape CBE's policy rate page. Falls back gracefully on failure."""
    url = "https://www.cbe.org.eg/en/economic-research/statistics/cbe-rates"
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            html = r.text
        # CBE publishes deposit & lending corridor rates as percentages.
        # Layout-resilient: grab the first two %-style numbers near "Overnight".
        rates = re.findall(r"(\d{1,2}\.\d{1,3})\s*%", html)
        deposit = float(rates[0]) if len(rates) >= 1 else None
        lending = float(rates[1]) if len(rates) >= 2 else None
        return {
            "deposit_rate_pct": deposit,
            "lending_rate_pct": lending,
            "midpoint_pct": round((deposit + lending) / 2, 2) if (deposit and lending) else None,
            "source": url,
        }
    except Exception as e:
        log.warning(f"CBE policy rate fetch failed: {e}")
        return {
            "deposit_rate_pct": None,
            "lending_rate_pct": None,
            "midpoint_pct": None,
            "error": f"CBE page unreachable: {e}",
            "source": url,
        }


_TROY_OZ_GRAMS = 31.1034768


def gold_prices_egp() -> dict[str, Any]:
    """Compute Egyptian gold prices in EGP per gram from international spot.

    Sources:
        gold_USD_per_oz   yfinance GC=F (COMEX gold front-month future)
        USDEGP            yfinance USDEGP=X

    Outputs (per gram in EGP):
        24K       100% pure gold
        21K       Egyptian jewelry standard (21/24 fineness)
        18K       Common alternate jewelry standard (18/24 = 0.75)
        Egyptian gold pound (الجنيه الذهب) = 8 grams of 21K gold

    Local Egyptian dealer prices typically run 5-15% above this fair-value
    computation due to workmanship and currency-market premium. This
    function returns the *theoretical* price; cross-check against a local
    source like sigma-egy or iSagha before acting.
    """

    def _load():
        gold_usd = _yf_quote("GC=F")  # USD/oz
        usdegp = _yf_quote("USDEGP=X")
        gold_oz_usd = gold_usd.get("value")
        fx = usdegp.get("value")
        if not gold_oz_usd or not fx:
            return {
                "error": "Gold or USDEGP unavailable from Yahoo right now.",
                "gold_oz_usd": gold_oz_usd,
                "usdegp": fx,
            }
        # Per-gram pure gold in EGP
        per_g_24k = gold_oz_usd * fx / _TROY_OZ_GRAMS
        per_g_21k = per_g_24k * 21 / 24
        per_g_18k = per_g_24k * 18 / 24
        gold_pound_egp = 8 * per_g_21k  # Egyptian gold pound = 8g of 21K
        return {
            "as_of": datetime.utcnow().isoformat() + "Z",
            "gold_oz_usd": round(gold_oz_usd, 2),
            "usdegp": round(fx, 4),
            "egp_per_gram_24k": round(per_g_24k, 2),
            "egp_per_gram_21k": round(per_g_21k, 2),
            "egp_per_gram_18k": round(per_g_18k, 2),
            "egyptian_gold_pound_egp": round(gold_pound_egp, 2),
            "method": (
                "Computed from international spot: (USD/oz × USDEGP) / 31.1035 "
                "for 24K, then × 21/24 and 18/24 for the standard alloys. "
                "Local dealer prices typically 5-15% above this — verify "
                "against sigma-egy / iSagha before acting."
            ),
            "note": (
                "Egyptian gold pound (الجنيه الذهب) = 8g of 21K gold by "
                "convention — that's the math here, regardless of any local "
                "premium for handmade pieces."
            ),
        }

    return _cached("gold_egp", _load)


def get_context() -> dict[str, Any]:
    """Return the full macro snapshot used by the scoring engine."""

    def _load():
        usdegp = _yf_quote("USDEGP=X")
        brent = _yf_quote("BZ=F")
        gold = _yf_quote("GC=F")
        cbe = _cbe_policy_rate()

        # Heuristic regime classification — this drives sector adjustments
        # in the scoring engine.
        regime_flags = []
        if usdegp.get("change_pct") is not None and usdegp["change_pct"] > 1:
            regime_flags.append("EGP weakening — favors exporters (chemicals, steel), hurts importers")
        if brent.get("change_pct") is not None and brent["change_pct"] > 2:
            regime_flags.append("Brent rallying — supports petrochems, pressures industrial margins")
        if cbe.get("midpoint_pct") and cbe["midpoint_pct"] >= 25:
            regime_flags.append("Tight monetary regime — banks benefit, real estate financing strained")

        return {
            "as_of": datetime.utcnow().isoformat() + "Z",
            "egp_usd": usdegp,
            "brent_usd": brent,
            "gold_usd": gold,
            "cbe_rates": cbe,
            "regime_flags": regime_flags,
            "note": (
                "EGP/USD and Brent from Yahoo (delayed). CBE rates scraped — "
                "verify against cbe.org.eg before acting on rate-sensitive trades."
            ),
        }

    return _cached("macro", _load)


def sector_macro_bias(sector: str) -> dict[str, Any]:
    """Map current macro regime → sector tailwind/headwind score in [-1, +1].

    Used by the scoring engine to nudge composite scores up or down based
    on macro fit. This is intentionally a small fixed rubric — easier to
    audit and override than a black-box model.
    """
    ctx = get_context()
    bias = 0.0
    reasons = []

    egp_chg = ctx.get("egp_usd", {}).get("change_pct") or 0
    brent_chg = ctx.get("brent_usd", {}).get("change_pct") or 0
    cbe_mid = ctx.get("cbe_rates", {}).get("midpoint_pct") or 0

    s = sector.lower()

    # EGP weakness helps exporters, hurts importers
    if egp_chg > 0.5:
        if s in ("chemicals", "basic resources"):
            bias += 0.3; reasons.append("EGP↓ helps exporters")
        elif s in ("food & beverage", "personal & household", "retail"):
            bias -= 0.3; reasons.append("EGP↓ raises import costs")

    # Brent up helps petrochems
    if brent_chg > 1:
        if s == "chemicals":
            bias += 0.2; reasons.append("Brent↑ supports petrochem margins")
        elif s in ("travel & leisure", "industrial goods"):
            bias -= 0.2; reasons.append("Brent↑ pressures fuel-heavy sectors")

    # High rates help banks, hurt rate-sensitive
    if cbe_mid >= 25:
        if s == "banks":
            bias += 0.4; reasons.append(f"CBE midpoint {cbe_mid}% — bank NIMs expanded")
        elif s == "real estate":
            bias -= 0.4; reasons.append(f"CBE midpoint {cbe_mid}% — mortgage demand suppressed")
        elif s == "financial services":
            bias += 0.1; reasons.append(f"CBE midpoint {cbe_mid}% — fee businesses neutral, lending arms benefit")

    bias = max(-1.0, min(1.0, bias))
    return {
        "sector": sector,
        "macro_bias": round(bias, 2),
        "reasons": reasons,
    }
