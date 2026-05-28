"""Liquidity & capacity — ADV checks, participation caps, slippage.

EGX is thin. A 1M EGP trade in many EGX 100 names will move price
1-3% just on entry. This module adds the missing capacity gate that
turns "buy 412 shares" into "buy 412 shares OR cap by 15% of ADV,
whichever is smaller, and add 0.X% expected slippage to the cost basis."

Slippage estimate (simple but honest):
    expected_slippage_bps = participation_pct * 8 + 5

i.e. each 1% of ADV ≈ 8 bps of impact, plus a 5 bp baseline spread.
This is the Almgren-Chriss reduced form — good enough for sizing,
not for execution.
"""
from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.liquidity")


def get_adv(user_ticker: str, days: int = 20) -> dict[str, Any]:
    """Return average daily volume and turnover for the trailing N days."""
    canonical, yahoo, _ = resolve_ticker(user_ticker)
    try:
        df = yf.Ticker(yahoo).history(period=f"{max(days * 2, 60)}d", interval="1d")
        if df is None or df.empty:
            return {"ticker": canonical, "error": "no history"}
        recent = df.tail(days)
        avg_vol = float(recent["Volume"].mean())
        avg_close = float(recent["Close"].mean())
        adv_egp = avg_vol * avg_close
        median_vol = float(recent["Volume"].median())
        zero_volume_days = int((recent["Volume"] == 0).sum())
        return {
            "ticker": canonical,
            "yahoo_symbol": yahoo,
            "lookback_days": days,
            "avg_daily_volume_shares": int(avg_vol),
            "median_daily_volume_shares": int(median_vol),
            "avg_close_egp": round(avg_close, 4),
            "avg_daily_turnover_egp": round(adv_egp, 0),
            "zero_volume_days": zero_volume_days,
            "is_thin": zero_volume_days > days * 0.2 or adv_egp < 100_000,
        }
    except Exception as e:
        return {"ticker": canonical, "error": str(e)}


def check_capacity(
    user_ticker: str,
    intended_shares: int,
    max_participation_pct: float = 15.0,
    days: int = 20,
) -> dict[str, Any]:
    """Verify that a planned order respects ADV participation limits.

    Args:
        user_ticker: EGX code.
        intended_shares: Shares you plan to buy/sell.
        max_participation_pct: Cap on participation in ADV. Default 15%.
        days: Lookback for ADV computation. Default 20.

    Returns:
        Dict with: feasible (bool), max_safe_shares, participation_pct,
        estimated_slippage_bps, estimated_slippage_egp, adv_breakdown.
    """
    adv = get_adv(user_ticker, days=days)
    if "error" in adv:
        return {"ticker": adv.get("ticker"), "error": adv["error"]}

    avg_vol = adv["avg_daily_volume_shares"]
    if avg_vol <= 0:
        return {
            "ticker": adv["ticker"],
            "feasible": False,
            "reason": "zero or near-zero ADV — name is effectively untradeable",
            "adv": adv,
        }

    participation_pct = (intended_shares / avg_vol) * 100
    max_safe_shares = int(avg_vol * max_participation_pct / 100)
    feasible = intended_shares <= max_safe_shares

    # Slippage model: 8 bps per 1% participation + 5 bps spread baseline
    capped_pct = min(participation_pct, max_participation_pct)
    slippage_bps = capped_pct * 8 + 5
    slippage_egp = (slippage_bps / 10000) * intended_shares * adv["avg_close_egp"]

    return {
        "ticker": adv["ticker"],
        "intended_shares": intended_shares,
        "feasible": feasible,
        "participation_pct_of_adv": round(participation_pct, 2),
        "max_participation_cap_pct": max_participation_pct,
        "max_safe_shares": max_safe_shares,
        "is_thin": adv["is_thin"],
        "estimated_slippage_bps": round(slippage_bps, 1),
        "estimated_slippage_egp": round(slippage_egp, 2),
        "adv_breakdown": adv,
        "recommendation": (
            "OK to trade in one fill." if feasible
            else f"Cap order at {max_safe_shares} shares OR work the order over "
                 f"{int(participation_pct / max_participation_pct + 1)} sessions."
        ),
    }
