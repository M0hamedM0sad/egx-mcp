"""Short-term Monte Carlo simulator with technical edge overlay.

Goal: rank EGX names by probability of a meaningful upside move within
1-5 trading days. Two stages:

  1. Bootstrap simulation
     - Pull the last `lookback_days` of daily returns from yfinance.
     - Sample `horizon_days` returns with replacement, `n_paths` times.
     - Compound to terminal prices to build an empirical distribution.

  2. Edge overlay
     - Technical state (RSI, trend, acceleration) shifts the daily drift.
       RSI < 30 → mean-revert higher (oversold bounce candidate).
       RSI > 75 → mean-revert lower (overbought).
       Above MA20 with MA20 > MA50 → uptrend tailwind.
       Last 5d return > 1.5× recent average → acceleration signal.

The output for each ticker:
    expected_return_pct, prob_up_2pct, prob_up_5pct, prob_down_5pct,
    p10 / p50 / p90 terminal prices, sharpe-style imminent_move_score,
    drivers (the signals that fired), edge_drift_pct (the daily drift bias).

Bootstrap is honest: it uses the *actual* recent return distribution,
including fat tails and skew. Parametric (Black-Scholes) MC would
under-estimate jump risk on EGX names.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any

from .universe import EGX_UNIVERSE, resolve_ticker
from . import technicals
from . import egx_listing
from . import investing

log = logging.getLogger("egx-mcp.simulation")


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1])."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _edge_overlay(user_ticker: str) -> dict[str, Any]:
    """Compute the daily drift adjustment from technicals, plus drivers."""
    drift = 0.0
    drivers = []

    try:
        t = technicals.compute(user_ticker, period="3mo")
    except Exception as e:
        return {"drift_pct": 0.0, "drivers": [f"technicals failed: {e}"]}

    ind = t.get("indicators") or {}
    price = t.get("price")

    rsi = ind.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            drift += 0.003; drivers.append(f"RSI {rsi:.1f} oversold → mean-revert higher (+0.3%/day)")
        elif rsi < 40:
            drift += 0.001; drivers.append(f"RSI {rsi:.1f} weak → mild bounce bias (+0.1%/day)")
        elif rsi > 75:
            drift -= 0.003; drivers.append(f"RSI {rsi:.1f} overbought → mean-revert lower (-0.3%/day)")
        elif rsi > 65:
            drift -= 0.001; drivers.append(f"RSI {rsi:.1f} extended → mild fade bias (-0.1%/day)")

    sma_20 = ind.get("sma_20")
    sma_50 = ind.get("sma_50")
    if price and sma_20 and sma_50:
        if price > sma_20 > sma_50:
            drift += 0.0015; drivers.append("Price > MA20 > MA50 → uptrend tailwind (+0.15%/day)")
        elif price < sma_20 < sma_50:
            drift -= 0.0015; drivers.append("Price < MA20 < MA50 → downtrend headwind (-0.15%/day)")

    macd = ind.get("macd")
    macd_sig = ind.get("macd_signal")
    macd_hist = ind.get("macd_histogram")
    if macd is not None and macd_sig is not None and macd_hist is not None:
        if macd > macd_sig and macd_hist > 0:
            drift += 0.001; drivers.append("MACD bullish & expanding → momentum (+0.1%/day)")
        elif macd < macd_sig and macd_hist < 0:
            drift -= 0.001; drivers.append("MACD bearish & expanding → momentum (-0.1%/day)")

    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    if price and bb_upper and price > bb_upper:
        drift -= 0.001; drivers.append("Above upper Bollinger → extended (-0.1%/day)")
    elif price and bb_lower and price < bb_lower:
        drift += 0.002; drivers.append("Below lower Bollinger → snap-back candidate (+0.2%/day)")

    # Cap the overlay so it can't dominate the empirical distribution
    drift = max(-0.006, min(0.006, drift))

    return {"drift_pct": round(drift, 5), "drivers": drivers}


def simulate_one(
    user_ticker: str,
    horizon_days: int = 5,
    n_paths: int = 2000,
    lookback_days: int = 60,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a bootstrap MC for one ticker and return a probabilistic forecast."""
    canonical, yahoo, name = resolve_ticker(user_ticker)

    # Reliable, glitch-guarded daily history (investing.com primary, Yahoo
    # fallback, zero-volume bars dropped). One fetch supplies both the return
    # distribution and the baseline close, so the forecast can never anchor on
    # a stale or carry-forward bar.
    df = investing.daily_history(canonical, lookback_days=max(lookback_days, 60) + 60)
    closes = df["Close"].dropna().tolist() if not df.empty else []
    if len(closes) < 21:
        return {"ticker": canonical, "error": "Not enough return history for simulation"}

    rets = [closes[i] / closes[i - 1] - 1
            for i in range(1, len(closes)) if closes[i - 1] > 0][-lookback_days:]
    if len(rets) < 20:
        return {"ticker": canonical, "error": "Not enough return history for simulation"}

    last_price = closes[-1]
    baseline_date = df.index[-1].strftime("%Y-%m-%d")

    overlay = _edge_overlay(canonical)
    drift = overlay["drift_pct"]

    rng = random.Random(seed)
    terminals = []
    up_0 = up_2 = up_5 = down_2 = down_5 = 0
    for _ in range(n_paths):
        p = last_price
        for _step in range(horizon_days):
            r = rng.choice(rets) + drift
            p *= (1 + r)
        terminals.append(p)
        chg = p / last_price - 1
        if chg > 0: up_0 += 1
        if chg >= 0.02: up_2 += 1
        if chg >= 0.05: up_5 += 1
        if chg <= -0.02: down_2 += 1
        if chg <= -0.05: down_5 += 1

    terminals.sort()
    p10 = _percentile(terminals, 0.10)
    p50 = _percentile(terminals, 0.50)
    p90 = _percentile(terminals, 0.90)
    mean_terminal = sum(terminals) / len(terminals)

    # Headline point forecast = MEDIAN terminal: compounded bootstrap paths
    # are right-skewed, so the mean overstates names with hot trailing
    # windows. The mean stays available as expected_terminal_price.
    expected_return_pct = (p50 / last_price - 1) * 100
    # Empirical stdev of terminal returns
    rets_terminal = [(t / last_price - 1) for t in terminals]
    mean_r = sum(rets_terminal) / len(rets_terminal)
    var_r = sum((r - mean_r) ** 2 for r in rets_terminal) / len(rets_terminal)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.0001

    prob_up = up_0 / n_paths
    prob_up_2 = up_2 / n_paths
    prob_up_5 = up_5 / n_paths
    prob_down_2 = down_2 / n_paths
    prob_down_5 = down_5 / n_paths

    # Imminent move score: reward expected return × P(up >2%), penalize vol
    imminent_score = 100 * (mean_r * prob_up_2) / (std_r + 0.001)

    return {
        "ticker": canonical,
        "name": name,
        "yahoo_symbol": yahoo,
        "sector": EGX_UNIVERSE.get(canonical, {}).get("sector"),
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "lookback_days_used": len(rets),
        "baseline_date": baseline_date,
        "current_price": round(last_price, 4),
        "expected_terminal_price": round(mean_terminal, 4),
        "expected_return_pct": round(expected_return_pct, 2),
        "p10_price": round(p10, 4),
        "p50_price": round(p50, 4),
        "p90_price": round(p90, 4),
        "p10_return_pct": round((p10 / last_price - 1) * 100, 2),
        "p90_return_pct": round((p90 / last_price - 1) * 100, 2),
        "prob_up": round(prob_up, 3),
        "prob_up_2pct": round(prob_up_2, 3),
        "prob_up_5pct": round(prob_up_5, 3),
        "prob_down_2pct": round(prob_down_2, 3),
        "prob_down_5pct": round(prob_down_5, 3),
        "edge_drift_pct_per_day": drift * 100,
        "edge_drivers": overlay["drivers"],
        "imminent_move_score": round(imminent_score, 2),
        "method": (
            f"Bootstrap MC: {n_paths} paths × {horizon_days} days, sampling "
            f"with replacement from last {len(rets)} daily returns + technical edge overlay."
        ),
    }


def scan_universe(
    horizon_days: int = 5,
    n_paths: int = 2000,
    lookback_days: int = 60,
    min_prob_up_2pct: float = 0.0,
    min_expected_return_pct: float = 0.0,
    seed: int | None = 42,
    full_market: bool = False,
) -> dict[str, Any]:
    """Run simulate_one across the universe and rank.

    Args:
        full_market: If True, scans the validated extended EGX universe
            (~70 names — every EGX 100 component yfinance returns data
            for). If False (default), scans only the 29 hand-curated
            names where we have full sector taxonomy.

    Returns a sorted list with the strongest short-term upside candidates
    first, plus a top-5 spotlight for quick consumption.
    """
    if full_market:
        tickers = egx_listing.get_full_universe()
        if not tickers:
            tickers = [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]
        candidates = [(t, EGX_UNIVERSE.get(t, {}).get("sector", "Unknown")) for t in tickers]
    else:
        candidates = [(t, m["sector"]) for t, m in EGX_UNIVERSE.items()
                      if m["sector"] != "Index"]

    rows = []
    skipped = []
    for ticker, sector in candidates:
        if sector == "Index":
            continue
        try:
            r = simulate_one(
                ticker,
                horizon_days=horizon_days,
                n_paths=n_paths,
                lookback_days=lookback_days,
                seed=seed,
            )
        except Exception as e:
            skipped.append({"ticker": ticker, "error": str(e)})
            continue
        if "error" in r:
            skipped.append({"ticker": ticker, "error": r["error"]})
            continue
        if r["prob_up_2pct"] < min_prob_up_2pct:
            continue
        if r["expected_return_pct"] < min_expected_return_pct:
            continue
        rows.append(r)

    # Rank by imminent move score (combines expected return, hit-rate, vol)
    rows.sort(key=lambda r: r["imminent_move_score"], reverse=True)

    spotlight = []
    for r in rows[:5]:
        spotlight.append({
            "ticker": r["ticker"],
            "sector": r["sector"],
            "current_price": r["current_price"],
            "expected_return_pct": r["expected_return_pct"],
            "prob_up_2pct": r["prob_up_2pct"],
            "prob_up_5pct": r["prob_up_5pct"],
            "p90_return_pct": r["p90_return_pct"],
            "imminent_move_score": r["imminent_move_score"],
            "top_drivers": r["edge_drivers"][:3],
        })

    return {
        "horizon_days": horizon_days,
        "n_paths_per_ticker": n_paths,
        "universe_mode": "full_market" if full_market else "curated",
        "universe_scanned": len(candidates),
        "ranked_count": len(rows),
        "skipped_count": len(skipped),
        "top_5": spotlight,
        "ranked": rows,
        "skipped": skipped,
        "method": (
            "Each ticker simulated by bootstrap MC over the last "
            f"{lookback_days} daily returns, with a technical edge "
            "overlay (RSI / trend / MACD / Bollinger) shifting the daily "
            "drift in [-0.6%, +0.6%]. Ranked by "
            "expected_return × P(up>2%) / std(terminal_return)."
        ),
    }
