"""Portfolio risk module — VaR, CVaR, drawdown, circuit breaker.

Historical-simulation VaR/CVaR (no parametric assumptions, no scipy).
For a portfolio of tickers and weights, build the daily portfolio
return series from the joint history and compute:

    VaR_q       q-quantile of the loss distribution
    CVaR_q      mean loss in the q-tail (Expected Shortfall)
    max_dd      worst peak-to-trough on the equity curve
    sharpe_ann  excess return over T-bills, annualized

Plus a circuit-breaker rule: if rolling 20-day drawdown > threshold,
recommend cutting gross exposure.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import risk_free
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.risk")


def _portfolio_returns(tickers: list[str], weights: list[float], lookback_days: int) -> pd.Series:
    """Build daily portfolio return series from constituent histories."""
    rets = {}
    for tk in tickers:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(period=f"{lookback_days * 2}d", interval="1d")
            if h is None or h.empty:
                continue
            rets[tk] = h["Close"].pct_change()
        except Exception as e:
            log.warning(f"history fetch failed for {tk}: {e}")
    if not rets:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rets).dropna()
    df = df.tail(lookback_days)
    # Align weights to columns actually present
    w_map = dict(zip(tickers, weights))
    cols = list(df.columns)
    w_arr = np.array([w_map[c] for c in cols], dtype=float)
    if w_arr.sum() == 0:
        return pd.Series(dtype=float)
    w_arr = w_arr / w_arr.sum()
    port = (df.values @ w_arr)
    return pd.Series(port, index=df.index, name="portfolio")


def portfolio_risk(
    tickers: list[str],
    weights: list[float] | None = None,
    lookback_days: int = 252,
    confidence: float = 0.95,
    horizon_days: int = 1,
    nav_egp: float | None = None,
) -> dict[str, Any]:
    """Compute VaR / CVaR / drawdown / Sharpe for a portfolio."""
    n = len(tickers)
    if n == 0:
        return {"error": "no tickers"}
    if weights is None:
        weights = [1 / n] * n
    if len(weights) != n:
        return {"error": "weights length mismatch"}

    port = _portfolio_returns(tickers, weights, lookback_days)
    if len(port) < 30:
        return {"error": f"insufficient overlapping history ({len(port)} bars)"}

    # Scale daily returns to horizon by sqrt(t) — a standard VaR convention
    if horizon_days > 1:
        scaled = port * (horizon_days ** 0.5)
    else:
        scaled = port

    losses = -scaled.values
    losses_sorted = np.sort(losses)
    var_idx = int(np.ceil(confidence * len(losses_sorted))) - 1
    var = float(losses_sorted[var_idx])
    cvar = float(losses_sorted[var_idx:].mean())

    # Drawdown on the equity curve
    equity = (1 + port).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1)
    max_dd = float(drawdown.min())
    current_dd = float(drawdown.iloc[-1])

    # Rolling 20-day drawdown (early-warning)
    roll_dd = drawdown.tail(20).min()

    # Sharpe (excess over T-bills)
    rf = risk_free.get_rate()
    daily_rf = rf["daily_rate_pct"] / 100
    excess = port - daily_rf
    sharpe = float(excess.mean() / excess.std() * (252 ** 0.5)) if excess.std() > 0 else 0.0

    # Annualized vol
    vol_ann = float(port.std() * (252 ** 0.5))

    # Circuit breaker
    breaker_threshold = -0.08
    breaker_active = roll_dd <= breaker_threshold
    breaker_msg = (
        f"CIRCUIT BREAKER: rolling 20d DD = {roll_dd:.2%} ≤ {breaker_threshold:.0%}. "
        "Cut gross exposure by 50% until drawdown contracts."
        if breaker_active else
        f"OK: rolling 20d DD = {roll_dd:.2%} above {breaker_threshold:.0%} threshold."
    )

    payload = {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "weights": [round(w / sum(weights), 4) for w in weights],
        "lookback_days_used": int(len(port)),
        "horizon_days": horizon_days,
        "confidence": confidence,

        f"VaR_{int(confidence * 100)}_pct": round(var * 100, 3),
        f"CVaR_{int(confidence * 100)}_pct": round(cvar * 100, 3),
        "annualized_volatility_pct": round(vol_ann * 100, 2),
        "annualized_sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "current_drawdown_pct": round(current_dd * 100, 2),
        "rolling_20d_drawdown_pct": round(float(roll_dd) * 100, 2),

        "circuit_breaker_threshold_pct": breaker_threshold * 100,
        "circuit_breaker_active": breaker_active,
        "circuit_breaker_message": breaker_msg,

        "risk_free_rate_pct": rf["rate_pct"],
        "risk_free_source": rf["source"],
    }

    if nav_egp:
        payload["VaR_egp"] = round(var * nav_egp, 0)
        payload["CVaR_egp"] = round(cvar * nav_egp, 0)
        payload["max_dd_egp"] = round(max_dd * nav_egp, 0)

    return payload
