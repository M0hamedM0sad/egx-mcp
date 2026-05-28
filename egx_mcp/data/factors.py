"""Factor exposure decomposition — what is the portfolio actually betting on?

Regresses each ticker's daily returns on a small set of EGX-relevant
macro factors and reports the betas. Aggregating with portfolio weights
gives the **net** factor exposure of the book — the only honest answer
to "am I unintentionally long oil / short EGP / long rates?"

Factor basket (all live from yfinance):
    EGX30   ^CASE30        market beta
    EGP     USDEGP=X       FX beta — positive means stock rises when EGP weakens
    BRENT   BZ=F           oil beta
    GOLD    GC=F           safe-haven beta
    EM      EEM            EM equity beta — captures regional risk-on/off

OLS via numpy normal equations — no scipy dependency, no overfitting
(5 factors against 60-90 daily observations is fine).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.factors")


_FACTORS = {
    # egx30 is built synthetically (Yahoo's ^CASE30 only returns 1 bar)
    "egx30":  "__synthetic_egx30__",
    "egp":    "USDEGP=X",
    "brent":  "BZ=F",
    "gold":   "GC=F",
    "em":     "EEM",
}


def _synthetic_egx30_close() -> pd.Series:
    """Build the same equal-weighted basket used by the regime module."""
    from .regime import _fetch_egx30_history
    df, _ = _fetch_egx30_history()
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df["Close"]


def _aligned_returns(symbols: dict[str, str], lookback_days: int) -> pd.DataFrame:
    """Pull daily returns; normalize every index to tz-naive *date*
    timestamps so that Cairo / London / NY trading calendars line up.

    Outer-join, forward-fill the non-equity factors, then drop rows
    where the equity (`y`) is still NaN.
    """
    series = []
    for name, sym in symbols.items():
        try:
            if sym == "__synthetic_egx30__":
                close = _synthetic_egx30_close()
                if close.empty:
                    continue
            else:
                h = yf.Ticker(sym).history(period=f"{lookback_days * 3}d", interval="1d")
                if h is None or h.empty:
                    continue
                close = h["Close"].copy()
            # Normalize to tz-naive midnight (date-only)
            idx = pd.to_datetime(close.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            idx = pd.to_datetime(idx.date)
            close.index = idx
            close = close[~close.index.duplicated(keep="last")]
            ret = close.pct_change().rename(name)
            series.append(ret)
        except Exception as e:
            log.warning(f"factor pull failed for {sym}: {e}")
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1, join="outer").sort_index()
    # Forward-fill non-equity factors so EGX trading-day rows have factor values
    if "y" in out.columns:
        non_y = [c for c in out.columns if c != "y"]
        out[non_y] = out[non_y].ffill()
        out = out.dropna(subset=["y"])
    out = out.dropna()
    return out.tail(lookback_days)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit y = X @ beta + e via normal equations. Returns (beta, r_squared)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def ticker_factor_exposure(user_ticker: str, lookback_days: int = 90) -> dict[str, Any]:
    """Single-ticker factor regression."""
    canonical, yahoo, _ = resolve_ticker(user_ticker)
    symbols = {"y": yahoo, **_FACTORS}
    rets = _aligned_returns(symbols, lookback_days)
    if rets.empty or len(rets) < 30 or "y" not in rets.columns:
        return {"ticker": canonical, "error": "insufficient overlapping return history"}

    y = rets["y"].values
    factor_cols = [c for c in _FACTORS.keys() if c in rets.columns]
    X = rets[factor_cols].values
    beta, r2 = _ols(y, X)
    intercept = float(beta[0])
    factor_betas = {col: round(float(beta[i + 1]), 4) for i, col in enumerate(factor_cols)}

    return {
        "ticker": canonical,
        "lookback_days": int(len(rets)),
        "alpha_daily_pct": round(intercept * 100, 4),
        "r_squared": round(r2, 4),
        "factor_betas": factor_betas,
        "interpretation": _interpret(factor_betas),
    }


def _interpret(b: dict[str, float]) -> list[str]:
    notes = []
    if "egx30" in b:
        if b["egx30"] >= 1.2:
            notes.append(f"High-beta name (EGX30 β={b['egx30']:.2f}) — amplifies market moves")
        elif b["egx30"] <= 0.5:
            notes.append(f"Low-beta name (EGX30 β={b['egx30']:.2f}) — defensive")
    if b.get("egp", 0) >= 0.3:
        notes.append(f"EGP-weakening winner (β={b['egp']:.2f}) — exporters / hard-currency revenue")
    elif b.get("egp", 0) <= -0.3:
        notes.append(f"EGP-strength winner (β={b['egp']:.2f}) — importer / EGP cost base")
    if b.get("brent", 0) >= 0.2:
        notes.append(f"Oil-leveraged (β={b['brent']:.2f})")
    elif b.get("brent", 0) <= -0.2:
        notes.append(f"Oil-hurt (β={b['brent']:.2f}) — fuel-cost sensitive")
    if b.get("gold", 0) >= 0.2:
        notes.append(f"Safe-haven correlated (β={b['gold']:.2f})")
    return notes


def portfolio_factor_exposure(
    tickers: list[str],
    weights: list[float] | None = None,
    lookback_days: int = 90,
) -> dict[str, Any]:
    """Aggregate factor exposures across a portfolio.

    Returns weighted-sum betas plus per-ticker breakdown so the PM can
    see which positions are driving each exposure.
    """
    n = len(tickers)
    if n == 0:
        return {"error": "no tickers"}
    if weights is None:
        weights = [1 / n] * n
    if len(weights) != n:
        return {"error": "weights length mismatch"}

    total = sum(weights)
    if total <= 0:
        return {"error": "weights sum to zero or negative"}
    weights = [w / total for w in weights]

    per_ticker = []
    agg = {f: 0.0 for f in _FACTORS}
    agg_alpha = 0.0
    agg_r2 = 0.0
    valid = 0
    for tk, w in zip(tickers, weights):
        exp = ticker_factor_exposure(tk, lookback_days=lookback_days)
        if "error" in exp:
            per_ticker.append({"ticker": tk, "weight": w, "error": exp["error"]})
            continue
        per_ticker.append({
            "ticker": exp["ticker"],
            "weight": round(w, 4),
            "betas": exp["factor_betas"],
            "alpha_daily_pct": exp["alpha_daily_pct"],
            "r_squared": exp["r_squared"],
        })
        for f, b in exp["factor_betas"].items():
            agg[f] += w * b
        agg_alpha += w * exp["alpha_daily_pct"]
        agg_r2 += w * exp["r_squared"]
        valid += 1

    portfolio_betas = {f: round(v, 4) for f, v in agg.items()}
    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "n_positions": n,
        "n_with_data": valid,
        "lookback_days": lookback_days,
        "portfolio_factor_betas": portfolio_betas,
        "portfolio_alpha_daily_pct": round(agg_alpha, 4),
        "weighted_avg_r_squared": round(agg_r2, 4),
        "interpretation": _interpret(portfolio_betas),
        "per_ticker": per_ticker,
    }
