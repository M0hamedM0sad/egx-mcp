"""Market regime classifier — bull / bear / high-vol / sideways.

A simple, transparent rule-based classifier on EGX 30 daily returns.
The same scoring weights don't work in every regime: momentum dominates
in bulls, mean-reversion in sideways, defensives in high-vol. This
module returns the current regime + suggested weight overrides for
the scoring engine.

Inputs (computed live from EGX 30):
    - 60-day cumulative return
    - 60-day annualized volatility
    - 200-day trend slope (price > MA200?)
    - Drawdown from 1-year high

Regimes:
    BULL          ret_60d > +5% AND vol < 30% AND price > MA200
    HIGH_VOL      vol >= 40%
    BEAR          ret_60d < -5% OR drawdown_1y < -15%
    SIDEWAYS      everything else

Each regime ships a `weight_override` dict the scoring engine can
multiply into the base 30/25/25/20 weights.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import yfinance as yf

log = logging.getLogger("egx-mcp.regime")


# Scoring weight bias by regime — multiplied with base weights then renormalized
_REGIME_WEIGHTS = {
    "BULL":     {"valuation": 0.8, "quality": 0.9, "momentum": 1.4, "risk": 0.9},
    "BEAR":     {"valuation": 1.3, "quality": 1.4, "momentum": 0.7, "risk": 1.1},
    "HIGH_VOL": {"valuation": 1.0, "quality": 1.3, "momentum": 0.7, "risk": 1.5},
    "SIDEWAYS": {"valuation": 1.2, "quality": 1.1, "momentum": 0.9, "risk": 1.0},
}

_REGIME_DESCRIPTIONS = {
    "BULL": "Trending higher with manageable vol — momentum names lead.",
    "BEAR": "Index falling or in 15%+ drawdown — quality and value defensible, fade momentum.",
    "HIGH_VOL": "Realized vol >40% — risk and quality dominate; cut gross exposure.",
    "SIDEWAYS": "Range-bound — slight bias to mean-reversion and value.",
}


def _fetch_egx30_history() -> Any:
    """Try the EGX 30 spot, fall back to ETF, then build a synthetic
    equal-weighted basket from the curated EGX universe.

    Yahoo returns only one bar for `^CASE30` and `EGS69491M015.CA` —
    documented limitation. The synthetic basket is the only reliable
    way to get a daily EGX-market series.
    """
    import pandas as pd
    from .universe import EGX_UNIVERSE

    for sym in ("^CASE30", "EGS69491M015.CA", "EGX30.CA"):
        try:
            df = yf.Ticker(sym).history(period="1y", interval="1d")
            if df is not None and not df.empty and len(df) >= 60:
                return df, sym
        except Exception as e:
            log.warning(f"history fetch failed for {sym}: {e}")

    # Synthetic: equal-weight basket of the most-liquid curated names
    proxy_set = ["COMI", "HDBK", "CIRA", "SWDY", "ETEL", "ABUK", "EFID",
                 "TMGH", "ORWE", "FWRY", "EAST", "MFPC", "EIPI"]
    closes = {}
    for tk in proxy_set:
        meta = EGX_UNIVERSE.get(tk)
        if not meta:
            continue
        try:
            h = yf.Ticker(meta["yahoo"]).history(period="1y", interval="1d")
            if h is None or h.empty or len(h) < 60:
                continue
            s = h["Close"].copy()
            # Normalize to tz-naive date index so downstream comparisons work
            idx = pd.to_datetime(s.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            s.index = pd.to_datetime(idx.date)
            closes[tk] = s[~s.index.duplicated(keep="last")]
        except Exception:
            continue
    if not closes:
        return None, None
    panel = pd.DataFrame(closes).sort_index().dropna(how="all")
    # Build a basket level (rebased to 100) from equal-weighted daily returns
    rets = panel.pct_change().mean(axis=1).fillna(0)
    level = (1 + rets).cumprod() * 100
    synthetic = pd.DataFrame({"Close": level})
    return synthetic, f"synthetic_basket({len(closes)}_names)"


def classify(index_symbol: str | None = None) -> dict[str, Any]:
    """Classify the current EGX regime from EGX 30 daily history."""
    df, used_symbol = _fetch_egx30_history()
    if df is None:
        return {"regime": "UNKNOWN", "error": "no EGX 30 source returned ≥60 bars"}
    try:
        closes = df["Close"].dropna()
    except Exception as e:
        return {"regime": "UNKNOWN", "error": str(e)}

    last_price = float(closes.iloc[-1])
    last_60 = closes.iloc[-60:] if len(closes) >= 60 else closes
    last_252 = closes

    # 60-day cumulative return
    ret_60 = (float(last_60.iloc[-1]) / float(last_60.iloc[0]) - 1) * 100 if len(last_60) > 1 else 0.0

    # Annualized vol from daily returns
    daily_rets = last_60.pct_change().dropna()
    vol_ann = float(daily_rets.std() * (252 ** 0.5) * 100) if len(daily_rets) > 1 else 0.0

    # MA200 trend
    sma_200 = float(last_252.tail(200).mean()) if len(last_252) >= 200 else float(last_252.mean())
    above_ma200 = last_price > sma_200

    # 1-year drawdown
    high_1y = float(last_252.max())
    drawdown_1y = (last_price / high_1y - 1) * 100

    # Decision tree
    if vol_ann >= 40:
        regime = "HIGH_VOL"
    elif ret_60 < -5 or drawdown_1y < -15:
        regime = "BEAR"
    elif ret_60 > 5 and vol_ann < 30 and above_ma200:
        regime = "BULL"
    else:
        regime = "SIDEWAYS"

    return {
        "regime": regime,
        "description": _REGIME_DESCRIPTIONS[regime],
        "weight_override": _REGIME_WEIGHTS[regime],
        "metrics": {
            "egx30_last": round(last_price, 2),
            "ret_60d_pct": round(ret_60, 2),
            "annualized_vol_pct": round(vol_ann, 2),
            "sma_200": round(sma_200, 2),
            "above_ma200": above_ma200,
            "drawdown_1y_pct": round(drawdown_1y, 2),
        },
        "source_symbol": used_symbol,
        "as_of": datetime.utcnow().isoformat() + "Z",
    }
