"""Persistent price + driver cache, sourced from investing.com.

Makes the model "know" prices and all price-derived drivers offline. A
single `refresh()` pulls daily OHLCV for the whole universe plus the five
macro factors, computes each name's driver profile (factor betas, R²,
volatility, drawdown, momentum), and writes it all to disk. Every other
tool can then read from the cache without touching the network — so the
behavior/factor tools keep working even when the live feed is down.

Cache file: egx_mcp/data/price_cache.json
  {
    refreshed_at, lookback_days, date_range,
    prices:  {TICKER: [ {date,open,high,low,close,volume,change_pct}, ... ]},
    factors: {egx30: [closes], egp: [...], ...} keyed by date list,
    drivers: {TICKER: {market_beta, factor_betas, r_squared, ...}},
  }
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import investing
from .factors import _FACTORS, _ols

log = logging.getLogger("egx-mcp.price_cache")

_CACHE_PATH = Path(__file__).parent / "price_cache.json"

# Factor key -> the resolver symbol investing.resolve_pair_id understands.
_FACTOR_KEYS = ["egx30", "egp", "brent", "gold", "em"]


def _closes_by_date(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {r["date"]: r["close"] for r in rows}


def _ffill_on(target_dates: list[str], series: dict[str, float]) -> list[float | None]:
    """For each target date, the series close on-or-before it (forward fill)."""
    s_dates = sorted(series)
    out: list[float | None] = []
    j = 0
    last = None
    for d in target_dates:
        while j < len(s_dates) and s_dates[j] <= d:
            last = series[s_dates[j]]
            j += 1
        out.append(last)
    return out


def _returns(closes: list[float | None]) -> np.ndarray:
    arr = np.array([c if c is not None else np.nan for c in closes], dtype=float)
    rets = arr[1:] / arr[:-1] - 1
    return rets


def _risk_from_closes(closes: list[float]) -> dict[str, Any]:
    c = np.array(closes, dtype=float)
    if len(c) < 2:
        return {}
    rets = c[1:] / c[:-1] - 1
    running_max = np.maximum.accumulate(c)
    drawdown = (c / running_max - 1) * 100
    return {
        "trailing_return_pct": round((c[-1] / c[0] - 1) * 100, 2),
        "annualized_volatility_pct": round(float(np.std(rets)) * (252 ** 0.5) * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()), 2),
        "momentum_20d_pct": round((c[-1] / c[-21] - 1) * 100, 2) if len(c) > 21 else None,
        "momentum_60d_pct": round((c[-1] / c[-61] - 1) * 100, 2) if len(c) > 61 else None,
    }


def _compute_drivers(
    stock_rows: list[dict[str, Any]],
    factor_series: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Regress the stock's returns on the macro factors using cached closes."""
    if len(stock_rows) < 30:
        return {"error": "insufficient history"}

    dates = [r["date"] for r in stock_rows]
    stock_closes = [r["close"] for r in stock_rows]
    y = _returns(stock_closes)

    factor_cols: list[str] = []
    factor_ret_cols: list[np.ndarray] = []
    for key in _FACTOR_KEYS:
        series = factor_series.get(key)
        if not series:
            continue
        ffilled = _ffill_on(dates, series)
        fr = _returns(ffilled)
        if np.isnan(fr).all():
            continue
        factor_cols.append(key)
        factor_ret_cols.append(fr)

    if not factor_cols:
        return {"error": "no factor data"}

    X = np.column_stack(factor_ret_cols)
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y_clean, X_clean = y[mask], X[mask]
    if len(y_clean) < 30:
        return {"error": "insufficient overlapping returns"}

    beta, r2 = _ols(y_clean, X_clean)
    factor_betas = {col: round(float(beta[i + 1]), 4) for i, col in enumerate(factor_cols)}

    out = {
        "market_beta": factor_betas.get("egx30"),
        "factor_betas": factor_betas,
        "r_squared": round(r2, 4),
        "idiosyncratic_pct": round((1 - r2) * 100, 1),
        "alpha_daily_pct": round(float(beta[0]) * 100, 4),
        "n_obs": int(len(y_clean)),
    }
    out.update(_risk_from_closes(stock_closes))
    return out


def refresh(universe: str = "extended", lookback_days: int = 400, throttle_s: float = 0.4) -> dict[str, Any]:
    """Pull prices for the universe + factors, compute drivers, write cache."""
    if universe == "curated":
        from .universe import EGX_UNIVERSE
        symbols = [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]
    else:
        from .egx_listing import get_full_universe
        symbols = get_full_universe()

    # Factors first
    factor_series: dict[str, dict[str, float]] = {}
    factor_prices: dict[str, list[dict[str, Any]]] = {}
    for key in _FACTOR_KEYS:
        rows = investing.fetch_history(key.upper(), lookback_days=lookback_days)
        if rows:
            factor_series[key] = _closes_by_date(rows)
            factor_prices[key] = rows
        time.sleep(throttle_s)

    prices: dict[str, list[dict[str, Any]]] = {}
    drivers: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for sym in symbols:
        rows = investing.fetch_history(sym, lookback_days=lookback_days)
        if not rows:
            failed.append(sym)
            time.sleep(throttle_s)
            continue
        prices[sym] = rows
        drivers[sym] = _compute_drivers(rows, factor_series)
        time.sleep(throttle_s)

    all_dates = [d for rows in prices.values() for d in (rows[0]["date"], rows[-1]["date"])]
    payload = {
        "refreshed_at": datetime.utcnow().isoformat() + "Z",
        "lookback_days": lookback_days,
        "universe": universe,
        "date_range": {"start": min(all_dates), "end": max(all_dates)} if all_dates else None,
        "n_tickers": len(prices),
        "n_failed": len(failed),
        "failed": failed,
        "factor_pairids": {k: investing.resolve_pair_id(k.upper()) for k in _FACTOR_KEYS},
        "factors": factor_prices,
        "prices": prices,
        "drivers": drivers,
    }
    _CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "status": "ok",
        "cache_path": str(_CACHE_PATH),
        "refreshed_at": payload["refreshed_at"],
        "date_range": payload["date_range"],
        "n_tickers": len(prices),
        "n_failed": len(failed),
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Read API — used by behavior/factor tools for offline access
# ---------------------------------------------------------------------------

_MEM: dict[str, Any] | None = None
_MEM_MTIME: float = 0.0


def _load() -> dict[str, Any] | None:
    global _MEM, _MEM_MTIME
    if not _CACHE_PATH.exists():
        return None
    mtime = _CACHE_PATH.stat().st_mtime
    if _MEM is None or mtime != _MEM_MTIME:
        try:
            _MEM = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            _MEM_MTIME = mtime
        except Exception as e:
            log.warning(f"cache load failed: {e}")
            return None
    return _MEM


def is_available() -> bool:
    return _load() is not None


def meta() -> dict[str, Any]:
    c = _load()
    if not c:
        return {"available": False}
    return {
        "available": True,
        "refreshed_at": c.get("refreshed_at"),
        "date_range": c.get("date_range"),
        "n_tickers": c.get("n_tickers"),
        "universe": c.get("universe"),
    }


def get_prices(ticker: str) -> list[dict[str, Any]]:
    c = _load() or {}
    return c.get("prices", {}).get(ticker.upper(), [])


def get_drivers(ticker: str) -> dict[str, Any] | None:
    c = _load() or {}
    return c.get("drivers", {}).get(ticker.upper())


def get_quote(ticker: str) -> dict[str, Any]:
    rows = get_prices(ticker)
    if not rows:
        return {"ticker": ticker.upper(), "error": "not in cache"}
    last = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    return {
        "ticker": ticker.upper(),
        "price": last["close"],
        "previous_close": prev["close"] if prev else None,
        "change_pct": last.get("change_pct"),
        "date": last["date"],
        "volume": last["volume"],
        "source": "price_cache (investing.com)",
    }


def cached_tickers() -> list[str]:
    c = _load() or {}
    return sorted(c.get("prices", {}).keys())
