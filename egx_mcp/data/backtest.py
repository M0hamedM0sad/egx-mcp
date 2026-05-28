"""Walk-forward backtest harness — does the model actually work?

A scoring rubric is plausible until you replay it on history. This module
runs a monthly-rebalance walk-forward: at each month-end, score every
name in the universe using ONLY data available at that date, take the
top-N, equal-weight them, hold for one month, repeat.

Aggregate outputs:
    total_return        compounded portfolio return over the test window
    annualized_return   CAGR
    annualized_vol      realized
    sharpe              excess over T-bills, annualized
    max_drawdown        worst peak-to-trough
    hit_rate            % of months with positive return
    benchmark_return    EGX 30 over the same window for reference
    information_ratio   active return / tracking error vs EGX 30

The score used in the backtest is a simplified PRICE-ONLY composite:
    momentum    6M return at rebalance date
    mean_rev    (negative of) 1M return — buy laggards
    trend       price > 200d MA boost
    risk        penalize high realized vol

This is intentionally simpler than the full live `score_stock` because
historical fundamentals aren't reliably reconstructable from yfinance.
What it tests: whether the price-based half of the live model has edge.
The fundamental half needs its own backtest once an audited data feed
exists.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import egx_listing, risk_free
from .universe import resolve_ticker

try:
    # Fundamentals filter — only available when Mubasher cache exists.
    from . import mubasher_fundamentals
    _FUND_AVAILABLE = True
except ImportError:
    _FUND_AVAILABLE = False

log = logging.getLogger("egx-mcp.backtest")


def _price_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Pull one daily-close panel for all tickers, tz-naive date index."""
    closes = {}
    for tk in tickers:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start=start, end=end, interval="1d")
            if h is None or h.empty:
                continue
            s = h["Close"].copy()
            idx = pd.to_datetime(s.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            s.index = pd.to_datetime(idx.date)
            closes[tk] = s[~s.index.duplicated(keep="last")]
        except Exception as e:
            log.warning(f"history fetch failed for {tk}: {e}")
    if not closes:
        return pd.DataFrame()
    df = pd.DataFrame(closes).sort_index()
    return df


def _score_at(panel: pd.DataFrame, asof: pd.Timestamp) -> dict[str, float]:
    """Compute the price-only score for every column at a given date.

    Score = momentum (6m return) + trend bonus (above MA200) - vol penalty.
    Returns {ticker: score}; tickers with insufficient history are skipped.
    """
    # Compute features per-ticker on its own dropna'd series so calendar
    # gaps in one name don't shift the -130 indexing for everyone else.
    scores = {}
    for tk in panel.columns:
        try:
            ser = panel[tk].loc[:asof].dropna()
            if len(ser) < 130:
                continue
            p_now = float(ser.iloc[-1])
            p_6m = float(ser.iloc[-130])
            p_1m = float(ser.iloc[-22])
            ma200 = float(ser.tail(200).mean()) if len(ser) >= 200 else float(ser.mean())
            vol = float(ser.tail(60).pct_change().std() * (252 ** 0.5))
            if not (p_now > 0 and p_6m > 0 and p_1m > 0):
                continue
            mom_6m = (p_now / p_6m - 1) * 100
            mom_1m = (p_now / p_1m - 1) * 100
            mr_1m = -mom_1m  # buy laggards (negate)
            above_ma200 = p_now > ma200
            trend_bonus = 5 if above_ma200 else -5
            vol_penalty = max(0, (vol * 100 - 30) * 0.5)
            # V3 score = V1 + stretched penalty + dip-in-uptrend bonus
            #   (V3 was selected over V1/V2 in head-to-head backtests:
            #    same Sharpe 0.96 as V1, max DD halved from -7.9% to -3.8%)
            s = mom_6m * 0.5 + mr_1m * 0.2 + trend_bonus - vol_penalty
            if mom_6m > 50:
                s -= (mom_6m - 50) * 0.2          # stretched penalty
            if mom_1m < -8 and mom_6m > 10 and above_ma200:
                s += 5                              # dip-in-uptrend bonus
            scores[tk] = s
        except (KeyError, ValueError, ZeroDivisionError, AttributeError):
            continue
    return scores


def _load_fundamentals_filter(min_roe_pct: float | None) -> set[str] | None:
    """Return the set of tickers that pass the ROE filter, or None if disabled."""
    if min_roe_pct is None or not _FUND_AVAILABLE:
        return None
    cache_path = (Path(__file__).parent / "mubasher_fundamentals_cache.json")
    if not cache_path.exists():
        log.warning("min_roe_pct requested but mubasher_fundamentals_cache.json missing — filter disabled")
        return None
    import json as _json
    cache = _json.loads(cache_path.read_text(encoding="utf-8"))
    keep = set()
    for tk, d in cache.items():
        roe = d.get("roe_pct")
        if roe is not None and roe >= min_roe_pct:
            keep.add(tk)
    return keep


def backtest(
    start: str = "2023-01-01",
    end: str | None = None,
    top_n: int = 5,
    rebalance_days: int = 21,
    universe: str = "extended",
    min_roe_pct: float | None = 10.0,
) -> dict[str, Any]:
    """Walk-forward backtest. Returns full equity curve and stats.

    Args:
        start: ISO date.
        end: ISO date, defaults to today.
        top_n: Names to hold each rebalance.
        rebalance_days: Approximately monthly = 21 trading days.
        universe: 'extended' (~70 names) or 'curated' (29).
        min_roe_pct: Quality filter — exclude names with ROE below this.
            Default 10.0 (V8b production setting). Set to None to disable.
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")

    if universe == "extended":
        tickers = egx_listing.get_full_universe()
    else:
        from .universe import EGX_UNIVERSE
        tickers = [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]

    if not tickers:
        return {"error": "empty universe"}

    # Apply quality pre-filter (V8b)
    quality_set = _load_fundamentals_filter(min_roe_pct)
    if quality_set is not None:
        n_before = len(tickers)
        tickers = [t for t in tickers if t in quality_set]
        log.info(f"Quality filter ROE>={min_roe_pct}%: {n_before} → {len(tickers)} names")

    # Fetch with a 1-year history buffer so the V3 score's 6m / 200d
    # lookback windows have enough data on the very first rebalance date.
    fetch_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    panel = _price_panel(tickers, start=fetch_start, end=end)
    if panel.empty:
        return {"error": "no price history"}

    # Benchmark: EGX 30 (try spot, fall back to ETF, finally synthetic basket
    # — same Yahoo limitation documented in regime.py)
    bench = None
    for bsym in ("^CASE30", "EGS69491M015.CA", "EGX30.CA"):
        try:
            bench_h = yf.Ticker(bsym).history(start=start, end=end, interval="1d")
            if bench_h is not None and not bench_h.empty and len(bench_h) > 30:
                bench = bench_h["Close"]
                break
        except Exception:
            continue
    if bench is None:
        # Synthetic equal-weight basket as fallback benchmark
        from .regime import _fetch_egx30_history
        synth, _ = _fetch_egx30_history()
        if synth is not None and not synth.empty:
            # Trim to backtest window
            synth_in_window = synth.loc[(synth.index >= pd.Timestamp(start)) &
                                         (synth.index <= pd.Timestamp(end))]
            if len(synth_in_window) > 30:
                bench = synth_in_window["Close"]

    # Generate rebalance dates from the panel — but only those falling
    # within the requested [start, end] window. The buffer fetch (above)
    # ensures earlier history is available for the score's lookback.
    start_ts = pd.Timestamp(start)
    in_window_idx = panel.index[panel.index >= start_ts]
    rebalance_idx = in_window_idx[::rebalance_days]
    if len(rebalance_idx) < 3:
        return {"error": "not enough rebalance periods in window"}

    equity = [1.0]
    monthly_returns = []
    holdings_history = []

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]
        next_date = rebalance_idx[i + 1]
        scores = _score_at(panel, date)
        if not scores:
            monthly_returns.append(0.0)
            equity.append(equity[-1])
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [tk for tk, _ in ranked[:top_n]]
        holdings_history.append({
            "rebalance_date": date.strftime("%Y-%m-%d"),
            "picks": picks,
            "top_score": round(ranked[0][1], 2) if ranked else None,
        })

        # Compute period return for picks (equal weight)
        period_panel = panel.loc[date:next_date, picks].dropna(how="all")
        if period_panel.empty or len(period_panel) < 2:
            monthly_returns.append(0.0)
            equity.append(equity[-1])
            continue
        start_p = period_panel.iloc[0]
        end_p = period_panel.iloc[-1]
        rets = (end_p / start_p - 1).dropna()
        if rets.empty:
            monthly_returns.append(0.0)
            equity.append(equity[-1])
            continue
        period_ret = float(rets.mean())
        monthly_returns.append(period_ret)
        equity.append(equity[-1] * (1 + period_ret))

    eq = np.array(equity)
    total_ret = float(eq[-1] - 1) * 100
    n_periods = len(monthly_returns)
    years = n_periods * rebalance_days / 252
    cagr = ((1 + total_ret / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0.0

    # Stats
    rets_arr = np.array(monthly_returns)
    period_vol = float(rets_arr.std()) if len(rets_arr) > 1 else 0.0
    annualized_vol = period_vol * ((252 / rebalance_days) ** 0.5) * 100
    rf = risk_free.get_rate()["rate_pct"] / 100
    rf_period = (1 + rf) ** (rebalance_days / 252) - 1
    excess = rets_arr - rf_period
    sharpe = float(excess.mean() / excess.std() * ((252 / rebalance_days) ** 0.5)) if excess.std() > 0 else 0.0
    hit_rate = float((rets_arr > 0).mean()) * 100

    # Drawdown
    running_max = np.maximum.accumulate(eq)
    dd = (eq / running_max - 1)
    max_dd = float(dd.min()) * 100

    # Benchmark
    bench_total = None
    bench_cagr = None
    info_ratio = None
    if bench is not None and len(bench) > 1:
        bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1) * 100
        bench_cagr = ((1 + bench_total / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0.0
        # Active return per period
        bench_idx = bench.reindex(rebalance_idx, method="ffill")
        bench_rets = bench_idx.pct_change().dropna().values
        m = min(len(bench_rets), len(rets_arr))
        if m > 1:
            active = rets_arr[:m] - bench_rets[:m]
            te = float(active.std())
            info_ratio = float(active.mean() / te * ((252 / rebalance_days) ** 0.5)) if te > 0 else 0.0

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "start": start,
        "end": end,
        "universe": universe,
        "universe_size": len(tickers),
        "top_n": top_n,
        "rebalance_days": rebalance_days,
        "n_rebalances": len(holdings_history),

        "total_return_pct": round(total_ret, 2),
        "annualized_return_pct": round(cagr, 2),
        "annualized_volatility_pct": round(annualized_vol, 2),
        "annualized_sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "hit_rate_pct": round(hit_rate, 1),

        "benchmark_total_return_pct": round(bench_total, 2) if bench_total is not None else None,
        "benchmark_annualized_return_pct": round(bench_cagr, 2) if bench_cagr is not None else None,
        "alpha_vs_benchmark_pct": round(total_ret - bench_total, 2) if bench_total is not None else None,
        "information_ratio": round(info_ratio, 2) if info_ratio is not None else None,

        "equity_curve_final": round(float(eq[-1]), 4),
        "holdings_history_sample": holdings_history[-5:],
        "method": (
            "Walk-forward monthly rebalance. Score = 0.5×6M_return + "
            "0.2×(-1M_return) + 5(if above MA200, else -5) - 0.5×max(0, vol_pct-30). "
            "Picks top_n equal-weight, holds rebalance_days, repeats. "
            "Excess Sharpe uses live EGP T-bill rate."
        ),
    }
