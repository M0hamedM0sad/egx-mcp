"""Walk-forward backtest of the chairman's verdict bands.

The live `scoring.score_stock` and `debate.debate` chairman pull current
Yahoo fundamentals + current sentiment, so a true historical replay of
the full agentic stack would carry lookahead bias. This module isolates
the part that IS point-in-time clean: the V3 price-only score from
`backtest._score_at`, mapped each period into chairman-style verdict
bands (BUY / ACCUMULATE / HOLD / REDUCE / AVOID) by within-period
percentile rank.

Two outputs:

  1. Per-band grading. For every (ticker, date, verdict) we record the
     forward `rebalance_days` return. Aggregated by band: hit rate,
     mean return, count. Tells you whether the chairman's mapping has
     historical edge — does BUY actually beat AVOID by enough to matter?

  2. BUY-only portfolio simulation. At each rebalance, equal-weight the
     names whose verdict is BUY (or BUY+ACCUMULATE), hold for the
     period, repeat. Compared to EGX 30 over the same window.

This is intentionally narrower than the full V8b backtest — it grades
the *verdict mapping*, not the production strategy. Use both reads:
backtest.backtest() for the strategy, this for the agentic layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import backtest as bt_mod
from . import egx_listing, risk_free
from .universe import EGX_UNIVERSE, resolve_ticker

log = logging.getLogger("egx-mcp.agentic_backtest")


# Default percentile cutoffs per rebalance — chairman-style bands.
# Top 10% of names that period = BUY, next 15% = ACCUMULATE, etc.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "BUY":         (0.90, 1.00),
    "ACCUMULATE":  (0.75, 0.90),
    "HOLD":        (0.25, 0.75),
    "REDUCE":      (0.10, 0.25),
    "AVOID":       (0.00, 0.10),
}


def _band_for_pct(pct: float, bands: dict[str, tuple[float, float]]) -> str:
    for name, (lo, hi) in bands.items():
        # Top band is closed on the right
        if (lo <= pct < hi) or (hi == 1.0 and pct >= lo):
            return name
    return "HOLD"


def _benchmark_series(start: str, end: str) -> pd.Series | None:
    """Best-effort EGX 30 close series. Same fallback chain as backtest.py."""
    for sym in ("^CASE30", "EGS69491M015.CA", "EGX30.CA"):
        try:
            h = yf.Ticker(sym).history(start=start, end=end, interval="1d")
            if h is not None and not h.empty and len(h) > 30:
                idx = pd.to_datetime(h.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                s = h["Close"].copy()
                s.index = pd.to_datetime(idx.date)
                return s[~s.index.duplicated(keep="last")]
        except Exception:
            continue
    return None


def backtest_agentic(
    start: str = "2023-01-01",
    end: str | None = None,
    rebalance_days: int = 21,
    universe: str = "extended",
    bands: dict[str, tuple[float, float]] | None = None,
    buy_only_top_n: int | None = 5,
    include_accumulate: bool = True,
) -> dict[str, Any]:
    """Walk-forward grading of chairman verdict bands.

    Args:
        start: ISO date for the test window.
        end: ISO date, default today.
        rebalance_days: Period in trading days. Default 21 (~monthly).
        universe: 'extended' (~70) or 'curated' (29).
        bands: Override DEFAULT_BANDS (each entry is (lo_pct, hi_pct)).
        buy_only_top_n: Cap on BUY+ACCUMULATE portfolio size per period.
            None = uncapped (every BUY-band name held). Default 5.
        include_accumulate: Treat ACCUMULATE as buy-side too. Default True.

    Returns:
        Dict with: by_verdict (n, hit_rate, mean_return), buy_portfolio
        (return / sharpe / DD vs EGX 30), per-period verdict counts,
        sample picks, method.
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    bands = bands or DEFAULT_BANDS

    if universe == "extended":
        tickers = egx_listing.get_full_universe()
    else:
        tickers = [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]

    if not tickers:
        return {"error": "empty universe"}

    fetch_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    panel = bt_mod._price_panel(tickers, start=fetch_start, end=end)
    if panel.empty:
        return {"error": "no price history"}

    bench = _benchmark_series(start, end)

    start_ts = pd.Timestamp(start)
    in_window = panel.index[panel.index >= start_ts]
    rebalance_idx = in_window[::rebalance_days]
    if len(rebalance_idx) < 3:
        return {"error": "not enough rebalance periods in window"}

    # Per-band aggregators
    band_buckets: dict[str, list[float]] = {b: [] for b in bands}
    period_summaries: list[dict[str, Any]] = []
    buy_period_returns: list[float] = []
    buy_equity = [1.0]

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]
        next_date = rebalance_idx[i + 1]
        scores = bt_mod._score_at(panel, date)
        if not scores:
            buy_period_returns.append(0.0)
            buy_equity.append(buy_equity[-1])
            continue

        # Rank to percentile within the period
        items = sorted(scores.items(), key=lambda x: x[1])  # ascending
        n = len(items)
        ranks = {tk: (i + 1) / n for i, (tk, _) in enumerate(items)}

        verdicts: dict[str, str] = {tk: _band_for_pct(p, bands) for tk, p in ranks.items()}

        # Forward returns for every scored name
        period_panel = panel.loc[date:next_date].dropna(how="all")
        if period_panel.empty or len(period_panel) < 2:
            buy_period_returns.append(0.0)
            buy_equity.append(buy_equity[-1])
            continue
        start_p = period_panel.iloc[0]
        end_p = period_panel.iloc[-1]
        rets_full = (end_p / start_p - 1)

        # Bucket each scored name into its band
        period_counts = {b: 0 for b in bands}
        for tk, verdict in verdicts.items():
            r = rets_full.get(tk)
            if r is None or pd.isna(r):
                continue
            band_buckets[verdict].append(float(r))
            period_counts[verdict] += 1

        # BUY-side portfolio (BUY, optionally + ACCUMULATE)
        side_set = {"BUY"}
        if include_accumulate:
            side_set.add("ACCUMULATE")
        buy_names = [tk for tk, v in verdicts.items() if v in side_set]
        # Rank within BUY by raw score, then cap
        buy_names = sorted(buy_names, key=lambda tk: scores[tk], reverse=True)
        if buy_only_top_n is not None:
            buy_names = buy_names[:buy_only_top_n]

        if buy_names:
            r_arr = [rets_full.get(tk) for tk in buy_names]
            r_arr = [float(x) for x in r_arr if x is not None and not pd.isna(x)]
            period_ret = float(np.mean(r_arr)) if r_arr else 0.0
        else:
            period_ret = 0.0

        buy_period_returns.append(period_ret)
        buy_equity.append(buy_equity[-1] * (1 + period_ret))

        period_summaries.append({
            "date": date.strftime("%Y-%m-%d"),
            "n_scored": n,
            "counts": period_counts,
            "buy_picks": buy_names,
            "buy_return_pct": round(period_ret * 100, 2),
        })

    # Per-band stats
    by_verdict: dict[str, dict[str, Any]] = {}
    for band, returns in band_buckets.items():
        if not returns:
            by_verdict[band] = {"n": 0}
            continue
        arr = np.array(returns)
        by_verdict[band] = {
            "n": int(len(arr)),
            "mean_return_pct": round(float(arr.mean()) * 100, 3),
            "median_return_pct": round(float(np.median(arr)) * 100, 3),
            "hit_rate_pct": round(float((arr > 0).mean()) * 100, 1),
            "stdev_pct": round(float(arr.std()) * 100, 3),
        }

    # BUY portfolio summary
    eq = np.array(buy_equity)
    rets_arr = np.array(buy_period_returns)
    n_periods = len(rets_arr)
    years = n_periods * rebalance_days / 252
    total_ret = float(eq[-1] - 1) * 100
    cagr = ((1 + total_ret / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0.0
    period_vol = float(rets_arr.std()) if len(rets_arr) > 1 else 0.0
    annualized_vol = period_vol * ((252 / rebalance_days) ** 0.5) * 100

    rf = risk_free.get_rate()["rate_pct"] / 100
    rf_period = (1 + rf) ** (rebalance_days / 252) - 1
    excess = rets_arr - rf_period
    sharpe = float(excess.mean() / excess.std() * ((252 / rebalance_days) ** 0.5)) if excess.std() > 0 else 0.0
    hit_rate_portfolio = float((rets_arr > 0).mean()) * 100

    running_max = np.maximum.accumulate(eq)
    dd = (eq / running_max - 1)
    max_dd = float(dd.min()) * 100

    bench_total = None
    bench_cagr = None
    info_ratio = None
    if bench is not None and len(bench) > 1:
        bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1) * 100
        bench_cagr = ((1 + bench_total / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0.0
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
        "rebalance_days": rebalance_days,
        "n_rebalances": len(period_summaries),

        "by_verdict": by_verdict,
        "monotonicity_check": _monotonicity(by_verdict),

        "buy_portfolio": {
            "side": "BUY+ACCUMULATE" if include_accumulate else "BUY only",
            "top_n_cap": buy_only_top_n,
            "total_return_pct": round(total_ret, 2),
            "annualized_return_pct": round(cagr, 2),
            "annualized_volatility_pct": round(annualized_vol, 2),
            "annualized_sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "hit_rate_pct": round(hit_rate_portfolio, 1),
        },

        "benchmark_egx30": {
            "total_return_pct": round(bench_total, 2) if bench_total is not None else None,
            "annualized_return_pct": round(bench_cagr, 2) if bench_cagr is not None else None,
            "alpha_pct": round(total_ret - bench_total, 2) if bench_total is not None else None,
            "information_ratio": round(info_ratio, 2) if info_ratio is not None else None,
        },

        "bands_used": {b: list(rng) for b, rng in bands.items()},
        "period_samples_recent": period_summaries[-3:],

        "method": (
            "Within each rebalance period, score every name in the universe "
            "with the V3 price-only score (point-in-time clean), rank to "
            "percentile, map to chairman bands. For each (name, period, "
            "verdict) record forward return. Aggregate by band to test the "
            "monotonicity assumption (BUY > ACCUMULATE > HOLD > REDUCE > "
            "AVOID). Separately, simulate a BUY-side equal-weight portfolio "
            "with `buy_only_top_n` cap and compare to EGX 30."
        ),
        "caveat": (
            "Fundamentals, sentiment, calendar, and macro inputs of the "
            "live chairman are NOT replayed here — only the price-momentum "
            "half is point-in-time clean. The full live system has more "
            "filters; this isolates the verdict-mapping edge."
        ),
    }


def _monotonicity(by_verdict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check whether mean returns decline across BUY → AVOID."""
    order = ["BUY", "ACCUMULATE", "HOLD", "REDUCE", "AVOID"]
    means = []
    for v in order:
        mu = (by_verdict.get(v) or {}).get("mean_return_pct")
        if mu is None:
            return {"checked": False, "reason": f"empty band: {v}"}
        means.append((v, mu))

    pairs = list(zip(means, means[1:]))
    decreasing = all(a[1] >= b[1] for a, b in pairs)
    spread = round(means[0][1] - means[-1][1], 3)
    return {
        "checked": True,
        "ordered_means": means,
        "monotone_decreasing": decreasing,
        "buy_minus_avoid_pct": spread,
        "interpretation": (
            "Spread > 0 means BUY band outperforms AVOID band on average. "
            "monotone_decreasing=True is the gold-standard signal that the "
            "verdict mapping is meaningful, not noise."
        ),
    }
