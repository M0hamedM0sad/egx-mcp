"""V8 — fundamentals as a quality filter, not a primary signal.

Rationale: V7's blended approach mixes signals across the entire universe,
exposing the model to look-ahead bias even when fundamentals are weakly
informative. V8 uses fundamentals only to *exclude* low-quality names from
the candidate pool that V3's price signal then ranks.

Two filter modes:
    quality_filter  exclude bottom X% on ROE — kicks out junk
    value_filter    exclude top X% on P/E (most expensive) — kicks out hype
    both            apply both

The price signal (V3) is unchanged — fundamentals only narrow the universe.
This minimizes look-ahead exposure (fundamentals only act on the EXCLUSION
of names; the actual return ranking comes from price).
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from egx_mcp.data import egx_listing, risk_free
from egx_mcp.data.universe import resolve_ticker

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

START, END, HOLDOUT_START = "2024-01-01", "2026-04-30", "2025-07-01"


@dataclass
class FilterCfg:
    # V3 price weights
    w_mom6m: float = 0.5
    w_mr1m: float = 0.2
    trend_bonus: float = 5.0
    vol_cutoff_pct: float = 30.0
    vol_slope: float = 0.5
    stretched_thresh: float = 50.0
    stretched_slope: float = 0.2
    dip_thresh_1m: float = -8.0
    dip_thresh_6m: float = 10.0
    dip_bonus: float = 5.0
    # Fundamental filters
    exclude_roe_below: float | None = None     # drop ROE < X%
    exclude_pe_above: float | None = None      # drop P/E > X (junk-expensive)
    exclude_pb_above: float | None = None
    require_positive_eps: bool = False
    # Portfolio
    top_n: int = 5
    rebal_days: int = 21


def _norm(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _features_price(closes, asof):
    sub = closes.loc[:asof].dropna()
    if len(sub) < 130:
        return None
    p_now = float(sub.iloc[-1])
    p_6m = float(sub.iloc[-130])
    p_1m = float(sub.iloc[-22])
    ma200 = float(sub.tail(200).mean()) if len(sub) >= 200 else float(sub.mean())
    vol = float(sub.tail(60).pct_change().std() * (252 ** 0.5))
    if not (p_now > 0 and p_6m > 0 and p_1m > 0):
        return None
    return {"mom_6m": (p_now/p_6m - 1)*100, "mom_1m": (p_now/p_1m - 1)*100,
            "above_ma200": p_now > ma200, "vol_pct": vol*100}


def _price_score(f, c):
    base = (f["mom_6m"] * c.w_mom6m + (-f["mom_1m"]) * c.w_mr1m
            + (c.trend_bonus if f["above_ma200"] else -c.trend_bonus)
            - max(0, (f["vol_pct"] - c.vol_cutoff_pct) * c.vol_slope))
    if f["mom_6m"] > c.stretched_thresh:
        base -= (f["mom_6m"] - c.stretched_thresh) * c.stretched_slope
    if f["mom_1m"] < c.dip_thresh_1m and f["mom_6m"] > c.dip_thresh_6m and f["above_ma200"]:
        base += c.dip_bonus
    return base


def _passes_filter(tk: str, fund: dict, c: FilterCfg) -> bool:
    d = fund.get(tk, {})
    if c.exclude_roe_below is not None:
        roe = d.get("roe")
        if roe is None or roe < c.exclude_roe_below:
            return False
    if c.exclude_pe_above is not None:
        pe = d.get("pe")
        if pe is None or pe > c.exclude_pe_above:
            return False
    if c.exclude_pb_above is not None:
        pb = d.get("pb")
        if pb is None or pb > c.exclude_pb_above:
            return False
    if c.require_positive_eps:
        eps = d.get("eps")
        if eps is None or eps <= 0:
            return False
    return True


def _load_fund():
    cache = json.loads(
        (Path(__file__).parent.parent / "egx_mcp" / "data"
         / "mubasher_fundamentals_cache.json").read_text(encoding="utf-8")
    )
    return {tk: {"pe": d.get("pe_ratio"), "pb": d.get("pb_ratio"),
                 "roe": d.get("roe_pct"), "eps": d.get("trailing_eps")}
            for tk, d in cache.items()}


def run(cp, fund, cfg, start_date=None):
    panel = cp.loc[start_date:] if start_date else cp
    rebalance_idx = panel.index[::cfg.rebal_days]
    if len(rebalance_idx) < 3:
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods", "vol", "n_eligible")}
    equity = [1.0]
    period_rets = []
    eligible_counts = []
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct/100) ** (cfg.rebal_days/252) - 1

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]
        next_date = rebalance_idx[i+1]

        scores = {}
        for tk in panel.columns:
            if not _passes_filter(tk, fund, cfg):
                continue
            f = _features_price(cp[tk], date)
            if f is None:
                continue
            scores[tk] = _price_score(f, cfg)
        eligible_counts.append(len(scores))

        if not scores:
            period_rets.append(0); equity.append(equity[-1]); continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [tk for tk, _ in ranked[:cfg.top_n]]
        try:
            sub = cp.loc[date:next_date, picks].dropna(how="all")
            if len(sub) < 2:
                period_rets.append(0); equity.append(equity[-1]); continue
            r = (sub.iloc[-1] / sub.iloc[0] - 1).dropna()
            period_ret = float(r.mean()) if not r.empty else 0
        except Exception:
            period_ret = 0
        period_rets.append(period_ret)
        equity.append(equity[-1] * (1 + period_ret))

    eq = np.array(equity); rets_arr = np.array(period_rets)
    if len(rets_arr) < 2:
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods", "vol", "n_eligible")}
    total = (eq[-1] - 1) * 100
    years = len(rets_arr) * cfg.rebal_days / 252
    cagr = ((1 + total/100) ** (1/max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol_ann = rets_arr.std() * ((252/cfg.rebal_days)**0.5) * 100
    excess = rets_arr - rf_period
    sharpe = excess.mean()/excess.std() * ((252/cfg.rebal_days)**0.5) if excess.std() > 0 else 0
    hit = (rets_arr > 0).mean() * 100
    rmax = np.maximum.accumulate(eq); dd = (eq/rmax - 1)
    max_dd = dd.min() * 100
    calmar = abs(cagr / max_dd) if max_dd < 0 else 999
    return {"cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
            "hit_rate": hit, "n_periods": len(rets_arr), "vol": vol_ann,
            "n_eligible": np.mean(eligible_counts)}


def buy_hold(cp, start_date=None):
    panel = cp.loc[start_date:] if start_date else cp
    panel = panel.dropna(how="all")
    rets = panel.pct_change().dropna(how="all").mean(axis=1).fillna(0)
    eq = (1 + rets).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    years = len(rets) / 252
    cagr = ((1 + total/100) ** (1/max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol = float(rets.std() * (252**0.5)) * 100
    rmax = eq.cummax(); dd = (eq/rmax - 1)
    max_dd = float(dd.min()) * 100
    rf = risk_free.get_rate()["rate_pct"]; daily_rf = (1+rf/100)**(1/252) - 1
    excess = rets - daily_rf
    sharpe = float(excess.mean()/excess.std() * (252**0.5)) if excess.std() > 0 else 0
    return {"cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
            "calmar": abs(cagr/max_dd) if max_dd < 0 else 0, "vol": vol}


def main():
    print("Loading prices and fundamentals...")
    universe = egx_listing.get_full_universe()
    cp_d = {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start=START, end=END, interval="1d")
            if h is None or h.empty:
                continue
            cp_d[tk] = _norm(h["Close"])
        except Exception:
            continue
    cp = pd.DataFrame(cp_d).sort_index()
    fund = _load_fund()
    print(f"  Prices: {len(cp.columns)} names, fundamentals: "
          f"{sum(1 for d in fund.values() if d.get('roe') is not None)} with ROE\n")

    print(f"{'Strategy':<40}{'Univ':>5}{'CAGR':>8}{'DD':>8}{'Sharpe':>8}{'Calm':>7}"
          f"{'HCAGR':>8}{'HDD':>8}{'HSharpe':>9}{'HCalm':>7}")
    print("-" * 105)

    scenarios = [
        ("V3 baseline (no filter)",            FilterCfg()),
        ("V8a: ROE > 5% filter",               FilterCfg(exclude_roe_below=5)),
        ("V8b: ROE > 10% filter",              FilterCfg(exclude_roe_below=10)),
        ("V8c: ROE > 15% filter",              FilterCfg(exclude_roe_below=15)),
        ("V8d: positive EPS filter",           FilterCfg(require_positive_eps=True)),
        ("V8e: ROE > 10 + P/E < 30",           FilterCfg(exclude_roe_below=10, exclude_pe_above=30)),
        ("V8f: ROE > 15 + P/E < 25",           FilterCfg(exclude_roe_below=15, exclude_pe_above=25)),
        ("V8g: ROE > 10 + P/E < 30 + +EPS",    FilterCfg(exclude_roe_below=10, exclude_pe_above=30, require_positive_eps=True)),
    ]

    for label, c in scenarios:
        f = run(cp, fund, c)
        h = run(cp, fund, c, start_date=HOLDOUT_START)
        print(f"{label:<40}{int(f['n_eligible']):>5}"
              f"{f['cagr']:>+7.1f}%{f['max_dd']:>+7.1f}%{f['sharpe']:>8.2f}{f['calmar']:>7.1f}"
              f"{h['cagr']:>+7.1f}%{h['max_dd']:>+7.1f}%{h['sharpe']:>9.2f}{h['calmar']:>7.1f}")

    bf = buy_hold(cp); bh = buy_hold(cp, start_date=HOLDOUT_START)
    print(f"{'Buy & Hold (market)':<40}{68:>5}"
          f"{bf['cagr']:>+7.1f}%{bf['max_dd']:>+7.1f}%{bf['sharpe']:>8.2f}{bf['calmar']:>7.1f}"
          f"{bh['cagr']:>+7.1f}%{bh['max_dd']:>+7.1f}%{bh['sharpe']:>9.2f}{bh['calmar']:>7.1f}")
    print()


if __name__ == "__main__":
    main()
