"""V3 vs V5 (= V3 with top_n=10) — targeted A/B test."""
from __future__ import annotations

# Imports BEFORE stdout wrap so module-level prints in dependencies don't
# get a wrapper that's later invalidated.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from egx_mcp.data import egx_listing, risk_free
from egx_mcp.data.universe import resolve_ticker

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

START, END, HOLDOUT_START = "2024-01-01", "2026-04-30", "2025-07-01"


@dataclass
class Config:
    w_mom6m: float = 0.5
    w_mr1m: float = 0.2
    w_mom3m: float = 0.0
    trend_bonus: float = 5.0
    vol_cutoff_pct: float = 30.0
    vol_slope: float = 0.5
    stretched_thresh: float = 50.0
    stretched_slope: float = 0.2
    dip_thresh_1m: float = -8.0
    dip_thresh_6m: float = 10.0
    dip_bonus: float = 5.0
    top_n: int = 5
    rebal_days: int = 21
    min_adv_egp: float = 0


def _norm(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _features(closes, asof):
    sub = closes.loc[:asof].dropna()
    if len(sub) < 130:
        return None
    p_now = float(sub.iloc[-1])
    p_6m = float(sub.iloc[-130])
    p_3m = float(sub.iloc[-65])
    p_1m = float(sub.iloc[-22])
    ma200 = float(sub.tail(200).mean()) if len(sub) >= 200 else float(sub.mean())
    vol = float(sub.tail(60).pct_change().std() * (252 ** 0.5))
    if not (p_now > 0 and p_6m > 0 and p_3m > 0 and p_1m > 0):
        return None
    return {"p_now": p_now, "mom_6m": (p_now/p_6m - 1)*100, "mom_3m": (p_now/p_3m - 1)*100,
            "mom_1m": (p_now/p_1m - 1)*100, "above_ma200": p_now > ma200, "vol_pct": vol*100}


def _score(f, c):
    base = (f["mom_6m"] * c.w_mom6m + (-f["mom_1m"]) * c.w_mr1m
            + f["mom_3m"] * c.w_mom3m
            + (c.trend_bonus if f["above_ma200"] else -c.trend_bonus)
            - max(0, (f["vol_pct"] - c.vol_cutoff_pct) * c.vol_slope))
    if f["mom_6m"] > c.stretched_thresh:
        base -= (f["mom_6m"] - c.stretched_thresh) * c.stretched_slope
    if f["mom_1m"] < c.dip_thresh_1m and f["mom_6m"] > c.dip_thresh_6m and f["above_ma200"]:
        base += c.dip_bonus
    return base


def _adv_at(vp, cp, asof, days=20):
    sub_v = vp.loc[:asof].tail(days)
    sub_c = cp.loc[:asof].tail(days)
    if sub_v.empty:
        return pd.Series(dtype=float)
    return (sub_v.mean() * sub_c.mean()).fillna(0)


def run(cp, vp, cfg, start_date=None):
    panel = cp.loc[start_date:] if start_date else cp
    rebalance_idx = panel.index[::cfg.rebal_days]
    if len(rebalance_idx) < 3:
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods")}
    equity = [1.0]
    period_rets = []
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct/100) ** (cfg.rebal_days/252) - 1
    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]; next_date = rebalance_idx[i+1]
        if cfg.min_adv_egp > 0:
            adv = _adv_at(vp, cp, date, 20)
            eligible = set(adv[adv >= cfg.min_adv_egp].index)
        else:
            eligible = set(panel.columns)
        scores = {}
        for tk in panel.columns:
            if tk not in eligible: continue
            f = _features(cp[tk], date)
            if f is None: continue
            scores[tk] = _score(f, cfg)
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
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods")}
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
            "hit_rate": hit, "n_periods": len(rets_arr), "vol": vol_ann}


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
    calmar = abs(cagr/max_dd) if max_dd < 0 else 0
    return {"cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar, "vol": vol}


print("Loading prices...")
universe = egx_listing.get_full_universe()
cp_d, vp_d = {}, {}
for tk in universe:
    _, yahoo, _ = resolve_ticker(tk)
    try:
        h = yf.Ticker(yahoo).history(start=START, end=END, interval="1d")
        if h is None or h.empty: continue
        cp_d[tk] = _norm(h["Close"])
        vp_d[tk] = _norm(h["Volume"])
    except Exception:
        continue
cp = pd.DataFrame(cp_d).sort_index()
vp = pd.DataFrame(vp_d).sort_index()
print(f"  Loaded {len(cp.columns)} names\n")

print(f"{'Strategy':<28}{'CAGR':>9}{'DD':>8}{'Sharpe':>8}{'Calmar':>8}"
      f"{'HCAGR':>9}{'HDD':>8}{'HShar':>8}{'HCalm':>8}")
print("-" * 95)

scenarios = [
    ("V3 baseline (top5)",        Config(top_n=5)),
    ("V5: V3 + top10",            Config(top_n=10)),
    ("V5 + liquidity floor",      Config(top_n=10, min_adv_egp=500_000)),
    ("V5 + liq + 3m signal",      Config(top_n=10, min_adv_egp=500_000, w_mom3m=0.15)),
    ("V5 + liq + biweekly",       Config(top_n=10, min_adv_egp=500_000, rebal_days=10)),
    ("V5 (top7)",                 Config(top_n=7)),
    ("V6: top10 + 3m + biweekly", Config(top_n=10, w_mom3m=0.15, rebal_days=10, min_adv_egp=500_000)),
]
for label, c in scenarios:
    f = run(cp, vp, c)
    h = run(cp, vp, c, start_date=HOLDOUT_START)
    print(f"{label:<28}{f['cagr']:>+8.1f}%{f['max_dd']:>+7.1f}%{f['sharpe']:>8.2f}"
          f"{f['calmar']:>8.1f}{h['cagr']:>+8.1f}%{h['max_dd']:>+7.1f}%"
          f"{h['sharpe']:>8.2f}{h['calmar']:>8.1f}")

bf = buy_hold(cp); bh = buy_hold(cp, start_date=HOLDOUT_START)
print(f"{'Buy & Hold (market)':<28}{bf['cagr']:>+8.1f}%{bf['max_dd']:>+7.1f}%"
      f"{bf['sharpe']:>8.2f}{bf['calmar']:>8.1f}"
      f"{bh['cagr']:>+8.1f}%{bh['max_dd']:>+7.1f}%{bh['sharpe']:>8.2f}{bh['calmar']:>8.1f}")
print()
