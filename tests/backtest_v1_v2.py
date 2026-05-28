"""Full historical backtest of V1 vs V2 — Jan 2024 → today.

Walk-forward monthly rebalance, top-5 equal-weight, both scoring rubrics,
same universe. The single OOS week is N=1; this aggregates ~28 rebalances
to give a real Sharpe/DD/hit-rate comparison.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import egx_listing, risk_free
from egx_mcp.data.universe import resolve_ticker

START = "2024-01-01"
END = "2026-04-30"
TOP_N = 5
REBAL = 21


def _features(closes: pd.Series, asof: pd.Timestamp) -> dict | None:
    sub = closes.loc[:asof].dropna()
    if len(sub) < 130:
        return None
    p_now = float(sub.iloc[-1])
    p_6m = float(sub.iloc[-130])
    p_1m = float(sub.iloc[-22])
    p_5d = float(sub.iloc[-6]) if len(sub) >= 6 else p_1m
    ma200 = float(sub.tail(200).mean()) if len(sub) >= 200 else float(sub.mean())
    vol = float(sub.tail(60).pct_change().std() * (252 ** 0.5))
    if not (p_now > 0 and p_6m > 0 and p_1m > 0 and p_5d > 0):
        return None
    return {
        "p_now": p_now, "mom_6m": (p_now / p_6m - 1) * 100,
        "mom_1m": (p_now / p_1m - 1) * 100,
        "mom_5d": (p_now / p_5d - 1) * 100,
        "above_ma200": p_now > ma200, "vol_pct": vol * 100,
    }


def score_v1(f: dict) -> float:
    return f["mom_6m"] * 0.5 + (-f["mom_1m"]) * 0.2 + (5 if f["above_ma200"] else -5) - max(0, (f["vol_pct"] - 30) * 0.5)


def score_v2(f: dict) -> float:
    base = (f["mom_6m"] * 0.3 + (-f["mom_1m"]) * 0.2 + (-f["mom_5d"]) * 0.3
            + (5 if f["above_ma200"] else -5) - max(0, (f["vol_pct"] - 30) * 0.5))
    if f["mom_6m"] > 50:
        base -= (f["mom_6m"] - 50) * 0.2
    if f["mom_1m"] < -8 and f["mom_6m"] > 10 and f["above_ma200"]:
        base += 5
    return base


def score_v3(f: dict) -> float:
    """V3 = V1 + only the parts of V2 that survived the stretched-name test.

    Same momentum weight as V1 (0.5) — the alpha source we don't want to lose.
    Keeps the 1M mean-reversion at 0.2.
    Adds the stretched penalty (proven to help on the short side).
    Adds the dip-in-uptrend bonus.
    """
    base = (f["mom_6m"] * 0.5 + (-f["mom_1m"]) * 0.2
            + (5 if f["above_ma200"] else -5)
            - max(0, (f["vol_pct"] - 30) * 0.5))
    if f["mom_6m"] > 50:
        base -= (f["mom_6m"] - 50) * 0.2
    if f["mom_1m"] < -8 and f["mom_6m"] > 10 and f["above_ma200"]:
        base += 5
    return base


def run_backtest(label: str, scorer):
    panel = pd.DataFrame(panel_data).sort_index()
    rebalance_idx = panel.index[::REBAL]
    if len(rebalance_idx) < 3:
        print("Not enough rebalances")
        return

    equity = [1.0]
    period_rets = []
    monthly_picks = []

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]
        next_date = rebalance_idx[i + 1]
        scores = {}
        for tk, closes in panel.items():
            f = _features(closes, date)
            if f is None:
                continue
            scores[tk] = scorer(f)
        if not scores:
            period_rets.append(0)
            equity.append(equity[-1])
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [tk for tk, _ in ranked[:TOP_N]]
        monthly_picks.append({"date": date.strftime("%Y-%m-%d"), "picks": picks})
        try:
            sub = panel.loc[date:next_date, picks].dropna(how="all")
            if len(sub) < 2:
                period_rets.append(0); equity.append(equity[-1]); continue
            r = (sub.iloc[-1] / sub.iloc[0] - 1).dropna()
            period_ret = float(r.mean()) if not r.empty else 0
        except Exception:
            period_ret = 0
        period_rets.append(period_ret)
        equity.append(equity[-1] * (1 + period_ret))

    eq = np.array(equity)
    rets_arr = np.array(period_rets)
    total_ret = (eq[-1] - 1) * 100
    years = len(rets_arr) * REBAL / 252
    cagr = ((1 + total_ret / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol_ann = rets_arr.std() * ((252 / REBAL) ** 0.5) * 100 if len(rets_arr) > 1 else 0
    rf = risk_free.get_rate()["rate_pct"] / 100
    rf_period = (1 + rf) ** (REBAL / 252) - 1
    excess = rets_arr - rf_period
    sharpe = excess.mean() / excess.std() * ((252 / REBAL) ** 0.5) if excess.std() > 0 else 0
    hit_rate = (rets_arr > 0).mean() * 100
    running_max = np.maximum.accumulate(eq)
    dd = (eq / running_max - 1)
    max_dd = dd.min() * 100

    return {
        "label": label, "total_ret": total_ret, "cagr": cagr, "vol": vol_ann,
        "sharpe": sharpe, "hit_rate": hit_rate, "max_dd": max_dd,
        "n_periods": len(rets_arr), "final_equity": float(eq[-1]),
        "monthly_picks": monthly_picks,
    }


# Pull universe once
print("Loading prices for full universe...")
universe = egx_listing.get_full_universe()
panel_data = {}
for tk in universe:
    _, yahoo, _ = resolve_ticker(tk)
    try:
        h = yf.Ticker(yahoo).history(start=START, end=END, interval="1d")
        if h is None or h.empty:
            continue
        s = h["Close"].copy()
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = pd.to_datetime(idx.date)
        panel_data[tk] = s[~s.index.duplicated(keep="last")]
    except Exception:
        continue
print(f"Loaded {len(panel_data)} names\n")

print(f"Running V1 backtest...")
v1 = run_backtest("V1 (current)", score_v1)
print(f"Running V2 backtest...")
v2 = run_backtest("V2 (MR-aware)", score_v2)
print(f"Running V3 backtest...")
v3 = run_backtest("V3 (V1 + stretched penalty + dip bonus)", score_v3)

print("\n" + "=" * 80)
print(f"BACKTEST: {START} → {END}, top-{TOP_N} monthly rebalance")
print("=" * 80)
print(f"  {'Metric':<28}{'V1':>13}{'V2':>13}{'V3':>13}{'V3-V1':>10}")
print("-" * 80)
for k, lbl in [("total_ret", "Total return %"), ("cagr", "CAGR %"),
               ("vol", "Annualized vol %"), ("sharpe", "Sharpe (excess)"),
               ("hit_rate", "Hit rate %"), ("max_dd", "Max drawdown %"),
               ("final_equity", "Final equity (1.0 start)")]:
    if k == "final_equity":
        fmt = "{:>12.4f} "
    else:
        fmt = "{:>+12.2f} "
    print(f"  {lbl:<28}{fmt.format(v1[k])}{fmt.format(v2[k])}{fmt.format(v3[k])}"
          f"{v3[k] - v1[k]:>+9.2f}")
print(f"  {'Periods':<28}{v1['n_periods']:>13}{v2['n_periods']:>13}{v3['n_periods']:>13}")
print()
