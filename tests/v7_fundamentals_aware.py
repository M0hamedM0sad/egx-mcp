"""V7 — fundamentals-aware scoring.

Combines V3's price signal with a fundamentals score built from Mubasher's
audited P/E, P/B, ROE, EPS, BVPS data.

Caveat: fundamentals are loaded as the *current* Mubasher snapshot — they
don't change over the backtest window. This introduces moderate look-ahead
bias since today's fundamentals partially reflect what's happened in the
test period. To mitigate:
  1. ROE and P/E for EGX names move slowly relative to a 28-month window.
  2. The hardest test is the 10-month holdout where the bias is smallest.
  3. We compare against V3 (which has zero look-ahead) — V7 must beat V3
     by enough margin that even with a haircut for bias, it's still better.

Score:
    V7 = 0.6 × price_score (V3) + 0.4 × fundamentals_score

Fundamentals score (0-100) = rank-based composite:
    rank_PE   ascending (low P/E = high rank = high score)
    rank_PB   ascending
    rank_ROE  descending
    Final = 100 × avg(ranks_normalized)

Plus the same stretched-name guard from V3.
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
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
class Config:
    w_price: float = 0.6
    w_fund: float = 0.4
    # Price score weights (V3)
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
    # Fundamentals score weights
    fw_pe: float = 0.4    # higher weight on cheapness
    fw_pb: float = 0.25
    fw_roe: float = 0.35
    # Sanity bounds — values outside get clipped before ranking
    pe_floor: float = 1.0
    pe_cap: float = 100.0
    pb_floor: float = 0.1
    pb_cap: float = 30.0
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


def _load_fundamentals():
    cache = json.loads(
        (Path(__file__).parent.parent / "egx_mcp" / "data"
         / "mubasher_fundamentals_cache.json").read_text(encoding="utf-8")
    )
    out = {}
    for tk, d in cache.items():
        out[tk] = {
            "pe":  d.get("pe_ratio"),
            "pb":  d.get("pb_ratio"),
            "roe": d.get("roe_pct"),
        }
    return out


def _fund_score_for_universe(fund: dict, c: Config, eligible: set) -> dict:
    """Rank-based 0-100 score for each eligible ticker."""
    rows = []
    for tk in eligible:
        d = fund.get(tk, {})
        pe, pb, roe = d.get("pe"), d.get("pb"), d.get("roe")
        if pe is not None:
            pe = max(c.pe_floor, min(c.pe_cap, pe))
        if pb is not None:
            pb = max(c.pb_floor, min(c.pb_cap, pb))
        rows.append({"tk": tk, "pe": pe, "pb": pb, "roe": roe})

    df = pd.DataFrame(rows).set_index("tk")
    n = len(df)
    if n == 0:
        return {}

    # Rank: low P/E good (ascending), low P/B good, high ROE good (descending)
    df["r_pe"] = df["pe"].rank(method="average", ascending=True, na_option="bottom")
    df["r_pb"] = df["pb"].rank(method="average", ascending=True, na_option="bottom")
    df["r_roe"] = df["roe"].rank(method="average", ascending=False, na_option="bottom")

    # Normalize ranks to 0-100 (best rank = 100)
    df["s_pe"]  = 100 * (n - df["r_pe"] + 1) / n
    df["s_pb"]  = 100 * (n - df["r_pb"] + 1) / n
    df["s_roe"] = 100 * (n - df["r_roe"] + 1) / n

    df["fund_score"] = (df["s_pe"] * c.fw_pe + df["s_pb"] * c.fw_pb
                          + df["s_roe"] * c.fw_roe) / (c.fw_pe + c.fw_pb + c.fw_roe)
    return df["fund_score"].to_dict()


def run(cp, fund, cfg, start_date=None, score_mode="v7"):
    """score_mode in {'v3', 'v7', 'fund_only'}"""
    panel = cp.loc[start_date:] if start_date else cp
    rebalance_idx = panel.index[::cfg.rebal_days]
    if len(rebalance_idx) < 3:
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods", "vol")}

    equity = [1.0]
    period_rets = []
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct/100) ** (cfg.rebal_days/252) - 1

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]
        next_date = rebalance_idx[i+1]

        # Compute price scores for everyone
        price_scores = {}
        for tk in panel.columns:
            f = _features_price(cp[tk], date)
            if f is None:
                continue
            price_scores[tk] = _price_score(f, cfg)

        if not price_scores:
            period_rets.append(0); equity.append(equity[-1]); continue

        eligible = set(price_scores.keys())

        # Compute fundamentals scores
        if score_mode in ("v7", "fund_only"):
            fund_scores = _fund_score_for_universe(fund, cfg, eligible)
        else:
            fund_scores = {}

        # Combine
        if score_mode == "v3":
            combined = price_scores
        elif score_mode == "fund_only":
            combined = fund_scores
        else:  # v7
            combined = {}
            for tk in eligible:
                p = price_scores.get(tk, 0)
                ff = fund_scores.get(tk, 50)  # neutral if missing
                combined[tk] = cfg.w_price * p + cfg.w_fund * ff

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
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
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods", "vol")}

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
    fund = _load_fundamentals()
    print(f"  Prices: {len(cp.columns)} names, fundamentals: "
          f"{sum(1 for d in fund.values() if d.get('roe') is not None)} with ROE")
    print()

    cfg_v3 = Config(w_price=1.0, w_fund=0.0)
    cfg_v7_balanced = Config(w_price=0.6, w_fund=0.4)
    cfg_v7_heavy = Config(w_price=0.4, w_fund=0.6)
    cfg_v7_light = Config(w_price=0.8, w_fund=0.2)
    cfg_fund_only = Config(w_price=0.0, w_fund=1.0)

    print(f"{'Strategy':<32}{'CAGR':>8}{'DD':>8}{'Sharpe':>8}{'Calmar':>8}"
          f"{'HCAGR':>8}{'HDD':>8}{'HSharpe':>9}{'HCalm':>8}")
    print("-" * 95)

    scenarios = [
        ("V3 baseline (price only)", cfg_v3, "v3"),
        ("V7 light (80% px / 20% fund)", cfg_v7_light, "v7"),
        ("V7 balanced (60/40)", cfg_v7_balanced, "v7"),
        ("V7 heavy (40/60)", cfg_v7_heavy, "v7"),
        ("Fundamentals only", cfg_fund_only, "fund_only"),
    ]

    for label, c, mode in scenarios:
        f = run(cp, fund, c, score_mode=mode)
        h = run(cp, fund, c, start_date=HOLDOUT_START, score_mode=mode)
        print(f"{label:<32}{f['cagr']:>+7.1f}%{f['max_dd']:>+7.1f}%"
              f"{f['sharpe']:>8.2f}{f['calmar']:>8.1f}"
              f"{h['cagr']:>+7.1f}%{h['max_dd']:>+7.1f}%"
              f"{h['sharpe']:>9.2f}{h['calmar']:>8.1f}")

    bf = buy_hold(cp); bh = buy_hold(cp, start_date=HOLDOUT_START)
    print(f"{'Buy & Hold (market)':<32}{bf['cagr']:>+7.1f}%{bf['max_dd']:>+7.1f}%"
          f"{bf['sharpe']:>8.2f}{bf['calmar']:>8.1f}"
          f"{bh['cagr']:>+7.1f}%{bh['max_dd']:>+7.1f}%"
          f"{bh['sharpe']:>9.2f}{bh['calmar']:>8.1f}")
    print()


if __name__ == "__main__":
    main()
