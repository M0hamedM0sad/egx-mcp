"""W1 — Weekly trading model search.

The monthly V8b uses 6M momentum (slow) + ROE filter. That signal-to-noise
is wrong for a 5-day horizon — short-term moves are dominated by:

    - 5-10 day momentum and mean reversion
    - Volume confirmation (high-volume moves persist; low-volume mean-revert)
    - Breakout / breakdown of recent ranges (20-day high/low)
    - Position relative to short MAs (MA5, MA20)

This script searches the parameter space for a weekly score that beats the
synthetic-basket market on a 5-day rebalance horizon, validated walk-forward.

Score formula (all weights tunable):
    s = w_mom5 * mom_5d
      + w_mr1 * (-mom_1d)              # mean-revert daily noise
      + w_vol_conf * volume_ratio      # high-vol = strong signal
      + w_breakout * breakout_signal   # +1 if 20d high broken, -1 if 20d low
      + w_trend * (1 if above MA20)
      - w_vol_pen * realized_vol_pct
      + stretched_penalty (for >25% in 5d)
      + dip_in_uptrend (down 5%+ but above MA20)
"""
from __future__ import annotations

import io
import json
import random
import sys
import time
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

START = "2024-01-01"
END = "2026-04-30"
HOLDOUT_START = "2025-07-01"


@dataclass
class W1Config:
    # Weekly score weights
    w_mom5: float = 1.0           # 5-day momentum
    w_mom20: float = 0.3          # 20-day momentum (intermediate)
    w_mr1: float = 0.5            # 1-day mean reversion
    w_vol_conf: float = 2.0       # volume confirmation multiplier
    w_breakout: float = 5.0       # breakout/breakdown signal
    w_trend: float = 3.0          # above MA20 bonus
    w_vol_pen: float = 0.0        # realized vol penalty (off by default)
    stretched_5d_thresh: float = 12.0   # 5d return above this = stretched
    stretched_5d_slope: float = 0.5
    dip_thresh_5d: float = -5.0
    dip_bonus: float = 3.0
    min_volume_ratio: float = 0.5  # require vol >= 50% of ADV (skip dead names)
    min_roe_pct: float = 10.0      # quality filter
    top_n: int = 5
    rebal_days: int = 5            # WEEKLY


def _norm(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _features_weekly(closes: pd.Series, volumes: pd.Series, asof: pd.Timestamp):
    cl = closes.loc[:asof].dropna()
    if len(cl) < 60:
        return None
    p_now = float(cl.iloc[-1])
    p_1d = float(cl.iloc[-2]) if len(cl) >= 2 else p_now
    p_5d = float(cl.iloc[-6]) if len(cl) >= 6 else p_now
    p_20d = float(cl.iloc[-21]) if len(cl) >= 21 else p_now
    if not (p_now > 0 and p_1d > 0 and p_5d > 0 and p_20d > 0):
        return None

    high_20 = float(cl.tail(20).max())
    low_20 = float(cl.tail(20).min())
    ma5 = float(cl.tail(5).mean())
    ma20 = float(cl.tail(20).mean())

    # Volume ratio: today's volume vs trailing 20-day average
    vol = volumes.loc[:asof].dropna()
    today_vol = float(vol.iloc[-1]) if len(vol) >= 1 else 0
    avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else max(today_vol, 1)
    vol_ratio = today_vol / max(avg_vol, 1)

    # Realized vol over 20 days
    rvol = float(cl.tail(20).pct_change().std() * (252 ** 0.5)) if len(cl) >= 20 else 0.4

    # Breakout signal: +1 if today's price = recent 20d high, -1 if = recent low
    if p_now >= high_20 * 0.999:
        breakout = 1.0
    elif p_now <= low_20 * 1.001:
        breakout = -1.0
    else:
        breakout = 0.0

    return {
        "p_now": p_now,
        "mom_1d": (p_now / p_1d - 1) * 100,
        "mom_5d": (p_now / p_5d - 1) * 100,
        "mom_20d": (p_now / p_20d - 1) * 100,
        "above_ma20": p_now > ma20,
        "above_ma5": p_now > ma5,
        "vol_ratio": vol_ratio,
        "rvol_pct": rvol * 100,
        "breakout": breakout,
        "high_20": high_20,
        "low_20": low_20,
    }


def _w1_score(f, c: W1Config):
    s = (f["mom_5d"] * c.w_mom5
         + f["mom_20d"] * c.w_mom20
         + (-f["mom_1d"]) * c.w_mr1
         + c.w_vol_conf * (f["vol_ratio"] - 1.0)         # signed: above 1 boosts, below 1 cuts
         + c.w_breakout * f["breakout"]
         + (c.w_trend if f["above_ma20"] else -c.w_trend))
    s -= c.w_vol_pen * f["rvol_pct"]
    if f["mom_5d"] > c.stretched_5d_thresh:
        s -= (f["mom_5d"] - c.stretched_5d_thresh) * c.stretched_5d_slope
    if f["mom_5d"] < c.dip_thresh_5d and f["above_ma20"]:
        s += c.dip_bonus
    return s


def _load_quality(min_roe_pct: float):
    cache = json.loads(
        (Path(__file__).parent.parent / "egx_mcp" / "data"
         / "mubasher_fundamentals_cache.json").read_text(encoding="utf-8")
    )
    return {tk for tk, d in cache.items()
            if d.get("roe_pct") is not None and d["roe_pct"] >= min_roe_pct}


def run(cp, vp, cfg: W1Config, start_date=None, quality_set=None):
    panel = cp.loc[start_date:] if start_date else cp
    rebalance_idx = panel.index[::cfg.rebal_days]
    if len(rebalance_idx) < 3:
        return {k: 0 for k in ("cagr", "max_dd", "sharpe", "calmar", "hit_rate", "n_periods", "vol")}
    eligible_cols = list(cp.columns)
    if quality_set is not None:
        eligible_cols = [t for t in eligible_cols if t in quality_set]

    equity = [1.0]
    period_rets = []
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct/100) ** (cfg.rebal_days/252) - 1

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]; next_date = rebalance_idx[i+1]
        scores = {}
        for tk in eligible_cols:
            if tk not in cp.columns or tk not in vp.columns:
                continue
            f = _features_weekly(cp[tk], vp[tk], date)
            if f is None:
                continue
            if f["vol_ratio"] < cfg.min_volume_ratio:
                continue
            scores[tk] = _w1_score(f, cfg)
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


def buy_hold(cp, start_date=None, rebal_days=5):
    """Weekly returns of equal-weighted basket — fair benchmark for weekly horizon."""
    panel = cp.loc[start_date:] if start_date else cp
    panel = panel.dropna(how="all")
    if panel.empty:
        return {}
    rebalance_idx = panel.index[::rebal_days]
    if len(rebalance_idx) < 3:
        return {}
    period_rets = []
    for i in range(len(rebalance_idx) - 1):
        d0 = rebalance_idx[i]; d1 = rebalance_idx[i+1]
        sub = panel.loc[d0:d1].dropna(how="all")
        if len(sub) < 2:
            period_rets.append(0); continue
        r = (sub.iloc[-1] / sub.iloc[0] - 1).dropna()
        period_rets.append(float(r.mean()) if not r.empty else 0)
    eq = [1.0]
    for r in period_rets:
        eq.append(eq[-1] * (1 + r))
    eq = np.array(eq); rets_arr = np.array(period_rets)
    if len(rets_arr) < 2:
        return {}
    total = (eq[-1] - 1) * 100
    years = len(rets_arr) * rebal_days / 252
    cagr = ((1 + total/100) ** (1/max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol_ann = rets_arr.std() * ((252/rebal_days)**0.5) * 100
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct/100) ** (rebal_days/252) - 1
    excess = rets_arr - rf_period
    sharpe = excess.mean()/excess.std() * ((252/rebal_days)**0.5) if excess.std() > 0 else 0
    rmax = np.maximum.accumulate(eq); dd = (eq/rmax - 1)
    max_dd = dd.min() * 100
    return {"cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
            "calmar": abs(cagr/max_dd) if max_dd < 0 else 0,
            "hit_rate": (rets_arr > 0).mean() * 100, "vol": vol_ann,
            "n_periods": len(rets_arr)}


def main():
    print(f"Loading prices and volumes ({START} → {END})...")
    universe = egx_listing.get_full_universe()
    cp_d, vp_d = {}, {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start="2023-01-01", end=END, interval="1d")
            if h is None or h.empty:
                continue
            cp_d[tk] = _norm(h["Close"])
            vp_d[tk] = _norm(h["Volume"])
        except Exception:
            continue
    cp = pd.DataFrame(cp_d).sort_index()
    vp = pd.DataFrame(vp_d).sort_index()
    quality_set = _load_quality(10.0)
    print(f"  {len(cp.columns)} names, {len(quality_set)} pass ROE>=10% filter\n")

    # 1. Baseline benchmark and a few hand-picked configs
    print("=" * 110)
    print("STAGE 1 — Hand-picked weekly configs vs market (weekly buy-hold)")
    print("=" * 110)
    print(f"{'Strategy':<42}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'Hit%':>7}")

    bench = buy_hold(cp, start_date=START, rebal_days=5)
    print(f"{'Market: weekly buy-hold (68 names)':<42}"
          f"{bench['cagr']:>+8.1f}%{bench['vol']:>+7.1f}%{bench['sharpe']:>8.2f}"
          f"{bench['max_dd']:>+8.1f}%{bench['calmar']:>8.2f}{bench['hit_rate']:>+6.1f}%")

    configs = [
        ("W1-A: pure 5d momentum", W1Config(w_mom5=1.0, w_mom20=0, w_mr1=0, w_vol_conf=0, w_breakout=0, w_trend=0)),
        ("W1-B: 5d mom + trend filter", W1Config(w_mom5=1.0, w_mom20=0, w_mr1=0, w_vol_conf=0, w_breakout=0, w_trend=3)),
        ("W1-C: 5d mom + breakout", W1Config(w_mom5=1.0, w_mom20=0, w_mr1=0, w_vol_conf=0, w_breakout=5, w_trend=0)),
        ("W1-D: full multi-factor (defaults)", W1Config()),
        ("W1-E: heavy mean-rev (5d)", W1Config(w_mom5=-0.5, w_mr1=1.0, w_vol_conf=0, w_breakout=0, w_trend=0)),
        ("W1-F: 20d mom dominant", W1Config(w_mom5=0.3, w_mom20=1.0, w_breakout=2, w_trend=2)),
        ("W1-G: pure breakout system", W1Config(w_mom5=0, w_mom20=0, w_mr1=0, w_vol_conf=2, w_breakout=10, w_trend=2)),
        ("W1-H: combined pull-back", W1Config(w_mom5=0.5, w_mom20=0.5, w_mr1=0.3, w_vol_conf=1.0, w_breakout=2, w_trend=2, dip_bonus=5)),
    ]

    for label, c in configs:
        r = run(cp, vp, c, start_date=START, quality_set=quality_set)
        marker = " *" if r['cagr'] > bench['cagr'] and r['sharpe'] > bench['sharpe'] else "  "
        print(f"{label:<42}{r['cagr']:>+8.1f}%{r['vol']:>+7.1f}%{r['sharpe']:>8.2f}"
              f"{r['max_dd']:>+8.1f}%{r['calmar']:>8.2f}{r['hit_rate']:>+6.1f}%{marker}")

    # 2. Random search to find best config
    print("\n" + "=" * 110)
    print("STAGE 2 — Random search (60 configs) — sorted by Sharpe")
    print("=" * 110)

    grid = {
        "w_mom5":      [0.3, 0.5, 0.8, 1.0, 1.5],
        "w_mom20":     [0.0, 0.2, 0.5, 0.8],
        "w_mr1":       [0.0, 0.3, 0.5, 1.0],
        "w_vol_conf":  [0.0, 1.0, 2.0, 3.0],
        "w_breakout":  [0.0, 3.0, 5.0, 8.0],
        "w_trend":     [0.0, 2.0, 3.0, 5.0],
        "stretched_5d_thresh": [10, 15, 20],
        "stretched_5d_slope": [0.3, 0.5, 0.8],
        "dip_bonus":   [0, 3, 5],
        "top_n":       [3, 5, 7, 10],
    }
    rng = random.Random(42)
    candidates = [W1Config(**{k: rng.choice(v) for k, v in grid.items()}) for _ in range(60)]
    results = []
    t0 = time.time()
    for i, c in enumerate(candidates):
        r = run(cp, vp, c, start_date=START, quality_set=quality_set)
        h = run(cp, vp, c, start_date=HOLDOUT_START, quality_set=quality_set)
        results.append({"cfg": c, "full": r, "hold": h})
        if (i+1) % 15 == 0:
            print(f"  {i+1}/60 ({time.time()-t0:.0f}s)")

    qualified = [x for x in results
                 if x["full"]["sharpe"] > 0.4 and x["full"]["max_dd"] > -25
                 and x["hold"]["sharpe"] > 0.4 and x["hold"]["max_dd"] > -15]
    qualified.sort(key=lambda x: x["hold"]["sharpe"] + x["full"]["sharpe"], reverse=True)

    print(f"\n  Qualified configs (Sharpe>0.4 in both windows): {len(qualified)}/{len(results)}\n")
    print(f"  {'Rk':<3}{'Full CAGR':>11}{'Full Sh':>9}{'Full DD':>9}"
          f"{'Hold CAGR':>11}{'Hold Sh':>9}{'Hold DD':>9}"
          f"{'mom5':>6}{'mom20':>7}{'mr1':>6}{'vol':>6}{'brk':>6}{'trd':>6}{'topN':>6}")
    print("  " + "-" * 95)
    for i, x in enumerate(qualified[:8], 1):
        c = x["cfg"]; f = x["full"]; h = x["hold"]
        print(f"  {i:<3}{f['cagr']:>+10.1f}%{f['sharpe']:>9.2f}{f['max_dd']:>+8.1f}%"
              f"{h['cagr']:>+10.1f}%{h['sharpe']:>9.2f}{h['max_dd']:>+8.1f}%"
              f"{c.w_mom5:>6.1f}{c.w_mom20:>7.1f}{c.w_mr1:>6.1f}{c.w_vol_conf:>6.1f}"
              f"{c.w_breakout:>6.1f}{c.w_trend:>6.1f}{c.top_n:>6}")

    # 3. Final comparison: best config vs market on all windows
    if qualified:
        print("\n" + "=" * 110)
        print("STAGE 3 — Best W1 config vs Market across multiple windows")
        print("=" * 110)
        best = qualified[0]["cfg"]
        windows = [
            ("Full window (28mo)",  START),
            ("2024 only",            "2024-01-01"),
            ("2025 only",            "2025-01-01"),
            ("Last 12mo",            "2025-04-30"),
            ("Holdout 10mo OOS",    "2025-07-01"),
            ("Last 6mo",             "2025-10-30"),
        ]
        end_dates = {
            "Full window (28mo)":  END,
            "2024 only":            "2024-12-31",
            "2025 only":            "2025-12-31",
            "Last 12mo":            END,
            "Holdout 10mo OOS":    END,
            "Last 6mo":             END,
        }
        print(f"\n  {'Window':<22}{'W1 CAGR':>11}{'Mkt CAGR':>11}{'Alpha':>10}"
              f"{'W1 Sh':>8}{'Mkt Sh':>8}{'W1 DD':>9}{'Mkt DD':>9}{'W1 Hit%':>9}")
        print("  " + "-" * 99)
        for label, start in windows:
            end = end_dates[label]
            w1 = run(cp.loc[:end], vp.loc[:end], best, start_date=start, quality_set=quality_set)
            mkt = buy_hold(cp.loc[:end], start_date=start, rebal_days=5)
            if not w1 or not mkt:
                continue
            alpha = w1["cagr"] - mkt["cagr"]
            print(f"  {label:<22}{w1['cagr']:>+10.1f}%{mkt['cagr']:>+10.1f}%"
                  f"{alpha:>+9.1f}pp{w1['sharpe']:>8.2f}{mkt['sharpe']:>8.2f}"
                  f"{w1['max_dd']:>+8.1f}%{mkt['max_dd']:>+8.1f}%{w1['hit_rate']:>+8.1f}%")
        print(f"\n  Winning config:")
        c = best
        print(f"    w_mom5={c.w_mom5}, w_mom20={c.w_mom20}, w_mr1={c.w_mr1}, "
              f"w_vol_conf={c.w_vol_conf}, w_breakout={c.w_breakout}, w_trend={c.w_trend}")
        print(f"    stretched_5d>{c.stretched_5d_thresh} slope={c.stretched_5d_slope}, "
              f"dip_bonus={c.dip_bonus}, top_n={c.top_n}")
    print()


if __name__ == "__main__":
    main()
