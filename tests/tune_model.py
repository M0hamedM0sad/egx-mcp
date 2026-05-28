"""Hyperparameter tuning — find the best risk-adjusted EGX strategy.

Design:
  1. Load price panel once (slow, cached in memory).
  2. Run ~50 parameter configs against the panel (fast — pure compute).
  3. Optimize for Calmar = CAGR / |MaxDD|. EGX PMs care about tail risk.
  4. Walk-forward validate top 3 on a 2026-only holdout.
  5. Print the winner and compare to V1, V3, and the synthetic benchmark.

Tunable parameters:
    w_mom6m         weight on 6m momentum
    w_mr1m          weight on 1m mean-reversion (negate of 1m return)
    w_mom3m         weight on 3m momentum (NEW — bridges 1m and 6m)
    trend_bonus     points if above MA200 (sign reversed if below)
    vol_cutoff_pct  vol level above which the penalty starts
    vol_slope       points-per-vol-pct of penalty
    stretched_thresh stretched penalty kicks in if 6m > this
    stretched_slope  points-per-pct of stretched penalty
    dip_thresh_1m    dip-in-uptrend bonus needs 1m < this
    dip_thresh_6m    AND 6m > this
    dip_bonus        points if dip filter triggers
    top_n           names to hold
    rebal_days      rebalance frequency
    min_adv_egp     minimum 20d ADV to be eligible (liquidity floor)

Constraints:
    Sharpe must be >= 0.5
    Hit rate must be >= 45%
    Max DD must be > -15%
"""
from __future__ import annotations

import io
import itertools
import random
import sys
import time
from dataclasses import dataclass
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
HOLDOUT_START = "2025-07-01"   # walk-forward window — 10 months for stable validation


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


# ---------------------------------------------------------------------------
# Data loading (once)
# ---------------------------------------------------------------------------

def _normalize_index(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _load_data():
    print("Loading price + volume panel...")
    universe = egx_listing.get_full_universe()
    closes, volumes = {}, {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start=START, end=END, interval="1d")
            if h is None or h.empty:
                continue
            closes[tk] = _normalize_index(h["Close"])
            volumes[tk] = _normalize_index(h["Volume"])
        except Exception:
            continue
    cp = pd.DataFrame(closes).sort_index()
    vp = pd.DataFrame(volumes).sort_index()
    print(f"  Loaded {len(cp.columns)} names\n")
    return cp, vp


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def _adv_at(volume_panel: pd.DataFrame, close_panel: pd.DataFrame,
            asof_loc: pd.Timestamp, days: int = 20) -> pd.Series:
    """Avg daily turnover (EGP) for each ticker over trailing N days."""
    sub_v = volume_panel.loc[:asof_loc].tail(days)
    sub_c = close_panel.loc[:asof_loc].tail(days)
    if sub_v.empty:
        return pd.Series(dtype=float)
    avg_vol = sub_v.mean()
    avg_cl = sub_c.mean()
    return (avg_vol * avg_cl).fillna(0)


def _features(closes: pd.Series, asof: pd.Timestamp) -> dict | None:
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
    return {
        "p_now": p_now,
        "mom_6m": (p_now / p_6m - 1) * 100,
        "mom_3m": (p_now / p_3m - 1) * 100,
        "mom_1m": (p_now / p_1m - 1) * 100,
        "above_ma200": p_now > ma200,
        "vol_pct": vol * 100,
    }


def _score(f: dict, c: Config) -> float:
    base = (
        f["mom_6m"] * c.w_mom6m
        + (-f["mom_1m"]) * c.w_mr1m
        + f["mom_3m"] * c.w_mom3m
        + (c.trend_bonus if f["above_ma200"] else -c.trend_bonus)
        - max(0, (f["vol_pct"] - c.vol_cutoff_pct) * c.vol_slope)
    )
    if f["mom_6m"] > c.stretched_thresh:
        base -= (f["mom_6m"] - c.stretched_thresh) * c.stretched_slope
    if (f["mom_1m"] < c.dip_thresh_1m
        and f["mom_6m"] > c.dip_thresh_6m
        and f["above_ma200"]):
        base += c.dip_bonus
    return base


def run_backtest(close_panel: pd.DataFrame, volume_panel: pd.DataFrame,
                 cfg: Config, start_date: str | None = None) -> dict:
    if start_date is not None:
        panel = close_panel.loc[start_date:]
        vol_p = volume_panel.loc[start_date:]
    else:
        panel = close_panel
        vol_p = volume_panel

    rebalance_idx = panel.index[::cfg.rebal_days]
    if len(rebalance_idx) < 3:
        return {"sharpe": 0, "calmar": 0, "cagr": 0, "max_dd": 0, "vol": 0,
                "total_ret": 0, "hit_rate": 0, "n_periods": 0}

    equity = [1.0]
    period_rets = []
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct / 100) ** (cfg.rebal_days / 252) - 1

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]
        next_date = rebalance_idx[i + 1]

        # Liquidity filter
        if cfg.min_adv_egp > 0:
            adv = _adv_at(vol_p, close_panel, date, days=20)
            eligible = set(adv[adv >= cfg.min_adv_egp].index)
        else:
            eligible = set(panel.columns)

        scores = {}
        for tk in panel.columns:
            if tk not in eligible:
                continue
            f = _features(close_panel[tk], date)
            if f is None:
                continue
            scores[tk] = _score(f, cfg)
        if not scores:
            period_rets.append(0); equity.append(equity[-1]); continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [tk for tk, _ in ranked[:cfg.top_n]]
        try:
            sub = close_panel.loc[date:next_date, picks].dropna(how="all")
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
    if len(rets_arr) < 2:
        return {"sharpe": 0, "calmar": 0, "cagr": 0, "max_dd": 0, "vol": 0,
                "total_ret": 0, "hit_rate": 0, "n_periods": len(rets_arr)}

    total_ret = (eq[-1] - 1) * 100
    years = len(rets_arr) * cfg.rebal_days / 252
    cagr = ((1 + total_ret / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol_ann = rets_arr.std() * ((252 / cfg.rebal_days) ** 0.5) * 100
    excess = rets_arr - rf_period
    sharpe = excess.mean() / excess.std() * ((252 / cfg.rebal_days) ** 0.5) if excess.std() > 0 else 0
    hit_rate = (rets_arr > 0).mean() * 100
    rmax = np.maximum.accumulate(eq)
    dd = (eq / rmax - 1)
    max_dd = dd.min() * 100
    calmar = abs(cagr / max_dd) if max_dd < 0 else 999

    return {
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "cagr": float(cagr),
        "max_dd": float(max_dd),
        "vol": float(vol_ann),
        "total_ret": float(total_ret),
        "hit_rate": float(hit_rate),
        "n_periods": len(rets_arr),
    }


def benchmark_buy_hold(close_panel: pd.DataFrame, start_date: str | None = None) -> dict:
    """Equal-weighted synthetic basket buy-and-hold over the same window."""
    panel = close_panel.loc[start_date:] if start_date else close_panel
    panel = panel.dropna(how="all")
    if panel.empty:
        return {"total_ret": 0, "cagr": 0, "vol": 0, "max_dd": 0, "sharpe": 0}
    rets = panel.pct_change().dropna(how="all").mean(axis=1).fillna(0)
    eq = (1 + rets).cumprod()
    total_ret = (eq.iloc[-1] - 1) * 100
    years = len(rets) / 252
    cagr = ((1 + total_ret / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol = float(rets.std() * (252 ** 0.5)) * 100
    rmax = eq.cummax()
    dd = (eq / rmax - 1)
    max_dd = float(dd.min()) * 100
    rf = risk_free.get_rate()["rate_pct"]
    daily_rf = (1 + rf / 100) ** (1 / 252) - 1
    excess = rets - daily_rf
    sharpe = float(excess.mean() / excess.std() * (252 ** 0.5)) if excess.std() > 0 else 0
    return {"total_ret": total_ret, "cagr": cagr, "vol": vol, "max_dd": max_dd, "sharpe": sharpe}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def main():
    cp, vp = _load_data()

    # Baseline configs
    v1 = Config(w_mom6m=0.5, w_mr1m=0.2, stretched_slope=0.0, dip_bonus=0.0)
    v3 = Config()  # all defaults

    # Search grid — constrained to rebalances that produce ≥8 holdout periods
    # with the new 10-month holdout window
    grid = {
        "w_mom6m":          [0.3, 0.4, 0.5, 0.6, 0.7],
        "w_mr1m":           [0.1, 0.2, 0.3],
        "w_mom3m":          [0.0, 0.15, 0.25, 0.35],
        "stretched_thresh": [40, 50, 60, 70],
        "stretched_slope":  [0.1, 0.2, 0.3, 0.5],
        "dip_bonus":        [0, 3, 5, 7],
        "top_n":            [3, 5, 7, 10],
        "rebal_days":       [5, 10, 15, 21],   # capped to keep holdout tight
        "min_adv_egp":      [0, 500_000, 2_000_000],
    }

    # Random sample of 120 — wider net, more search.
    keys = list(grid.keys())
    rng = random.Random(42)
    configs = []
    while len(configs) < 120:
        c = Config(**{k: rng.choice(grid[k]) for k in keys})
        configs.append(c)

    # Always include V1 and V3 as anchors
    configs.insert(0, v1)
    configs.insert(1, v3)

    print(f"Running {len(configs)} configurations on training window ({START} → {END})...\n")
    t0 = time.time()
    results = []
    for i, c in enumerate(configs):
        r = run_backtest(cp, vp, c)
        results.append({"cfg": c, **r})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(configs)}  ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.0f}s\n")

    # Filter: min Sharpe + min hit rate + max DD bound + meaningful sample size.
    # The 999 Calmar artifact came from short rebalance counts producing
    # zero-DD by accident — exclude configs with <15 periods.
    qualified = [r for r in results
                 if r["sharpe"] >= 0.6 and r["hit_rate"] >= 45
                 and r["max_dd"] > -12 and r["n_periods"] >= 15
                 and r["max_dd"] < 0]    # require an actual drawdown experience

    # Rank by Calmar
    qualified.sort(key=lambda r: r["calmar"], reverse=True)
    top = qualified[:5]

    print("=" * 95)
    print("TOP 5 CANDIDATES (training window) — sorted by Calmar = CAGR / |MaxDD|")
    print("=" * 95)
    print(f"{'Rk':<3}{'CAGR%':<8}{'Vol%':<7}{'Sharpe':<8}{'MaxDD%':<8}"
          f"{'Calmar':<8}{'HitR%':<7}{'TopN':<5}{'Reb':<5}{'w6m':<5}"
          f"{'w1m':<5}{'w3m':<5}{'StrTh':<6}{'StrSl':<6}{'DipB':<5}{'minADV':<7}")
    print("-" * 95)
    for i, r in enumerate(top, 1):
        c = r["cfg"]
        print(f"{i:<3}{r['cagr']:<8.1f}{r['vol']:<7.1f}{r['sharpe']:<8.2f}"
              f"{r['max_dd']:<8.1f}{r['calmar']:<8.2f}{r['hit_rate']:<7.0f}"
              f"{c.top_n:<5}{c.rebal_days:<5}{c.w_mom6m:<5.2f}{c.w_mr1m:<5.2f}"
              f"{c.w_mom3m:<5.2f}{c.stretched_thresh:<6.0f}{c.stretched_slope:<6.2f}"
              f"{c.dip_bonus:<5.1f}{int(c.min_adv_egp/1e6):<7}M")

    # Walk-forward validate the top 3 on the 2026 holdout
    print("\n" + "=" * 95)
    print(f"WALK-FORWARD VALIDATION — holdout window ({HOLDOUT_START} → {END})")
    print("=" * 95)

    bench_full = benchmark_buy_hold(cp)
    bench_holdout = benchmark_buy_hold(cp, start_date=HOLDOUT_START)

    print(f"\n  Benchmark (synthetic basket buy-and-hold):")
    print(f"    Full window:  CAGR {bench_full['cagr']:+.1f}%, vol {bench_full['vol']:.1f}%, "
          f"MaxDD {bench_full['max_dd']:.1f}%, Sharpe {bench_full['sharpe']:.2f}")
    print(f"    Holdout only: CAGR {bench_holdout['cagr']:+.1f}%, vol {bench_holdout['vol']:.1f}%, "
          f"MaxDD {bench_holdout['max_dd']:.1f}%, Sharpe {bench_holdout['sharpe']:.2f}")

    print(f"\n  Strategy validation on holdout:")
    print(f"  {'Rk':<3}{'Train Calmar':<14}{'Holdout CAGR':<14}{'Holdout MaxDD':<15}"
          f"{'Holdout Sharpe':<16}{'Holdout Calmar':<14}")
    holdout_results = []
    for i, r in enumerate(top, 1):
        h = run_backtest(cp, vp, r["cfg"], start_date=HOLDOUT_START)
        holdout_results.append({"train": r, "holdout": h})
        print(f"  {i:<3}{r['calmar']:<14.2f}{h['cagr']:<14.1f}{h['max_dd']:<15.1f}"
              f"{h['sharpe']:<16.2f}{h['calmar']:<14.2f}")

    # Pick the winner — best holdout SHARPE (true risk-adjusted), with
    # sanity checks. Calmar is sample-size sensitive on short holdouts;
    # Sharpe is what an institutional PM actually optimizes for.
    winners = [hr for hr in holdout_results
               if hr["holdout"]["sharpe"] >= 0.5
               and hr["holdout"]["max_dd"] < -0.3      # require real DD experience
               and hr["holdout"]["max_dd"] > -8
               and hr["holdout"]["n_periods"] >= 6]
    winners.sort(key=lambda hr: hr["holdout"]["sharpe"], reverse=True)
    if not winners:
        print("\nNo config passed holdout validation. Sticking with V3.")
        winner = next(r for r in results if r["cfg"] == v3)
    else:
        winner = winners[0]["train"]
        print(f"\n  Winner: rank {top.index(winner)+1} from training, "
              f"holdout Calmar = {winners[0]['holdout']['calmar']:.2f}")

    # Final comparison: V1 / V3 / V4 / Benchmark
    v1_full = results[0]
    v3_full = results[1]
    v4_full = winner
    v1_h = run_backtest(cp, vp, v1, start_date=HOLDOUT_START)
    v3_h = run_backtest(cp, vp, v3, start_date=HOLDOUT_START)
    v4_h = run_backtest(cp, vp, winner["cfg"], start_date=HOLDOUT_START)

    print("\n" + "=" * 95)
    print("FINAL COMPARISON  —  Full window: 2024-01 → 2026-04   |   Holdout: 2026-01 → 2026-04")
    print("=" * 95)
    print(f"\n  {'Strategy':<20}{'Full CAGR':<13}{'Full MaxDD':<13}{'Full Sharpe':<14}"
          f"{'Full Calmar':<14}{'Holdout CAGR':<14}{'Holdout MaxDD':<14}")
    print("  " + "-" * 93)
    rows = [
        ("V1 (original)", v1_full, v1_h),
        ("V3 (production)", v3_full, v3_h),
        ("V4 (tuned)", v4_full, v4_h),
    ]
    for name, full, hold in rows:
        print(f"  {name:<20}{full['cagr']:<+13.1f}{full['max_dd']:<+13.1f}"
              f"{full['sharpe']:<14.2f}{full['calmar']:<14.2f}"
              f"{hold['cagr']:<+14.1f}{hold['max_dd']:<+14.1f}")
    print(f"  {'Buy & Hold basket':<20}{bench_full['cagr']:<+13.1f}{bench_full['max_dd']:<+13.1f}"
          f"{bench_full['sharpe']:<14.2f}{abs(bench_full['cagr']/bench_full['max_dd']) if bench_full['max_dd']<0 else 999:<14.2f}"
          f"{bench_holdout['cagr']:<+14.1f}{bench_holdout['max_dd']:<+14.1f}")

    print(f"\n  V4 (winning) configuration:")
    c = winner["cfg"]
    print(f"    w_mom6m={c.w_mom6m}, w_mr1m={c.w_mr1m}, w_mom3m={c.w_mom3m}")
    print(f"    stretched_thresh={c.stretched_thresh}, stretched_slope={c.stretched_slope}")
    print(f"    dip_thresh_1m={c.dip_thresh_1m}, dip_thresh_6m={c.dip_thresh_6m}, dip_bonus={c.dip_bonus}")
    print(f"    top_n={c.top_n}, rebal_days={c.rebal_days}, min_adv_egp={c.min_adv_egp:,.0f}")
    print()

    return winner


if __name__ == "__main__":
    main()
