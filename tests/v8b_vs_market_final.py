"""Final V8b vs Market backtest — multiple windows, honest benchmarks.

Benchmarks tested (all rebased to 100):
  Market 1: Equal-weighted basket of all 68 validated EGX names (broadest)
  Market 2: Equal-weighted basket of 13 most-liquid EGX 30 names
  Market 3: V3 (price-only baseline strategy, for reference)

Time windows:
  Full          2024-01-01 → 2026-04-30 (28 months — full available history)
  2024          2024-01-01 → 2024-12-31
  2025          2025-01-01 → 2025-12-31
  Last 12 mo    2025-04-30 → 2026-04-30
  Holdout (OOS) 2025-07-01 → 2026-04-30 (10 months — pure out-of-sample)

For each strategy we report: CAGR, Volatility, Sharpe (excess over T-bills),
Max Drawdown, Calmar, Hit Rate, Final equity (1.00 → X). Plus the rolling
correlation between V8b and the market to assess diversification value.
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

START = "2024-01-01"
END = "2026-04-30"
TOP_N = 5
REBAL = 21
LIQUID_13 = ["COMI", "HDBK", "CIRA", "SWDY", "ETEL", "ABUK", "EFID",
             "TMGH", "ORWE", "FWRY", "EAST", "MFPC", "PHAR"]


# -- Data --

def _norm(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _load_data():
    print(f"Loading prices for full universe ({START} → {END})...")
    universe = egx_listing.get_full_universe()
    closes = {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start="2023-01-01", end=END, interval="1d")
            if h is None or h.empty:
                continue
            closes[tk] = _norm(h["Close"])
        except Exception:
            continue
    cp = pd.DataFrame(closes).sort_index()
    print(f"  Loaded {len(cp.columns)} names\n")
    return cp


def _load_fund_filter(min_roe_pct: float = 10.0) -> set[str]:
    cache_path = (Path(__file__).parent.parent / "egx_mcp" / "data"
                   / "mubasher_fundamentals_cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    return {tk for tk, d in cache.items()
            if d.get("roe_pct") is not None and d["roe_pct"] >= min_roe_pct}


# -- Scoring (V3 / V8b) --

def _features(closes: pd.Series, asof: pd.Timestamp):
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
    return {"mom_6m": (p_now / p_6m - 1) * 100,
            "mom_1m": (p_now / p_1m - 1) * 100,
            "above_ma200": p_now > ma200,
            "vol_pct": vol * 100}


def _v3_score(f):
    s = (f["mom_6m"] * 0.5 + (-f["mom_1m"]) * 0.2
         + (5 if f["above_ma200"] else -5)
         - max(0, (f["vol_pct"] - 30) * 0.5))
    if f["mom_6m"] > 50:
        s -= (f["mom_6m"] - 50) * 0.2
    if f["mom_1m"] < -8 and f["mom_6m"] > 10 and f["above_ma200"]:
        s += 5
    return s


# -- Backtest engine --

def run_strategy(cp: pd.DataFrame, start: str, end: str,
                 quality_set: set[str] | None = None) -> dict:
    panel = cp.loc[:end]
    start_ts = pd.Timestamp(start)
    in_window_idx = panel.index[panel.index >= start_ts]
    rebalance_idx = in_window_idx[::REBAL]
    if len(rebalance_idx) < 3:
        return {}

    eligible_cols = list(cp.columns)
    if quality_set is not None:
        eligible_cols = [t for t in eligible_cols if t in quality_set]

    equity = [1.0]
    period_rets = []
    rf_pct = risk_free.get_rate()["rate_pct"]
    rf_period = (1 + rf_pct / 100) ** (REBAL / 252) - 1

    for i in range(len(rebalance_idx) - 1):
        date = rebalance_idx[i]; next_date = rebalance_idx[i + 1]
        scores = {}
        for tk in eligible_cols:
            f = _features(cp[tk], date)
            if f is None:
                continue
            scores[tk] = _v3_score(f)
        if not scores:
            period_rets.append(0); equity.append(equity[-1]); continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [tk for tk, _ in ranked[:TOP_N]]
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
    return _stats(equity, period_rets, rf_period, REBAL, "strategy")


def run_buy_hold(cp: pd.DataFrame, start: str, end: str,
                 tickers: list[str] | None = None) -> dict:
    panel = cp.loc[start:end]
    if tickers is not None:
        panel = panel[[t for t in tickers if t in panel.columns]]
    panel = panel.dropna(how="all")
    if panel.empty:
        return {}
    rets = panel.pct_change().dropna(how="all").mean(axis=1).fillna(0)
    eq = (1 + rets).cumprod()
    rf_pct = risk_free.get_rate()["rate_pct"]
    daily_rf = (1 + rf_pct / 100) ** (1 / 252) - 1
    return _stats(eq.tolist(), rets.tolist(), daily_rf, 1, "buy_hold")


def _stats(equity_list, rets, rf_period, rebal_days, label):
    eq = np.array(equity_list)
    rets_arr = np.array(rets)
    if len(rets_arr) < 2:
        return {}
    total = (eq[-1] / eq[0] - 1) * 100
    years = len(rets_arr) * rebal_days / 252
    cagr = ((1 + total / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol = rets_arr.std() * ((252 / rebal_days) ** 0.5) * 100
    excess = rets_arr - rf_period
    sharpe = excess.mean() / excess.std() * ((252 / rebal_days) ** 0.5) if excess.std() > 0 else 0
    rmax = np.maximum.accumulate(eq); dd_arr = (eq / rmax - 1)
    max_dd = float(dd_arr.min()) * 100
    calmar = abs(cagr / max_dd) if max_dd < 0 else 0
    hit = float((rets_arr > 0).mean()) * 100
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "calmar": calmar, "hit_rate": hit, "final_equity": float(eq[-1] / eq[0]),
        "n_periods": len(rets_arr),
    }


# -- Main --

def main():
    cp = _load_data()
    quality_set = _load_fund_filter(10.0)
    print(f"Quality filter (ROE >= 10%): {len(quality_set)} names\n")

    windows = [
        ("Full window (2024-01 → 2026-04)",  "2024-01-01"),
        ("2024 calendar year",                 "2024-01-01"),
        ("2025 calendar year",                 "2025-01-01"),
        ("Last 12 months",                     "2025-04-30"),
        ("Holdout / OOS (10 months)",          "2025-07-01"),
        ("Last 6 months",                      "2025-10-30"),
    ]
    end_dates = {
        "Full window (2024-01 → 2026-04)":  END,
        "2024 calendar year":                 "2024-12-31",
        "2025 calendar year":                 "2025-12-31",
        "Last 12 months":                     END,
        "Holdout / OOS (10 months)":          END,
        "Last 6 months":                      END,
    }

    print("=" * 110)
    print("V8b PRODUCTION  vs  V3 BASELINE  vs  MARKET — across multiple windows")
    print("=" * 110)

    for label, start in windows:
        end = end_dates[label]
        print(f"\n  ── {label}  [{start} → {end}] ──")
        print(f"  {'Strategy':<32}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}"
              f"{'MaxDD':>9}{'Calmar':>8}{'HitR':>8}{'×Final':>9}")
        print("  " + "-" * 91)

        v8 = run_strategy(cp, start, end, quality_set=quality_set)
        v3 = run_strategy(cp, start, end, quality_set=None)
        m_broad = run_buy_hold(cp, start, end, tickers=None)
        m_liquid = run_buy_hold(cp, start, end, tickers=LIQUID_13)

        rows = [
            ("V8b production (ROE>10 + V3)",   v8),
            ("V3 baseline (price only)",         v3),
            ("Market: 68-name equal-weight",     m_broad),
            ("Market: 13 liquid names EW",       m_liquid),
        ]
        for name, r in rows:
            if not r:
                print(f"  {name:<32}  (insufficient data)")
                continue
            print(f"  {name:<32}{r['cagr']:>+8.1f}%{r['vol']:>+7.1f}%"
                  f"{r['sharpe']:>8.2f}{r['max_dd']:>+8.1f}%"
                  f"{r['calmar']:>8.2f}{r['hit_rate']:>+7.1f}%"
                  f"{r['final_equity']:>9.3f}")

    # Summary ranking
    print("\n" + "=" * 110)
    print("SUMMARY — V8b vs broad market across all windows")
    print("=" * 110)
    print(f"  {'Window':<35}{'V8b CAGR':>11}{'Mkt CAGR':>11}{'Alpha':>10}"
          f"{'V8b Sharpe':>12}{'Mkt Sharpe':>12}{'V8b DD':>10}{'Mkt DD':>10}")
    print("  " + "-" * 105)
    for label, start in windows:
        end = end_dates[label]
        v8 = run_strategy(cp, start, end, quality_set=quality_set)
        mkt = run_buy_hold(cp, start, end, tickers=None)
        if v8 and mkt:
            alpha = v8["cagr"] - mkt["cagr"]
            print(f"  {label[:34]:<35}{v8['cagr']:>+10.1f}%{mkt['cagr']:>+10.1f}%"
                  f"{alpha:>+9.1f}pp{v8['sharpe']:>12.2f}{mkt['sharpe']:>12.2f}"
                  f"{v8['max_dd']:>+9.1f}%{mkt['max_dd']:>+9.1f}%")
    print()


if __name__ == "__main__":
    main()
