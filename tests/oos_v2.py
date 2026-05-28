"""V1 vs V2 head-to-head — same out-of-sample week, two scoring rubrics.

V1 (current production):
    0.5 × mom_6m + 0.2 × mr_1m + trend - vol_pen

V2 (mean-reversion-aware):
    0.3 × mom_6m + 0.2 × mr_1m + 0.3 × mr_5d + trend - vol_pen
    - stretched_penalty   # if 6m > 50, subtract (6m - 50) × 0.2
    + dip_in_uptrend      # +5 if 1m < -8 AND 6m > 10 AND above MA200

Same cutoff, same window, same universe — only the scoring formula differs.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import egx_listing, risk_free
from egx_mcp.data.universe import resolve_ticker

CUTOFF = "2026-04-23"
WINDOW_END = "2026-04-30"
TOP_N = 5

print("\n" + "=" * 80)
print("V1 vs V2 — out-of-sample head-to-head")
print(f"  Cutoff: {CUTOFF}   Window: through {WINDOW_END}")
print("=" * 80 + "\n")


def _fetch(symbol: str) -> pd.Series:
    try:
        h = yf.Ticker(symbol).history(start="2025-01-01", end="2026-05-01", interval="1d")
        if h is None or h.empty:
            return pd.Series(dtype=float)
        s = h["Close"].copy()
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = pd.to_datetime(idx.date)
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return pd.Series(dtype=float)


def _features(closes: pd.Series, cutoff: pd.Timestamp) -> dict | None:
    sub = closes.loc[:cutoff].dropna()
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
        "p_now": p_now,
        "mom_6m": (p_now / p_6m - 1) * 100,
        "mom_1m": (p_now / p_1m - 1) * 100,
        "mom_5d": (p_now / p_5d - 1) * 100,
        "above_ma200": p_now > ma200,
        "vol_pct": vol * 100,
    }


def _v1(f: dict) -> float:
    """Original price-only score."""
    mr_1m = -f["mom_1m"]
    trend = 5 if f["above_ma200"] else -5
    vol_pen = max(0, (f["vol_pct"] - 30) * 0.5)
    return f["mom_6m"] * 0.5 + mr_1m * 0.2 + trend - vol_pen


def _v2(f: dict) -> float:
    """V2 — mean-reversion-aware score."""
    mr_1m = -f["mom_1m"]
    mr_5d = -f["mom_5d"]
    trend = 5 if f["above_ma200"] else -5
    vol_pen = max(0, (f["vol_pct"] - 30) * 0.5)

    base = f["mom_6m"] * 0.3 + mr_1m * 0.2 + mr_5d * 0.3 + trend - vol_pen

    # Stretched penalty — fade names already up too much over 6m
    if f["mom_6m"] > 50:
        base -= (f["mom_6m"] - 50) * 0.2

    # Dip-in-uptrend bonus — buy beaten-down names whose long-term trend is intact
    if f["mom_1m"] < -8 and f["mom_6m"] > 10 and f["above_ma200"]:
        base += 5

    return base


# Pull all data once
print("Loading prices...")
universe = egx_listing.get_full_universe()
data = {}
for tk in universe:
    _, yahoo, _ = resolve_ticker(tk)
    closes = _fetch(yahoo)
    if not closes.empty:
        data[tk] = closes

cutoff_ts = pd.Timestamp(CUTOFF)
end_ts = pd.Timestamp(WINDOW_END)

rows = []
for tk, closes in data.items():
    f = _features(closes, cutoff_ts)
    if f is None:
        continue
    pre = closes.loc[:cutoff_ts].dropna()
    end = closes.loc[:end_ts].dropna()
    if pre.empty or end.empty or pre.index[-1] >= end.index[-1]:
        continue
    p0 = float(pre.iloc[-1])
    p1 = float(end.iloc[-1])
    actual = (p1 / p0 - 1) * 100
    rows.append({
        "ticker": tk,
        "v1": _v1(f),
        "v2": _v2(f),
        "actual": actual,
        "mom_6m": f["mom_6m"],
        "mom_1m": f["mom_1m"],
        "mom_5d": f["mom_5d"],
    })

print(f"Names in scope: {len(rows)}\n")

basket = sum(r["actual"] for r in rows) / len(rows)
rf_week = ((1 + risk_free.get_rate()["rate_pct"] / 100) ** (5 / 252) - 1) * 100


def report(label, key):
    ranked = sorted(rows, key=lambda r: r[key], reverse=True)
    top = ranked[:TOP_N]
    bot = ranked[-TOP_N:]
    top_ret = sum(r["actual"] for r in top) / len(top)
    bot_ret = sum(r["actual"] for r in bot) / len(bot)

    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  Top {TOP_N} by score:")
    print(f"  {'Tic':<7}{'Score':<8}{'mom_6m':<10}{'mom_1m':<10}{'mom_5d':<10}{'actual':<9}")
    for r in top:
        flag = "✓" if r["actual"] > basket else " "
        print(f"  {r['ticker']:<7}{r[key]:<8.1f}"
              f"{r['mom_6m']:>+7.1f}%  {r['mom_1m']:>+7.1f}%  {r['mom_5d']:>+7.1f}%  "
              f"{r['actual']:>+6.2f}%  {flag}")
    print(f"\n  Bottom {TOP_N} (model said AVOID):")
    for r in bot:
        print(f"  {r['ticker']:<7}{r[key]:<8.1f}"
              f"{r['mom_6m']:>+7.1f}%  {r['mom_1m']:>+7.1f}%  {r['mom_5d']:>+7.1f}%  "
              f"{r['actual']:>+6.2f}%")
    print(f"\n  Top-5 portfolio return:    {top_ret:+.2f}%")
    print(f"  Bottom-5 actual return:    {bot_ret:+.2f}%")
    print(f"  Long-top / short-bottom:   {top_ret - bot_ret:+.2f} pp")
    print(f"  Active vs basket ({basket:+.2f}%): {top_ret - basket:+.2f} pp")
    print(f"  Active vs T-bills ({rf_week:+.2f}%): {top_ret - rf_week:+.2f} pp")
    return {"top": top_ret, "bot": bot_ret, "spread": top_ret - bot_ret,
            "vs_basket": top_ret - basket, "vs_rf": top_ret - rf_week}


v1 = report("V1 — current production score", "v1")
print()
v2 = report("V2 — mean-reversion-aware score", "v2")

print("\n" + "=" * 70)
print("  HEAD-TO-HEAD")
print("=" * 70)
print(f"  {'Metric':<35}{'V1':>9}{'V2':>9}{'Δ':>9}")
print(f"  {'Top-5 return':<35}{v1['top']:>+8.2f}%{v2['top']:>+8.2f}%{v2['top']-v1['top']:>+8.2f}pp")
print(f"  {'Bottom-5 return':<35}{v1['bot']:>+8.2f}%{v2['bot']:>+8.2f}%{v2['bot']-v1['bot']:>+8.2f}pp")
print(f"  {'Long-top / short-bottom spread':<35}{v1['spread']:>+8.2f}{v2['spread']:>+8.2f}{v2['spread']-v1['spread']:>+8.2f}pp")
print(f"  {'Alpha vs basket':<35}{v1['vs_basket']:>+8.2f}{v2['vs_basket']:>+8.2f}{v2['vs_basket']-v1['vs_basket']:>+8.2f}pp")
print(f"  {'Alpha vs T-bills':<35}{v1['vs_rf']:>+8.2f}{v2['vs_rf']:>+8.2f}{v2['vs_rf']-v1['vs_rf']:>+8.2f}pp")
print()
