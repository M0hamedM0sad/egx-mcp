"""Day-trade backtest — does buying at open and selling at close on EGX work?

We have daily OHLC data, so "day trade" here = enter at OPEN of day T,
exit at CLOSE of day T. Different selection rules tested:

  S1  Top W1 score from prior close — ride yesterday's signal into today
  S2  Gap-and-go — buy names that gapped up >1% at open, ride to close
  S3  Gap-down reversal — buy names that gapped down >1%, bet on bounce
  S4  Volume-confirmed breakout — yesterday's volume spike + breakout, hold today
  S5  Pre-earnings drift — names approaching earnings, momentum runs into print
  S6  Overnight gap (different — close to next open, not intraday)

For each strategy we apply realistic transaction costs:
   commission 0.30% + stamp 0.05% + spread 0.50% + slippage 0.30%
   = 1.15% round-trip (slightly aggressive — typical mid-cap EGX)

We compare gross and net (after-cost) returns to:
   - Buy & hold the same names overnight + 1 day (5-day swing)
   - W1 weekly model (validated)
   - Broad market buy-hold

Output: full statistics + verdict on whether intraday strategies beat the
weekly model AFTER costs, for any window in the 28-month sample.
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
HOLDOUT_START = "2025-07-01"

# Realistic EGX round-trip cost (in %)
COST_LOW = 0.6     # large-cap, tight spread, limit orders
COST_MID = 1.15    # typical mid-cap, market orders
COST_HIGH = 2.0    # small-cap, wide spread, market orders


def _norm(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def load_ohlcv():
    print(f"Loading OHLCV panel ({START} → {END})...")
    universe = egx_listing.get_full_universe()
    O, H, L, C, V = {}, {}, {}, {}, {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start="2023-01-01", end=END, interval="1d")
            if h is None or h.empty:
                continue
            O[tk] = _norm(h["Open"])
            H[tk] = _norm(h["High"])
            L[tk] = _norm(h["Low"])
            C[tk] = _norm(h["Close"])
            V[tk] = _norm(h["Volume"])
        except Exception:
            continue
    return (pd.DataFrame(O).sort_index(),
            pd.DataFrame(H).sort_index(),
            pd.DataFrame(L).sort_index(),
            pd.DataFrame(C).sort_index(),
            pd.DataFrame(V).sort_index())


def load_quality_set():
    cache = (Path(__file__).parent.parent / "egx_mcp" / "data"
             / "mubasher_fundamentals_cache.json")
    if not cache.exists():
        return set()
    data = json.loads(cache.read_text(encoding="utf-8"))
    return {tk for tk, d in data.items()
            if d.get("roe_pct") is not None and d["roe_pct"] >= 10}


# -- Features (same as W1) computed from prior close --
def w1_score(closes, volumes, asof):
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
    ma20 = float(cl.tail(20).mean())
    vol = volumes.loc[:asof].dropna()
    today_vol = float(vol.iloc[-1]) if len(vol) >= 1 else 0
    avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else max(today_vol, 1)
    vol_ratio = today_vol / max(avg_vol, 1)
    breakout = 1.0 if p_now >= high_20 * 0.999 else (-1.0 if p_now <= low_20 * 1.001 else 0)
    mom_5d = (p_now / p_5d - 1) * 100
    mom_1d = (p_now / p_1d - 1) * 100
    above_ma20 = p_now > ma20
    s = (mom_5d * 1.5 + (-mom_1d) * 0.5 + 3 * breakout
         + (3 if above_ma20 else -3))
    if mom_5d > 10:
        s -= (mom_5d - 10) * 0.3
    if mom_5d < -5 and above_ma20:
        s += 5
    return {"score": s, "vol_ratio": vol_ratio, "mom_1d": mom_1d, "mom_5d": mom_5d,
            "breakout": breakout, "above_ma20": above_ma20}


# -- Day-trade backtest engines --

def bt_intraday(strategy, opens, closes, volumes, quality_set, start, end,
                top_n=5, cost_pct=COST_MID):
    """Generic intraday backtest: pick names by `strategy(asof)`, buy at next
    open, sell at next close, apply round-trip cost."""
    panel = closes.loc[start:end]
    dates = panel.index
    eligible_cols = [c for c in panel.columns if (not quality_set) or c in quality_set]
    daily_rets = []
    daily_winrates = []
    for i in range(1, len(dates) - 1):
        prior = dates[i]            # prior close — use this for signal
        today = dates[i + 1]        # buy at open of today, sell at close
        if today not in opens.index:
            continue
        try:
            picks = strategy(prior, opens, closes, volumes, eligible_cols, top_n)
        except Exception:
            continue
        if not picks:
            daily_rets.append(0); continue
        # Compute today's open-to-close return for each pick
        rets = []
        for tk in picks:
            try:
                op = float(opens.loc[today, tk])
                cp = float(closes.loc[today, tk])
                if op > 0 and cp > 0:
                    rets.append(cp / op - 1)
            except (KeyError, ValueError):
                continue
        if rets:
            avg = float(np.mean(rets))
            cost = cost_pct / 100  # round-trip
            net = avg - cost
            daily_rets.append(net)
            daily_winrates.append(1.0 if avg > cost else 0.0)
        else:
            daily_rets.append(0)

    eq = [1.0]
    for r in daily_rets:
        eq.append(eq[-1] * (1 + r))
    eq = np.array(eq); rets_arr = np.array(daily_rets)
    if len(rets_arr) < 5:
        return {}
    total = (eq[-1] - 1) * 100
    years = len(rets_arr) / 252
    cagr = ((1 + total / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol = rets_arr.std() * (252 ** 0.5) * 100
    rf = risk_free.get_rate()["rate_pct"] / 100
    daily_rf = (1 + rf) ** (1 / 252) - 1
    excess = rets_arr - daily_rf
    sharpe = excess.mean() / excess.std() * (252 ** 0.5) if excess.std() > 0 else 0
    rmax = np.maximum.accumulate(eq); dd = (eq / rmax - 1)
    max_dd = float(dd.min()) * 100
    hit = float((rets_arr > 0).mean()) * 100
    avg_daily = float(rets_arr.mean()) * 100
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "hit_rate": hit, "avg_daily_pct": avg_daily, "n_days": len(rets_arr),
        "total_return_pct": total, "final_equity": float(eq[-1]),
    }


# -- Selection strategies --

def s1_top_w1(prior, opens, closes, volumes, eligible, top_n):
    scores = {}
    for tk in eligible:
        if tk not in closes.columns or tk not in volumes.columns:
            continue
        s = w1_score(closes[tk], volumes[tk], prior)
        if s is not None:
            scores[tk] = s["score"]
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tk for tk, _ in ranked[:top_n]]


def s2_gap_and_go(prior, opens, closes, volumes, eligible, top_n, gap_min=0.01):
    """Buy names with overnight gap-up >1%, expecting continuation."""
    if prior not in closes.index:
        return []
    next_dates = closes.index[closes.index > prior]
    if len(next_dates) == 0:
        return []
    today = next_dates[0]
    candidates = []
    for tk in eligible:
        try:
            prior_close = float(closes.loc[prior, tk])
            today_open = float(opens.loc[today, tk])
            if prior_close > 0 and today_open > 0:
                gap = today_open / prior_close - 1
                if gap >= gap_min:
                    candidates.append((tk, gap))
        except (KeyError, ValueError):
            continue
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [tk for tk, _ in candidates[:top_n]]


def s3_gap_down_reversal(prior, opens, closes, volumes, eligible, top_n, gap_max=-0.01):
    """Buy names that gapped down >1%, betting on intraday recovery."""
    if prior not in closes.index:
        return []
    next_dates = closes.index[closes.index > prior]
    if len(next_dates) == 0:
        return []
    today = next_dates[0]
    candidates = []
    for tk in eligible:
        try:
            prior_close = float(closes.loc[prior, tk])
            today_open = float(opens.loc[today, tk])
            if prior_close > 0 and today_open > 0:
                gap = today_open / prior_close - 1
                if gap <= gap_max:
                    candidates.append((tk, gap))
        except (KeyError, ValueError):
            continue
    candidates.sort(key=lambda x: x[1])  # most negative first
    return [tk for tk, _ in candidates[:top_n]]


def s4_volume_breakout(prior, opens, closes, volumes, eligible, top_n):
    """Yesterday: high volume + breakout. Today: ride the momentum."""
    scores = {}
    for tk in eligible:
        if tk not in closes.columns:
            continue
        cl = closes[tk].loc[:prior].dropna()
        vl = volumes[tk].loc[:prior].dropna()
        if len(cl) < 60 or len(vl) < 60:
            continue
        try:
            p_now = float(cl.iloc[-1])
            high_20 = float(cl.tail(20).max())
            today_vol = float(vl.iloc[-1])
            avg_vol = float(vl.tail(20).mean())
            if avg_vol <= 0 or today_vol <= 0:
                continue
            vol_ratio = today_vol / avg_vol
            is_breakout = p_now >= high_20 * 0.999
            if is_breakout and vol_ratio >= 1.5:
                scores[tk] = vol_ratio
        except (ValueError, KeyError):
            continue
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tk for tk, _ in ranked[:top_n]]


def s5_overnight_only(prior, opens, closes, volumes, eligible, top_n):
    """Buy at TODAY's CLOSE, sell at NEXT day's OPEN — captures overnight drift.

    Different from intraday — held overnight, not within session. Uses W1 score.
    Marked specially in the result.
    """
    return s1_top_w1(prior, opens, closes, volumes, eligible, top_n)


def bt_overnight(opens, closes, volumes, quality_set, start, end,
                 top_n=5, cost_pct=COST_MID):
    """Buy at close, sell at next open. Applies same costs."""
    panel = closes.loc[start:end]
    dates = panel.index
    eligible = [c for c in panel.columns if (not quality_set) or c in quality_set]
    daily_rets = []
    for i in range(0, len(dates) - 1):
        today = dates[i]      # signal + buy at this close
        next_d = dates[i + 1]  # sell at next open
        if next_d not in opens.index:
            continue
        try:
            picks = s1_top_w1(today, opens, closes, volumes, eligible, top_n)
        except Exception:
            continue
        if not picks:
            daily_rets.append(0); continue
        rets = []
        for tk in picks:
            try:
                cp = float(closes.loc[today, tk])
                op = float(opens.loc[next_d, tk])
                if cp > 0 and op > 0:
                    rets.append(op / cp - 1)
            except (KeyError, ValueError):
                continue
        if rets:
            avg = float(np.mean(rets))
            net = avg - cost_pct / 100
            daily_rets.append(net)
        else:
            daily_rets.append(0)

    eq = [1.0]
    for r in daily_rets:
        eq.append(eq[-1] * (1 + r))
    eq = np.array(eq); rets_arr = np.array(daily_rets)
    if len(rets_arr) < 5:
        return {}
    total = (eq[-1] - 1) * 100
    years = len(rets_arr) / 252
    cagr = ((1 + total / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol = rets_arr.std() * (252 ** 0.5) * 100
    rf = risk_free.get_rate()["rate_pct"] / 100
    daily_rf = (1 + rf) ** (1 / 252) - 1
    excess = rets_arr - daily_rf
    sharpe = excess.mean() / excess.std() * (252 ** 0.5) if excess.std() > 0 else 0
    rmax = np.maximum.accumulate(eq); dd = (eq / rmax - 1)
    max_dd = float(dd.min()) * 100
    hit = float((rets_arr > 0).mean()) * 100
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "hit_rate": hit, "avg_daily_pct": float(rets_arr.mean()) * 100,
        "n_days": len(rets_arr), "total_return_pct": total, "final_equity": float(eq[-1]),
    }


def buy_hold_market(closes, start, end):
    panel = closes.loc[start:end].dropna(how="all")
    rets = panel.pct_change().dropna(how="all").mean(axis=1).fillna(0)
    eq = (1 + rets).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    years = len(rets) / 252
    cagr = ((1 + total / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    vol = float(rets.std() * (252 ** 0.5)) * 100
    rmax = eq.cummax(); dd = (eq / rmax - 1)
    max_dd = float(dd.min()) * 100
    rf = risk_free.get_rate()["rate_pct"]; daily_rf = (1 + rf / 100) ** (1 / 252) - 1
    excess = rets - daily_rf
    sharpe = float(excess.mean() / excess.std() * (252 ** 0.5)) if excess.std() > 0 else 0
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
            "hit_rate": float((rets > 0).mean()) * 100,
            "n_days": len(rets), "final_equity": float(eq.iloc[-1])}


def main():
    O, H, L, C, V = load_ohlcv()
    quality_set = load_quality_set()
    print(f"  Loaded {len(C.columns)} names, {len(quality_set)} pass ROE>=10%\n")

    windows = [
        ("Full window (28mo)",   START, END),
        ("Holdout (10mo OOS)",   HOLDOUT_START, END),
        ("Last 6 months",         "2025-10-30", END),
    ]

    print("=" * 115)
    print("DAY-TRADE BACKTEST — every strategy net of 1.15% round-trip costs (mid EGX cost estimate)")
    print("=" * 115)

    for label, start, end in windows:
        print(f"\n  ── {label}  [{start} → {end}] ──")
        print(f"  {'Strategy':<42}{'CAGR':>10}{'Vol':>8}{'Sharpe':>8}"
              f"{'MaxDD':>9}{'Hit%':>7}{'AvgDay%':>9}{'N':>5}")
        print("  " + "-" * 96)

        # Market reference
        mkt = buy_hold_market(C, start, end)
        if mkt:
            print(f"  {'Buy-Hold market (no costs)':<42}"
                  f"{mkt['cagr']:>+9.1f}%{mkt['vol']:>+7.1f}%{mkt['sharpe']:>8.2f}"
                  f"{mkt['max_dd']:>+8.1f}%{mkt['hit_rate']:>+6.1f}%{'—':>9}{mkt['n_days']:>5}")

        strategies = [
            ("S1: Top-5 W1 — open→close",
             lambda p, o, c, v, e, n: s1_top_w1(p, o, c, v, e, n), 5),
            ("S1b: Top-3 W1 — open→close",
             lambda p, o, c, v, e, n: s1_top_w1(p, o, c, v, e, n), 3),
            ("S1c: Top-1 W1 — open→close",
             lambda p, o, c, v, e, n: s1_top_w1(p, o, c, v, e, n), 1),
            ("S2: Gap-up >1% momentum",
             s2_gap_and_go, 5),
            ("S3: Gap-down >1% reversal",
             s3_gap_down_reversal, 5),
            ("S4: Vol-confirmed breakout",
             s4_volume_breakout, 5),
        ]
        for slabel, strat, n in strategies:
            r = bt_intraday(strat, O, C, V, quality_set, start, end,
                            top_n=n, cost_pct=COST_MID)
            if r:
                marker = " ✓" if r["cagr"] > mkt["cagr"] else "  "
                print(f"  {slabel:<42}"
                      f"{r['cagr']:>+9.1f}%{r['vol']:>+7.1f}%{r['sharpe']:>8.2f}"
                      f"{r['max_dd']:>+8.1f}%{r['hit_rate']:>+6.1f}%"
                      f"{r['avg_daily_pct']:>+8.3f}%{r['n_days']:>5}{marker}")
            else:
                print(f"  {slabel:<42}  (insufficient data)")

        # Overnight strategy
        ov = bt_overnight(O, C, V, quality_set, start, end, top_n=5, cost_pct=COST_MID)
        if ov:
            marker = " ✓" if ov["cagr"] > mkt["cagr"] else "  "
            print(f"  {'S6: Overnight (close→next open)':<42}"
                  f"{ov['cagr']:>+9.1f}%{ov['vol']:>+7.1f}%{ov['sharpe']:>8.2f}"
                  f"{ov['max_dd']:>+8.1f}%{ov['hit_rate']:>+6.1f}%"
                  f"{ov['avg_daily_pct']:>+8.3f}%{ov['n_days']:>5}{marker}")

    # Cost sensitivity — how much does cost matter?
    print("\n" + "=" * 115)
    print("COST SENSITIVITY — Top-5 W1 open→close on full window")
    print("=" * 115)
    for cost_label, cost in [("Best case (0.6% large-cap, limit orders)", COST_LOW),
                              ("Mid case (1.15% typical EGX)", COST_MID),
                              ("Worst case (2.0% small-cap, market orders)", COST_HIGH),
                              ("ZERO costs (theoretical max edge)", 0.0)]:
        r = bt_intraday(s1_top_w1, O, C, V, quality_set, START, END,
                        top_n=5, cost_pct=cost)
        print(f"  {cost_label:<48}: CAGR {r['cagr']:>+7.1f}%, "
              f"Sharpe {r['sharpe']:>5.2f}, Hit {r['hit_rate']:>5.1f}%, "
              f"AvgDay {r['avg_daily_pct']:>+6.3f}%")
    print()


if __name__ == "__main__":
    main()
