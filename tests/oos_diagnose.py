"""Diagnostic: why did the model miss last week's winners?

Same data & scoring as oos_last_week.py, but instead of ranking by model score,
rank by ACTUAL realized return and show what the model said about each.
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import curl_cffi.requests as _curl_requests

_orig_session_init = _curl_requests.Session.__init__
def _patched_session_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_session_init(self, *args, **kwargs)
_curl_requests.Session.__init__ = _patched_session_init

import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import egx_listing
from egx_mcp.data.universe import resolve_ticker

CUTOFF = pd.Timestamp("2026-05-14")
WINDOW_END = pd.Timestamp("2026-05-21")
LOOKBACK = 60
N_PATHS = 1500


def _fetch(symbol: str) -> pd.Series:
    try:
        h = yf.Ticker(symbol).history(start="2025-01-01", end="2026-05-22", interval="1d")
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


def _score(closes: pd.Series, cutoff: pd.Timestamp):
    """Same composite the OOS test uses. Returns (score, components)."""
    sub = closes.loc[:cutoff].dropna()
    if len(sub) < 130:
        return None, None
    p_now = float(sub.iloc[-1])
    p_6m = float(sub.iloc[-130])
    p_1m = float(sub.iloc[-22])
    ma200 = float(sub.tail(200).mean()) if len(sub) >= 200 else float(sub.mean())
    vol = float(sub.tail(60).pct_change().std() * (252 ** 0.5))
    if not (p_now > 0 and p_6m > 0 and p_1m > 0):
        return None, None
    mom_6m = (p_now / p_6m - 1) * 100
    mr_1m = -(p_now / p_1m - 1) * 100  # mean reversion: negative 1m return is positive signal
    trend = 5 if p_now > ma200 else -5
    vol_pen = max(0, (vol * 100 - 30) * 0.5)
    score = mom_6m * 0.5 + mr_1m * 0.2 + trend - vol_pen
    return score, {
        "mom_6m": mom_6m,
        "ret_1m": -mr_1m,   # display the actual 1m return, not the flipped sign
        "trend": trend,
        "vol_ann_%": vol * 100,
        "vol_pen": vol_pen,
    }


universe = egx_listing.get_full_universe()
rows = []
for tk in universe:
    _, yahoo, _ = resolve_ticker(tk)
    closes = _fetch(yahoo)
    if closes.empty:
        continue
    score, comps = _score(closes, CUTOFF)
    if score is None:
        continue
    pre = closes.loc[:CUTOFF].dropna()
    end = closes.loc[:WINDOW_END].dropna()
    if pre.empty or end.empty or pre.index[-1] >= end.index[-1]:
        continue
    p0 = float(pre.iloc[-1])
    p1 = float(end.iloc[-1])
    actual = (p1 / p0 - 1) * 100
    rows.append({"ticker": tk, "score": score, "actual": actual, **comps})

rows.sort(key=lambda r: r["actual"], reverse=True)
n = len(rows)

print(f"\nActual top-10 performers last week (n={n} validated names):\n")
print(f"{'Rank':<5}{'Tic':<7}{'Actual':<10}{'Score':<8}{'ModelRank':<11}{'mom_6m':<10}{'ret_1m':<10}{'vol%':<8}{'trend'}")
print("-" * 80)
# Build score rank lookup
by_score = sorted(rows, key=lambda r: r["score"], reverse=True)
score_rank = {r["ticker"]: i + 1 for i, r in enumerate(by_score)}
for i, r in enumerate(rows[:10], 1):
    mr = score_rank[r["ticker"]]
    print(f"{i:<5}{r['ticker']:<7}{r['actual']:>+7.2f}%  {r['score']:<8.1f}#{mr:<10}{r['mom_6m']:>+7.1f}%  {r['ret_1m']:>+7.1f}%  {r['vol_ann_%']:<8.1f}{r['trend']:+d}")

print(f"\nActual bottom-10 performers last week:\n")
print(f"{'Rank':<5}{'Tic':<7}{'Actual':<10}{'Score':<8}{'ModelRank':<11}{'mom_6m':<10}{'ret_1m':<10}{'vol%':<8}{'trend'}")
print("-" * 80)
for i, r in enumerate(rows[-10:], n - 9):
    mr = score_rank[r["ticker"]]
    print(f"{i:<5}{r['ticker']:<7}{r['actual']:>+7.2f}%  {r['score']:<8.1f}#{mr:<10}{r['mom_6m']:>+7.1f}%  {r['ret_1m']:>+7.1f}%  {r['vol_ann_%']:<8.1f}{r['trend']:+d}")

# Stat: Spearman-style — is score predictive at all?
import statistics
pairs = sorted(rows, key=lambda r: r["score"], reverse=True)
ranks_by_score = {r["ticker"]: i for i, r in enumerate(pairs)}
ranks_by_actual = {r["ticker"]: i for i, r in enumerate(sorted(rows, key=lambda r: r["actual"], reverse=True))}
diffs = [ranks_by_score[t] - ranks_by_actual[t] for t in ranks_by_score]
n_d = len(diffs)
spearman = 1 - 6 * sum(d * d for d in diffs) / (n_d * (n_d ** 2 - 1))
print(f"\nSpearman rank correlation (score vs actual return): {spearman:+.3f}")
print(f"  Interpretation: 0 = noise, +1 = perfect, -1 = anti-predictive\n")

# Sanity: top-5 by score (what model picked) vs top-5 by actual
top5_score = [r["ticker"] for r in by_score[:5]]
top5_actual = [r["ticker"] for r in rows[:5]]
overlap = set(top5_score) & set(top5_actual)
print(f"Model top-5 picks: {top5_score}")
print(f"Actual top-5:      {top5_actual}")
print(f"Overlap: {len(overlap)} / 5\n")
