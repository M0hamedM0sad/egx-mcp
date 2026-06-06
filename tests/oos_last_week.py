"""Out-of-sample test — pretend you only had data through last Thursday.

Dates are computed dynamically each run:
  Cutoff   = the most recent Thursday before last week
  Window   = the 5 EGX trading sessions (Sun–Thu) immediately before today

For each name in the validated universe:
  1. Score using ONLY data through the cutoff (price-based composite).
  2. Run the bootstrap MC simulator as if today were the cutoff day.
  3. Take the actual realized return over the test window.
  4. Compare: pick rank, forecast E[ret], actual ret, hit?

Aggregate:
  - top-5 portfolio return (equal-weight) vs synthetic EGX basket
  - top-5 vs T-bill week
  - hit rate of P(up>2%) > 0.5 calls
  - mean absolute forecast error

Results are appended to logs/oos_weekly_log.jsonl so performance
accumulates week-by-week and is committed back by the CI loop.
"""
from __future__ import annotations

import io
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import curl_cffi.requests as _curl_requests

# curl_cffi with impersonate=chrome uses BoringSSL which ignores both cert env
# vars and explicit cacert paths. Disable verify so yfinance can reach Yahoo's
# crumb endpoint. Public market data, no creds — acceptable here.
_orig_session_init = _curl_requests.Session.__init__

def _patched_session_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_session_init(self, *args, **kwargs)

_curl_requests.Session.__init__ = _patched_session_init

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import egx_listing, risk_free
from egx_mcp.data.universe import resolve_ticker

# ---------------------------------------------------------------------------
# Dynamic date computation — no hardcoded dates
# ---------------------------------------------------------------------------

def _last_egx_thursday(ref: date | None = None) -> date:
    """Return the most recent Thursday (last EGX week-end) at or before ref."""
    ref = ref or date.today()
    # Monday=0 … Thursday=3 … Sunday=6
    days_back = (ref.weekday() - 3) % 7
    return ref - timedelta(days=days_back)

_today = date.today()
_window_end_d  = _last_egx_thursday(_today)          # last Thu  (end of last week)
_cutoff_d      = _window_end_d - timedelta(days=7)   # Thu before that (cutoff)
_fetch_end_d   = _window_end_d + timedelta(days=3)   # safety margin for yfinance

CUTOFF     = str(_cutoff_d)       # e.g. "2026-05-28"
WINDOW_END = str(_window_end_d)   # e.g. "2026-06-04"
_FETCH_END = str(_fetch_end_d)    # e.g. "2026-06-07"

TOP_N    = 5
N_PATHS  = 1500
LOOKBACK = 60   # bars for bootstrap

print(f"\n{'=' * 80}")
print(f"OUT-OF-SAMPLE WALK-FORWARD")
print(f"  Training cutoff: {CUTOFF} (Thu)   — model sees only data ≤ this date")
print(f"  Test window:     {CUTOFF}+1 → {WINDOW_END}  (last week's 5 EGX sessions)")
print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch(symbol: str) -> pd.Series:
    """Pull daily history wide enough to cover the full test window."""
    try:
        h = yf.Ticker(symbol).history(start="2025-01-01", end=_FETCH_END, interval="1d")
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


def _price_only_score(closes: pd.Series, cutoff: pd.Timestamp) -> float | None:
    """Same composite the backtest harness uses, computed at cutoff."""
    sub = closes.loc[:cutoff].dropna()
    if len(sub) < 130:
        return None
    p_now = float(sub.iloc[-1])
    p_6m  = float(sub.iloc[-130])
    p_1m  = float(sub.iloc[-22])
    ma200 = float(sub.tail(200).mean()) if len(sub) >= 200 else float(sub.mean())
    vol   = float(sub.tail(60).pct_change().std() * (252 ** 0.5))
    if not (p_now > 0 and p_6m > 0 and p_1m > 0):
        return None
    mom_6m  = (p_now / p_6m - 1) * 100
    mr_1m   = -(p_now / p_1m - 1) * 100
    trend   = 5 if p_now > ma200 else -5
    vol_pen = max(0, (vol * 100 - 30) * 0.5)
    return mom_6m * 0.5 + mr_1m * 0.2 + trend - vol_pen


def _bootstrap_forecast(closes: pd.Series, cutoff: pd.Timestamp, horizon_days: int) -> dict:
    """Bootstrap MC starting at cutoff price, sampling from pre-cutoff returns."""
    sub = closes.loc[:cutoff].dropna()
    if len(sub) < 30:
        return {"e_ret": None, "p_up_2": None, "p10": None, "p90": None}
    rets = sub.pct_change().dropna().tail(LOOKBACK).tolist()
    if len(rets) < 20:
        return {"e_ret": None, "p_up_2": None, "p10": None, "p90": None}
    last = float(sub.iloc[-1])
    rng  = random.Random(42)
    terminals: list[float] = []
    up_2 = 0
    for _ in range(N_PATHS):
        p = last
        for _step in range(horizon_days):
            p *= (1 + rng.choice(rets))
        terminals.append(p)
        if p / last - 1 >= 0.02:
            up_2 += 1
    terminals.sort()
    return {
        "e_ret":  (sum(terminals) / len(terminals) / last - 1) * 100,
        "p_up_2": up_2 / N_PATHS,
        "p10":    (terminals[int(0.10 * N_PATHS)] / last - 1) * 100,
        "p90":    (terminals[int(0.90 * N_PATHS)] / last - 1) * 100,
    }


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

print("Loading price history for validated EGX universe...")
universe = egx_listing.get_full_universe()
print(f"Universe size: {len(universe)}\n")

cutoff_ts = pd.Timestamp(CUTOFF)
end_ts    = pd.Timestamp(WINDOW_END)

results: list[dict] = []
for tk in universe:
    _, yahoo, _ = resolve_ticker(tk)
    closes = _fetch(yahoo)
    if closes.empty:
        continue
    score = _price_only_score(closes, cutoff_ts)
    if score is None:
        continue

    pre = closes.loc[:cutoff_ts].dropna()
    end = closes.loc[:end_ts].dropna()
    if pre.empty or end.empty or pre.index[-1] >= end.index[-1]:
        continue
    p0 = float(pre.iloc[-1])
    p1 = float(end.iloc[-1])
    actual_ret_pct = (p1 / p0 - 1) * 100

    fc = _bootstrap_forecast(closes, cutoff_ts, horizon_days=5)

    results.append({
        "ticker":               tk,
        "score":                score,
        "cutoff_price":         p0,
        "end_price":            p1,
        "actual_ret_pct":       actual_ret_pct,
        "forecast_e_ret_pct":   fc["e_ret"],
        "forecast_p_up_2pct":   fc["p_up_2"],
        "forecast_p10":         fc["p10"],
        "forecast_p90":         fc["p90"],
        "in_90pct_ci": (fc["p10"] is not None
                        and fc["p10"] <= actual_ret_pct <= fc["p90"]),
    })

results.sort(key=lambda r: r["score"], reverse=True)

# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

rf       = risk_free.get_rate()["rate_pct"]
rf_week  = ((1 + rf / 100) ** (5 / 252) - 1) * 100

basket_returns = [r["actual_ret_pct"] for r in results]
benchmark_ret  = sum(basket_returns) / len(basket_returns) if basket_returns else None

top     = results[:TOP_N]
top_ret = sum(r["actual_ret_pct"] for r in top) / len(top) if top else None

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

print(f"Model would have picked these top {TOP_N} on {CUTOFF}:\n")
print(f"{'Rank':<5}{'Tic':<7}{'Score':<8}{'Forecast':<12}{'Actual':<10}{'P(>2%)':<9}{'In 90% CI'}")
print("-" * 72)
for i, r in enumerate(top, 1):
    flag  = "✓" if r["in_90pct_ci"] else "✗"
    p_up  = (f"{r['forecast_p_up_2pct']*100:.0f}%"
             if r["forecast_p_up_2pct"] is not None else "—")
    print(f"{i:<5}{r['ticker']:<7}{r['score']:<8.1f}"
          f"{r['forecast_e_ret_pct']:>+7.2f}%   "
          f"{r['actual_ret_pct']:>+7.2f}%  {p_up:<9}{flag}")

print("\n" + "=" * 72)
print(f"AGGREGATE RESULTS — week of {CUTOFF} → {WINDOW_END}")
print("=" * 72)
print(f"  Top-{TOP_N} portfolio return:    {top_ret:+.2f}%")
print(f"  EGX synthetic basket (n={len(results)}):  {benchmark_ret:+.2f}%")
print(f"  EGP T-bill weekly:           {rf_week:+.3f}%")
print(f"  Active vs basket:            {top_ret - benchmark_ret:+.2f} pp")
print(f"  Active vs T-bills:           {top_ret - rf_week:+.2f} pp")

top_beat = sum(1 for r in top if r["actual_ret_pct"] > benchmark_ret)
top_pos  = sum(1 for r in top if r["actual_ret_pct"] > 0)
print(f"\n  Picks that beat basket: {top_beat} / {TOP_N}")
print(f"  Picks with positive return: {top_pos} / {TOP_N}")

n_with_fc = [r for r in results if r["forecast_e_ret_pct"] is not None]
mae    = sum(abs(r["actual_ret_pct"] - r["forecast_e_ret_pct"]) for r in n_with_fc) / len(n_with_fc)
me     = sum(r["actual_ret_pct"] - r["forecast_e_ret_pct"] for r in n_with_fc) / len(n_with_fc)
in_ci  = sum(1 for r in results if r["in_90pct_ci"]) / len(results) * 100

print(f"\n  Bootstrap MC across {len(results)} names:")
print(f"    Mean absolute error: {mae:.2f}pp (predicted vs actual)")
print(f"    Mean error (bias):   {me:+.2f}pp")
print(f"    Actual within 90% CI: {in_ci:.0f}% of names")
print(f"    (Calibrated would be 80%; below = overconfident, above = underconfident)\n")

bottom  = results[-TOP_N:]
bot_ret = sum(r["actual_ret_pct"] for r in bottom) / len(bottom)
print(f"Model would have AVOIDED these bottom {TOP_N} on {CUTOFF}:\n")
print(f"{'Rank':<7}{'Tic':<7}{'Score':<8}{'Forecast':<12}{'Actual':<10}")
print("-" * 50)
for i, r in enumerate(bottom, len(results) - TOP_N + 1):
    print(f"{i:<7}{r['ticker']:<7}{r['score']:<8.1f}"
          f"{r['forecast_e_ret_pct']:>+7.2f}%   "
          f"{r['actual_ret_pct']:>+7.2f}%")
print(f"\n  Bottom-{TOP_N} actual return: {bot_ret:+.2f}%")
print(f"  Long top / short bottom spread: {top_ret - bot_ret:+.2f} pp")
print()

# ---------------------------------------------------------------------------
# Persist weekly result to logs/oos_weekly_log.jsonl
# ---------------------------------------------------------------------------

_LOG = Path(__file__).parent.parent / "logs" / "oos_weekly_log.jsonl"
_LOG.parent.mkdir(parents=True, exist_ok=True)

_record = {
    "cutoff":           CUTOFF,
    "window_end":       WINDOW_END,
    "n_universe":       len(universe),
    "n_scored":         len(results),
    "top5_return_pct":  round(top_ret, 4) if top_ret is not None else None,
    "basket_return_pct":round(benchmark_ret, 4) if benchmark_ret is not None else None,
    "tbill_week_pct":   round(rf_week, 4),
    "alpha_vs_basket":  round(top_ret - benchmark_ret, 4) if (top_ret is not None and benchmark_ret is not None) else None,
    "picks_beat_basket":top_beat,
    "picks_positive":   top_pos,
    "mc_mae_pp":        round(mae, 4),
    "mc_bias_pp":       round(me, 4),
    "pct_in_90ci":      round(in_ci, 1),
    "ls_spread_pp":     round(top_ret - bot_ret, 4) if top_ret is not None else None,
    "top_picks":        [{"ticker": r["ticker"], "score": round(r["score"], 2),
                          "actual_ret_pct": round(r["actual_ret_pct"], 4)}
                         for r in top],
}

# Append (one JSON object per line). Skip if this cutoff week is already logged.
existing = _LOG.read_text(encoding="utf-8").splitlines() if _LOG.exists() else []
already_logged = any(
    json.loads(ln).get("cutoff") == CUTOFF
    for ln in existing if ln.strip()
)
if already_logged:
    print(f"(Week {CUTOFF} already in oos_weekly_log.jsonl — not duplicated)")
else:
    with _LOG.open("a", encoding="utf-8") as _f:
        _f.write(json.dumps(_record, ensure_ascii=False) + "\n")
    print(f"Appended result to {_LOG}")
