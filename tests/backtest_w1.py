"""Full backtest — production W1 weekly model vs the EGX market basket.

Weekly Thursday rebalance: rank the validated universe with the deployed W1
score (egx_mcp.data.weekly — same quality + volume eligibility, same
holiday-week relax fallback), hold the top 5 equal-weight for one week.
Benchmark = equal-weight basket of every name with data that week (the
synthetic market, since no EGX30 index series exists on Yahoo).

Honesty rules baked in:
  - GROSS and NET equity curves. Net charges per-side costs on actual
    turnover (default 0.30%/side ≈ EGX retail commission + spread drag).
  - The W1 config was tuned on data through 2026-04. Everything before
    2026-05 is IN-SAMPLE for the model; the per-window table separates the
    pure out-of-sample tail.
  - T-bill comparison uses the live CBE rate when reachable, else 25%.

    python -m tests.backtest_w1
    python -m tests.backtest_w1 --start 2024-01-01 --cost-bps 30
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import curl_cffi.requests as _curl_requests

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
from egx_mcp.data.weekly import W1Config, _features as _w1_features, \
    _score as _w1_score, _load_quality_set

TOP_N = 5


def _norm(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _thursdays(start: date, end: date) -> list[date]:
    d = start + timedelta(days=(3 - start.weekday()) % 7)  # first Thursday
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def _week_return(cl: pd.Series, cut: pd.Timestamp, nxt: pd.Timestamp) -> float | None:
    pre = cl.loc[:cut].dropna()
    post = cl.loc[:nxt].dropna()
    if pre.empty or post.empty or pre.index[-1] >= post.index[-1]:
        return None
    return float(post.iloc[-1]) / float(pre.iloc[-1]) - 1


def _stats(weekly_rets: list[float], rf_week: float) -> dict:
    r = np.array(weekly_rets)
    if len(r) < 2:
        return {}
    eq = np.cumprod(1 + r)
    total = (eq[-1] - 1) * 100
    years = len(r) * 7 / 365.25
    cagr = ((1 + total / 100) ** (1 / max(years, 1e-9)) - 1) * 100
    vol = r.std() * (52 ** 0.5) * 100
    ex = r - rf_week
    sharpe = ex.mean() / ex.std() * (52 ** 0.5) if ex.std() > 0 else 0.0
    rmax = np.maximum.accumulate(eq)
    max_dd = ((eq / rmax) - 1).min() * 100
    return {"total": total, "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_dd": max_dd, "hit": (r > 0).mean() * 100, "n": len(r)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--cost-bps", type=float, default=30.0,
                    help="per-side transaction cost in basis points")
    args = ap.parse_args()

    start_d = date.fromisoformat(args.start)
    last_thu = date.today() - timedelta(days=(date.today().weekday() - 3) % 7)
    fetch_start = str(start_d - timedelta(days=200))
    cost = args.cost_bps / 10_000

    print(f"\nW1 weekly backtest {start_d} → {last_thu}  "
          f"(costs {args.cost_bps:.0f}bps/side)")
    print("Loading close+volume history for the validated universe...")
    universe = egx_listing.get_full_universe()
    cfg = W1Config()
    quality = _load_quality_set(cfg.min_roe_pct)

    closes_d, volumes_d = {}, {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start=fetch_start, interval="1d")
            if h is None or h.empty:
                continue
            closes_d[tk] = _norm(h["Close"])
            volumes_d[tk] = _norm(h["Volume"])
        except Exception:
            continue
    print(f"  {len(closes_d)} names with data "
          f"({len(quality)} pass ROE>={cfg.min_roe_pct:.0f}%)\n")

    thursdays = _thursdays(start_d, last_thu)
    rows = []          # one row per graded week
    prev_picks: set[str] = set()
    relaxed_weeks = 0

    for i in range(len(thursdays) - 1):
        cut = pd.Timestamp(thursdays[i])
        nxt = pd.Timestamp(thursdays[i + 1])

        scored, actuals = [], {}
        for tk, cl in closes_d.items():
            wr = _week_return(cl, cut, nxt)
            if wr is None:
                continue
            actuals[tk] = wr
            f = _w1_features(cl, volumes_d[tk], cut)
            if f is None:
                continue
            if quality and tk not in quality:
                continue
            scored.append((tk, _w1_score(f, cfg),
                           f["vol_ratio"] >= cfg.min_volume_ratio))

        if len(actuals) < 20 or not scored:
            continue
        eligible = [s for s in scored if s[2]]
        if len(eligible) < TOP_N:           # production relax fallback
            eligible = scored
            relaxed_weeks += 1
        eligible.sort(key=lambda x: x[1], reverse=True)
        picks = {tk for tk, _, _ in eligible[:TOP_N]}

        strat = sum(actuals[tk] for tk in picks) / len(picks)
        basket = sum(actuals.values()) / len(actuals)
        # Turnover: fraction of the book replaced this week (both sides traded)
        turnover = len(picks - prev_picks) / TOP_N if prev_picks else 1.0
        net = strat - turnover * 2 * cost
        prev_picks = picks

        rows.append({"cutoff": thursdays[i], "strat": strat, "net": net,
                     "basket": basket, "turnover": turnover})

    if not rows:
        print("No graded weeks — network problem?")
        return 1

    try:
        rf_pct = risk_free.get_rate()["rate_pct"]
    except Exception:
        rf_pct = 25.0
    rf_week = (1 + rf_pct / 100) ** (1 / 52) - 1

    windows = [
        ("Full window", None),
        ("2024 (in-sample for W1 tuning)", ("2024-01-01", "2024-12-31")),
        ("2025 (in-sample for W1 tuning)", ("2025-01-01", "2025-12-31")),
        ("2026 YTD (tuned through Apr)", ("2026-01-01", None)),
        ("2026-05+ (pure out-of-sample)", ("2026-05-01", None)),
    ]

    print(f"{'Window':<34}{'':>4}{'W1 gross':>10}{'W1 net':>9}{'Market':>9}"
          f"{'α net':>8}{'Sh g':>6}{'Sh mkt':>7}{'DD g':>8}{'DD mkt':>8}")
    print("-" * 105)
    for label, rng in windows:
        sub = rows
        if rng:
            lo = date.fromisoformat(rng[0])
            hi = date.fromisoformat(rng[1]) if rng[1] else date.today()
            sub = [r for r in rows if lo <= r["cutoff"] <= hi]
        if len(sub) < 4:
            print(f"{label:<34}  (only {len(sub)} weeks — skipped)")
            continue
        g = _stats([r["strat"] for r in sub], rf_week)
        nt = _stats([r["net"] for r in sub], rf_week)
        mk = _stats([r["basket"] for r in sub], rf_week)
        print(f"{label:<34}{g['n']:>3}w{g['cagr']:>+9.1f}%{nt['cagr']:>+8.1f}%"
              f"{mk['cagr']:>+8.1f}%{nt['cagr']-mk['cagr']:>+7.1f}p"
              f"{g['sharpe']:>6.2f}{mk['sharpe']:>7.2f}"
              f"{g['max_dd']:>+7.1f}%{mk['max_dd']:>+7.1f}%")

    full_g = _stats([r["strat"] for r in rows], rf_week)
    full_n = _stats([r["net"] for r in rows], rf_week)
    full_m = _stats([r["basket"] for r in rows], rf_week)
    avg_to = sum(r["turnover"] for r in rows) / len(rows)
    weekly_alpha = [r["strat"] - r["basket"] for r in rows]
    wa = np.array(weekly_alpha)
    t_stat = wa.mean() / (wa.std(ddof=1) / np.sqrt(len(wa))) if len(wa) > 2 else 0.0

    print("-" * 105)
    print(f"\nFull window ({full_g['n']} weeks):")
    print(f"  Total return  — W1 gross {full_g['total']:+.0f}%   "
          f"W1 net {full_n['total']:+.0f}%   market {full_m['total']:+.0f}%   "
          f"T-bills ≈ {((1+rf_week)**full_g['n']-1)*100:+.0f}%")
    print(f"  Weekly hit vs market: {sum(1 for a in weekly_alpha if a > 0)}"
          f"/{len(weekly_alpha)} weeks   mean weekly alpha (gross): "
          f"{wa.mean()*100:+.2f}pp   t-stat: {t_stat:.2f}")
    print(f"  Avg turnover: {avg_to*100:.0f}%/week → cost drag ≈ "
          f"{avg_to*2*cost*52*100:.1f}%/yr at {args.cost_bps:.0f}bps/side")
    if relaxed_weeks:
        print(f"  Volume filter relaxed (holiday weeks): {relaxed_weeks}")
    print(f"\n  t-stat reading: |t| ≥ 2 ≈ statistically real edge; below that the "
          f"alpha is within noise.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
