"""Multi-week walk-forward — old monthly composite vs production W1, plus
mean-vs-median bootstrap point-forecast comparison.

The weekly OOS log had a single week in it, scored with a monthly-style
composite the briefing doesn't actually use. This harness replays the last
N completed EGX weeks (Thu→Thu) and, at each cutoff, ranks the universe two
ways using only pre-cutoff data:

    OLD : 0.5*mom_6m + 0.2*mr_1m + trend(±5) - vol_pen   (what oos_last_week
          used to test)
    W1  : the deployed weekly model from egx_mcp.data.weekly, with its
          quality (ROE>=10%) and volume eligibility filters

and grades both top-5 baskets against the equal-weight universe basket.
It also runs the bootstrap MC per name per week and scores the MEAN vs the
MEDIAN terminal as point forecasts (MAE vs realized), plus 80% CI coverage.

    python -m tests.oos_multiweek            # default: last 12 weeks
    python -m tests.oos_multiweek --weeks 8
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from datetime import date, timedelta
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
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import egx_listing
from egx_mcp.data.universe import resolve_ticker
from egx_mcp.data.weekly import W1Config, _features as _w1_features, \
    _score as _w1_score, _load_quality_set, eligibility as _w1_eligibility

TOP_N = 5
N_PATHS = 1000
LOOKBACK = 60


def _last_thursday(ref: date) -> date:
    return ref - timedelta(days=(ref.weekday() - 3) % 7)


def _norm(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _old_score(closes: pd.Series, cutoff: pd.Timestamp) -> float | None:
    sub = closes.loc[:cutoff].dropna()
    if len(sub) < 130:
        return None
    p_now = float(sub.iloc[-1])
    p_6m = float(sub.iloc[-130])
    p_1m = float(sub.iloc[-22])
    ma200 = float(sub.tail(200).mean()) if len(sub) >= 200 else float(sub.mean())
    vol = float(sub.tail(60).pct_change().std() * (252 ** 0.5))
    if not (p_now > 0 and p_6m > 0 and p_1m > 0):
        return None
    mom_6m = (p_now / p_6m - 1) * 100
    mr_1m = -(p_now / p_1m - 1) * 100
    trend = 5 if p_now > ma200 else -5
    vol_pen = max(0, (vol * 100 - 30) * 0.5)
    return mom_6m * 0.5 + mr_1m * 0.2 + trend - vol_pen


def _bootstrap(closes: pd.Series, cutoff: pd.Timestamp, horizon: int = 5) -> dict | None:
    sub = closes.loc[:cutoff].dropna()
    if len(sub) < 30:
        return None
    rets = sub.pct_change().dropna().tail(LOOKBACK).tolist()
    if len(rets) < 20:
        return None
    last = float(sub.iloc[-1])
    rng = random.Random(42)
    terminals = []
    for _ in range(N_PATHS):
        p = last
        for _step in range(horizon):
            p *= (1 + rng.choice(rets))
        terminals.append(p)
    terminals.sort()
    return {
        "mean":   (sum(terminals) / len(terminals) / last - 1) * 100,
        "median": (terminals[N_PATHS // 2] / last - 1) * 100,
        "p10":    (terminals[int(0.10 * N_PATHS)] / last - 1) * 100,
        "p90":    (terminals[int(0.90 * N_PATHS)] / last - 1) * 100,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=12)
    args = ap.parse_args()

    last_thu = _last_thursday(date.today() - timedelta(days=1))
    cutoffs = [last_thu - timedelta(days=7 * (i + 1)) for i in range(args.weeks)]
    cutoffs.reverse()
    fetch_start = str(cutoffs[0] - timedelta(days=420))

    print(f"\nWalk-forward over {args.weeks} weeks: "
          f"{cutoffs[0]} → {last_thu}")
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

    weeks = []
    fc_rows = []  # per name-week forecast comparison
    for cut in cutoffs:
        cut_ts = pd.Timestamp(cut)
        end_ts = pd.Timestamp(cut + timedelta(days=7))
        rows = []
        for tk, cl in closes_d.items():
            pre = cl.loc[:cut_ts].dropna()
            post = cl.loc[:end_ts].dropna()
            if pre.empty or post.empty or pre.index[-1] >= post.index[-1]:
                continue
            actual = (float(post.iloc[-1]) / float(pre.iloc[-1]) - 1) * 100

            old = _old_score(cl, cut_ts)
            f = _w1_features(cl, volumes_d[tk], cut_ts)
            w1 = _w1_score(f, cfg) if f is not None else None
            eligible = (f is not None and all(_w1_eligibility(f, cfg).values())
                        and ((not quality) or tk in quality))
            rows.append({"tk": tk, "old": old, "w1": w1,
                         "eligible": eligible, "actual": actual})

            fc = _bootstrap(cl, cut_ts)
            if fc is not None:
                fc_rows.append({**fc, "actual": actual})

        if len(rows) < 20:
            continue
        basket = sum(r["actual"] for r in rows) / len(rows)
        old_rank = sorted((r for r in rows if r["old"] is not None),
                          key=lambda r: r["old"], reverse=True)
        old_elig_rank = sorted((r for r in rows if r["old"] is not None and r["eligible"]),
                               key=lambda r: r["old"], reverse=True)
        w1_rank = sorted((r for r in rows if r["w1"] is not None and r["eligible"]),
                         key=lambda r: r["w1"], reverse=True)
        if len(w1_rank) < TOP_N or len(old_elig_rank) < TOP_N:
            # Holiday-shortened week: the single-session volume filter can
            # empty the eligible set. Production (weekly.rank_universe)
            # relaxes to quality-only eligibility — mirror that here.
            print(f"  ! {cut}: only {len(w1_rank)} volume-eligible names — "
                  "relaxed to quality-only (matches production fallback)")
            qual_rows = [r for r in rows
                         if (not quality) or r["tk"] in quality]
            w1_rank = sorted((r for r in qual_rows if r["w1"] is not None),
                             key=lambda r: r["w1"], reverse=True)
            old_elig_rank = sorted((r for r in qual_rows if r["old"] is not None),
                                   key=lambda r: r["old"], reverse=True)
            if len(w1_rank) < TOP_N or len(old_elig_rank) < TOP_N:
                print(f"  ! {cut}: still <{TOP_N} names — week skipped")
                continue
        old_top = sum(r["actual"] for r in old_rank[:TOP_N]) / TOP_N
        olde_top = sum(r["actual"] for r in old_elig_rank[:TOP_N]) / TOP_N
        w1_top = sum(r["actual"] for r in w1_rank[:TOP_N]) / TOP_N
        w1_bot = sum(r["actual"] for r in w1_rank[-TOP_N:]) / TOP_N
        weeks.append({"cutoff": str(cut), "n": len(rows),
                      "n_elig": len(w1_rank), "basket": basket,
                      "old_top": old_top, "olde_top": olde_top,
                      "w1_top": w1_top, "w1_bot": w1_bot})

    if not weeks:
        print("No graded weeks — network problem?")
        return 1

    print(f"{'Cutoff':<12}{'n':>4}{'elig':>5}{'Basket':>9}{'OLD α':>8}"
          f"{'OLDf α':>8}{'W1 α':>8}{'W1 L/S':>8}")
    print("-" * 70)
    for w in weeks:
        print(f"{w['cutoff']:<12}{w['n']:>4}{w['n_elig']:>5}{w['basket']:>+8.2f}%"
              f"{w['old_top']-w['basket']:>+7.2f}p"
              f"{w['olde_top']-w['basket']:>+7.2f}p"
              f"{w['w1_top']-w['basket']:>+7.2f}p"
              f"{w['w1_top']-w['w1_bot']:>+7.2f}p")

    n = len(weeks)
    old_alpha = [w["old_top"] - w["basket"] for w in weeks]
    olde_alpha = [w["olde_top"] - w["basket"] for w in weeks]
    w1_alpha = [w["w1_top"] - w["basket"] for w in weeks]
    ls = [w["w1_top"] - w["w1_bot"] for w in weeks]
    print("-" * 70)
    print(f"{'mean over ' + str(n) + 'w':<21}"
          f"{sum(w['basket'] for w in weeks)/n:>+8.2f}%"
          f"{sum(old_alpha)/n:>+7.2f}p{sum(olde_alpha)/n:>+7.2f}p"
          f"{sum(w1_alpha)/n:>+7.2f}p{sum(ls)/n:>+7.2f}p")
    print(f"\n  weeks beating basket — OLD: {sum(1 for a in old_alpha if a > 0)}/{n}"
          f"   OLD+filters: {sum(1 for a in olde_alpha if a > 0)}/{n}"
          f"   W1: {sum(1 for a in w1_alpha if a > 0)}/{n}"
          f"   W1 L/S > 0: {sum(1 for x in ls if x > 0)}/{n}")
    print("  (OLDf = old monthly composite restricted to the same quality+volume "
          "eligibility set W1 uses)")

    if fc_rows:
        m = len(fc_rows)
        mae_mean = sum(abs(r["actual"] - r["mean"]) for r in fc_rows) / m
        mae_med = sum(abs(r["actual"] - r["median"]) for r in fc_rows) / m
        bias_mean = sum(r["actual"] - r["mean"] for r in fc_rows) / m
        bias_med = sum(r["actual"] - r["median"] for r in fc_rows) / m
        cover = sum(1 for r in fc_rows if r["p10"] <= r["actual"] <= r["p90"]) / m
        print(f"\nBootstrap point forecast across {m} name-weeks:")
        print(f"  MAE  — mean terminal: {mae_mean:.2f}pp   median terminal: {mae_med:.2f}pp")
        print(f"  bias — mean terminal: {bias_mean:+.2f}pp   median terminal: {bias_med:+.2f}pp")
        print(f"  realized within q10..q90: {cover*100:.0f}% (nominal 80%)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
