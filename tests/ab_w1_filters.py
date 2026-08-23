"""A/B the W1 eligibility filters out-of-sample, week by week.

The liquidity and extension filters were chosen against the learning panel,
in-sample and across several tried variants. That is a sanity check, not
evidence. This replays the last N completed EGX weeks (Thu->Thu) and ranks the
same universe with the same scores under different filter sets, so the only
thing that varies is eligibility.

Paired by construction: every variant sees identical prices, identical scores
and identical weeks, so the per-week difference is the filter's effect and
nothing else. The sign test over weeks is the honest read at this sample size —
one +76% microcap dominates any mean.

    python -m tests.ab_w1_filters              # last 12 weeks
    python -m tests.ab_w1_filters --weeks 26
"""
from __future__ import annotations

import argparse
import io
import statistics as st
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    import curl_cffi.requests as _curl_requests

    _orig_session_init = _curl_requests.Session.__init__

    def _patched_session_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_session_init(self, *args, **kwargs)

    _curl_requests.Session.__init__ = _patched_session_init
except ImportError:
    pass

import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import egx_listing
from egx_mcp.data.universe import resolve_ticker
from egx_mcp.data.weekly import W1Config, _features as _w1_features, \
    _score as _w1_score, _load_quality_set, eligibility as _w1_eligibility

TOP_N = 5

# Each variant is the set of eligibility keys it enforces on top of quality.
VARIANTS = {
    "old (quality+volume)":        ("volume",),
    "+ extension only":            ("volume", "extension"),
    "+ liquidity only":            ("volume", "liquidity"),
    "NEW (both, as shipped)":      ("volume", "liquidity", "extension"),
}


def _norm(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _last_thursday(ref: date) -> date:
    return ref - timedelta(days=(ref.weekday() - 3) % 7)


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-of-sample A/B of the W1 filters.")
    ap.add_argument("--weeks", type=int, default=12)
    args = ap.parse_args()

    last_thu = _last_thursday(date.today() - timedelta(days=1))
    cutoffs = [last_thu - timedelta(days=7 * (i + 1)) for i in range(args.weeks)]
    cutoffs.reverse()

    cfg = W1Config()
    print(f"\nW1 FILTER A/B — {args.weeks} weeks, {cutoffs[0]} -> {last_thu}")
    print(f"  min_price_egp={cfg.min_price_egp}  min_turnover_egp={cfg.min_turnover_egp:,.0f}"
          f"  max_5d_runup_pct={cfg.max_5d_runup_pct}\n")

    universe = egx_listing.get_full_universe()
    quality = _load_quality_set(cfg.min_roe_pct)
    fetch_start = str(cutoffs[0] - timedelta(days=420))

    closes_d, volumes_d = {}, {}
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(start=fetch_start, interval="1d")
            if h is None or h.empty:
                continue
            closes_d[tk] = _norm(h["Close"])
            volumes_d[tk] = _norm(h["Volume"])
        except Exception:  # noqa: BLE001
            continue
    print(f"  {len(closes_d)} names with data ({len(quality)} pass ROE>={cfg.min_roe_pct:.0f}%)\n")

    per_week: list[dict] = []
    for cut in cutoffs:
        cut_ts, end_ts = pd.Timestamp(cut), pd.Timestamp(cut + timedelta(days=7))
        rows = []
        for tk, cl in closes_d.items():
            pre, post = cl.loc[:cut_ts].dropna(), cl.loc[:end_ts].dropna()
            if pre.empty or post.empty or pre.index[-1] >= post.index[-1]:
                continue
            f = _w1_features(cl, volumes_d[tk], cut_ts)
            if f is None:
                continue
            rows.append({
                "tk": tk,
                "score": _w1_score(f, cfg),
                "quality": (not quality) or tk in quality,
                "flags": _w1_eligibility(f, cfg),
                "actual": (float(post.iloc[-1]) / float(pre.iloc[-1]) - 1) * 100,
                "runup_5d": f["mom_5d"],
            })
        if len(rows) < 20:
            continue

        week = {"cutoff": str(cut), "n": len(rows),
                "basket": st.mean(r["actual"] for r in rows)}
        for name, keys in VARIANTS.items():
            sel = [r for r in rows if r["quality"] and all(r["flags"][k] for k in keys)]
            sel.sort(key=lambda r: r["score"], reverse=True)
            # Mirror production's holiday fallback: relax the single-session
            # volume filter, never liquidity or extension.
            if len(sel) < TOP_N:
                sel = [r for r in rows
                       if r["quality"] and all(r["flags"][k] for k in keys if k != "volume")]
                sel.sort(key=lambda r: r["score"], reverse=True)
            week[name] = st.mean(r["actual"] for r in sel[:TOP_N]) if len(sel) >= TOP_N else None
            week[f"{name}__picks"] = [r["tk"] for r in sel[:TOP_N]]
        per_week.append(week)

    if not per_week:
        print("No usable weeks.")
        return 1

    print(f"{'week':>12} {'basket':>8} " + " ".join(f"{n[:20]:>20}" for n in VARIANTS))
    for w in per_week:
        cells = " ".join(
            f"{w[n]:>19.2f}%" if w.get(n) is not None else f"{'n/a':>20}" for n in VARIANTS)
        print(f"{w['cutoff']:>12} {w['basket']:>7.2f}% {cells}")

    base_name = "old (quality+volume)"
    print(f"\n{'=' * 96}")
    print(f"{'variant':>24} {'weeks':>6} {'mean':>9} {'median':>9} {'>basket':>9} "
          f"{'vs old':>9} {'old wins':>9}")
    for name in VARIANTS:
        vals = [(w[name], w["basket"], w[base_name]) for w in per_week
                if w.get(name) is not None and w.get(base_name) is not None]
        if not vals:
            continue
        rets = [v[0] for v in vals]
        beat = 100 * sum(1 for r, b, _ in vals if r > b) / len(vals)
        diffs = [r - o for r, _, o in vals]
        wins = sum(1 for d in diffs if d > 0)
        print(f"{name:>24} {len(vals):>6} {st.mean(rets):>8.2f}% {st.median(rets):>8.2f}% "
              f"{beat:>8.1f}% {st.mean(diffs):>+8.2f}pp {wins:>4}/{len(diffs)}")
    print("=" * 96)
    print("'vs old' is the paired per-week difference; 'old wins' counts weeks the "
          "variant beat the incumbent (a sign test — with ~12 weeks, 9+ is the "
          "rough threshold for signal rather than noise).")

    changed = [w for w in per_week
               if set(w[f"{base_name}__picks"]) != set(w["NEW (both, as shipped)__picks"])]
    print(f"\nWeeks where the new filters changed the pick list: {len(changed)}/{len(per_week)}")
    for w in changed[:6]:
        print(f"  {w['cutoff']}: {w[f'{base_name}__picks']} -> "
              f"{w['NEW (both, as shipped)__picks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
