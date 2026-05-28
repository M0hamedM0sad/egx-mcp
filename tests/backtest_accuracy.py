"""Backtest the decision model's ACCURACY — are its verdicts right?

The portfolio backtests (backtest.py, agentic_backtest.py) answer "does the
strategy make money?". This answers the adjacent question: "when the model
says BUY, is it actually right?" — framed as classification accuracy, across
multiple holding horizons.

What "right" means here is BEATING THE BENCHMARK, not just going up. EGX is
in a strong bull (the sample window is broadly +30%), so an absolute hit-rate
flatters every band — a model that screamed BUY at everything would look
great. So the headline metric is excess return vs EGX 30:

  buy-side  (BUY, ACCUMULATE)  is correct when it BEATS the index
  sell-side (REDUCE, AVOID)    is correct when it LAGS the index
  HOLD                         makes no directional claim -> excluded

Outputs per horizon (5 / 21 / 63 trading days):
  - per-band: n, mean return, mean excess, hit-rate, beat-benchmark rate
  - directional accuracy (the headline number) + buy/sell precision
  - confusion matrix: band x {beat, lagged}
  - monotonicity of mean excess (BUY > ACCUMULATE > ... > AVOID)

HONEST SCOPE: this grades the point-in-time-clean price-momentum core
(backtest._score_at) mapped to chairman bands — the same engine
agentic_backtest trusts. The FULL live model adds fundamentals, macro,
catalyst, and the new sentiment layer, none of which are historically
replayable without lookahead. To measure those, snapshot live verdicts
(your daily briefings) and grade them forward — see SCOPE note at the end
of the run.

    python -m tests.backtest_accuracy
    python -m tests.backtest_accuracy --start 2023-01-01 --universe extended
    python -m tests.backtest_accuracy --horizons 5,21
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

from egx_mcp.data import backtest as bt_mod
from egx_mcp.data import egx_listing
from egx_mcp.data.agentic_backtest import DEFAULT_BANDS, _band_for_pct, _benchmark_series
from egx_mcp.data.universe import EGX_UNIVERSE

_BUY_SIDE = {"BUY", "ACCUMULATE"}
_SELL_SIDE = {"REDUCE", "AVOID"}
_BAND_ORDER = ["BUY", "ACCUMULATE", "HOLD", "REDUCE", "AVOID"]


def _regime_at(bench: pd.Series | None, date: pd.Timestamp) -> str:
    """Lookahead-free regime label from the benchmark up to `date`.

    BULL when price is above its ~200d mean AND trailing 60d return is
    positive; otherwise CORRECTION. Returns 'unknown' without a benchmark
    or enough history. Used to test whether the model's edge survives
    outside the bull (the #1 way EGX models flatter themselves)."""
    if bench is None:
        return "unknown"
    ser = bench.loc[:date].dropna()
    if len(ser) < 60:
        return "unknown"
    p = float(ser.iloc[-1])
    ma = float(ser.tail(200).mean())
    ret60 = p / float(ser.iloc[-60]) - 1
    return "bull" if (p > ma and ret60 > 0) else "correction"


def _accuracy(calls: list[dict]) -> dict:
    """Directional accuracy over a list of {band, excess} call records."""
    dirn = [c for c in calls if c["band"] in _BUY_SIDE or c["band"] in _SELL_SIDE]
    if not dirn:
        return {"n": 0, "accuracy_pct": None}
    correct = sum(1 for c in dirn
                  if (c["band"] in _BUY_SIDE and c["excess"] > 0)
                  or (c["band"] in _SELL_SIDE and c["excess"] < 0))
    return {"n": len(dirn), "accuracy_pct": round(correct / len(dirn) * 100, 1)}


def _grade_horizon(panel: pd.DataFrame, bench: pd.Series | None,
                   start: str, horizon: int,
                   holdout_start: pd.Timestamp | None = None) -> dict:
    """Walk forward in `horizon`-day steps; grade every verdict vs benchmark."""
    start_ts = pd.Timestamp(start)
    in_window = panel.index[panel.index >= start_ts]
    rebalance_idx = in_window[::horizon]
    if len(rebalance_idx) < 3:
        return {"error": f"not enough periods at horizon={horizon}"}

    # Per-band accumulators of (forward_return, excess_vs_benchmark)
    rets: dict[str, list[float]] = {b: [] for b in DEFAULT_BANDS}
    excess: dict[str, list[float]] = {b: [] for b in DEFAULT_BANDS}
    calls: list[dict] = []  # per (name, period) record, tagged regime + segment
    n_periods = 0

    for i in range(len(rebalance_idx) - 1):
        date, next_date = rebalance_idx[i], rebalance_idx[i + 1]
        scores = bt_mod._score_at(panel, date)
        if not scores:
            continue

        # Rank to within-period percentile, then to a chairman band.
        items = sorted(scores.items(), key=lambda x: x[1])
        n = len(items)
        verdicts = {tk: _band_for_pct((j + 1) / n, DEFAULT_BANDS)
                    for j, (tk, _) in enumerate(items)}

        period_panel = panel.loc[date:next_date].dropna(how="all")
        if len(period_panel) < 2:
            continue
        fwd = (period_panel.iloc[-1] / period_panel.iloc[0] - 1)

        # Benchmark return over the same dates (0 if unavailable).
        bench_ret = 0.0
        if bench is not None:
            b = bench.reindex(period_panel.index, method="ffill").dropna()
            if len(b) >= 2 and b.iloc[0] > 0:
                bench_ret = float(b.iloc[-1] / b.iloc[0] - 1)

        regime = _regime_at(bench, date)
        segment = "OOS" if (holdout_start is not None and date >= holdout_start) else "IS"
        for tk, band in verdicts.items():
            r = fwd.get(tk)
            if r is None or pd.isna(r):
                continue
            ex = float(r) - bench_ret
            rets[band].append(float(r))
            excess[band].append(ex)
            calls.append({"band": band, "excess": ex, "regime": regime, "segment": segment})
        n_periods += 1

    # Per-band stats
    by_band: dict[str, dict] = {}
    for band in DEFAULT_BANDS:
        r = np.array(rets[band])
        e = np.array(excess[band])
        if len(r) == 0:
            by_band[band] = {"n": 0}
            continue
        by_band[band] = {
            "n": int(len(r)),
            "mean_ret_pct": round(float(r.mean()) * 100, 2),
            "mean_excess_pct": round(float(e.mean()) * 100, 2),
            "hit_rate_pct": round(float((r > 0).mean()) * 100, 1),
            "beat_bench_pct": round(float((e > 0).mean()) * 100, 1),
        }

    # Directional accuracy: buy-side correct if it beat the index; sell-side
    # correct if it lagged. HOLD excluded (no directional claim).
    buy_e = np.array([x for b in _BUY_SIDE for x in excess[b]])
    sell_e = np.array([x for b in _SELL_SIDE for x in excess[b]])
    buy_correct = int((buy_e > 0).sum())
    sell_correct = int((sell_e < 0).sum())
    n_dir = len(buy_e) + len(sell_e)
    accuracy = (buy_correct + sell_correct) / n_dir if n_dir else 0.0

    # Monotonicity of mean excess across the band order.
    means = [by_band[b].get("mean_excess_pct") for b in _BAND_ORDER]
    monotone = all(
        a is not None and b is not None and a >= b
        for a, b in zip(means, means[1:])
    )
    spread = (round(means[0] - means[-1], 2)
              if means[0] is not None and means[-1] is not None else None)

    # A3: accuracy conditioned on regime and on in-sample vs out-of-sample.
    by_regime = {reg: _accuracy([c for c in calls if c["regime"] == reg])
                 for reg in sorted({c["regime"] for c in calls})}
    by_segment = {seg: _accuracy([c for c in calls if c["segment"] == seg])
                  for seg in sorted({c["segment"] for c in calls})}

    return {
        "horizon_days": horizon,
        "n_periods": n_periods,
        "by_band": by_band,
        "directional_accuracy_pct": round(accuracy * 100, 1),
        "buy_precision_pct": round(buy_correct / len(buy_e) * 100, 1) if len(buy_e) else None,
        "sell_precision_pct": round(sell_correct / len(sell_e) * 100, 1) if len(sell_e) else None,
        "buy_n": int(len(buy_e)),
        "sell_n": int(len(sell_e)),
        "monotone_excess": monotone,
        "buy_minus_avoid_excess_pct": spread,
        "by_regime": by_regime,
        "by_segment": by_segment,
    }


def _print_horizon(res: dict) -> None:
    if "error" in res:
        print(f"\n  horizon: {res['error']}")
        return
    h = res["horizon_days"]
    print(f"\n{'=' * 72}")
    print(f"HORIZON = {h} trading days   ({res['n_periods']} rebalance periods)")
    print('=' * 72)

    hdr = f"{'band':>11} | {'n':>5} {'mean_ret':>9} {'mean_exc':>9} {'hit%':>6} {'beat_bm%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for b in _BAND_ORDER:
        s = res["by_band"].get(b, {"n": 0})
        if s["n"] == 0:
            print(f"{b:>11} | {0:>5}  (empty)")
            continue
        print(f"{b:>11} | {s['n']:>5} {s['mean_ret_pct']:>8.2f}% {s['mean_excess_pct']:>8.2f}% "
              f"{s['hit_rate_pct']:>5.1f}% {s['beat_bench_pct']:>8.1f}%")

    print(f"\n  Directional accuracy (vs benchmark): "
          f"{res['directional_accuracy_pct']:.1f}%   "
          f"[buy-side {res['buy_precision_pct']}% of {res['buy_n']}, "
          f"sell-side {res['sell_precision_pct']}% of {res['sell_n']}]")
    print(f"  Monotone excess (BUY>...>AVOID): {res['monotone_excess']}   "
          f"BUY-minus-AVOID excess: {res['buy_minus_avoid_excess_pct']}%")

    reg = res.get("by_regime", {})
    if reg:
        parts = [f"{k}={v['accuracy_pct']}% (n={v['n']})" for k, v in reg.items()]
        print(f"  Accuracy by regime: {'   '.join(parts)}")
    seg = res.get("by_segment", {})
    if seg and "OOS" in seg:
        parts = [f"{k}={v['accuracy_pct']}% (n={v['n']})" for k, v in seg.items()]
        print(f"  In-sample vs out-of-sample: {'   '.join(parts)}  "
              "<- OOS holding near IS is the trust signal")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest decision-model verdict accuracy.")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--universe", choices=["extended", "curated"], default="extended")
    ap.add_argument("--horizons", default="5,21,63",
                    help="Comma-separated holding horizons in trading days.")
    ap.add_argument("--holdout-months", type=int, default=None,
                    help="Reserve the last N months as out-of-sample; reports IS vs OOS "
                         "accuracy so you can see if the edge is real or curve-fit.")
    args = ap.parse_args()

    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    if args.universe == "extended":
        tickers = egx_listing.get_full_universe()
    else:
        tickers = [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]
    if not tickers:
        print("Empty universe.")
        return 1

    print(f"Backtesting verdict accuracy — universe={args.universe} ({len(tickers)} names), "
          f"window {args.start}..{end}, horizons={horizons}")
    print("Fetching price panel (one network pull, reused across horizons)...")

    # One panel + benchmark, reused for every horizon.
    fetch_start = (datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    panel = bt_mod._price_panel(tickers, start=fetch_start, end=end)
    if panel.empty:
        print("No price history fetched (network / SSL issue?). Nothing to grade.")
        return 1
    bench = _benchmark_series(args.start, end)
    if bench is None:
        print("WARNING: EGX 30 benchmark unavailable — excess metrics fall back to "
              "absolute (everything compared to 0%). Interpret with care in a bull market.")

    holdout_start = None
    if args.holdout_months:
        holdout_start = panel.index[-1] - pd.Timedelta(days=args.holdout_months * 30)
        print(f"Out-of-sample holdout: from {holdout_start.date()} onward "
              f"(last {args.holdout_months} months).")

    for h in horizons:
        _print_horizon(_grade_horizon(panel, bench, args.start, h, holdout_start))

    print(f"\n{'#' * 72}")
    print("# SCOPE — what this does and does NOT cover")
    print('#' * 72)
    print("# Grades the point-in-time-clean PRICE-MOMENTUM core (backtest._score_at)")
    print("# mapped to chairman bands. The full live model also uses fundamentals,")
    print("# macro, catalysts, and sentiment — NONE replayable without lookahead.")
    print("# To grade the FULL model incl. the new sentiment layer: persist daily")
    print("# briefing verdicts (you have briefings/*.json) and grade them forward")
    print("# against realized returns once you have 30+ snapshots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
