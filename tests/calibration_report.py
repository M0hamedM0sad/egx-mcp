"""A2 — Calibration: does higher conviction actually mean higher accuracy?

A model you can rely on isn't just accurate — it knows when it's unsure. This
reads the graded verdict dataset produced by tests/grade_briefings.py and
checks whether the model's stated conviction (high / medium / low / weekly)
lines up with realized hit-rate. A model that's 60% accurate but well-
calibrated is more usable than a 70% one that's confidently wrong half the
time it claims "high".

Reports, over elapsed (graded) calls only:
  - hit-rate, beat-benchmark rate, mean excess, n — per conviction bucket
  - per verdict band (BUY / ACCUMULATE / HOLD / REDUCE / AVOID / WEEKLY_BUY)
  - a monotonicity verdict: does accuracy rise with conviction?

    python -m tests.calibration_report
    python -m tests.calibration_report --in logs/graded_verdicts.jsonl

Needs a meaningful sample (aim for 30+ graded calls per bucket) before the
numbers mean anything — it prints n so you can judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_BUY_SIDE = {"BUY", "ACCUMULATE", "WEEKLY_BUY"}
_SELL_SIDE = {"REDUCE", "AVOID", "SELL"}
# Conviction order from most to least confident, for the monotonicity check.
_CONV_ORDER = ["high", "medium", "low"]


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stats(rows: list[dict]) -> dict | None:
    """Hit-rate / beat-bench / mean-excess over graded, directional rows."""
    sub = [r for r in rows if r.get("outcome") == "graded" and r.get("correct") is not None]
    if not sub:
        return None
    n = len(sub)
    hit = sum(1 for r in sub if r["correct"]) / n
    exc = [r["excess_pct"] for r in sub if r.get("excess_pct") is not None]
    beat = (sum(1 for e in exc if e > 0) / len(exc)) if exc else None
    mean_exc = (sum(exc) / len(exc)) if exc else None
    return {
        "n": n,
        "accuracy_pct": round(hit * 100, 1),
        "beat_bench_pct": round(beat * 100, 1) if beat is not None else None,
        "mean_excess_pct": round(mean_exc, 2) if mean_exc is not None else None,
    }


def _table(title: str, groups: dict[str, list[dict]], order: list[str]) -> dict[str, dict]:
    print(f"\n{title}")
    print(f"  {'bucket':>12} | {'n':>4} {'accuracy':>9} {'beat_bm':>8} {'mean_exc':>9}")
    print("  " + "-" * 48)
    out: dict[str, dict] = {}
    keys = order + [k for k in groups if k not in order]
    for k in keys:
        s = _stats(groups.get(k, []))
        out[k] = s
        if not s:
            continue
        beat = f"{s['beat_bench_pct']:.1f}%" if s["beat_bench_pct"] is not None else "  n/a"
        exc = f"{s['mean_excess_pct']:+.2f}%" if s["mean_excess_pct"] is not None else "   n/a"
        print(f"  {k:>12} | {s['n']:>4} {s['accuracy_pct']:>8.1f}% {beat:>8} {exc:>9}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Conviction calibration report (A2).")
    ap.add_argument("--in", dest="inp",
                    default=str(Path(__file__).parent.parent / "logs" / "graded_verdicts.jsonl"))
    ap.add_argument("--horizon", type=int, default=None,
                    help="Restrict to one horizon (e.g. 5). Default: all.")
    args = ap.parse_args()

    path = Path(args.inp)
    if not path.exists():
        print(f"Graded dataset not found: {path}\nRun: python -m tests.grade_briefings")
        return 1

    rows = _load(path)
    if args.horizon is not None:
        rows = [r for r in rows if r.get("horizon_days") == args.horizon]

    graded = [r for r in rows if r.get("outcome") == "graded" and r.get("correct") is not None]
    print(f"Loaded {len(rows)} rows ({len(graded)} graded directional calls"
          + (f", horizon={args.horizon}d" if args.horizon else "") + ")")
    if not graded:
        print("Nothing graded yet — let more time elapse since the briefings, then re-run "
              "tests.grade_briefings.")
        return 0

    by_conv: dict[str, list[dict]] = {}
    by_verdict: dict[str, list[dict]] = {}
    for r in graded:
        by_conv.setdefault((r.get("conviction") or "unknown"), []).append(r)
        by_verdict.setdefault(r["verdict"], []).append(r)

    conv_stats = _table("CALIBRATION BY CONVICTION", by_conv, _CONV_ORDER)
    _table("ACCURACY BY VERDICT BAND", by_verdict,
           ["BUY", "ACCUMULATE", "HOLD", "REDUCE", "AVOID", "WEEKLY_BUY"])

    # Monotonicity: accuracy should fall from high -> medium -> low.
    accs = [(c, conv_stats.get(c)) for c in _CONV_ORDER if conv_stats.get(c)]
    print(f"\n{'#' * 50}")
    if len(accs) >= 2:
        seq = [a[1]["accuracy_pct"] for a in accs]
        monotone = all(x >= y for x, y in zip(seq, seq[1:]))
        print(f"# Calibration check: {' >= '.join(f'{c}({s:.0f}%)' for (c, _), s in zip(accs, seq))}")
        print(f"# Conviction-monotone (confident = more accurate): {monotone}")
        if not monotone:
            print("# -> Conviction is NOT tracking accuracy. Either the sample is too small,")
            print("#    or the conviction logic needs recalibration before you trust it.")
    else:
        print("# Not enough distinct conviction buckets with data to judge calibration yet.")
    print('#' * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
