"""Statistical reliability validation — is the model's edge REAL or luck?

The reliability scorecard (scripts/reliability_status.py) checks the graded
verdicts against flat thresholds (accuracy >= 55%, mean excess sign). That is
not enough to RELY on the model, for two reasons this harness fixes:

  1. SIGN-CONVENTION TRAP. "mean excess vs benchmark" does not flip sign for
     sell-side calls. A REDUCE that is WRONG (the stock soars, beating the
     index) shows a *positive* excess and flatters the headline. The honest
     metric is the direction-aware SIGNED EDGE:
         buy-side  -> +excess_pct   (you wanted it to beat the index)
         sell-side -> -excess_pct   (you wanted it to lag the index)
     A positive mean signed edge is the only thing that means "the calls,
     taken in their stated direction, added value."

  2. NO UNCERTAINTY, NO INDEPENDENCE. A point estimate off N rows says nothing
     about whether it would survive another sample. Worse, the rows are NOT
     independent: the same (date, ticker) call is graded at both 5d and 21d,
     and every call in one briefing shares that day's market. So the EFFECTIVE
     sample is the number of distinct briefing dates, not the row count. This
     harness reports that, runs a cluster bootstrap by briefing date for the
     signed-edge CI, and an exact binomial test on the hit-rate.

Output is a single honest verdict: EDGE_CONFIRMED / INCONCLUSIVE /
NEGATIVE_EDGE / INSUFFICIENT_SAMPLE — plus the breakdowns that justify it.

    python -m tests.validate_edge
    python -m tests.validate_edge --in logs/graded_verdicts.jsonl --boot 20000
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
_DEFAULT_IN = ROOT / "logs" / "graded_verdicts.jsonl"

_BUY = {"BUY", "ACCUMULATE", "WEEKLY_BUY"}
_SELL = {"REDUCE", "AVOID", "SELL"}

# Reliability gates. Independence-aware: clusters are distinct briefing dates.
_MIN_CALLS = 30          # graded directional rows before any stat is reported
_MIN_CLUSTERS = 8        # distinct briefing dates for a CI to be trustworthy
_CONV_ORDER = ["high", "medium", "low", "weekly"]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _directional(rows: list[dict]) -> list[dict]:
    """Graded rows that make a directional claim, annotated with signed edge."""
    out = []
    for r in rows:
        if r.get("outcome") != "graded" or r.get("correct") is None:
            continue
        v = r.get("verdict")
        exc = r.get("excess_pct")
        if exc is None or v not in _BUY | _SELL:
            continue
        r = dict(r)
        r["signed_edge"] = exc if v in _BUY else -exc
        r["side"] = "buy" if v in _BUY else "sell"
        out.append(r)
    return out


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (no scipy) for k successes in n at p."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    # two-sided: sum of all outcomes no more likely than the observed one
    return min(1.0, sum(pr for pr in probs if pr <= obs + 1e-12))


def _cluster_bootstrap_ci(rows: list[dict], key: str, n_boot: int,
                          seed: int = 12345) -> tuple[float, float, float]:
    """Mean of `key` with a 95% CI, resampling whole briefing-date clusters.

    Clustering by briefing date is what makes the CI honest: it propagates the
    fact that calls sharing a date are not independent draws.
    """
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cluster[r["briefing_date"]].append(r[key])
    clusters = list(by_cluster.values())
    point = float(np.mean([x for c in clusters for x in c]))
    if len(clusters) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(clusters)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = [x for i in idx for x in clusters[i]]
        means[b] = np.mean(vals)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    if math.isnan(lo):
        return f"{point:+.2f}% (CI n/a — too few clusters)"
    return f"{point:+.2f}%  95% CI [{lo:+.2f}%, {hi:+.2f}%]"


def _hitrate_block(rows: list[dict], label: str) -> list[str]:
    n = len(rows)
    k = sum(1 for r in rows if r["correct"])
    if n == 0:
        return [f"{label:18} n=0"]
    acc = k / n * 100
    p = _binom_two_sided_p(k, n)
    star = "  (p<0.05)" if p < 0.05 else ""
    return [f"{label:18} n={n:3}  hit={acc:5.1f}%  (binomial vs 50%: p={p:.3f}){star}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(_DEFAULT_IN))
    ap.add_argument("--boot", type=int, default=20000, help="bootstrap resamples")
    args = ap.parse_args()

    rows = _directional(_load(Path(args.infile)))
    print("=" * 72)
    print("EGX MODEL — STATISTICAL RELIABILITY VALIDATION")
    print("=" * 72)

    if not rows:
        print("\nNo graded directional calls found. Run tests.grade_briefings first.")
        return 1

    clusters = sorted({r["briefing_date"] for r in rows})
    n, n_clu = len(rows), len(clusters)
    print(f"\ngraded directional calls : {n}")
    print(f"distinct briefing dates  : {n_clu}  -> EFFECTIVE independent sample")
    print(f"dates: {', '.join(clusters)}")
    if n_clu < _MIN_CLUSTERS:
        print(f"(!) only {n_clu} independent days. Row count {n} overstates the sample: "
              f"each\n    call is graded at 2 horizons and calls share a market day. "
              f"Treat every\n    statistic below as provisional until >= {_MIN_CLUSTERS} dates exist.")

    # --- 1. Direction-aware signed edge (the honest edge metric) -------------
    naive = float(np.mean([r["excess_pct"] for r in rows]))
    pt, lo, hi = _cluster_bootstrap_ci(rows, "signed_edge", args.boot)
    print("\n" + "-" * 72)
    print("1. SIGNED EDGE  (excess in the call's own direction; the metric that")
    print("   actually means 'the calls added value')")
    print("-" * 72)
    print(f"  naive mean excess (sign-blind, MISLEADING) : {naive:+.2f}%")
    print(f"  mean SIGNED edge (direction-aware, HONEST) : {_fmt_ci(pt, lo, hi)}")
    edge_pos = (not math.isnan(lo)) and lo > 0
    edge_neg = (not math.isnan(hi)) and hi < 0
    if edge_pos:
        print("  -> CI strictly above 0: edge is positive beyond resampling noise.")
    elif edge_neg:
        print("  -> CI strictly below 0: calls LOST value vs the index, beyond noise.")
    else:
        print("  -> CI spans 0: cannot distinguish the edge from luck at this sample.")

    # --- 2. Directional hit-rate, exact binomial -----------------------------
    print("\n" + "-" * 72)
    print("2. DIRECTIONAL HIT-RATE  (exact binomial vs a 50% coin)")
    print("-" * 72)
    out = _hitrate_block(rows, "pooled")
    for hd in sorted({r["horizon_days"] for r in rows}):
        sub = [r for r in rows if r["horizon_days"] == hd]
        out += _hitrate_block(sub, f"horizon {hd}d")
    for side in ("buy", "sell"):
        sub = [r for r in rows if r["side"] == side]
        if sub:
            out += _hitrate_block(sub, f"{side}-side")
    for ln in out:
        print("  " + ln)

    # --- 3. Calibration: conviction should track accuracy --------------------
    print("\n" + "-" * 72)
    print("3. CALIBRATION  (does higher conviction = higher hit-rate?)")
    print("-" * 72)
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_conv[r.get("conviction") or "unknown"].append(r)
    present = [c for c in _CONV_ORDER if c in by_conv] + \
              [c for c in by_conv if c not in _CONV_ORDER]
    accs = []
    for c in present:
        sub = by_conv[c]
        acc = sum(1 for r in sub if r["correct"]) / len(sub) * 100
        se = float(np.mean([r["signed_edge"] for r in sub]))
        accs.append((c, acc, len(sub)))
        print(f"  {c:10} n={len(sub):3}  hit={acc:5.1f}%  signed_edge={se:+.2f}%")
    ranked = [a for c, a, k in accs if c in ("high", "medium", "low") and k >= 5]
    monotone = len(ranked) >= 2 and all(x >= y for x, y in zip(ranked, ranked[1:]))
    if len(ranked) >= 2:
        print(f"  conviction tracks accuracy (monotone): {monotone}")
    else:
        print("  not enough graded calls in >=2 ranked buckets to judge calibration.")

    # --- 4. Regime robustness ------------------------------------------------
    print("\n" + "-" * 72)
    print("4. REGIME ROBUSTNESS  (edge split by the benchmark's own direction")
    print("   over each call's window — a bull-only edge is the #1 EGX blow-up)")
    print("-" * 72)
    up = [r for r in rows if (r.get("bench_return_pct") or 0) > 0]
    dn = [r for r in rows if (r.get("bench_return_pct") or 0) <= 0]
    for label, sub in (("bench up   (bull)", up), ("bench down (corr)", dn)):
        if sub:
            se = float(np.mean([r["signed_edge"] for r in sub]))
            acc = sum(1 for r in sub if r["correct"]) / len(sub) * 100
            print(f"  {label}: n={len(sub):3}  hit={acc:5.1f}%  signed_edge={se:+.2f}%")
        else:
            print(f"  {label}: n=0  (no calls in this regime — robustness UNTESTED here)")

    # --- Verdict -------------------------------------------------------------
    print("\n" + "=" * 72)
    if n < _MIN_CALLS or n_clu < _MIN_CLUSTERS:
        verdict = "INSUFFICIENT_SAMPLE"
        msg = (f"Need >= {_MIN_CALLS} calls across >= {_MIN_CLUSTERS} distinct dates "
               f"(have {n} across {n_clu}). The edge is UNPROVEN — not negative, "
               "just unmeasured. Keep accumulating briefings.")
    elif edge_pos:
        verdict = "EDGE_CONFIRMED"
        msg = "Signed edge is positive beyond resampling noise. Reliable only within a "\
              "risk-managed, human-supervised process — re-validate after regime change."
    elif edge_neg:
        verdict = "NEGATIVE_EDGE"
        msg = "The calls, taken in their stated direction, LOST to the index beyond "\
              "noise. Do not rely on the verdicts; investigate before trading them."
    else:
        verdict = "INCONCLUSIVE"
        msg = "Sample is adequate but the edge is statistically indistinguishable from "\
              "zero. Decision-support only; do not size up on conviction yet."
    print(f"VERDICT: {verdict}")
    for line in (msg[i:i + 70] for i in range(0, len(msg), 70)):
        print("  " + line)
    print("=" * 72)
    # Exit non-zero unless an edge is actually confirmed, so CI/automation can gate.
    return 0 if verdict == "EDGE_CONFIRMED" else 2


if __name__ == "__main__":
    sys.exit(main())
