"""Run the short-term Monte Carlo across the EGX universe.

    python -m tests.scan_short_term
    python -m tests.scan_short_term 1   # 1-day horizon
    python -m tests.scan_short_term 5   # default

Prints a ranked table of names with the highest probability of an
upside move within the chosen horizon.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import simulation


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Scanning EGX universe — Monte Carlo horizon = {horizon} day(s)\n")

    out = simulation.scan_universe(
        horizon_days=horizon,
        n_paths=2000,
        lookback_days=60,
        min_prob_up_2pct=0.0,           # show all
        min_expected_return_pct=-100.0,  # show all
        seed=42,
        full_market=True,                # ~70 names instead of 29
    )

    rows = out["ranked"]
    print(f"Universe scanned: {out['universe_scanned']}")
    print(f"Successfully simulated: {out['ranked_count']}")
    print(f"Skipped: {out['skipped_count']}")
    if out["skipped"]:
        print(f"  Skipped tickers: {[s['ticker'] for s in out['skipped']]}\n")

    print("=" * 110)
    print(f"RANKED — {horizon}-day horizon — sorted by imminent_move_score")
    print("=" * 110)
    hdr = (
        f"{'Rank':<5}{'Ticker':<8}{'Sector':<22}"
        f"{'Price':>9}{'E[ret]%':>10}{'P(>+2%)':>10}{'P(>+5%)':>10}"
        f"{'P(<-5%)':>10}{'p90 ret%':>10}{'Score':>9}"
    )
    print(hdr)
    print("-" * 110)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<5}{r['ticker']:<8}{(r['sector'] or '')[:21]:<22}"
            f"{r['current_price']:>9.2f}{r['expected_return_pct']:>10.2f}"
            f"{r['prob_up_2pct']:>10.2%}{r['prob_up_5pct']:>10.2%}"
            f"{r['prob_down_5pct']:>10.2%}{r['p90_return_pct']:>10.2f}"
            f"{r['imminent_move_score']:>9.1f}"
        )

    print("\n" + "=" * 110)
    print("TOP 5 — DRIVERS")
    print("=" * 110)
    for r in rows[:5]:
        print(f"\n{r['ticker']}  ({r['sector']})  score={r['imminent_move_score']}")
        print(f"  Price: {r['current_price']} | E[ret]: {r['expected_return_pct']:+.2f}%"
              f" | P(up>2%): {r['prob_up_2pct']:.2%} | P(up>5%): {r['prob_up_5pct']:.2%}")
        print(f"  90% CI: [{r['p10_price']:.2f} , {r['p90_price']:.2f}]"
              f"  ({r['p10_return_pct']:+.1f}% to {r['p90_return_pct']:+.1f}%)")
        print(f"  Edge drift: {r['edge_drift_pct_per_day']:+.3f}%/day")
        if r["edge_drivers"]:
            for d in r["edge_drivers"]:
                print(f"    • {d}")
    print()


if __name__ == "__main__":
    main()
