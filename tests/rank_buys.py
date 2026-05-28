"""Rank the curated universe by composite score, then run the full
decide() pipeline on the top names. Use this to answer the question:
"what should I buy tomorrow?"

    python -m tests.rank_buys
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Force UTF-8 stdout so EGX-related unicode (≤, %, etc.) doesn't crash
# the Windows cp1252 console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import scoring, decision
from egx_mcp.data.universe import EGX_UNIVERSE


def main():
    print("Ranking curated EGX universe...\n")
    results = []
    for ticker, meta in EGX_UNIVERSE.items():
        if meta["sector"] == "Index":
            continue
        try:
            sc = scoring.score_stock(ticker)
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")
            continue
        if "error" in sc:
            print(f"  {ticker}: skip ({sc['error']})")
            continue
        results.append({
            "ticker": ticker,
            "name": sc.get("name"),
            "sector": sc.get("sector"),
            "price": sc.get("price"),
            "score": sc.get("composite_score"),
            "valuation": sc["subscores"]["valuation"]["score"],
            "quality": sc["subscores"]["quality"]["score"],
            "momentum": sc["subscores"]["momentum"]["score"],
            "risk": sc["subscores"]["risk"]["score"],
        })
        print(f"  {ticker:6s} {sc.get('composite_score'):5.1f}  ({sc.get('sector')})")

    results.sort(key=lambda r: r["score"] or 0, reverse=True)

    print("\n" + "=" * 70)
    print("RANKED BY COMPOSITE SCORE")
    print("=" * 70)
    print(f"{'Rank':<5}{'Ticker':<8}{'Score':<8}{'Val':<6}{'Qual':<6}{'Mom':<6}{'Risk':<6}{'Sector':<20}")
    for i, r in enumerate(results, 1):
        print(f"{i:<5}{r['ticker']:<8}{r['score']:<8.1f}"
              f"{r['valuation']:<6.0f}{r['quality']:<6.0f}"
              f"{r['momentum']:<6.0f}{r['risk']:<6.0f}{r['sector']:<20}")

    print("\n" + "=" * 70)
    print("FULL DECISION ON TOP 5")
    print("=" * 70)
    for r in results[:5]:
        try:
            d = decision.decide(r["ticker"], portfolio_value_egp=500_000, risk_pct=1.0)
        except Exception as e:
            print(f"\n{r['ticker']}: decide() failed: {e}")
            continue
        print(f"\n--- {d['ticker']} ({d['name']}) ---")
        print(f"  Verdict:    {d['verdict']:12s}  conviction={d['conviction']}")
        print(f"  Score:      {d['composite_score']}")
        print(f"  Price:      {d['price']} EGP")
        if d.get("fair_value_estimate"):
            print(f"  Fair value: {d['fair_value_estimate']}  ({d['upside_pct']:+.1f}% upside)")
        if d.get("suggested_levels"):
            sl = d["suggested_levels"]
            print(f"  Entry/Stop/Target: {sl['entry']} / {sl['stop_loss']} / {sl['target']}")
            print(f"  Shares: {sl['shares']} ({sl['position_weight_pct']}% of NAV) — risk {d['composite_score'] and ''}")
        print(f"  Drivers:")
        for line in (d.get("key_drivers") or [])[:4]:
            print(f"    + {line}")
        if d.get("key_risks"):
            print(f"  Risks:")
            for line in d["key_risks"][:3]:
                print(f"    - {line}")
        if d.get("blocking_catalysts"):
            print(f"  BLOCKING: {d['blocking_catalysts']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
