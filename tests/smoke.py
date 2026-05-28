"""Smoke test — runs the data layer and decision layer directly without the MCP transport.

Use this to verify that yfinance, the disclosures scraper, the portfolio
CSV loader, and the new decision tools all work in your environment
before debugging Claude Desktop.

    python -m tests.smoke
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import (
    market, technicals, universe, portfolio,
    fundamentals, macro, scoring, peers, sizing, decision,
)
from egx_mcp.data import calendar as cal_mod


def _print(label: str, payload: dict, limit: int = 1500) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(json.dumps(payload, indent=2, default=str)[:limit])


def main() -> int:
    print("Running EGX MCP smoke tests...\n")

    failures = []

    def _try(label, fn, limit=1500):
        try:
            result = fn()
            payload = result if isinstance(result, dict) else {"value": result}
            _print(label, payload, limit=limit)
            return result
        except Exception as e:
            print(f"\nFAIL {label}: {e}")
            failures.append(label)
            return None

    # --- Raw data layer ---
    _try("get_quote('CIRA')", lambda: market.get_quote("CIRA"))
    _try("get_index('EGX30')", lambda: market.get_index("EGX30"))

    h = _try("get_history('COMI', '3mo')",
             lambda: {k: (v if k != "rows" else f"{len(v)} rows")
                      for k, v in market.get_history("COMI", period="3mo").items()})

    _try("compute_technicals('COMI')", lambda: technicals.compute("COMI", period="6mo"))
    _try("list_egx_stocks(sector='Real Estate')",
         lambda: universe.list_stocks(sector="Real Estate"))

    sample = Path(__file__).parent.parent / "sample_portfolio.csv"
    if sample.exists():
        _try("portfolio_summary(sample_portfolio.csv)",
             lambda: {k: (v if k != "positions" else f"{len(v)} positions")
                      for k, v in portfolio.summary(csv_path=str(sample)).items()})

    # --- Decision layer ---
    print("\n\n" + "#" * 60)
    print("# DECISION LAYER")
    print("#" * 60)

    _try("get_fundamentals('CIRA')", lambda: fundamentals.get_fundamentals("CIRA"))
    _try("get_macro_context()", lambda: macro.get_context())

    _try("score_stock('COMI')",
         lambda: {k: v for k, v in scoring.score_stock("COMI").items()
                  if k != "subscores"} | {"subscores_summary": {
                      cat: f"{sub['score']} ({len(sub.get('notes', []))} notes)"
                      for cat, sub in scoring.score_stock("COMI").get("subscores", {}).items()
                  }})

    _try("compare_peers('COMI')",
         lambda: {k: v for k, v in peers.compare("COMI", max_peers=5).items()
                  if k != "peers"} | {"peers_summary": [
                      f"{p['ticker']} score={p.get('composite_score')}"
                      for p in peers.compare("COMI", max_peers=5).get("peers", [])
                  ]})

    _try("position_size('SWDY', 500000, risk_pct=1)",
         lambda: sizing.position_size("SWDY", 500_000, risk_pct=1))

    _try("get_catalyst_calendar('CIRA')", lambda: cal_mod.get_calendar("CIRA"))

    # The headline tool
    _try("decide('COMI', portfolio_value_egp=500000)",
         lambda: decision.decide("COMI", portfolio_value_egp=500_000),
         limit=3000)

    print(f"\n\n{'=' * 60}")
    if failures:
        print(f"SMOKE FAILED ({len(failures)} errors): {', '.join(failures)}")
        return 1
    print("SMOKE PASSED — all tools responded.")
    print('=' * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
