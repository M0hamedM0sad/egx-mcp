"""Smoke test for the portfolio-manager layer.

    python -m tests.smoke_pm

Exercises every new module: risk_free, regime, liquidity, factors,
risk, optimizer, backtest. Uses a small candidate basket so the test
finishes in reasonable time.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import (
    risk_free, regime, liquidity, factors,
    risk as risk_mod, optimizer, backtest,
)


def show(label, payload, limit=2000):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(json.dumps(payload, indent=2, default=str)[:limit])


def main():
    print("Running PM-layer smoke tests...\n")
    failures = []

    def trial(label, fn):
        try:
            r = fn()
            show(label, r)
            return r
        except Exception as e:
            print(f"\nFAIL {label}: {type(e).__name__}: {e}")
            failures.append(label)
            return None

    trial("get_egp_risk_free_rate()", lambda: risk_free.get_rate())
    trial("detect_market_regime()", lambda: regime.classify())
    trial("check_liquidity('COMI', 1000)", lambda: liquidity.check_capacity("COMI", 1000))

    sample_basket = ["COMI", "HDBK", "ABUK", "CIRA", "SWDY"]

    trial(f"portfolio_factor_exposure({sample_basket})",
          lambda: factors.portfolio_factor_exposure(sample_basket))

    trial(f"portfolio_risk({sample_basket}, NAV=500K)",
          lambda: risk_mod.portfolio_risk(sample_basket, nav_egp=500_000))

    trial(f"optimize_portfolio({sample_basket}, min_variance, NAV=500K)",
          lambda: optimizer.optimize(sample_basket, method="min_variance", nav_egp=500_000))

    trial(f"optimize_portfolio({sample_basket}, risk_parity)",
          lambda: optimizer.optimize(sample_basket, method="risk_parity", target_vol_pct=25.0))

    expected = {"COMI": 18.0, "HDBK": 30.0, "ABUK": 22.0, "CIRA": 12.0, "SWDY": 8.0}
    trial(f"optimize_portfolio({sample_basket}, tangency, with expected returns)",
          lambda: optimizer.optimize(sample_basket, method="tangency",
                                      expected_returns_pct=expected, nav_egp=500_000))

    print("\n# Backtest will take ~30-60s on the curated universe (smaller, faster)\n")
    trial("backtest_strategy(2024-01-01 → today, top_n=5, curated)",
          lambda: backtest.backtest(start="2024-01-01", top_n=5,
                                     rebalance_days=21, universe="curated"))

    print(f"\n{'=' * 70}")
    if failures:
        print(f"PM SMOKE FAILED ({len(failures)}): {failures}")
        return 1
    print("PM SMOKE PASSED — all 8 PM-layer modules responded.")
    print('=' * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
