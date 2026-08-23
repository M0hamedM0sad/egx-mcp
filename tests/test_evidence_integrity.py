"""Regression tests for the evidence-integrity and tiering changes.

These cover the three ways the gate could previously reach a confident verdict
from evidence it should not have trusted: corporate-action rows graded as
returns, a sample blended across model versions, and a positive mean carried by
one lucky date.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from egx_mcp.data import price_sanity, reliability, sizing


def _row(day: str, verdict: str, correct: bool, *, excess: float = 1.0,
         score: float = 60.0, conviction: str = "medium",
         outcome: str = "graded", version: str | None = None) -> dict:
    row = {
        "source": "v8b", "outcome": outcome, "horizon_days": 21,
        "briefing_date": day, "verdict": verdict, "correct": correct,
        "conviction": conviction, "excess_pct": excess, "score": score,
    }
    if version is not None:
        row["model_version"] = version
    return row


def _gate_over(rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "graded.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        with patch.object(reliability, "_GRADED", path):
            return reliability.status()


def _days(n: int) -> list[str]:
    return [(date.today() - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


class PriceSanityTests(unittest.TestCase):
    def test_flags_a_session_outside_the_daily_band(self) -> None:
        s = pd.Series([161.92, 160.0, 82.70, 83.0],
                      index=pd.to_datetime(["2026-06-24", "2026-06-25",
                                            "2026-06-26", "2026-06-29"]))
        brk = price_sanity.find_break(s)
        self.assertIsNotNone(brk)
        self.assertEqual(brk["date"], "2026-06-26")
        self.assertLess(brk["pct"], -25)

    def test_ordinary_limit_moves_are_not_flagged(self) -> None:
        s = pd.Series([10.0, 10.9, 11.9, 13.0],
                      index=pd.to_datetime(["2026-06-24", "2026-06-25",
                                            "2026-06-26", "2026-06-29"]))
        self.assertIsNone(price_sanity.find_break(s))

    def test_negative_and_zero_prices_are_dropped(self) -> None:
        s = pd.Series([1.0, -0.34, 0.0, 2.0],
                      index=pd.to_datetime(["2026-06-24", "2026-06-25",
                                            "2026-06-26", "2026-06-29"]))
        self.assertEqual(list(price_sanity.clean_series(s)), [1.0, 2.0])


class EvidenceScopeTests(unittest.TestCase):
    def test_quarantined_rows_never_reach_the_gate(self) -> None:
        days = _days(12)
        good = [_row(d, "BUY", True, excess=2.0) for d in days]
        poisoned = [_row(days[0], "BUY", False, excess=-271.0, outcome="quarantined")]
        gate = _gate_over(good + poisoned)
        self.assertEqual(gate["directional_calls"], len(good))
        self.assertGreater(gate["mean_date_signed_edge_pct"], 0)

    def test_evidence_is_scoped_to_the_active_model_version(self) -> None:
        days = _days(10)
        old = [_row(d, "BUY", False, excess=-5.0, version="old") for d in days]
        new = [_row(d, "BUY", True, excess=2.0, version="v2") for d in days]
        with patch.object(reliability, "active_model_version", return_value="v2"):
            gate = _gate_over(old + new)
        self.assertTrue(gate["version_filtered"])
        self.assertEqual(gate["directional_calls"], len(new))
        self.assertEqual(gate["directional_accuracy_pct"], 100.0)

    def test_unstamped_history_falls_back_to_every_row(self) -> None:
        gate = _gate_over([_row(d, "BUY", True) for d in _days(10)])
        self.assertFalse(gate["version_filtered"])
        self.assertEqual(gate["directional_calls"], 10)


class SignedEdgeIntervalTests(unittest.TestCase):
    def test_one_lucky_date_does_not_pass_the_edge_check(self) -> None:
        days = _days(12)
        rows = [_row(d, "BUY", False, excess=-1.0) for d in days[:-1]]
        rows.append(_row(days[-1], "BUY", True, excess=500.0))   # the outlier
        gate = _gate_over(rows)
        self.assertGreater(gate["mean_date_signed_edge_pct"], 0)  # mean says yes
        self.assertFalse(gate["checks"]["positive_signed_edge"])  # interval says no


class TierTests(unittest.TestCase):
    def _cross_section(self, n_dates: int, aligned: bool) -> list[dict]:
        """`aligned`: high score -> high excess (a real ranking edge)."""
        rows = []
        for day in _days(n_dates):
            for i in range(10):
                excess = (i - 4.5) if aligned else (4.5 - i)
                rows.append(_row(day, "HOLD", False, excess=excess, score=50 + i))
        return rows

    def test_ranking_edge_reaches_tier_1(self) -> None:
        gate = _gate_over(self._cross_section(reliability.MIN_RANKED_DATES, aligned=True))
        self.assertEqual(gate["tier"], 1)
        self.assertEqual(gate["mode"], "satellite_capped")
        self.assertFalse(gate["passed"])          # tier 2 still requires the rest
        self.assertGreater(gate["rank_ic"]["mean"], 0)

    def test_inverted_ranking_stays_tier_0(self) -> None:
        gate = _gate_over(self._cross_section(reliability.MIN_RANKED_DATES, aligned=False))
        self.assertEqual(gate["tier"], 0)
        self.assertEqual(gate["mode"], "research_only")

    def test_too_few_dates_stays_tier_0(self) -> None:
        gate = _gate_over(self._cross_section(reliability.MIN_RANKED_DATES - 5, aligned=True))
        self.assertEqual(gate["tier"], 0)


class StopDistanceTests(unittest.TestCase):
    def test_atr_wider_than_price_cannot_produce_a_negative_stop(self) -> None:
        quote = {"ticker": "GTWL", "name": "Golden Tex", "price": 29.84}
        tech = {"indicators": {"atr_14": 17.30}}     # 58% of price — broken series
        with patch.object(sizing.market, "get_quote", return_value=quote), \
             patch.object(sizing.technicals, "compute", return_value=tech), \
             patch.object(sizing.liquidity, "check_capacity",
                          return_value={"max_safe_shares": None}), \
             patch.object(sizing.risk_free, "excess_return_pct", return_value=0.0):
            out = sizing.position_size("GTWL", portfolio_value_egp=500_000)
        self.assertGreater(out["stop_loss_price"], 0)
        self.assertTrue(out["stop_distance_capped"])
        self.assertIn("unadjusted split", out["stop_distance_capped_note"])


if __name__ == "__main__":
    unittest.main()
