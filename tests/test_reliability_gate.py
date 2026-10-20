"""Regression tests for the fail-closed decision-reliability gate."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from egx_mcp.data import decision, reliability
from scripts import learn


def _row(day: str, verdict: str, correct: bool, conviction: str = "medium",
         excess: float = 1.0) -> dict:
    return {
        "source": "v8b", "outcome": "graded", "horizon_days": 21,
        "briefing_date": day, "verdict": verdict, "correct": correct,
        "conviction": conviction, "excess_pct": excess,
    }


class ReliabilityGateTests(unittest.TestCase):
    def test_missing_evidence_is_research_only(self) -> None:
        with patch.object(reliability, "_GRADED", Path("does-not-exist.jsonl")):
            gate = reliability.status()
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["mode"], "research_only")
        self.assertIn("sample_size", gate["failed_checks"])

    def test_reliable_live_evidence_can_pass(self) -> None:
        rows = []
        for day_index in range(10):
            day = (date.today() - timedelta(days=9 - day_index)).isoformat()
            rows += [
                _row(day, "BUY", True, "high", 2.0),
                _row(day, "BUY", True, "medium", 1.0),
                _row(day, "BUY", day_index < 6, "medium", 1.0 if day_index < 6 else -0.5),
                _row(day, "REDUCE", day_index < 6, "low", -1.0 if day_index < 6 else 0.5),
            ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graded.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.object(reliability, "_GRADED", path):
                gate = reliability.status()
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["mode"], "actionable")

    def test_unproven_model_abstains_from_buy_side_call(self) -> None:
        assessment = decision._assess(
            "BUY", "high", 80.0, {"valuation": 80.0, "quality": 80.0}, 20.0,
            {"level": "high"}, {"passed": False}, 1.0, 0.0,
        )
        self.assertEqual(assessment["verdict"], "ABSTAIN")
        self.assertFalse(assessment["actionable"])
        # The withheld call survives as a shadow so it can still be graded.
        self.assertTrue(assessment["gate_suppressed"])
        self.assertEqual(assessment["provisional_verdict"], "BUY")
        self.assertEqual(assessment["shadow_conviction"], "high")

    def test_low_data_abstain_is_not_a_shadow_call(self) -> None:
        assessment = decision._assess(
            "BUY", "high", 80.0, {}, 20.0, {"level": "low"}, {"passed": False}, 1.0, 0.0,
        )
        self.assertEqual(assessment["verdict"], "ABSTAIN")
        self.assertFalse(assessment["gate_suppressed"])

    def test_shadow_call_is_graded_as_the_withheld_verdict(self) -> None:
        from tests import grade_briefings as gb
        rows = gb._extract_verdicts({"v8b_verdicts": [
            {"ticker": "AAA", "v8b_verdict": "ABSTAIN", "v8b_conviction": "none",
             "v8b_shadow_verdict": "ACCUMULATE", "v8b_shadow_conviction": "medium"},
            {"ticker": "BBB", "v8b_verdict": "REDUCE", "v8b_conviction": "medium"},
        ]})
        self.assertEqual([(r["verdict"], r["conviction"], r["shadow"]) for r in rows],
                         [("ACCUMULATE", "medium", True), ("REDUCE", "medium", False)])

    def test_unadjusted_split_is_excluded_not_graded(self) -> None:
        import pandas as pd
        from tests import grade_briefings as gb
        idx = pd.bdate_range("2026-06-01", periods=30)
        px = [160.0] * 10 + [80.0] * 20          # 2:1 split on day 10
        panel = pd.DataFrame({"HDBK": px}, index=idx)
        rows = [{"briefing_date": "2026-06-01", "ticker": "HDBK", "source": "v8b",
                 "verdict": "ACCUMULATE", "conviction": "medium"}]
        (rec,) = gb._grade(rows, panel, None, [21])
        self.assertEqual(rec["outcome"], "excluded_corporate_action")
        self.assertIsNone(rec["correct"])

    def test_binomial_p_value_survives_large_samples(self) -> None:
        from tests.validate_edge import _binom_two_sided_p
        self.assertAlmostEqual(_binom_two_sided_p(5, 10), 1.0, places=6)
        p = _binom_two_sided_p(1285, 2457)   # n that overflowed in CI
        self.assertTrue(0.0 < p < 0.05)

    def test_learning_loop_refuses_parameter_change_when_gate_is_open(self) -> None:
        with patch.object(learn.reliability, "status", return_value={"passed": False}):
            proposal = learn._build_proposal()
        self.assertEqual(proposal["status"], "blocked_by_reliability")
        self.assertEqual(proposal["recommendation"], "KEEP_CURRENT")


if __name__ == "__main__":
    unittest.main()
