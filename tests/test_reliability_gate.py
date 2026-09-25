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

    def test_learning_loop_refuses_parameter_change_when_gate_is_open(self) -> None:
        with patch.object(learn.reliability, "status", return_value={"passed": False}):
            proposal = learn._build_proposal()
        self.assertEqual(proposal["status"], "blocked_by_reliability")
        self.assertEqual(proposal["recommendation"], "KEEP_CURRENT")

    def test_one_outlier_date_cannot_flip_the_edge_check(self) -> None:
        # Nine dates lose 1%; one date's call "wins" +200%. The mean-date edge
        # is positive, but most dates destroyed value, so the check must fail.
        rows = []
        for day_index in range(10):
            day = (date.today() - timedelta(days=9 - day_index)).isoformat()
            rows += [_row(day, "BUY", day_index == 9, "high", 200.0 if day_index == 9 else -1.0)
                     for _ in range(5)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graded.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch.object(reliability, "_GRADED", path):
                gate = reliability.status()
        self.assertGreater(gate["mean_date_signed_edge_pct"], 0)
        self.assertLess(gate["median_date_signed_edge_pct"], 0)
        self.assertFalse(gate["checks"]["positive_signed_edge"])


class MedianBenchmarkTests(unittest.TestCase):
    """Grading uses the universe median so a random call is right 50% of the time."""

    def test_skewed_universe_grades_against_median_not_mean(self) -> None:
        import numpy as np
        import pandas as pd
        from tests import grade_briefings as gb

        idx = pd.bdate_range("2026-06-01", periods=30)
        # 27 names drift down 0.1%/day, 3 rocket 3%/day: the mean is dragged far
        # above what a typical name does.
        upanel = pd.DataFrame(
            {f"S{i}": 100 * np.cumprod(np.r_[1.0, np.full(29, 1.03 if i < 3 else 0.999)])
             for i in range(30)}, index=idx)
        # S5 lagged the (outlier-inflated) mean but beat the median.
        upanel["S5"] = 100 * np.cumprod(np.r_[1.0, np.full(29, 1.0005)])
        rows = [{"briefing_date": "2026-06-01", "ticker": "S5", "source": "v8b",
                 "verdict": "ACCUMULATE"}]
        g = gb._grade(rows, upanel[["S5"]], gb._synthetic_basket(upanel), [21], upanel)[0]
        self.assertEqual(g["bench_kind"], "universe_median")
        self.assertGreater(g["excess_pct"], 0)
        self.assertLess(g["excess_vs_mean_pct"], 0)
        self.assertTrue(g["correct"])

    def test_small_universe_falls_back_to_basket(self) -> None:
        import pandas as pd
        from tests import grade_briefings as gb

        idx = pd.bdate_range("2026-06-01", periods=30)
        upanel = pd.DataFrame({f"S{i}": [100.0 + d for d in range(30)] for i in range(5)},
                              index=idx)
        rows = [{"briefing_date": "2026-06-01", "ticker": "S1", "source": "v8b",
                 "verdict": "REDUCE"}]
        g = gb._grade(rows, upanel[["S1"]], gb._synthetic_basket(upanel), [21], upanel)[0]
        self.assertEqual(g["bench_kind"], "index_or_basket")


class NewsAwareGradingTests(unittest.TestCase):
    def test_chairman_verdicts_are_extracted_with_sentiment(self) -> None:
        from tests import grade_briefings as gb

        payload = {
            "v8b_verdicts": [{"ticker": "AAA", "v8b_verdict": "HOLD", "v8b_score": 55}],
            "chairman_per_pick": {
                "AAA": {"verdict": "ACCUMULATE", "conviction": "medium", "edge": 0.3,
                        "sentiment_label": "bullish", "sentiment_score": 1.0},
                "BBB": {"error": "timeout"},
            },
        }
        rows = gb._extract_verdicts(payload)
        chair = [r for r in rows if r["source"] == "chairman"]
        self.assertEqual(len(chair), 1)
        self.assertEqual(chair[0]["verdict"], "ACCUMULATE")
        self.assertEqual(chair[0]["sentiment_label"], "bullish")
        self.assertTrue(set(chair[0]) <= set(gb._FIELDS) | {"ticker"})

    def test_gate_ignores_chairman_rows(self) -> None:
        rows = [{**_row((date.today() - timedelta(days=d)).isoformat(), "BUY", True, "high", 5.0),
                 "source": "chairman"} for d in range(20)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graded.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with patch.object(reliability, "_GRADED", path):
                gate = reliability.status()
        self.assertEqual(gate["directional_calls"], 0)


if __name__ == "__main__":
    unittest.main()
