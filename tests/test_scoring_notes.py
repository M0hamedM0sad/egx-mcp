"""Regression tests for scoring notes and how decide() presents them.

The bug these pin: key_drivers/key_risks were split by keyword-matching the
note text, so the momentum stretch penalty — the largest single deduction the
model makes — was reported as a reason to BUY a name the same score sent to
AVOID. Sign now comes from the points awarded, not the wording.
"""
from __future__ import annotations

import unittest

from egx_mcp.data import scoring
from egx_mcp.data.decision import split_notes

_MED = {"median_pe": 37.17, "median_pb": 5.59, "median_roe_pct": 13.12}

# A BIOC-shaped name: up ~700% in 6M, expensive, violently volatile.
_STRETCHED = {"summary": {"return_pct": 701.43, "max_drawdown_pct": -20.0,
                          "annualized_volatility_pct": 104.39}}
_STRETCHED_IND = {"indicators": {"rsi_14": 81.42, "macd": 40.0, "macd_signal": 30.0,
                                 "sma_50": 159.7, "sma_200": 88.7}}


class LedgerTests(unittest.TestCase):
    def test_every_note_carries_a_delta(self) -> None:
        for res in (scoring._score_valuation({"pe_ratio": 159.8, "pb_ratio": 21.8}, _MED),
                    scoring._score_quality({"roe_pct": 13.67}, _MED),
                    scoring._score_momentum(_STRETCHED, _STRETCHED_IND),
                    scoring._score_risk(_STRETCHED)):
            self.assertEqual(len(res["notes"]), len(res["note_deltas"]))

    def test_score_reconciles_to_the_deltas(self) -> None:
        """The ledger must explain the score exactly, or the split is a lie."""
        for res in (scoring._score_valuation({"pe_ratio": 12.0, "pb_ratio": 0.8,
                                              "dividend_yield_pct": 9.0}, _MED),
                    scoring._score_quality({"roe_pct": 25.0, "profit_margin_pct": 22.0,
                                            "debt_to_equity": 15.0}, _MED),
                    scoring._score_momentum(_STRETCHED, _STRETCHED_IND),
                    scoring._score_risk(_STRETCHED)):
            expected = round(max(0, min(100, 50.0 + sum(res["note_deltas"]))), 1)
            self.assertEqual(res["score"], expected)

    def test_stretch_penalty_is_negative(self) -> None:
        mom = scoring._score_momentum(_STRETCHED, _STRETCHED_IND)
        stretch = [d for n, d in zip(mom["notes"], mom["note_deltas"]) if "stretched" in n]
        self.assertEqual(len(stretch), 1)
        self.assertLess(stretch[0], 0)
        self.assertEqual(mom["score"], 0.0)          # floored by the penalty


class SplitNotesTests(unittest.TestCase):
    def _subscores(self) -> dict:
        return {
            "valuation": scoring._score_valuation(
                {"pe_ratio": 159.82, "pb_ratio": 21.81}, _MED),
            "quality": scoring._score_quality({"roe_pct": 13.67}, _MED),
            "momentum": scoring._score_momentum(_STRETCHED, _STRETCHED_IND),
            "risk": scoring._score_risk(_STRETCHED),
        }

    def test_stretch_penalty_is_reported_as_a_risk(self) -> None:
        """The original bug: this landed in key_drivers."""
        drivers, risks, _ = split_notes(self._subscores())
        self.assertTrue(any("stretched" in r for r in risks))
        self.assertFalse(any("stretched" in d for d in drivers))

    def test_no_note_is_both_or_neither(self) -> None:
        subs = self._subscores()
        drivers, risks, neutral = split_notes(subs)
        total = sum(len(s["notes"]) for s in subs.values())
        self.assertEqual(len(drivers) + len(risks) + len(neutral), total)
        self.assertFalse(set(drivers) & set(risks))

    def test_zero_delta_notes_are_neutral_not_drivers(self) -> None:
        """'P/E unavailable' awards nothing — it is not a reason to buy."""
        subs = {"valuation": scoring._score_valuation({"pb_ratio": None}, _MED)}
        drivers, risks, neutral = split_notes(subs)
        self.assertTrue(any("unavailable" in n for n in neutral))
        self.assertFalse(any("unavailable" in d for d in drivers))
        self.assertFalse(any("unavailable" in r for r in risks))

    def test_falls_back_to_wording_without_deltas(self) -> None:
        """A pre-ledger payload must still classify, including 'stretched'."""
        legacy = {"momentum": {"score": 0.0, "notes": [
            "6M +701.43% — stretched (-260.6 pts)", "MACD bullish cross"]}}
        drivers, risks, _ = split_notes(legacy)
        self.assertTrue(any("stretched" in r for r in risks))
        self.assertTrue(any("MACD bullish" in d for d in drivers))

    def test_tolerates_malformed_subscores(self) -> None:
        self.assertEqual(split_notes({}), ([], [], []))
        self.assertEqual(split_notes({"valuation": None}), ([], [], []))


if __name__ == "__main__":
    unittest.main()
