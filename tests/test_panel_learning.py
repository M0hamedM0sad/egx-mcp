"""Regression tests for the market-data panel learning loop.

The correctness-critical part is the purge/embargo. A 21-session forward label
means a training row dated within ~22 sessions of the test window already
"knows" returns from inside it; without the embargo the CV would score a leak
and happily propose weights fitted on the future.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import learn_panel as L


def _dates(n: int) -> list[str]:
    """n weekly (Thursday) rebalance dates as ISO strings."""
    from datetime import date, timedelta
    d0 = date(2024, 1, 4)
    return [str(d0 + timedelta(days=7 * i)) for i in range(n)]


def _row(date_str: str, v: float, q: float, m: float, r: float, excess: float) -> dict:
    return {"date": date_str, "ticker": f"T{v:.0f}{m:.0f}",
            "sub_valuation": v, "sub_quality": q, "sub_momentum": m, "sub_risk": r,
            f"excess_{L._HORIZON}d_pct": excess}


class FoldTests(unittest.TestCase):
    def test_embargo_drops_leaking_training_dates(self) -> None:
        dates = _dates(60)
        folds = L._folds(dates)
        self.assertGreaterEqual(len(folds), 2)
        embargo = -(-L._EMBARGO_SESSIONS // 5)
        for train, test in folds:
            gap = dates.index(test[0]) - dates.index(train[-1]) - 1
            self.assertGreaterEqual(
                gap, embargo,
                f"only {gap} dates between train end and test start; "
                f"the {L._HORIZON}-session label would leak")

    def test_train_and_test_never_overlap(self) -> None:
        dates = _dates(60)
        for train, test in L._folds(dates):
            self.assertFalse(set(train) & set(test))
            self.assertLess(max(train), min(test))   # strictly time-ordered

    def test_too_few_dates_yields_no_usable_folds(self) -> None:
        self.assertLess(len(L._folds(_dates(4))), 2)


def _noise_panel(seed: int = 0, n_dates: int = 60, n_names: int = 20) -> list[dict]:
    """Features and label drawn independently — no learnable relationship."""
    import random
    rnd = random.Random(seed)
    return [_row(d, rnd.uniform(0, 100), rnd.uniform(0, 100),
                 rnd.uniform(0, 100), rnd.uniform(0, 100), rnd.gauss(0, 5))
            for d in _dates(n_dates) for _ in range(n_names)]


class GuardrailTests(unittest.TestCase):
    def test_noise_panel_makes_no_proposal(self) -> None:
        """Sub-scores unrelated to the label must not produce a reweight.

        This is the overfitting canary: the cv fitter picks the best of ~800
        grid points, so on pure noise one of them will always look good on the
        selection folds. The untouched holdout is what stops it."""
        with patch.object(L, "_load_panel", return_value=_noise_panel()):
            p = L._build()
        self.assertIsNone(p["winning_fitter"])
        self.assertEqual(p["status"], "no_change")

    def test_noise_panel_is_rejected_across_seeds(self) -> None:
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                with patch.object(L, "_load_panel", return_value=_noise_panel(seed)):
                    p = L._build()
                self.assertIsNone(p["winning_fitter"])

    def test_insufficient_panel_is_reported_not_fitted(self) -> None:
        rows = [_row(d, 50, 50, 50, 50, 1.0) for d in _dates(5) for _ in range(10)]
        with patch.object(L, "_load_panel", return_value=rows):
            p = L._build()
        self.assertEqual(p["status"], "insufficient_panel")
        self.assertNotIn("fitters", p)


class ContaminationTests(unittest.TestCase):
    def test_lookahead_factors_are_flagged(self) -> None:
        with patch.object(L, "_load_panel", return_value=_noise_panel()):
            p = L._build()
        c = p["contamination"]
        self.assertEqual(set(c["contaminated"]), {"valuation", "quality"})
        for k in ("valuation", "quality"):
            self.assertFalse(c["per_factor_oos_ic"][k]["point_in_time"])
        for k in ("momentum", "risk"):
            self.assertTrue(c["per_factor_oos_ic"][k]["point_in_time"])
        # the clean reference must weight only the point-in-time factors
        self.assertEqual(c["clean_only_weights"]["valuation"], 0.0)
        self.assertEqual(c["clean_only_weights"]["quality"], 0.0)


class FundamentalsHistoryTests(unittest.TestCase):
    """`as_of` is what keeps valuation/quality honest — a lookup that reaches
    forward by even one snapshot puts the look-ahead straight back in."""

    HISTORY = {"COMI": [
        {"snapshot_date": "2026-01-15", "ticker": "COMI", "trailing_eps": 10.0},
        {"snapshot_date": "2026-04-20", "ticker": "COMI", "trailing_eps": 12.0},
        {"snapshot_date": "2026-07-30", "ticker": "COMI", "trailing_eps": 15.0},
    ]}

    def test_returns_most_recent_prior_snapshot(self) -> None:
        from scripts.snapshot_fundamentals import as_of
        self.assertEqual(as_of(self.HISTORY, "COMI", "2026-06-01")["trailing_eps"], 12.0)

    def test_never_reaches_into_the_future(self) -> None:
        from scripts.snapshot_fundamentals import as_of
        for day in ("2026-04-19", "2026-07-29"):
            eps = as_of(self.HISTORY, "COMI", day)["trailing_eps"]
            self.assertNotEqual(eps, 15.0, f"{day} saw a later snapshot")
        self.assertEqual(as_of(self.HISTORY, "COMI", "2026-04-19")["trailing_eps"], 10.0)

    def test_snapshot_date_itself_is_visible(self) -> None:
        from scripts.snapshot_fundamentals import as_of
        self.assertEqual(as_of(self.HISTORY, "COMI", "2026-04-20")["trailing_eps"], 12.0)

    def test_no_history_before_first_snapshot(self) -> None:
        from scripts.snapshot_fundamentals import as_of
        self.assertIsNone(as_of(self.HISTORY, "COMI", "2025-12-31"))
        self.assertIsNone(as_of(self.HISTORY, "NOPE", "2026-06-01"))

    def test_clean_row_pct_counts_only_uncontaminated(self) -> None:
        rows = ([{"pit_contaminated": []}] * 3
                + [{"pit_contaminated": ["valuation", "quality"]}] * 1)
        self.assertEqual(L._clean_row_pct(rows), 75.0)
        self.assertEqual(L._clean_row_pct([]), 0.0)


if __name__ == "__main__":
    unittest.main()
