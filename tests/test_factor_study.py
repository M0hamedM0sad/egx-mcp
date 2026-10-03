"""Offline tests for the point-in-time factor study and TradingView history."""
from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from egx_mcp.data import tv_history
from scripts import factor_study


def _market(n_names: int = 40, n_days: int = 700, seed: int = 1):
    """Synthetic market where next-month return rises with earnings yield."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    ey = np.linspace(-0.05, 0.25, n_names)          # fixed per name
    prices, hist = {}, {"tickers": {}}
    latest_q = date(2025, 6, 30)
    for k in range(n_names):
        tk = f"T{k:02d}"
        drift = 0.0004 * (ey[k] - ey.mean()) / ey.std()
        px = 10 * np.cumprod(1 + drift + rng.normal(0, 0.01, n_days))
        prices[tk] = [{"date": d.strftime("%Y-%m-%d"), "close": float(p), "volume": 1000}
                      for d, p in zip(days, px)]
        n_q = 16
        # Quarterly EPS so TTM EPS / ~price(10) ~= ey[k]
        hist["tickers"][tk] = {
            "fiscal_period_end_fq": latest_q.isoformat(),
            "q": {"eps": [ey[k] * 10 / 4] * n_q, "net_income": [ey[k] * 100] * n_q,
                  "revenue": [1000.0] * n_q, "total_assets": [2000.0] * n_q,
                  "total_debt": [500.0 + 10 * k] * n_q},
        }
    return prices, hist


class TvHistoryTests(unittest.TestCase):
    def test_quarter_ends_step_back_by_calendar_quarter(self) -> None:
        self.assertEqual(tv_history.quarter_ends(date(2025, 3, 31), 3),
                         [date(2025, 3, 31), date(2024, 12, 31), date(2024, 9, 30)])

    def test_pit_hides_quarters_until_the_reporting_lag_passes(self) -> None:
        entry = {"fiscal_period_end_fq": "2025-06-30",
                 "q": {"eps": [4.0, 3.0, 2.0, 1.0, 9.0], "total_assets": [5, 4, 3, 2, 1]}}
        table = tv_history.quarterly_table(entry)
        # On 2025-07-15 the June quarter is not public yet: latest visible is March.
        f = tv_history.pit(table, date(2025, 7, 15))
        self.assertEqual(f["period_end"], "2025-03-31")
        self.assertEqual(f["ttm_eps"], 3.0 + 2.0 + 1.0 + 9.0)
        self.assertEqual(f["total_assets"], 4)
        lag = tv_history.REPORTING_LAG_DAYS
        f = tv_history.pit(table, date(2025, 6, 30) + timedelta(days=lag))
        self.assertEqual(f["ttm_eps"], 4.0 + 3.0 + 2.0 + 1.0)
        self.assertIsNone(tv_history.pit(table, date(2024, 1, 1)))

    def test_ttm_needs_four_quarters(self) -> None:
        entry = {"fiscal_period_end_fq": "2025-06-30", "q": {"eps": [1.0, None, 1.0, 1.0]}}
        f = tv_history.pit(tv_history.quarterly_table(entry), date(2026, 1, 1))
        self.assertIsNone(f["ttm_eps"])

    def test_units_mismatch_is_flagged(self) -> None:
        entry = {"fiscal_period_end_fq": "2025-06-30", "close": 50.0,
                 "price_earnings_ttm": 10.0, "q": {"eps": [0.05] * 4}}   # ours: P/E 250
        self.assertTrue(tv_history.units_suspect(entry))
        entry["q"]["eps"] = [1.25] * 4                                    # ours: P/E 10
        self.assertFalse(tv_history.units_suspect(entry))


class FactorStudyTests(unittest.TestCase):
    def test_planted_value_signal_is_found_with_the_right_sign(self) -> None:
        prices, hist = _market()
        res = factor_study.run(prices, hist)
        summ, lines = factor_study.summarize(res)
        ey = summ["overall"]["earnings_yield"]
        self.assertGreater(ey["n"], 10)
        self.assertGreater(ey["mean"], 0.1)
        self.assertGreater(ey["t"], 2)
        self.assertGreater(summ["overall"]["value_composite"]["mean"], 0.1)
        self.assertTrue(any("earnings_yield" in line for line in lines))

    def test_dates_before_any_visible_quarter_do_not_crash(self) -> None:
        # Prices start 2023-01; with 8 quarters to 2025-06 nothing is public
        # until late 2023, so early rebalances have no fundamentals at all.
        prices, hist = _market()
        for k, entry in enumerate(hist["tickers"].values()):
            entry["q"] = {f: v[:8] for f, v in entry["q"].items()}
            if k % 2:
                entry["q"]["total_debt"] = [None] * 8        # TradingView gap
        res = factor_study.run(prices, hist)
        summ, _ = factor_study.summarize(res)
        self.assertGreater(summ["overall"]["earnings_yield"]["n"], 10)

    def test_labels_never_overlap(self) -> None:
        prices, hist = _market(n_days=400)
        dates = [pd.Timestamp(r["date"]) for r in factor_study.run(prices, hist)["per_date"]]
        gaps = [len(pd.bdate_range(a, b)) - 1 for a, b in zip(dates, dates[1:])]
        self.assertTrue(all(g >= factor_study.HORIZON for g in gaps))


if __name__ == "__main__":
    unittest.main()
