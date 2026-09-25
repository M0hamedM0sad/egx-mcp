"""Offline tests for the Fed series parser and the Fed event study."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from egx_mcp.data import fed
from scripts import fed_event_study as fes

_CSV = """observation_date,DFEDTARU
2019-07-30,2.50
2019-07-31,2.50
2019-08-01,2.25
2019-08-02,.
2019-08-05,2.25
2020-03-16,0.25
2022-03-17,0.50
2022-03-16,0.25
"""


class FedSeriesTests(unittest.TestCase):
    def test_parse_skips_missing_and_sorts(self) -> None:
        s = fed.parse_csv(_CSV)
        self.assertEqual(len(s), 7)
        self.assertEqual([d.isoformat() for d, _ in s][-2:], ["2022-03-16", "2022-03-17"])

    def test_decisions_are_changes_in_the_upper_bound(self) -> None:
        moves = fed.decisions(fed.parse_csv(_CSV))
        self.assertEqual([(m["effective_date"], m["change_bp"], m["action"]) for m in moves],
                         [("2019-08-01", -25, "cut"), ("2020-03-16", -200, "cut"),
                          ("2022-03-17", 25, "hike")])


class EventStudyTests(unittest.TestCase):
    def test_planted_post_cut_rally_is_found(self) -> None:
        rng = np.random.default_rng(3)
        cal = pd.bdate_range("2016-01-04", periods=1500)
        rets = pd.Series(rng.normal(0, 0.01, len(cal)), index=cal)
        cut_days = cal[200::120][:10]
        for d in cut_days:                       # +1%/day for 5 sessions after each cut
            pos = cal.get_loc(d)
            rets.iloc[pos:pos + 5] += 0.01
        market = (1 + rets).cumprod()
        fx = pd.Series(np.linspace(8, 50, len(cal)), index=cal)
        events = [{"effective_date": str(d.date()), "action": "cut", "change_bp": -25,
                   "upper_pct": 2.0} for d in cut_days]
        res = fes.study(market, fx, events)
        g = res["groups"]["cuts"]
        self.assertEqual(g["n"], 10)
        self.assertGreater(g["egx_5d"]["vs_normal_pct"], 3.0)
        self.assertGreater(g["egx_5d"]["t"], 2.0)
        self.assertNotIn("egx_5d", res["groups"]["hikes"])        # no hikes planted
        self.assertTrue(any("cuts" in line for line in fes.render(res)))

    def test_window_starts_at_the_close_before_the_effective_day(self) -> None:
        cal = pd.bdate_range("2020-01-06", periods=10)
        level = pd.Series(range(100, 110), index=cal, dtype=float)
        self.assertAlmostEqual(fes.window_returns(level, cal, 3, 1), 103 / 102 - 1)
        self.assertIsNone(fes.window_returns(level, cal, 0, 1))
        self.assertIsNone(fes.window_returns(level, cal, 8, 5))


if __name__ == "__main__":
    unittest.main()


class FetchRetryTests(unittest.TestCase):
    def test_retries_then_succeeds_and_raises_after_budget(self) -> None:
        from unittest.mock import patch

        class R:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("read timed out")
            return R(_CSV)

        with patch("httpx.get", side_effect=flaky), patch.object(fed.time, "sleep"), \
             patch.dict(fed._CACHE, clear=True):
            self.assertEqual(len(fed.fetch_series("DFEDTARU")), 7)
        self.assertEqual(calls["n"], 3)
        with patch("httpx.get", side_effect=TimeoutError("down")), \
             patch.object(fed.time, "sleep"), patch.dict(fed._CACHE, clear=True):
            with self.assertRaises(RuntimeError):
                fed.fetch_series("DFEDTARU")


class BriefingPathTests(unittest.TestCase):
    def test_current_tries_once_and_degrades(self) -> None:
        from unittest.mock import patch

        calls = {"n": 0}

        def down(*a, **k):
            calls["n"] += 1
            raise TimeoutError("down")

        with patch("httpx.get", side_effect=down), patch.object(fed.time, "sleep") as slept, \
             patch.dict(fed._CACHE, clear=True):
            out = fed.current()
        self.assertIsNone(out["upper_pct"])
        self.assertIn("error", out)
        self.assertEqual(calls["n"], 1)
        slept.assert_not_called()
