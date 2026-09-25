"""Driver profiles recover planted exposures and describe them correctly."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from egx_mcp.data import driver_profiles as dp


def _world(seed: int = 5):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=900)
    lvl = lambda r: pd.Series(100 * np.cumprod(1 + r), index=days)   # noqa: E731
    oil_r = rng.normal(0, 0.02, len(days))
    fx_r = rng.normal(0, 0.006, len(days))
    gold_r = rng.normal(0, 0.01, len(days))
    mkt_r = rng.normal(0, 0.01, len(days))
    stocks = {}
    for k in range(30):                                    # generic names: market only
        stocks[f"G{k}"] = lvl(mkt_r + rng.normal(0, 0.015, len(days)))
    stocks["OILY"] = lvl(mkt_r + 0.8 * oil_r + rng.normal(0, 0.01, len(days)))
    stocks["FXNEG"] = lvl(mkt_r - 2.0 * fx_r + rng.normal(0, 0.01, len(days)))
    stocks["NOISE"] = lvl(rng.normal(0, 0.03, len(days)))
    drivers = {"usdegp": lvl(fx_r), "brent": lvl(oil_r), "gold": lvl(gold_r)}
    return stocks, drivers


class DriverProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        stocks, drivers = _world()
        cls.res = dp.build(stocks, drivers, {"OILY": "Chemicals"})

    def test_planted_exposures_are_found(self) -> None:
        p = self.res["profiles"]
        oily = {d["driver"]: d for d in p["OILY"]["drivers"]}
        self.assertIn("brent", oily)
        self.assertEqual(oily["brent"]["sign"], "+")
        self.assertAlmostEqual(p["OILY"]["beta"]["brent"], 0.8, delta=0.2)
        fx = {d["driver"]: d for d in p["FXNEG"]["drivers"]}
        self.assertEqual(fx["usdegp"]["sign"], "-")
        self.assertNotIn("brent", {d["driver"] for d in p["NOISE"]["drivers"]})
        self.assertEqual(p["OILY"]["sector"], "Chemicals")

    def test_card_reads_today_as_tailwind_or_headwind(self) -> None:
        c = dp.describe("OILY", {"brent": 3.0, "usdegp": 0.1}, self.res)
        self.assertTrue(c["available"])
        self.assertTrue(any("with Brent oil" in line for line in c["lines"]))
        self.assertEqual(c["today"], ["Brent oil +3.0% today: tailwind"])
        c = dp.describe("FXNEG", {"usdegp": 1.2}, self.res)
        self.assertEqual(c["today"], ["USD/EGP +1.2% today: headwind"])
        self.assertFalse(dp.describe("UNKNOWN", {}, self.res)["available"])

    def test_short_history_is_skipped(self) -> None:
        stocks, drivers = _world()
        stocks = {"SHORT": stocks["OILY"].iloc[-200:], **{k: v for k, v in stocks.items() if k.startswith("G")}}
        self.assertNotIn("SHORT", dp.build(stocks, drivers)["profiles"])


if __name__ == "__main__":
    unittest.main()
