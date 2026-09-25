"""Tests for the walk-forward factor-weight proposal and the two scoring switches."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from egx_mcp.data import fundamentals, model_params, scoring
from scripts import factor_study, propose_factor_weights as pfw
from tests.test_factor_study import _market


def _params(**over):
    return {**model_params.DEFAULTS, **over}


class GridTests(unittest.TestCase):
    def test_grid_is_the_constrained_simplex(self) -> None:
        g = pfw.grid()
        self.assertEqual(len(g), 969)          # C(19, 3): 0.80 in 0.05 steps over 4 weights
        for w in g:
            self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
            self.assertTrue(all(v >= pfw.MIN_WEIGHT - 1e-9 for v in w.values()))
            self.assertTrue(model_params._weights_ok(w))


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        prices, hist = _market(n_days=1500)          # 2023-2028, value drives returns
        # Make the quality proxies independent of value (in _market, net income
        # is proportional to EPS, so ROA / margin would carry the same signal).
        rng = np.random.default_rng(7)
        for entry in hist["tickers"].values():
            ni = float(rng.uniform(-50, 200))
            entry["q"]["net_income"] = [ni] * len(entry["q"]["net_income"])
            entry["q"]["total_debt"] = [float(rng.uniform(0, 1500))] * len(entry["q"]["total_debt"])
        cls.study = factor_study.run(prices, hist, keep_frames=True)
        cls.summary, _ = factor_study.summarize(cls.study)

    def test_walk_forward_only_trains_on_earlier_years(self) -> None:
        dates = pfw.prepare(self.study["per_date"])
        _, folds = pfw.walk_forward(dates, pfw.grid()[::25])
        self.assertTrue(folds)
        for f in folds:
            train_years = {d["year"] for d in dates if d["year"] < f["test_year"]}
            self.assertTrue(all(y < f["test_year"] for y in train_years))
            self.assertGreaterEqual(f["n_train"], pfw.MIN_TRAIN_DATES)

    def test_planted_value_signal_moves_weight_to_valuation(self) -> None:
        with patch.object(model_params, "load_params", return_value=_params()):
            p = pfw.build_proposal(self.study["per_date"], self.summary)
        self.assertEqual(p["recommendation"], "APPLY")
        w = p["changes"]["score_weights"]
        self.assertEqual(max(w, key=w.get), "valuation")
        self.assertTrue(model_params._weights_ok(w))
        self.assertTrue(p["changes"]["valuation_market_fallback"])
        wf = p["results"]["walk_forward"]
        self.assertGreater(wf["mean_ic"], p["results"]["current"]["mean_ic"])
        self.assertTrue(any("RECOMMENDATION" in line for line in pfw.render(p)))

    def test_regime_override_changes_weights(self) -> None:
        base = model_params.DEFAULTS["score_weights"]
        on = pfw._weights_for(base, "BULL", True)
        off = pfw._weights_for(base, "BULL", False)
        self.assertGreater(on[pfw.KEYS.index("momentum")], off[pfw.KEYS.index("momentum")])
        self.assertAlmostEqual(on.sum(), 1.0)

    def test_apply_writes_only_the_proposed_changes(self) -> None:
        saved = {}
        with patch.object(model_params, "load_params", return_value=_params()), \
             patch.object(model_params, "save_params", side_effect=saved.update):
            pfw.apply({"changes": {"regime_weight_overrides": False}, "generated_at": "2026-09-25"})
        self.assertFalse(saved["regime_weight_overrides"])
        self.assertEqual(saved["score_weights"], model_params.DEFAULTS["score_weights"])
        self.assertEqual(saved["verdict_thresholds"], model_params.DEFAULTS["verdict_thresholds"])


class ScoringSwitchTests(unittest.TestCase):
    """The switches default to today's behaviour and change it only when set."""

    def _score(self, **params):
        f = {"ticker": "XYZ", "name": "XYZ", "sector": None, "price": 10.0,
             "pe_ratio": 3.0, "pb_ratio": 0.5}
        hist = {"summary": {"return_pct": 5.0, "max_drawdown_pct": -5.0,
                            "annualized_volatility_pct": 25.0}}
        mkt = {"sector": "Market (all EGX)", "median_pe": 10.0, "median_pb": 2.0,
               "median_roe_pct": 15.0, "median_margin_pct": 10.0}
        with patch.object(model_params, "load_params", return_value=_params(**params)), \
             patch.object(fundamentals, "get_fundamentals", return_value=f), \
             patch.object(fundamentals, "market_medians", return_value=mkt), \
             patch.object(scoring.market, "get_history", return_value=hist), \
             patch.object(scoring.technicals, "compute", return_value={"indicators": {}}), \
             patch.object(scoring.regime, "classify", return_value={
                 "regime": "BULL", "weight_override": {"valuation": 0.8, "quality": 0.9,
                                                       "momentum": 1.4, "risk": 0.9}}), \
             patch.object(scoring.risk_free, "excess_return_pct", return_value=0.0):
            return scoring.score_stock("XYZ")

    def test_defaults_keep_current_behaviour(self) -> None:
        self.assertFalse(model_params.DEFAULTS["valuation_market_fallback"])
        self.assertTrue(model_params.DEFAULTS["regime_weight_overrides"])
        s = self._score()
        self.assertEqual(s["subscores"]["valuation"]["score"], 50)   # no sector -> neutral
        self.assertGreater(s["weights_used"]["momentum"], 0.25)       # BULL boost applied

    def test_market_fallback_values_unsectored_names(self) -> None:
        s = self._score(valuation_market_fallback=True)
        self.assertGreater(s["subscores"]["valuation"]["score"], 50)  # P/E 3 vs market 10
        self.assertEqual(s["sector_medians"]["sector"], "Market (all EGX)")

    def test_regime_overrides_can_be_switched_off(self) -> None:
        s = self._score(regime_weight_overrides=False)
        self.assertEqual(s["weights_used"], {k: round(v, 3) for k, v in
                                             model_params.DEFAULTS["score_weights"].items()})


class MarketMediansTests(unittest.TestCase):
    def test_medians_ignore_out_of_band_pe(self) -> None:
        rows = {"A": {"pe_ratio": 5.0, "roe_pct": 10.0}, "B": {"pe_ratio": 15.0},
                "C": {"pe_ratio": 2575.0}, "D": {"pb_ratio": 1.0}}
        with patch.object(fundamentals, "_load_overrides", return_value=rows), \
             patch.object(fundamentals, "_MARKET_MED", None):
            m = fundamentals.market_medians()
        self.assertEqual(m["median_pe"], 10.0)
        self.assertEqual(m["median_pb"], 1.0)
        self.assertEqual(m["peer_count"], 4)


if __name__ == "__main__":
    unittest.main()


class SectorSourceTests(unittest.TestCase):
    def test_curated_default_and_tradingview_switch(self) -> None:
        from egx_mcp.data import sectors

        self.assertEqual(model_params.DEFAULTS["sector_source"], "curated")
        with patch.object(model_params, "load_params", return_value=_params()):
            self.assertIsNone(sectors.sector_of("MPCO"))            # not curated
            self.assertEqual(sectors.sector_of("COMI"), "Banks")     # curated
        with patch.object(model_params, "load_params",
                          return_value=_params(sector_source="tradingview")):
            self.assertEqual(sectors.sector_of("MPCO"), "Food & Beverage")
            self.assertEqual(sectors.sector_of("EGTS"), "Travel & Leisure")   # hand-corrected
            self.assertEqual(sectors.sector_of("EFIH"), "Financial Services") # curated wins
        self.assertIn("COMI", sectors.peers("banks"))

    def test_sector_medians_need_five_peers(self) -> None:
        rows = {"B1": {"pe_ratio": 5.0}, "B2": {"pe_ratio": 6.0}, "B3": {"pe_ratio": 7.0},
                "B4": {"pe_ratio": 8.0}, "B5": {"pe_ratio": 9.0}, "B6": {"pb_ratio": 1.0}}
        from egx_mcp.data import sectors
        with patch.object(sectors, "peers", return_value=list(rows)), \
             patch.object(fundamentals, "_load_overrides", return_value=rows), \
             patch.object(model_params, "load_params",
                          return_value=_params(sector_source="tradingview")):
            m = fundamentals.sector_medians("Banks")
        self.assertEqual(m["median_pe"], 7.0)
        self.assertIsNone(m["median_pb"])      # one value < 5 peers -> market fallback fills it
