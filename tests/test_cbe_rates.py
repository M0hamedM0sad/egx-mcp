"""The CBE policy rate comes from the maintained decision file, not a scrape."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from egx_mcp.data import macro


class CbeRatesTests(unittest.TestCase):
    def test_committed_file_is_valid(self) -> None:
        data = json.loads(macro._CBE_FILE.read_text(encoding="utf-8"))
        dates = [d["date"] for d in data["decisions"]]
        self.assertEqual(dates, sorted(dates))
        for d in data["decisions"]:
            date.fromisoformat(d["date"])
            self.assertLess(d["overnight_deposit_pct"], d["overnight_lending_pct"])
            self.assertTrue(d["sources"], f"{d['date']} has no source")

    def test_latest_decision_on_or_before_today(self) -> None:
        r = macro._cbe_policy_rate(date(2026, 9, 25))
        self.assertEqual(r["decision_date"], "2026-09-24")
        self.assertEqual((r["deposit_rate_pct"], r["lending_rate_pct"], r["midpoint_pct"]),
                         (19.0, 20.0, 19.5))
        self.assertFalse(r["stale"])
        self.assertEqual(macro._cbe_policy_rate(date(2026, 9, 1))["decision_date"], "2026-08-20")

    def test_stale_and_missing(self) -> None:
        r = macro._cbe_policy_rate(date(2026, 12, 31))
        self.assertTrue(r["stale"])
        self.assertIn("warning", r)
        self.assertIsNone(macro._cbe_policy_rate(date(2020, 1, 1))["midpoint_pct"])
        with tempfile.TemporaryDirectory() as t:
            with patch.object(macro, "_CBE_FILE", Path(t) / "missing.json"):
                r = macro._cbe_policy_rate(date(2026, 9, 25))
        self.assertIsNone(r["midpoint_pct"])
        self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main()
