"""Offline tests for article publication-date extraction and its cache."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from egx_mcp.data import news_scrapers as ns


class ExtractPublishedTests(unittest.TestCase):
    def test_open_graph_meta(self) -> None:
        html = '<meta property="article:published_time" content="2026-09-21T10:15:00+03:00">'
        self.assertEqual(ns.extract_published(html), "2026-09-21T10:15:00")

    def test_json_ld_nested_graph(self) -> None:
        ld = {"@context": "https://schema.org",
              "@graph": [{"@type": "WebPage"}, {"@type": "NewsArticle",
                                                "datePublished": "2026-08-30"}]}
        html = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        self.assertEqual(ns.extract_published(html), "2026-08-30")

    def test_itemprop_and_time_tag(self) -> None:
        self.assertEqual(ns.extract_published(
            '<span itemprop="datePublished" content="2026-07-01 09:00"></span>'), "2026-07-01T09:00")
        self.assertEqual(ns.extract_published(
            '<article><time datetime="2026-06-15">15 June</time></article>'), "2026-06-15")

    def test_mubasher_markup(self) -> None:
        # Shaped like the live page (probe run 36138421781).
        html = ('<meta property="article:published_time" datetime="Fri Sep 25 15:46:59 UTC 2026" />'
                '<span class="mi-article__published-at"><i class="fa fa-calendar"></i>'
                '<time itemprop="datePublished" datetime="Fri Sep 25 15:46:59 UTC 2026">'
                '25 سبتمبر 2026 03:46 م</time></span>')
        self.assertEqual(ns.extract_published(html), "2026-09-25T15:46:59")
        only_time = ('<time itemprop="datePublished" datetime="Mon Aug  4 09:05:00 UTC 2026">'
                     '4 أغسطس 2026</time>')
        self.assertEqual(ns.extract_published(only_time), "2026-08-04T09:05:00")

    def test_no_date_and_invalid_date(self) -> None:
        self.assertIsNone(ns.extract_published("<html><p>no date here</p></html>"))
        self.assertIsNone(ns.extract_published('<meta name="date" content="2026-13-45">'))

    def test_broken_json_ld_is_skipped(self) -> None:
        html = ('<script type="application/ld+json">{not json</script>'
                '<time datetime="2026-05-05"></time>')
        self.assertEqual(ns.extract_published(html), "2026-05-05")


class _Resp:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text, self.status_code = text, status


class _Client:
    def __init__(self, pages: dict[str, str], calls: list[str]) -> None:
        self.pages, self.calls = pages, calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        self.calls.append(url)
        if url not in self.pages:
            return _Resp("", 404)
        return _Resp(self.pages[url])


class AddDatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(ns, "_DATE_CACHE_PATH", Path(self.tmp.name) / "c.json"),
                        patch.object(ns, "_DATE_CACHE", None),
                        patch.object(ns, "_fetches_done", 0)]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_dates_filled_cached_and_misses_not_refetched(self) -> None:
        pages = {"u1": '<meta property="article:published_time" content="2026-09-20T08:00">',
                 "u2": "<p>nothing</p>"}
        calls: list[str] = []
        items = [{"title": "a", "url": "u1", "date": None},
                 {"title": "b", "url": "u2", "date": None},
                 {"title": "c", "url": "u3", "date": None},            # 404
                 {"title": "d", "url": "u4", "date": "2026-01-01"}]    # already dated
        with patch.object(ns, "_client", lambda: _Client(pages, calls)):
            out = ns.add_dates(items)
            self.assertEqual(out[0]["date"], "2026-09-20")
            self.assertEqual(out[0]["published"], "2026-09-20T08:00")
            self.assertIsNone(out[1]["date"])
            self.assertIsNone(out[2]["date"])
            self.assertEqual(out[3]["date"], "2026-01-01")
            self.assertEqual(sorted(calls), ["u1", "u2", "u3"])
            # Second pass: everything comes from the cache, misses included.
            ns.add_dates([{"title": "a", "url": "u1", "date": None},
                          {"title": "b", "url": "u2", "date": None}])
            self.assertEqual(len(calls), 3)
            ns.add_dates([{"title": "c", "url": "u3", "date": None}])    # 404 is retried
            self.assertEqual(len(calls), 4)
        saved = json.loads((Path(self.tmp.name) / "c.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, {"u1": "2026-09-20T08:00", "u2": ""})   # 404 not cached

    def test_fragment_links_are_not_dated(self) -> None:
        calls: list[str] = []
        pages = {"https://e.com/edition#s1": '<meta name="date" content="2025-01-14">'}
        with patch.object(ns, "_client", lambda: _Client(pages, calls)):
            out = ns.add_dates([{"title": "t", "url": "https://e.com/edition#s1", "date": None}])
        self.assertIsNone(out[0]["date"])
        self.assertEqual(calls, [])

    def test_fetch_budget_is_respected(self) -> None:
        calls: list[str] = []
        items = [{"title": str(i), "url": f"x{i}", "date": None}
                 for i in range(ns._MAX_DATE_FETCHES + 10)]
        with patch.object(ns, "_client", lambda: _Client({}, calls)):
            ns.add_dates(items)
        self.assertEqual(len(calls), ns._MAX_DATE_FETCHES)


if __name__ == "__main__":
    unittest.main()


class SentimentFreshnessTests(unittest.TestCase):
    def test_only_recent_dated_headlines_are_scored(self) -> None:
        from datetime import datetime, timedelta

        from egx_mcp.data import sentiment

        today = datetime.utcnow().date()
        arts = {"articles": [
            {"title": "fresh", "date": (today - timedelta(days=2)).isoformat()},
            {"title": "old", "date": (today - timedelta(days=120)).isoformat()},
            {"title": "undated", "date": None},
        ]}
        scores = {"fresh": 0.5, "old": 1.0, "undated": -1.0}
        with patch.object(sentiment.news, "fetch", return_value=arts), \
             patch.object(sentiment, "_score_headline",
                          side_effect=lambda t, lang, b: (scores[t], [t])):
            out = sentiment.analyze_sentiment("COMI", lang="ar", max_age_days=7)
        self.assertEqual(out["headline_count"], 1)
        self.assertEqual(out["aggregate_score"], 0.5)
        self.assertEqual((out["stale_count"], out["undated_count"], out["listed_count"]), (1, 1, 3))
        self.assertEqual(out["bull_signals"], ["fresh"])       # the old +1.0 story is not a signal
        self.assertEqual({h["title"]: h["freshness"] for h in out["headlines"]},
                         {"fresh": "fresh", "old": "stale", "undated": "undated"})

    def test_nothing_fresh_reads_neutral(self) -> None:
        from egx_mcp.data import sentiment

        arts = {"articles": [{"title": "old", "date": "2020-01-01"}]}
        with patch.object(sentiment.news, "fetch", return_value=arts), \
             patch.object(sentiment, "_score_headline", return_value=(1.0, ["x"])):
            out = sentiment.analyze_sentiment("COMI", lang="ar")
        self.assertEqual(out["headline_count"], 0)
        self.assertEqual(out["aggregate_score"], 0.0)
        self.assertEqual(out["label"], sentiment._label(0.0))
