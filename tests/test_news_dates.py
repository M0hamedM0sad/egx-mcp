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
