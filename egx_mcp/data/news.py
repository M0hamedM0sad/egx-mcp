"""News adapter — multi-source for EGX.

English path:
  1. yfinance.Ticker.news — works for large caps with .CA suffix
  2. Investing.com / Enterprise / Daily News Egypt — fallback for thin
     names where Yahoo returns nothing

Arabic path:
  1. Mubasher per-symbol page — works for individual EGX names
  2. Mubasher /markets/EGX market page — replaced the dead
     /countries/eg/news endpoint

The dead URLs (/countries/eg/news, /news, /markets/EGX/news) all return
404 since 2026. The orchestrator in news_scrapers.py routes around them.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
import yfinance as yf
from bs4 import BeautifulSoup

from . import news_scrapers
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.news")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (egx-mcp/0.1; mailto:m0hamedm0sad@gmail.com)",
    "Accept-Language": "ar,en;q=0.9",
}


def fetch(ticker: str | None = None, lang: str = "en", limit: int = 10) -> dict[str, Any]:
    if lang == "ar":
        return _fetch_arabic(ticker, limit)
    return _fetch_english(ticker, limit)


def _fetch_english(ticker: str | None, limit: int) -> dict[str, Any]:
    """English path.

    Per-ticker queries: yfinance ONLY. If Yahoo returns nothing for a
    thin EGX name, return empty — never backfill with market-wide news,
    that would mislabel market headlines as belonging to a single stock
    and mislead the sentiment scorer.

    Market-wide queries (ticker is None): use the multi-source
    orchestrator (Mubasher /markets/EGX, Enterprise, Daily News Egypt).
    """
    canonical = None
    articles: list[dict[str, Any]] = []

    if ticker:
        canonical, yahoo, _ = resolve_ticker(ticker)
        try:
            t = yf.Ticker(yahoo)
            items = (t.news or [])[:limit]
        except Exception as e:
            log.warning(f"yfinance news failed for {yahoo}: {e}")
            items = []

        for item in items:
            content = item.get("content", item)
            title = content.get("title")
            if not title:
                continue
            url = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("link")
            publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else content.get("publisher")
            ts = content.get("pubDate") or content.get("providerPublishTime")
            if isinstance(ts, (int, float)):
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            elif isinstance(ts, str):
                date_str = ts[:10]
            else:
                date_str = None
            articles.append({
                "date": date_str,
                "source": publisher,
                "title": title,
                "url": url,
                "summary": content.get("summary"),
            })
        # No market-wide backfill for per-ticker queries — empty is honest.
    else:
        # Market-wide query — pull from the multi-source orchestrator.
        try:
            scraped = news_scrapers.fetch_market_multi(limit=limit)
        except Exception as e:
            log.warning(f"market scraper failed: {e}")
            scraped = []
        for item in scraped:
            articles.append({
                "date": item.get("date"),
                "source": item.get("source"),
                "title": item.get("title"),
                "url": item.get("url"),
                "summary": None,
            })

    return {
        "ticker": canonical,
        "lang": "en",
        "count": len(articles),
        "articles": articles[:limit],
    }


def _fetch_arabic(ticker: str | None, limit: int) -> dict[str, Any]:
    """Arabic path — Mubasher EGX market page or per-stock page.

    The market-wide /countries/eg/news URL went 404 in 2026; the
    multi-source orchestrator routes around this via /markets/EGX.
    """
    canonical = None
    articles: list[dict[str, Any]] = []

    if ticker:
        canonical, _, _ = resolve_ticker(ticker)
        try:
            stock_items = news_scrapers.fetch_mubasher_stock(canonical, limit=limit)
        except Exception as e:
            log.warning(f"Mubasher stock fetch failed for {canonical}: {e}")
            stock_items = []
        for item in stock_items:
            articles.append({
                "date": item.get("date"),
                "source": item.get("source", "Mubasher"),
                "title": item.get("title"),
                "url": item.get("url"),
                "summary": None,
            })
    else:
        try:
            market_items = news_scrapers.fetch_mubasher_market(limit=limit)
        except Exception as e:
            log.warning(f"Mubasher market fetch failed: {e}")
            market_items = []
        for item in market_items:
            articles.append({
                "date": item.get("date"),
                "source": item.get("source", "Mubasher"),
                "title": item.get("title"),
                "url": item.get("url"),
                "summary": None,
            })

    return {
        "ticker": canonical,
        "lang": "ar",
        "count": len(articles),
        "articles": articles[:limit],
    }
