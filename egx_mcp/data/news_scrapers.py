"""Per-source news scrapers for EGX.

Each adapter returns a list of {date, source, title, url} dicts, where
`title` is the short factual headline only (no article body) and `url`
links back to the original publisher. The shape mirrors news.py so the
multi-source merger can dedupe and sort cleanly.

Sources confirmed reachable (May 2026):
  - Mubasher EGX market page          /markets/EGX        AR
  - Mubasher per-stock                /markets/EGX/stocks/{tk}/news  AR
  - Investing.com EGX 100             /indices/egx-100-news          EN
  - Enterprise.press homepage         /                              EN
  - Daily News Egypt /business        /category/business/            EN

Each adapter is independently failure-tolerant — if its target is down,
it returns []. The orchestrator (news.py) calls all of them in
parallel-style sequence and merges the results.

Design rules:
  - Title only — never extract article body (fair-use).
  - Always emit a URL back to the source.
  - Skip empty / boilerplate titles ("read more", "click here").
  - Cap each adapter at `limit` items so a single chatty source can't
    crowd out the others.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Iterable

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger("egx-mcp.news_scrapers")


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,en;q=0.9",
}

_NOISE_RE = re.compile(
    r"^(read more|click here|see more|view all|next|previous|home|sign up)\s*$",
    re.IGNORECASE,
)


def _good_title(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 15 or len(t) > 240:
        return False
    if _NOISE_RE.match(t):
        return False
    return True


def _client() -> httpx.Client:
    return httpx.Client(timeout=15.0, headers=_HEADERS, follow_redirects=True)


def _absolute(href: str, base: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        # Strip path from base to get scheme://host
        m = re.match(r"^(https?://[^/]+)", base)
        if m:
            return m.group(1) + href
    return base.rstrip("/") + "/" + href.lstrip("/")


def _dedupe(items: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        title = (it.get("title") or "").strip().lower()
        # Dedupe on first 60 chars of title — handles minor publisher rewrites
        key = title[:60]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# Mubasher (AR)
# ---------------------------------------------------------------------------

def fetch_mubasher_market(limit: int = 8) -> list[dict[str, Any]]:
    """Mubasher EGX market page — Arabic. The /countries/eg/news URL went
    404 in 2026; /markets/EGX is the canonical replacement."""
    url = "https://www.mubasher.info/markets/EGX"
    try:
        with _client() as c:
            r = c.get(url)
            if r.status_code != 200:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"mubasher market fetch failed: {e}")
        return []

    items: list[dict[str, Any]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/news/" not in href:
            continue
        # Skip section pages (e.g. /news/islamic-finance with no numeric id)
        if not re.search(r"/news/\d+/", href):
            continue
        title = a.get_text(strip=True)
        if not _good_title(title):
            continue
        items.append({
            "date": None,
            "source": "Mubasher",
            "title": title,
            "url": _absolute(href, url),
            "lang": "ar",
        })
        if len(items) >= limit:
            break
    return items


def fetch_mubasher_stock(ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    """Mubasher per-stock news — Arabic."""
    url = f"https://www.mubasher.info/markets/EGX/stocks/{ticker}/news"
    try:
        with _client() as c:
            r = c.get(url)
            if r.status_code != 200:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"mubasher stock fetch failed for {ticker}: {e}")
        return []

    items: list[dict[str, Any]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(r"/news/\d+/", href):
            continue
        title = a.get_text(strip=True)
        if not _good_title(title):
            continue
        items.append({
            "date": None,
            "source": "Mubasher",
            "title": title,
            "url": _absolute(href, url),
            "lang": "ar",
        })
        if len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# Investing.com (EN) — EGX 100 news page
# ---------------------------------------------------------------------------

def fetch_investing_egx(limit: int = 8) -> list[dict[str, Any]]:
    """Investing.com EGX 100 news. Heavily JS-rendered — server-side
    HTML usually has nav links only. Kept as a stub that returns []
    rather than nav-noise. Re-enable once they ship SSR for the news
    cards or a feed endpoint we can hit cleanly.
    """
    return []


# ---------------------------------------------------------------------------
# Enterprise.press (EN) — Egypt-focused business newsletter
# ---------------------------------------------------------------------------

def fetch_enterprise(limit: int = 6) -> list[dict[str, Any]]:
    url = "https://enterprise.press/"
    try:
        with _client() as c:
            r = c.get(url)
            if r.status_code != 200:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"enterprise.press fetch failed: {e}")
        return []

    items: list[dict[str, Any]] = []
    # Enterprise.press lists posts as <article> or <h2><a>
    for h in soup.find_all(["h1", "h2", "h3"]):
        link = h.find("a", href=True)
        if not link:
            continue
        title = link.get_text(strip=True)
        if not _good_title(title):
            continue
        items.append({
            "date": None,
            "source": "Enterprise",
            "title": title,
            "url": _absolute(link["href"], url),
            "lang": "en",
        })
        if len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# Daily News Egypt /category/business/
# ---------------------------------------------------------------------------

def fetch_daily_news_egypt(limit: int = 6) -> list[dict[str, Any]]:
    url = "https://www.dailynewsegypt.com/category/business/"
    try:
        with _client() as c:
            r = c.get(url)
            if r.status_code != 200:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.warning(f"daily news egypt fetch failed: {e}")
        return []

    items: list[dict[str, Any]] = []
    for h in soup.find_all(["h2", "h3"]):
        link = h.find("a", href=True)
        if not link:
            continue
        title = link.get_text(strip=True)
        if not _good_title(title):
            continue
        href = link["href"]
        # Skip category and tag pages
        if "/category/" in href or "/tag/" in href:
            continue
        items.append({
            "date": None,
            "source": "Daily News Egypt",
            "title": title,
            "url": _absolute(href, url),
            "lang": "en",
        })
        if len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# Public orchestrators
# ---------------------------------------------------------------------------

def fetch_market_multi(limit: int = 12) -> list[dict[str, Any]]:
    """Pull market-wide EGX news from all reachable sources, dedupe."""
    chunk = max(3, limit // 3)
    items: list[dict[str, Any]] = []
    items.extend(fetch_mubasher_market(limit=chunk))
    items.extend(fetch_investing_egx(limit=chunk))
    items.extend(fetch_enterprise(limit=chunk))
    items.extend(fetch_daily_news_egypt(limit=chunk))
    return _dedupe(items)[:limit]


def fetch_stock_multi(ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    """Pull per-stock news with multi-source fallback. Always tries Mubasher
    Arabic per-stock; Investing.com search-style is omitted because their
    per-stock URLs are captcha-walled. Caller still has yfinance for EN."""
    chunk = max(3, limit // 2)
    items: list[dict[str, Any]] = []
    items.extend(fetch_mubasher_stock(ticker, limit=chunk))
    return _dedupe(items)[:limit]
