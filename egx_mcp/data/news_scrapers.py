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

import json
import logging
import re
from datetime import datetime
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Publication dates
# ---------------------------------------------------------------------------
# Listing pages carry no dates, so 533 of 534 briefing headlines had
# date=None: a months-old story read as today's news, and no study of
# news -> price was possible. Each article page is fetched once and its
# published date read from standard metadata; results (misses included) are
# cached so a headline is never refetched.

_DATE_CACHE_PATH = Path(__file__).resolve().parents[2] / "logs" / "news_dates_cache.json"
_DATE_CACHE: dict[str, str] | None = None
_MAX_DATE_FETCHES = 80          # per process: ~12 market + 5 picks x 6 stock headlines, x2 slack

_META_KEYS = ("article:published_time", "og:published_time", "og:article:published_time",
              "datepublished", "pubdate", "publishdate", "publish-date", "date",
              "dc.date.issued", "sailthru.date", "parsely-pub-date")
_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?))?")


# Mubasher writes datetime="Fri Sep 25 15:46:59 UTC 2026" (probed). The
# "UTC" label is wrong — the clock is Cairo local time — so only the date
# part should be trusted as exact.
_CTIME_RE = re.compile(r"[A-Z][a-z]{2} ([A-Z][a-z]{2}) +(\d{1,2}) (\d{2}:\d{2}:\d{2}) [A-Z]{2,5} (\d{4})")


def _iso(value: Any) -> str | None:
    text = str(value or "")
    c = _CTIME_RE.search(text)
    if c:
        try:
            d = datetime.strptime(f"{c.group(1)} {c.group(2)} {c.group(4)}", "%b %d %Y")
            return f"{d:%Y-%m-%d}T{c.group(3)}"
        except ValueError:
            pass
    m = _ISO_RE.search(text)
    if not m:
        return None
    try:
        datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError:
        return None
    return m.group(1) + (f"T{m.group(2)}" if m.group(2) else "")


def _jsonld_date(node: Any) -> str | None:
    if isinstance(node, dict):
        for k in ("datePublished", "dateCreated", "uploadDate"):
            if k in node and _iso(node[k]):
                return _iso(node[k])
        for v in node.values():
            found = _jsonld_date(v)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _jsonld_date(v)
            if found:
                return found
    return None


def extract_published(html: str) -> str | None:
    """Published date of an article page from meta tags, JSON-LD or <time>.
    Returns 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM[:SS]', else None."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or tag.get("itemprop") or "").lower()
        value = tag.get("content") or tag.get("datetime")      # Mubasher uses datetime=
        if key in _META_KEYS and _iso(value):
            return _iso(value)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            found = _jsonld_date(json.loads(script.string or ""))
        except (ValueError, TypeError):
            continue
        if found:
            return found
    for tag in soup.find_all(attrs={"itemprop": "datePublished"}):
        found = _iso(tag.get("content") or tag.get("datetime") or tag.get_text())
        if found:
            return found
    t = soup.find("time", datetime=True)
    return _iso(t["datetime"]) if t else None


def _load_date_cache() -> dict[str, str]:
    global _DATE_CACHE
    if _DATE_CACHE is None:
        try:
            _DATE_CACHE = json.loads(_DATE_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _DATE_CACHE = {}
    return _DATE_CACHE


def _save_date_cache() -> None:
    try:
        _DATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DATE_CACHE_PATH.write_text(json.dumps(_load_date_cache(), ensure_ascii=False),
                                    encoding="utf-8")
    except OSError as e:
        log.warning(f"news date cache not saved: {e}")


_fetches_done = 0


def add_dates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill each item's missing `date` from its article page (cached).
    An empty string in the cache records a page with no readable date."""
    global _fetches_done
    cache = _load_date_cache()
    # A '#fragment' link points into a shared page (e.g. an Enterprise
    # edition); that page's date is the edition template's, not the story's —
    # probed: every Enterprise story came back 2025-01-14. Leave those undated.
    todo = [it for it in items if not it.get("date") and it.get("url")
            and "#" not in it["url"] and it["url"] not in cache]
    if todo and _fetches_done < _MAX_DATE_FETCHES:
        with _client() as c:
            for it in todo:
                if _fetches_done >= _MAX_DATE_FETCHES:
                    break
                _fetches_done += 1
                try:
                    r = c.get(it["url"])
                    # A page that loads but has no date is cached as "" (no
                    # refetch); an HTTP error is not cached, so it is retried.
                    if r.status_code == 200:
                        cache[it["url"]] = extract_published(r.text) or ""
                except Exception as e:  # noqa: BLE001
                    log.warning(f"article date fetch failed for {it['url']}: {e}")
        _save_date_cache()
    for it in items:
        if not it.get("date") and "#" not in (it.get("url") or "") \
                and cache.get(it.get("url") or ""):
            it["date"] = cache[it["url"]][:10]
            it["published"] = cache[it["url"]]
    return items


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
    return add_dates(items)


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
    return add_dates(items)


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
    return add_dates(items)


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
    return add_dates(items)


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
