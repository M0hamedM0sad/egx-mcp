"""TradingView symbol-page adapter.

Pulls clean structured data from TradingView's per-symbol pages:

  https://www.tradingview.com/symbols/EGX-{TICKER}/

Two extraction paths, in order of reliability:

  1. JSON-LD <script type="application/ld+json"> blocks. TradingView
     publishes name, ticker, exchange, and a basic offer/price object
     here for SEO. This is structured data the site explicitly exposes
     for machine consumption — clean and stable.

  2. Embedded "price": "<value>" string in the SSR HTML. Used as a
     fallback when the JSON-LD offer block is missing or empty.

Used for:
  - EGX 30 index (`/symbols/EGX-EGX30/` and `/symbols/EGX30/`) as a
    cross-check against yfinance.
  - Per-stock spot price as a sanity-check against yfinance.
  - One-click chart deeplinks the briefing surfaces for each W1 pick
    so the reader can jump straight to the TradingView chart.

Not used for: EGX 70 / EGX 100 (TradingView returns 404 — they
don't carry those indices). Per-stock for thin EGX names varies.

Rate limit: callers should fan out gently — TV will captcha-wall a
machine that hammers their pages. Default cache TTL 30 minutes.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

log = logging.getLogger("egx-mcp.tv_scraper")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TTL = 1800  # 30 min

_PRICE_RE = re.compile(r'"price"\s*:\s*"([0-9.,]+)"')


def chart_url(ticker: str) -> str:
    """Return the TradingView deeplink for an EGX-listed ticker.

    The reader can click this to land on the live TV chart for the
    name, with TV's full set of indicators/community ideas alongside.
    """
    return f"https://www.tradingview.com/symbols/EGX-{ticker.upper()}/"


def _parse_jsonld(html: str) -> list[dict[str, Any]]:
    """Pull every JSON-LD block out of a TradingView SSR page."""
    out: list[dict[str, Any]] = []
    pattern = re.compile(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        re.DOTALL,
    )
    for m in pattern.findall(html):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            out.extend(o for o in obj if isinstance(o, dict))
    return out


def _extract_price(html: str, ld_blocks: list[dict[str, Any]]) -> float | None:
    """Best-effort price extraction. JSON-LD `offers.price` first; then
    the embedded `"price": "<value>"` string.
    """
    for obj in ld_blocks:
        offers = obj.get("offers")
        if isinstance(offers, dict):
            p = offers.get("price")
            try:
                if p is not None:
                    return float(str(p).replace(",", ""))
            except (TypeError, ValueError):
                pass
    m = _PRICE_RE.search(html)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def fetch_symbol(ticker: str) -> dict[str, Any]:
    """Fetch a TradingView symbol page and return parsed structured data.

    Args:
        ticker: EGX ticker (e.g. 'COMI', 'CIRA', 'EGX30').

    Returns:
        Dict with: ticker, name, exchange, price, currency,
        chart_url, source. Returns {error: ...} on failure rather than
        raising, so callers can fall through to other sources.
    """
    tk = ticker.strip().upper()
    cache_key = f"sym:{tk}"
    now = time.time()
    if cache_key in _CACHE:
        ts, val = _CACHE[cache_key]
        if now - ts < _TTL:
            return val

    url = chart_url(tk)
    try:
        with httpx.Client(timeout=15.0, headers=_HEADERS, follow_redirects=True) as c:
            r = c.get(url)
            if r.status_code != 200:
                payload = {"ticker": tk, "error": f"HTTP {r.status_code}", "source": url}
                _CACHE[cache_key] = (now, payload)
                return payload
            html = r.text
    except Exception as e:
        return {"ticker": tk, "error": f"{type(e).__name__}: {e}", "source": url}

    ld = _parse_jsonld(html)
    name = exchange = currency = None
    for obj in ld:
        if not name:
            name = obj.get("name")
        # tickerSymbol may be a top-level key or nested in identifier list
        ident = obj.get("identifier")
        if isinstance(ident, list):
            for i in ident:
                if isinstance(i, dict):
                    pid = (i.get("propertyID") or "").lower()
                    if pid in ("exchange", "exchangetimezone") and not exchange:
                        exchange = i.get("value")
        if not currency:
            offers = obj.get("offers")
            if isinstance(offers, dict):
                currency = offers.get("priceCurrency")

    price = _extract_price(html, ld)

    payload = {
        "ticker": tk,
        "name": name,
        "exchange": exchange,
        "price": price,
        "currency": currency,
        "chart_url": url,
        "source": "tradingview.com",
        "as_of_unix": now,
    }
    _CACHE[cache_key] = (now, payload)
    return payload


def fetch_egx30() -> dict[str, Any]:
    """TradingView's EGX 30 index page. Used as a cross-check against
    yfinance, which is unreliable for EGX index symbols.
    """
    return fetch_symbol("EGX30")
