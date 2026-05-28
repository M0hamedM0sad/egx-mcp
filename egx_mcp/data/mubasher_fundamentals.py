"""Mubasher fundamentals scraper — fills Yahoo's quality-data gap.

Yahoo doesn't return ROE / margins / leverage / dividend yield / EPS for
EGX names. Mubasher does (server-rendered HTML, no JS required).

Per-ticker URL pattern:
    https://english.mubasher.info/markets/EGX/stocks/{TICKER}/ratios

Tables parsed (best-effort, regex-tolerant to layout changes):
    ROE %                    → roe_pct
    Net Profit Margin %      → profit_margin_pct
    Debt to Equity           → debt_to_equity
    P/E                      → pe_ratio (cross-check vs Yahoo)
    P/B                      → pb_ratio
    Dividend Yield %         → dividend_yield_pct
    Earnings per Share       → trailing_eps
    Book Value per Share     → book_value_per_share

Cached to disk so we don't re-scrape on every run.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger("egx-mcp.mubasher_fundamentals")

_CACHE_PATH = Path(__file__).parent / "mubasher_fundamentals_cache.json"
_CACHE_TTL = 7 * 24 * 3600   # weekly refresh

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}
_TIMEOUT = 20.0


# Each row maps a regex-pattern label → output field name.
# Patterns are matched against the *normalized* row label (lowercase, spaces collapsed).
_ROW_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^roe\s*%?$"),                                     "roe_pct"),
    (re.compile(r"^return on equity"),                              "roe_pct"),
    (re.compile(r"^net profit margin"),                             "profit_margin_pct"),
    (re.compile(r"^operating margin"),                              "operating_margin_pct"),
    (re.compile(r"^gross margin"),                                  "gross_margin_pct"),
    (re.compile(r"^debt[\s/]+to[\s/]+equity"),                      "debt_to_equity"),
    (re.compile(r"^p/?e( ratio)?$"),                                "pe_ratio"),
    (re.compile(r"^price[\s/]*earnings"),                           "pe_ratio"),
    (re.compile(r"^p/?b( ratio)?$"),                                "pb_ratio"),
    (re.compile(r"^price[\s/]+to[\s/]+book"),                       "pb_ratio"),
    (re.compile(r"^dividend yield"),                                "dividend_yield_pct"),
    (re.compile(r"^earnings per share"),                            "trailing_eps"),
    (re.compile(r"^eps$"),                                          "trailing_eps"),
    (re.compile(r"^book value per share"),                          "book_value_per_share"),
    (re.compile(r"^current ratio"),                                 "current_ratio"),
]


def _parse_number(text: str) -> float | None:
    """Extract the first plausible number from an HTML cell."""
    if not text:
        return None
    text = text.replace(",", "").strip()
    # Allow trailing %
    m = re.match(r"^[-+]?(\d+(?:\.\d+)?)\s*%?$", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_market_cap(text: str) -> float | None:
    """Mubasher prints market cap as '423,777,438,000.00 Egyptian Pound'."""
    if not text:
        return None
    text = text.replace(",", "").strip()
    m = re.match(r"^[-+]?(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# Overview page label → (field name, parser function)
_OVERVIEW_LABELS = {
    "P/E Ratio":         ("pe_ratio", _parse_number),
    "P/B Ratio":         ("pb_ratio", _parse_number),
    "Market Cap":        ("market_cap", _parse_market_cap),
    "Dividend Yield":    ("dividend_yield_pct", _parse_number),
    "EV/EBITDA":         ("ev_ebitda", _parse_number),
    "Book Value (BVPS)": ("book_value_per_share", _parse_number),
    "BVPS":              ("book_value_per_share", _parse_number),
    "EPS":               ("trailing_eps", _parse_number),
    "Par Value":         ("par_value", _parse_number),
}


def _scrape_overview(html: str, out: dict) -> None:
    """Parse the overview spans on the main stock page."""
    soup = BeautifulSoup(html, "html.parser")
    for span_label in soup.find_all("span", class_="stock-overview__text"):
        label = span_label.get_text(strip=True)
        if label not in _OVERVIEW_LABELS:
            continue
        field, parser = _OVERVIEW_LABELS[label]
        if field in out:
            continue
        # The value lives in a sibling <span class="stock-overview__value">
        value_container = span_label.find_next("span", class_="stock-overview__value")
        if not value_container:
            continue
        num_span = value_container.find("span", class_="number")
        if num_span:
            v = parser(num_span.get_text(strip=True))
            if v is not None:
                out[field] = v


def _scrape_one(ticker: str) -> dict[str, Any]:
    """Scrape Mubasher's overview + ratios pages for a single ticker."""
    base_url = f"https://english.mubasher.info/markets/EGX/stocks/{ticker.upper()}"
    overview_url = base_url
    ratios_url = f"{base_url}/ratios"

    out: dict[str, Any] = {"ticker": ticker.upper(), "sources": []}

    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as c:
        # 1. Overview page — P/E, P/B, market cap, dividend yield
        try:
            r = c.get(overview_url)
            if r.status_code == 200:
                _scrape_overview(r.text, out)
                out["sources"].append(overview_url)
        except Exception as e:
            out.setdefault("errors", []).append(f"overview: {e}")

        # 2. Ratios page — ROE, ROA, EPS, growth metrics
        try:
            r = c.get(ratios_url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tr in soup.find_all("tr"):
                    cells = tr.find_all("td")
                    if len(cells) < 2:
                        continue
                    label = re.sub(r"\s+", " ", cells[0].get_text(strip=True).lower())
                    value_text = cells[1].get_text(strip=True)
                    for pat, field in _ROW_PATTERNS:
                        if pat.search(label):
                            num = _parse_number(value_text)
                            if num is not None and field not in out:
                                out[field] = num
                            break
                out["sources"].append(ratios_url)
        except Exception as e:
            out.setdefault("errors", []).append(f"ratios: {e}")

    return out


def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"cache write failed: {e}")


def get_fundamentals(ticker: str, force_refresh: bool = False) -> dict[str, Any]:
    """Get Mubasher fundamentals for one ticker, with disk caching."""
    cache = _load_cache()
    now = time.time()
    cached = cache.get(ticker.upper())
    if cached and not force_refresh:
        if now - cached.get("fetched_at", 0) < _CACHE_TTL:
            return cached

    data = _scrape_one(ticker)
    data["fetched_at"] = now
    cache[ticker.upper()] = data
    _save_cache(cache)
    return data


def scrape_universe(tickers: list[str], force_refresh: bool = False,
                     verbose: bool = False) -> dict[str, dict]:
    """Scrape every ticker, return {ticker: row}."""
    out = {}
    for tk in tickers:
        d = get_fundamentals(tk, force_refresh=force_refresh)
        if verbose:
            n_fields = sum(1 for k in d if k not in ("ticker", "source", "fetched_at", "error"))
            print(f"  {tk:6s}  {n_fields} fields  "
                  f"{'OK' if 'error' not in d else 'ERR: ' + d['error'][:40]}")
        out[tk] = d
    return out


if __name__ == "__main__":
    import sys
    from .egx_listing import get_full_universe

    universe = get_full_universe()
    print(f"Scraping Mubasher fundamentals for {len(universe)} EGX names...\n")
    results = scrape_universe(universe, force_refresh="--refresh" in sys.argv, verbose=True)

    fields = ["roe_pct", "profit_margin_pct", "debt_to_equity",
              "pe_ratio", "pb_ratio", "dividend_yield_pct",
              "trailing_eps", "book_value_per_share"]
    counts = {f: 0 for f in fields}
    for d in results.values():
        for f in fields:
            if d.get(f) is not None:
                counts[f] += 1
    print(f"\nCoverage report:")
    for f, c in counts.items():
        pct = c / len(results) * 100
        bar = "#" * int(pct / 2.5)
        print(f"  {f:<25} {c:>3}/{len(results)}  {pct:>5.1f}%  {bar}")
