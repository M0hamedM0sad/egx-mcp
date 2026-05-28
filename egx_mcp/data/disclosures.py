"""EGX disclosures scraper.

Pulls from egx.com.eg's disclosures portal. The endpoint is unofficial — EGX
periodically changes URL structure and the data tables are server-rendered
HTML, so we rely on BeautifulSoup parsing rather than a stable API.

If EGX changes their layout, update _DISCLOSURES_URL and the parser.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.disclosures")

_DISCLOSURES_URL = "https://www.egx.com.eg/en/Disclosure.aspx"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (egx-mcp/0.1; mailto:m0hamedm0sad@gmail.com)",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}
_TIMEOUT = 15.0


def fetch(ticker: str | None = None, days: int = 7) -> dict[str, Any]:
    """Fetch recent disclosures.

    EGX's disclosures page returns the most recent ~50 items by default;
    we parse and filter them locally.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    canonical = None
    if ticker:
        canonical, _, _ = resolve_ticker(ticker)

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
            r = client.get(_DISCLOSURES_URL)
            r.raise_for_status()
            html = r.text
    except Exception as e:
        log.error(f"Failed to fetch EGX disclosures: {e}")
        return {
            "ticker": canonical,
            "days": days,
            "count": 0,
            "disclosures": [],
            "error": (
                f"Could not reach egx.com.eg ({e}). The EGX disclosures portal "
                "is occasionally rate-limited or rendered behind a CDN that "
                "blocks bot traffic. Try again in a few minutes, or check "
                "https://www.egx.com.eg/en/Disclosure.aspx directly."
            ),
        }

    soup = BeautifulSoup(html, "html.parser")
    disclosures = []

    # EGX renders disclosures in a table with rows of: Date | Symbol | Title | Link
    # Layout has changed historically — we look for any table with date-like cells.
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            # Try to parse a date from the first cell
            date_text = cells[0].get_text(strip=True)
            disc_date = _try_parse_date(date_text)
            if disc_date is None:
                continue
            if disc_date < cutoff:
                continue

            symbol = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            title = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            link_tag = row.find("a", href=True)
            url = link_tag["href"] if link_tag else None
            if url and url.startswith("/"):
                url = "https://www.egx.com.eg" + url

            if canonical and canonical.upper() not in symbol.upper():
                continue

            disclosures.append({
                "date": disc_date.strftime("%Y-%m-%d"),
                "ticker": symbol,
                "title": title,
                "url": url,
            })

    # Dedupe and sort
    seen = set()
    unique = []
    for d in disclosures:
        key = (d["date"], d["ticker"], d["title"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    unique.sort(key=lambda d: d["date"], reverse=True)

    return {
        "ticker": canonical,
        "days": days,
        "count": len(unique),
        "disclosures": unique,
        "source": _DISCLOSURES_URL,
    }


def _try_parse_date(text: str) -> datetime | None:
    """EGX uses formats like '14/04/2026', '2026-04-14', '14 Apr 2026'."""
    text = text.strip()
    fmts = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"]
    for f in fmts:
        try:
            return datetime.strptime(text, f)
        except ValueError:
            continue
    return None
