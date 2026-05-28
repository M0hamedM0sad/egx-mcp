"""Official egx.com.eg adapter — Playwright-based.

The Egyptian Exchange site is firewalled by TSPD (Akamai-style anti-bot).
Plain `httpx` requests get back a 5,754-byte JavaScript challenge page,
not the actual content. Playwright runs a real Chromium that solves the
challenge, after which we extract the index/quote/disclosure tables.

Cost: each call spawns a headless Chromium (~1-3s warm, ~5s cold) and
holds ~150MB of RAM while open. Cache aggressively. The default TTL is
10 minutes per endpoint — long enough for the briefing to reuse a single
fetch across all sections, short enough that intraday price moves still
update.

Endpoints currently supported:
  fetch_indices()      EGX 30 / 30 TR / 70 EWI / 100 / FNTV / etc.
  fetch_market_watch() Top gainers / losers / most-active
  fetch_disclosures()  Replaces the broken httpx scraper

Failure mode: any function returns {error: "..."} on Playwright failure
or TSPD lockout, so callers can fall through to TV / yfinance.

Install requirement (one-time):
    pip install playwright
    playwright install chromium
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger("egx-mcp.egx_official")

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TTL = 600  # 10 minutes


# Playwright import is lazy — module loads even when the package isn't
# installed, so the rest of the project doesn't break.
def _playwright_ready() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


# A minimal user-agent that matches a recent Chromium. Don't override
# Playwright's UA defaults — they already pass TSPD's heuristics.
_VIEWPORT = {"width": 1366, "height": 900}
_NAV_TIMEOUT_MS = 30_000  # 30s — TSPD challenge can be slow on cold cache


def _fresh_page(playwright):
    """Spawn a stealthed page that survives TSPD.

    Uses playwright-stealth to patch the most common headless tells:
      - navigator.webdriver
      - chrome.runtime / chrome.csi
      - WebGL vendor / renderer
      - Plugin / mime-type lists
      - Permissions API
      - Various canvas / audio fingerprints
    """
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(
        viewport=_VIEWPORT,
        locale="en-US",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        java_script_enabled=True,
    )
    page = context.new_page()
    try:
        from playwright_stealth import Stealth
        # v2 API: apply evasion scripts to the page
        Stealth().apply_stealth_sync(page)
    except Exception as e:
        log.warning(f"playwright-stealth not applied: {e}")
    return browser, context, page


def _wait_through_tspd(page, max_attempts: int = 8) -> bool:
    """Block until the TSPD challenge clears.

    Strategy:
      - Poll document size every 2s (real pages > 30KB)
      - On each attempt, also poll for table elements (final content
        signal — TSPD challenge has zero <table> tags)
      - Reload halfway through if still stuck
    """
    for attempt in range(max_attempts):
        try:
            content_size = page.evaluate("document.documentElement.outerHTML.length")
            table_count = page.evaluate("document.querySelectorAll('table').length")
        except Exception:
            content_size = 0
            table_count = 0
        if content_size > 30_000 and table_count > 0:
            return True
        # Halfway through, give it a fresh reload — sometimes TSPD wedges
        if attempt == max_attempts // 2:
            try:
                page.reload(timeout=_NAV_TIMEOUT_MS, wait_until="load")
            except Exception:
                pass
        page.wait_for_timeout(2500)
    return False


def _cached(key: str, loader):
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _TTL:
            return val
    val = loader()
    _CACHE[key] = (now, val)
    return val


# ---------------------------------------------------------------------------
# Indices (EGX 30 / 70 / 100)
# ---------------------------------------------------------------------------

def fetch_indices() -> dict[str, Any]:
    """Pull the official EGX index values from the Indices page.

    Returns a dict keyed by index name (EGX30, EGX30TR, EGX70EWI, EGX100,
    FNTV, …) where each value has `price`, `change`, `change_pct`.
    """
    if not _playwright_ready():
        return {"error": "playwright not installed; run: pip install playwright && playwright install chromium"}

    def _load():
        from playwright.sync_api import sync_playwright
        url = "https://www.egx.com.eg/en/Indices.aspx"
        try:
            with sync_playwright() as p:
                browser, ctx, page = _fresh_page(p)
                try:
                    page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="networkidle")
                    if not _wait_through_tspd(page):
                        return {"error": "TSPD challenge did not clear", "source": url}
                    # Try a structured table extraction
                    rows = page.evaluate("""
                        () => {
                          const out = [];
                          const tables = document.querySelectorAll('table');
                          for (const t of tables) {
                            for (const tr of t.querySelectorAll('tr')) {
                              const cells = [...tr.querySelectorAll('td,th')]
                                .map(c => c.textContent.trim());
                              if (cells.length >= 3) out.push(cells);
                            }
                          }
                          return out;
                        }
                    """)
                finally:
                    ctx.close()
                    browser.close()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "source": url}

        # Parse rows — index name is usually first cell, value is a
        # numeric cell, change% has a '%' suffix.
        indices: dict[str, dict[str, Any]] = {}
        for cells in rows:
            if not cells:
                continue
            label = cells[0]
            if not label or len(label) < 3:
                continue
            # Try to find a numeric value and a percent change
            value = None
            change_pct = None
            for c in cells[1:]:
                if value is None:
                    try:
                        value = float(c.replace(",", ""))
                        continue
                    except ValueError:
                        pass
                if "%" in c:
                    try:
                        change_pct = float(c.replace("%", "").replace(",", "").strip())
                    except ValueError:
                        pass
            if value is not None:
                # Normalize the label — strip suffixes / dashes
                key = label.replace(" ", "").upper()
                indices[key] = {
                    "label": label,
                    "value": value,
                    "change_pct": change_pct,
                }
        return {
            "source": url,
            "as_of_unix": time.time(),
            "indices": indices,
            "row_count": len(rows),
        }

    return _cached("indices", _load)


# ---------------------------------------------------------------------------
# Market watch — top movers
# ---------------------------------------------------------------------------

def fetch_market_watch() -> dict[str, Any]:
    """Pull top gainers / losers / most-active from the Market Watch page."""
    if not _playwright_ready():
        return {"error": "playwright not installed"}

    def _load():
        from playwright.sync_api import sync_playwright
        url = "https://www.egx.com.eg/en/MostActive.aspx"
        try:
            with sync_playwright() as p:
                browser, ctx, page = _fresh_page(p)
                try:
                    page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="networkidle")
                    if not _wait_through_tspd(page):
                        return {"error": "TSPD challenge did not clear", "source": url}
                    rows = page.evaluate("""
                        () => {
                          const out = [];
                          for (const t of document.querySelectorAll('table')) {
                            for (const tr of t.querySelectorAll('tr')) {
                              const cells = [...tr.querySelectorAll('td,th')]
                                .map(c => c.textContent.trim());
                              if (cells.length >= 3) out.push(cells);
                            }
                          }
                          return out;
                        }
                    """)
                finally:
                    ctx.close()
                    browser.close()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "source": url}

        return {
            "source": url,
            "as_of_unix": time.time(),
            "rows": rows[:50],  # cap the payload
            "row_count": len(rows),
        }

    return _cached("market_watch", _load)


# ---------------------------------------------------------------------------
# Disclosures — replaces the broken httpx scraper
# ---------------------------------------------------------------------------

def fetch_disclosures() -> dict[str, Any]:
    """Pull the latest disclosures from the official disclosures page."""
    if not _playwright_ready():
        return {"error": "playwright not installed"}

    def _load():
        from playwright.sync_api import sync_playwright
        url = "https://www.egx.com.eg/en/disclosure.aspx"
        try:
            with sync_playwright() as p:
                browser, ctx, page = _fresh_page(p)
                try:
                    page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="networkidle")
                    if not _wait_through_tspd(page):
                        return {"error": "TSPD challenge did not clear", "source": url}
                    rows = page.evaluate("""
                        () => {
                          const out = [];
                          for (const t of document.querySelectorAll('table')) {
                            for (const tr of t.querySelectorAll('tr')) {
                              const cells = [...tr.querySelectorAll('td,th')]
                                .map(c => c.textContent.trim());
                              const links = [...tr.querySelectorAll('a')]
                                .map(a => a.href);
                              if (cells.length >= 2)
                                out.push({cells, links});
                            }
                          }
                          return out;
                        }
                    """)
                finally:
                    ctx.close()
                    browser.close()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "source": url}

        return {
            "source": url,
            "as_of_unix": time.time(),
            "rows": rows[:60],
            "row_count": len(rows),
        }

    return _cached("disclosures", _load)
