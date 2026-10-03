"""Collect English EGX headlines for labeling (read-only, run on CI).

Per-stock: yfinance news for every listing in egx_sectors.json (the path
news._fetch_english uses). Market-wide: the multi-source scraper, keeping
only headlines that are actually in English (Mubasher's market page is
Arabic). Prints one compact JSON line so it survives job-log limits.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yfinance as yf

from egx_mcp.data import news_scrapers

_ARABIC = re.compile(r"[؀-ۿ]")


def main() -> int:
    sectors = json.loads(Path("egx_mcp/data/egx_sectors.json").read_text(encoding="utf-8"))
    tickers = sorted(sectors["tickers"])
    out: dict[str, dict] = {}
    for tk in tickers:
        try:
            items = yf.Ticker(f"{tk}.CA").news or []
        except Exception as e:  # noqa: BLE001
            print(f"{tk}: {e}", file=sys.stderr)
            continue
        for item in items:
            c = item.get("content", item)
            title = (c.get("title") or "").strip()
            if title and not _ARABIC.search(title):
                out.setdefault(title, {"ticker": tk, "date": str(c.get("pubDate") or "")[:10]})
    try:
        market = news_scrapers.fetch_market_multi(limit=60)
    except Exception as e:  # noqa: BLE001
        print(f"market: {e}", file=sys.stderr)
        market = []
    for m in market:
        title = (m.get("title") or "").strip()
        if title and not _ARABIC.search(title):
            out.setdefault(title, {"ticker": "MKT", "date": m.get("date") or ""})
    print(f"collected {len(out)} English headlines from {len(tickers)} tickers + market")
    print("EN_HEADLINES_JSON=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
