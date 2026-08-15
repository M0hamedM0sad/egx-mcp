"""Convert Mubasher cache → audited fundamentals CSV.

Reads egx_mcp/data/mubasher_fundamentals_cache.json (built by running
`python -m egx_mcp.data.mubasher_fundamentals`) and writes the CSV
that fundamentals.py reads via $EGX_FUNDAMENTALS_CSV.

    python -m scripts.export_fundamentals_csv

Output: egx_fundamentals_audited.csv at repo root.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# reconfigure() rather than wrapping sys.stdout.buffer in a fresh TextIOWrapper:
# this module is imported by other scripts that do the same, and a second
# wrapper leaves the first unreferenced — TextIOWrapper closes the underlying
# buffer when collected, killing stdout mid-run. Mutating in place is idempotent.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = Path(__file__).parent.parent / "egx_mcp" / "data" / "mubasher_fundamentals_cache.json"
OUT = Path(__file__).parent.parent / "egx_fundamentals_audited.csv"

# Watchlist canonicals whose EGX/Mubasher code differs — the cache holds the
# row under the exchange code, but fundamentals.py looks up the canonical.
# (ESRS and IDHC have no alias: both delisted from EGX.)
ALIASES = {
    "SODIC": "OCDI",   # Sixth of October Development trades as OCDI
    "EIPI": "PHAR",    # EIPICO trades as PHAR
    "MNHD": "MASR",    # Madinet Nasr renamed Madinet Masr (MASR)
}

FIELDS = [
    "ticker",
    "trailing_eps",
    "book_value_per_share",
    "pe_ratio",
    "pb_ratio",
    "roe_pct",
    "profit_margin_pct",
    "debt_to_equity",
    "dividend_yield_pct",
    "market_cap",
]


def _tv_fill(cache: dict) -> None:
    """Fill fields Mubasher no longer serves from TradingView's scanner API.

    Mubasher's ratios page dropped Net Profit Margin, Debt/Equity and
    Dividend Yield (only ROE/ROA/EPS/growth remain), which capped every name
    at MEDIUM coverage. One scanner POST returns the whole Egyptian market.
    Mubasher values always win — TV only fills blanks."""
    import httpx
    from egx_mcp.data._certs import ensure_ca_bundle

    ensure_ca_bundle()
    cols = ["name", "net_margin_ttm", "net_margin_fy", "net_income_fy",
            "total_revenue_fy", "debt_to_equity_fq", "debt_to_equity",
            "dividend_yield_recent"]
    body = {"filter": [], "options": {"lang": "en"}, "markets": ["egypt"],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": cols, "range": [0, 400]}
    try:
        r = httpx.post("https://scanner.tradingview.com/egypt/scan", json=body,
                       timeout=30, headers={"User-Agent": "Mozilla/5.0 Chrome/120.0",
                                            "Origin": "https://www.tradingview.com"})
        r.raise_for_status()
        data = r.json()["data"]
    except Exception as e:  # noqa: BLE001
        print(f"TradingView fill skipped ({e}) — margins/D-E stay as scraped.")
        return

    filled = {"profit_margin_pct": 0, "debt_to_equity": 0, "dividend_yield_pct": 0}
    for row in data:
        tv = dict(zip(cols[1:], row["d"][1:]))
        entry = cache.get(row["d"][0])
        if entry is None:
            continue
        margin = tv.get("net_margin_ttm")
        if margin is None:
            margin = tv.get("net_margin_fy")
        if margin is None and tv.get("net_income_fy") is not None and tv.get("total_revenue_fy"):
            margin = 100 * tv["net_income_fy"] / tv["total_revenue_fy"]
        de = tv.get("debt_to_equity_fq")
        if de is None:
            de = tv.get("debt_to_equity")
        for field, val in (("profit_margin_pct", margin), ("debt_to_equity", de),
                           ("dividend_yield_pct", tv.get("dividend_yield_recent"))):
            if val is not None and entry.get(field) is None:
                entry[field] = round(val, 4)
                filled[field] += 1
    print("TradingView fill:", ", ".join(f"{k}+{v}" for k, v in filled.items()))


def main():
    if not CACHE.exists():
        print(f"Cache not found: {CACHE}\nRun: python -m egx_mcp.data.mubasher_fundamentals")
        sys.exit(1)

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    _tv_fill(cache)
    for canonical, code in ALIASES.items():
        if canonical not in cache and code in cache:
            cache[canonical] = {**cache[code], "ticker": canonical}
    rows = []
    for ticker, data in cache.items():
        if "error" in data and not any(data.get(f) for f in FIELDS[1:]):
            continue
        row = {"ticker": ticker}
        for f in FIELDS[1:]:
            v = data.get(f)
            row[f] = "" if v is None else v
        rows.append(row)

    rows.sort(key=lambda r: r["ticker"])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT}")
    print(f"\nTo activate as the audited override:")
    print(f"  Windows:  set EGX_FUNDAMENTALS_CSV={OUT}")
    print(f"  Bash:     export EGX_FUNDAMENTALS_CSV='{OUT}'")
    print(f"\nSample rows:")
    for r in rows[:5]:
        print(f"  {r}")


if __name__ == "__main__":
    main()
