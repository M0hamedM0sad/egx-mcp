"""Convert Mubasher cache → audited fundamentals CSV.

Reads egx_mcp/data/mubasher_fundamentals_cache.json (built by running
`python -m egx_mcp.data.mubasher_fundamentals`) and writes the CSV
that fundamentals.py reads via $EGX_FUNDAMENTALS_CSV.

    python -m scripts.export_fundamentals_csv

Output: egx_fundamentals_audited.csv at repo root.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = Path(__file__).parent.parent / "egx_mcp" / "data" / "mubasher_fundamentals_cache.json"
OUT = Path(__file__).parent.parent / "egx_fundamentals_audited.csv"

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


def main():
    if not CACHE.exists():
        print(f"Cache not found: {CACHE}\nRun: python -m egx_mcp.data.mubasher_fundamentals")
        sys.exit(1)

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
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
