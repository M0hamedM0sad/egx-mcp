"""Append today's fundamentals to an append-only history — the fix for the
panel's look-ahead.

`mubasher_fundamentals_cache.json` is a single current snapshot with one
`fetched_at` per ticker and no history, so scoring a 2024 date with it uses EPS
and book value that did not exist yet. Valuation and quality are therefore
contaminated in every panel row built before this script starts running.

This records what the fundamentals actually were on each date. Once history
accumulates, `build_panel` reads the most recent snapshot at or before each
rebalance date and those two sub-scores become point-in-time like momentum and
risk. Nothing retroactively fixes existing rows — the panel simply gets a
growing clean region, and the learner reports its size.

Storage is change-only: a row is appended just when a value differs from that
ticker's last recorded one. Fundamentals move on earnings, so daily runs cost a
few rows most days and capture the exact date each figure changed — which is
the whole point — instead of 250 identical rows per day.

    python -m scripts.snapshot_fundamentals            # snapshot from cache
    python -m scripts.snapshot_fundamentals --refresh  # rescrape first (slow)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# In-place reconfigure, not a fresh TextIOWrapper — see export_fundamentals_csv.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_fundamentals_csv import ALIASES, CACHE, FIELDS, _tv_fill  # noqa: E402

ROOT = Path(__file__).parent.parent
_HISTORY = ROOT / "logs" / "fundamentals_history.jsonl"

# The fields the sub-scorers actually read. market_cap is carried for context.
_TRACKED = [f for f in FIELDS if f != "ticker"]


def load_history() -> dict[str, list[dict]]:
    """{ticker: [rows sorted by snapshot_date]} — the point-in-time store."""
    out: dict[str, list[dict]] = {}
    if not _HISTORY.exists():
        return out
    for line in _HISTORY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.setdefault(r["ticker"], []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["snapshot_date"])
    return out


def as_of(history: dict[str, list[dict]], ticker: str, on: str) -> dict | None:
    """Most recent snapshot for `ticker` at or before `on`, else None.

    None means "no fundamentals were recorded yet on that date" — the caller
    must treat a fallback to the current snapshot as contaminated, never as a
    point-in-time read."""
    rows = history.get(ticker)
    if not rows:
        return None
    prior = [r for r in rows if r["snapshot_date"] <= on]
    return prior[-1] if prior else None


def _values(row: dict) -> tuple:
    return tuple(row.get(f) for f in _TRACKED)


def snapshot(refresh: bool = False, on: str | None = None) -> dict:
    if refresh:
        from egx_mcp.data import mubasher_fundamentals
        from egx_mcp.data.egx_listing import get_full_universe
        universe = get_full_universe()
        print(f"Rescraping {len(universe)} names ...")
        mubasher_fundamentals.scrape_universe(universe, force_refresh=True)

    if not CACHE.exists():
        raise SystemExit(f"No fundamentals cache at {CACHE}. Run with --refresh.")

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    _tv_fill(cache)                       # margin / D-E / yield from TradingView
    for canonical, code in ALIASES.items():
        if canonical not in cache and code in cache:
            cache[canonical] = {**cache[code], "ticker": canonical}

    stamp = on or str(date.today())
    history = load_history()
    appended, unchanged, skipped = [], 0, 0

    for ticker, data in sorted(cache.items()):
        row = {"snapshot_date": stamp, "ticker": ticker,
               **{f: data.get(f) for f in _TRACKED}}
        if all(row[f] is None for f in _TRACKED):
            skipped += 1                  # nothing worth recording
            continue
        prior = history.get(ticker)
        if prior and _values(prior[-1]) == _values(row):
            unchanged += 1
            continue
        if prior and prior[-1]["snapshot_date"] == stamp:
            continue                      # already snapshotted today; don't duplicate
        appended.append(row)

    if appended:
        _HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY.open("a", encoding="utf-8") as fh:
            for r in appended:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    history = load_history()
    covered = sum(1 for rows in history.values() if rows)
    first = min((r[0]["snapshot_date"] for r in history.values() if r), default=None)
    result = {"snapshot_date": stamp, "appended": len(appended),
              "unchanged": unchanged, "skipped_empty": skipped,
              "tickers_with_history": covered,
              "history_starts": first,
              "history_rows": sum(len(r) for r in history.values())}
    print(json.dumps(result, indent=2))
    if appended:
        print(f"\nAppended {len(appended)} changed row(s) -> {_HISTORY}")
    else:
        print("\nNo fundamentals changed since the last snapshot.")
    if first:
        print(f"Point-in-time fundamentals available from {first} onward; panel rows "
              f"before that date stay flagged as look-ahead-contaminated.")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Append today's fundamentals to history.")
    ap.add_argument("--refresh", action="store_true",
                    help="Rescrape Mubasher before snapshotting (slow, ~250 names).")
    ap.add_argument("--date", help="Override the snapshot date (YYYY-MM-DD).")
    args = ap.parse_args()
    snapshot(refresh=args.refresh, on=args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
