"""Rebuild egx_mcp/data/driver_profiles.json (weekly, on GitHub).

Uses the panel's price history (logs/panel_prices.json, refreshed by the
weekly panel build) plus Yahoo series for USD/EGP, Brent and gold.

    python -m scripts.build_driver_profiles
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import driver_profiles, sectors  # noqa: E402
from scripts.build_panel import _load_prices  # noqa: E402

_YAHOO = {"usdegp": "USDEGP=X", "brent": "BZ=F", "gold": "GC=F"}


def _yahoo_close(symbol: str) -> pd.Series:
    import yfinance as yf

    h = yf.Ticker(symbol).history(period="5y", interval="1d", auto_adjust=False)
    s = h["Close"].copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.normalize()
    return s.dropna()


def main() -> int:
    prices, source = _load_prices()
    closes = {}
    for tk, rows in prices.items():
        if tk == "EGX30" or not rows:
            continue
        s = pd.Series({r["date"]: r.get("close") for r in rows}, dtype="float64")
        s.index = pd.to_datetime(s.index)
        closes[tk] = s
    drivers = {k: _yahoo_close(v) for k, v in _YAHOO.items()}
    print(f"prices: {source} ({len(closes)} names); drivers: "
          + ", ".join(f"{k} {len(v)}d" for k, v in drivers.items()))
    res = driver_profiles.build(closes, drivers, sectors.all_sectors())
    profiles = res["profiles"]
    top = Counter(p["drivers"][0]["driver"] if p["drivers"] else "stock-specific"
                  for p in profiles.values())
    print(f"profiles: {len(profiles)}  as of {res['as_of']}  weekly driver sd %: "
          f"{res['driver_weekly_sd_pct']}")
    print(f"strongest driver per name: {dict(top)}")
    print("\nmedian weight of each driver in weekly moves, by sector (%):")
    print(f"  {'sector':22} {'n':>3} {'market':>7} {'USD/EGP':>8} {'oil':>5} {'gold':>5} {'own':>5}")
    for sec, w in sorted(res["sector_weights_pct"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {sec:22} {w['n']:>3} {w['market']:>7} {w['usdegp']:>8} {w['brent']:>5} "
              f"{w['gold']:>5} {w['stock_specific']:>5}")
    print()
    for tk in ("COMI", "ABUK", "SWDY", "TMGH", "AMOC", "EAST"):
        c = driver_profiles.describe(tk, data=res)
        print(f"  {tk}: " + (" | ".join(c["lines"]) if c["available"] else c["note"]))
    driver_profiles.PROFILE_PATH.write_text(json.dumps(res, indent=1, sort_keys=True),
                                            encoding="utf-8")
    print(f"wrote {driver_profiles.PROFILE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
