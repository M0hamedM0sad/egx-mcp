"""Expand the simulation universe to the full EGX market.

Master symbol list comes from TradingView's Egypt scanner (~289 names, the
whole exchange). Each is then validated against investing.com via
`investing.daily_history` — the model's ACTUAL fetch path — keeping only names
with >=30 real (non-zero-volume) daily bars, i.e. genuinely simulatable.

Writes the validated set to egx_universe_investing.json and reports how many of
today's ceiling-class movers (>=+9.5%) are now covered.

    python -m scripts.expand_universe
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data._certs import ensure_ca_bundle

ensure_ca_bundle()
from curl_cffi import requests as cr

from egx_mcp.data import investing

_OUT = Path(__file__).parent.parent / "egx_mcp" / "data" / "egx_universe_investing.json"


def _tv_stocks() -> dict[str, list]:
    """{ticker: [name, close, change, volume]} for all EGX stocks."""
    for _ in range(4):
        try:
            d = cr.post(
                "https://scanner.tradingview.com/egypt/scan",
                json={"columns": ["name", "close", "change", "volume", "type"],
                      "range": [0, 600]},
                impersonate="chrome", timeout=40,
            ).json()
            return {x["s"].split(":")[1]: x["d"] for x in d.get("data", [])
                    if x["d"][4] == "stock"}
        except Exception:
            time.sleep(2)
    raise SystemExit("TradingView scan failed after retries")


def main():
    stocks = _tv_stocks()
    print(f"EGX stocks on TradingView: {len(stocks)}")

    validated, nodata = [], []
    for i, tk in enumerate(stocks):
        try:
            df = investing.daily_history(tk, lookback_days=120)
            if len(df) >= 30:
                validated.append({
                    "ticker": tk,
                    "bars": int(len(df)),
                    "last_close": round(float(df["Close"].iloc[-1]), 4),
                    "last_date": df.index[-1].strftime("%Y-%m-%d"),
                })
            else:
                nodata.append(tk)
        except Exception:
            nodata.append(tk)
        time.sleep(0.3)
        if i % 25 == 0:
            print(f"  ...{i}/{len(stocks)} probed, {len(validated)} usable")

    _OUT.write_text(json.dumps(
        {"source": "tradingview master + investing.com validation",
         "validated": validated, "nodata": nodata}, indent=2), encoding="utf-8")

    print(f"\nVALIDATED (>=30 real-volume bars): {len(validated)}")
    print(f"No usable investing.com data:      {len(nodata)}")

    # Ceiling coverage check
    vset = {v["ticker"] for v in validated}
    ceil = sorted([(tk, d[2], d[3]) for tk, d in stocks.items()
                   if d[2] is not None and d[2] >= 9.5], key=lambda x: -x[1])
    print(f"\nToday's ceiling-class movers (>=+9.5%): {len(ceil)}")
    for tk, ch, vol in ceil:
        mark = "COVERED" if tk in vset else "missing"
        print(f"  {tk:<7} +{ch:>5.2f}%  vol={vol or 0:>14,.0f}   [{mark}]")
    print(f"\nSaved: {_OUT}")


if __name__ == "__main__":
    main()
