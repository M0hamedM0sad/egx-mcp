"""EGX30 proxy basket: read around Eid al-Fitr 2026 and Eid al-Adha 2026.

Yahoo's index history for EGX is unusable (README caveat). We hit the Yahoo
chart API directly (the system Python's trust store blocks yfinance/curl_cffi
SSL — likely corporate MITM — so we pass verify=False; data is public).

Builds an equal-weighted basket of liquid EGX30 constituents as a proxy.
"""
from __future__ import annotations
import sys
import datetime as dt
import pandas as pd
from curl_cffi import requests as creq

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 120)

BASKET = [
    "COMI.CA", "HDBK.CA", "CIEB.CA", "HRHO.CA", "CIRA.CA",
    "TMGH.CA", "PHDC.CA", "EMFD.CA", "MNHD.CA", "SWDY.CA",
    "ESRS.CA", "ABUK.CA", "MFPC.CA", "ETEL.CA", "EAST.CA",
    "FWRY.CA", "ORWE.CA", "JUFO.CA", "CLHO.CA", "EGTS.CA",
]

EID_FITR_WINDOW = ("2026-02-15", "2026-04-10")
EID_ADHA_WINDOW = ("2026-04-15", "2026-05-23")


def to_epoch(d: str) -> int:
    return int(dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_closes(symbol: str, start: str, end: str) -> pd.Series:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
    params = {
        "period1": to_epoch(start),
        "period2": to_epoch(end) + 86400,
        "interval": "1d",
        "events": "history",
    }
    r = creq.get(url, params=params, impersonate="chrome", verify=False, timeout=20)
    if r.status_code != 200:
        return pd.Series(dtype=float)
    js = r.json()
    res = js.get("chart", {}).get("result")
    if not res:
        return pd.Series(dtype=float)
    res = res[0]
    ts = res.get("timestamp") or []
    closes = res.get("indicators", {}).get("quote", [{}])[0].get("close") or []
    s = pd.Series(closes, index=pd.to_datetime(ts, unit="s").date)
    s = s.dropna()
    s.name = symbol
    return s


def build_basket(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    cols = {}
    for s in symbols:
        try:
            ser = fetch_closes(s, start, end)
            if not ser.empty:
                cols[s] = ser
            else:
                print(f"  warn: {s} returned no data", file=sys.stderr)
        except Exception as e:
            print(f"  warn: {s} failed: {e}", file=sys.stderr)
    return pd.DataFrame(cols).sort_index()


def equal_weight_index(closes: pd.DataFrame) -> pd.Series:
    # Normalize each name to its first valid close; average across names (equal weight).
    # Use forward-fill within each column so missing days don't drop the row.
    df = closes.copy()
    base = df.bfill().iloc[0]
    norm = df.divide(base, axis=1) * 100
    return norm.mean(axis=1, skipna=True)


def summarize(label: str, idx: pd.Series, closes: pd.DataFrame) -> None:
    print(f"\n=== {label} ===")
    if idx.empty:
        print("(no data)")
        return
    daily = pd.DataFrame({
        "proxy_close": idx.round(2),
        "pct_chg": (idx.pct_change() * 100).round(2),
        "n_names": closes.notna().sum(axis=1),
    })
    print(daily)
    first, last = idx.iloc[0], idx.iloc[-1]
    print(f"\nStart: {first:.2f}  End: {last:.2f}  High: {idx.max():.2f}  Low: {idx.min():.2f}  Window return: {(last/first-1)*100:+.2f}%")
    dates = list(idx.index)
    gaps = [((dates[i] - dates[i-1]).days, i) for i in range(1, len(dates))]
    if gaps:
        max_gap, max_idx = max(gaps, key=lambda x: x[0])
        print(f"Longest closure gap: {max_gap} days, between {dates[max_idx-1]} and {dates[max_idx]}")
        if max_gap >= 4:
            pre, post = idx.iloc[max_idx - 1], idx.iloc[max_idx]
            print(f"Pre-break close:  {pre:.2f}  ({dates[max_idx-1]})")
            print(f"Post-break close: {post:.2f}  ({dates[max_idx]})")
            print(f"Eid gap return:   {(post/pre-1)*100:+.2f}%")


def main() -> None:
    # Silence the InsecureRequestWarning since we're knowingly bypassing verification.
    import warnings, urllib3
    warnings.filterwarnings("ignore")
    try:
        urllib3.disable_warnings()
    except Exception:
        pass

    for label, (s, e) in [("EID AL-FITR 2026", EID_FITR_WINDOW),
                          ("EID AL-ADHA 2026 (pre-Eid run-up only)", EID_ADHA_WINDOW)]:
        print(f"\n############ {label}  [{s} -> {e}] ############")
        closes = build_basket(BASKET, s, e)
        if closes.empty:
            print("(no data fetched)")
            continue
        idx = equal_weight_index(closes)
        summarize(label, idx, closes)

        # also show per-name window return for color
        per_name = ((closes.bfill().iloc[-1] / closes.bfill().iloc[0]) - 1) * 100
        print("\nPer-name window return (%):")
        print(per_name.sort_values(ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
