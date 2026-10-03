"""Event study: what did US Fed rate moves do to USD/EGP and to EGX stocks?

Before the Fed enters the model it has to show an effect. For every FOMC
move since 2015 (a change in FRED's DFEDTARU, dated by effective day = the
first EGX session after the ~21:00-Cairo announcement), this measures:

  - EGX market: equal-weight daily return of the validated universe
    (logs/panel_prices.json), cumulative over 1, 5 and 21 sessions from the
    close BEFORE the effective day;
  - USD/EGP (Yahoo USDEGP=X) over the same sessions (+ = pound weakens);
  - both against the unconditional distribution of every same-length
    window, so "the market rose after cuts" is judged against how often it
    rises anyway. t = (event mean - all-window mean) / (all-window sd / sqrt n).

Split by hikes and cuts. Holds are not in the FRED series, so they are not
tested. Samples are small (tens of events): read effect sizes and signs,
not precise p-values. Survivor-only universe, as in the factor study.

    python -m scripts.fed_event_study
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import fed  # noqa: E402
from scripts.build_panel import _load_prices  # noqa: E402

ROOT = Path(__file__).parent.parent
_OUT_TXT = ROOT / "logs" / "fed_event_study_latest.txt"
_OUT_JSON = ROOT / "logs" / "fed_event_study_latest.json"
WINDOWS = (1, 5, 21)


def market_index(prices: dict[str, list[dict]]) -> pd.Series:
    """Equal-weight daily return of the universe, on sessions where >= 30% of
    names traded (same calendar rule as the factor study)."""
    closes = {}
    for tk, rows in prices.items():
        if tk == "EGX30" or not rows:
            continue
        s = pd.Series({r["date"]: r.get("close") for r in rows}, dtype="float64")
        closes[tk] = s
    px = pd.DataFrame(closes)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    px = px[px.notna().mean(axis=1) >= 0.3].ffill(limit=5)
    rets = px.pct_change(fill_method=None).clip(-0.5, 0.5)   # guard bad prints
    return (1 + rets.mean(axis=1).fillna(0)).cumprod()


def fx_series() -> pd.Series:
    import yfinance as yf

    h = yf.Ticker("USDEGP=X").history(period="max", interval="1d", auto_adjust=False)
    s = h["Close"].copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")].dropna()


def window_returns(level: pd.Series, calendar: pd.DatetimeIndex, start_pos: int, k: int) -> float | None:
    """Return from the close at calendar[start_pos-1] to calendar[start_pos-1+k],
    reading `level` at or before each calendar date."""
    if start_pos < 1 or start_pos - 1 + k >= len(calendar):
        return None
    a = level.asof(calendar[start_pos - 1])
    b = level.asof(calendar[start_pos - 1 + k])
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return float(b / a - 1)


def study(market: pd.Series, fx: pd.Series, events: list[dict]) -> dict:
    cal = market.index
    res: dict = {"events": [], "baseline": {}, "groups": {}}
    base = {}
    for name, level in (("egx", market), ("usdegp", fx)):
        for k in WINDOWS:
            vals = [window_returns(level, cal, p, k) for p in range(1, len(cal) - k, k)]
            vals = [v for v in vals if v is not None]
            base[(name, k)] = (float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals))
            res["baseline"][f"{name}_{k}d"] = {"mean_pct": 100 * base[(name, k)][0],
                                               "sd_pct": 100 * base[(name, k)][1],
                                               "n_windows": len(vals)}
    for ev in events:
        eff = pd.Timestamp(ev["effective_date"])
        if eff < cal[0] or eff > cal[-1]:
            continue
        pos = int(cal.searchsorted(eff))
        rec = {**ev, "egx_session": str(cal[pos].date()) if pos < len(cal) else None}
        for name, level in (("egx", market), ("usdegp", fx)):
            for k in WINDOWS:
                v = window_returns(level, cal, pos, k)
                rec[f"{name}_{k}d_pct"] = None if v is None else round(100 * v, 3)
        res["events"].append(rec)

    for grp, sel in (("all moves", lambda e: True), ("hikes", lambda e: e["action"] == "hike"),
                     ("cuts", lambda e: e["action"] == "cut")):
        evs = [e for e in res["events"] if sel(e)]
        g = {"n": len(evs)}
        for name in ("egx", "usdegp"):
            for k in WINDOWS:
                vals = [e[f"{name}_{k}d_pct"] / 100 for e in evs if e[f"{name}_{k}d_pct"] is not None]
                if len(vals) < 3:
                    continue
                m0, sd0, _ = base[(name, k)]
                m = float(np.mean(vals))
                g[f"{name}_{k}d"] = {
                    "mean_pct": 100 * m, "median_pct": 100 * float(np.median(vals)),
                    "vs_normal_pct": 100 * (m - m0),
                    "t": (m - m0) / (sd0 / math.sqrt(len(vals))) if sd0 > 0 else None,
                    "pct_up": 100 * float(np.mean([v > 0 for v in vals])), "n": len(vals)}
        res["groups"][grp] = g
    return res


def render(res: dict) -> list[str]:
    L = ["=" * 78, "FED RATE MOVES -> USD/EGP AND EGX  (event study, effective-date sessions)",
         "=" * 78, "EGX = equal-weight universe; USD/EGP + = pound weaker. 'vs normal' = event mean",
         "minus the mean of every same-length window; |t| > 2 ~ significant.", ""]
    for grp, g in res["groups"].items():
        L.append(f"{grp}  (n={g['n']})")
        for name, label in (("egx", "EGX stocks"), ("usdegp", "USD/EGP   ")):
            for k in WINDOWS:
                r = g.get(f"{name}_{k}d")
                if not r:
                    continue
                L.append(f"  {label} {k:>2}d: mean {r['mean_pct']:+6.2f}%  vs normal "
                         f"{r['vs_normal_pct']:+6.2f}%  t={r['t'] or 0:+5.2f}  up {r['pct_up']:3.0f}%")
        L.append("")
    L.append("events:")
    for e in res["events"]:
        L.append(f"  {e['effective_date']} {e['action']:4} {e['change_bp']:+4d}bp -> {e['upper_pct']:.2f}%"
                 f" | EGX 5d {e.get('egx_5d_pct') if e.get('egx_5d_pct') is not None else 'n/a':>7}"
                 f" | USDEGP 5d {e.get('usdegp_5d_pct') if e.get('usdegp_5d_pct') is not None else 'n/a':>7}")
    L += ["", "Caveats: tens of events; holds untested; EGP moves are dominated by CBE",
          "devaluation dates, which this study does not separate out."]
    return L


def main() -> int:
    upper = fed.fetch_series("DFEDTARU")
    events = fed.decisions(upper)
    print(f"FRED DFEDTARU: {len(upper)} days, {len(events)} moves "
          f"({events[0]['effective_date'] if events else '-'} .. "
          f"{events[-1]['effective_date'] if events else '-'})")
    prices, source = _load_prices()
    print(f"Prices: {source} ({len(prices)} series)")
    market = market_index(prices)
    fx = fx_series()
    print(f"EGX index {market.index[0].date()}..{market.index[-1].date()}; "
          f"USDEGP {fx.index[0].date()}..{fx.index[-1].date()}")
    res = study(market, fx, events)
    text = "\n".join(render(res))
    print(text)
    _OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    _OUT_TXT.write_text(text + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
