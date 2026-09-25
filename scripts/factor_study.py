"""Multi-year, point-in-time factor study for EGX: which inputs predicted the
next 21 sessions' relative return, and was it stable across years?

Live grading found the valuation and quality sub-scores NEGATIVELY related
to 21-session excess (rank IC -0.23 / -0.25 over 52 dates, Jun–Sep 2026).
Four months is one regime. This rebuilds the same kinds of signal back to
~2018 from TradingView's quarterly history (egx_mcp/data/tv_history.py),
each figure visible only REPORTING_LAG_DAYS after its quarter end, plus
price-only signals, and measures each one's cross-sectional rank IC.

Design:
  - rebalance every 21 sessions (non-overlapping labels, so the t-stat on the
    mean IC is not inflated by overlapping windows);
  - label = forward 21-session return from the NEXT session's close, minus
    the cross-sectional median (same benchmark as tests/grade_briefings.py);
  - IC = Spearman rank correlation within a date, averaged over dates;
    IC > 0 means "higher factor value -> better next-month relative return".

Signals (point-in-time):
  earnings_yield  TTM EPS / price           (valuation proxy: cheap = high)
  roa             TTM net income / assets   (quality proxy)
  net_margin      TTM net income / revenue  (quality proxy)
  low_leverage    -(total debt / assets)    (quality proxy)
  value_composite rank of earnings_yield
  quality_composite mean rank of roa, net_margin, low_leverage
  mom_6m, rev_1m, low_vol_3m, liquidity (log median traded value, 3m)

Known residual biases (printed with the results): TradingView serves today's
restated figures, and covers only currently listed names (survivorship).

    python -m scripts.factor_study                  # uses logs/panel_prices.json
    python -m scripts.factor_study --refresh-tv     # refetch TradingView history
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import tv_history  # noqa: E402
from scripts.build_panel import _load_prices  # noqa: E402

ROOT = Path(__file__).parent.parent
_OUT_TXT = ROOT / "logs" / "factor_study_latest.txt"
_OUT_JSON = ROOT / "logs" / "factor_study_latest.json"

HORIZON = 21
WARMUP = 130          # rows of history before the first rebalance (6m momentum)
MIN_NAMES = 20        # names with both factor and label for a date to count
FFILL_LIMIT = 5       # thin names: carry a close across at most 5 missing sessions

FUND_FACTORS = ["earnings_yield", "earnings_yield_in_sector", "roa", "net_margin", "low_leverage"]
MIN_SECTOR_NAMES = 5  # a sector-relative rank needs this many names with a value
PRICE_FACTORS = ["mom_6m", "rev_1m", "low_vol_3m", "liquidity"]
COMPOSITES = ["value_composite", "quality_composite"]
ALL_FACTORS = FUND_FACTORS + COMPOSITES + PRICE_FACTORS


def _frames(prices: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes, vols = {}, {}
    for tk, rows in prices.items():
        if tk == "EGX30" or not rows:
            continue
        idx = pd.to_datetime([r["date"] for r in rows])
        closes[tk] = pd.Series([r.get("close") for r in rows], index=idx, dtype="float64")
        vols[tk] = pd.Series([r.get("volume") or 0 for r in rows], index=idx, dtype="float64")
    raw = pd.DataFrame(closes).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    # Calendar: sessions where a meaningful share of the market traded.
    raw = raw[raw.notna().mean(axis=1) >= 0.3]
    vol = pd.DataFrame(vols).reindex(raw.index)
    return raw, raw.ffill(limit=FFILL_LIMIT), vol


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    ok = x.notna() & y.notna()
    if int(ok.sum()) < MIN_NAMES:
        return None
    a, b = x[ok].rank(), y[ok].rank()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _features_at(i: int, raw: pd.DataFrame, px: pd.DataFrame, vol: pd.DataFrame,
                 tables: dict[str, list], asof: date) -> pd.DataFrame:
    price = px.iloc[i]
    recent = raw.iloc[max(0, i - FFILL_LIMIT):i + 1].notna().any()   # traded lately
    rets = px.iloc[i - 63:i + 1].pct_change(fill_method=None)
    traded = (raw.iloc[i - 63:i + 1] * vol.iloc[i - 63:i + 1])
    f = pd.DataFrame(index=px.columns)
    f["mom_6m"] = price / px.iloc[i - 126] - 1
    f["rev_1m"] = -(price / px.iloc[i - 21] - 1)
    f["low_vol_3m"] = -rets.std()
    f["liquidity"] = np.log(traded.where(traded > 0).median())
    fund = {tk: tv_history.ratios(tv_history.pit(tables[tk], asof), price.get(tk))
            for tk in px.columns if tk in tables}
    # None (missing ratio) must become NaN: an all-None column is object dtype.
    fd = pd.DataFrame.from_dict(fund, orient="index").apply(pd.to_numeric, errors="coerce")
    for col in ("earnings_yield", "roa", "net_margin"):
        f[col] = fd[col] if col in fd else np.nan
    f["low_leverage"] = -fd["debt_to_assets"] if "debt_to_assets" in fd else np.nan
    f = f[recent & price.notna()]
    # Clip extreme ratios (tiny denominators) before ranking into composites.
    for col in ("earnings_yield", "roa", "net_margin"):
        f[col] = f[col].clip(-1, 1)
    # Cheapness within the name's own sector (egx_sectors.json): is comparing
    # a bank with banks more informative than comparing it with the market?
    from egx_mcp.data import sectors
    sec = pd.Series({tk: sectors.all_sectors().get(tk) for tk in f.index})
    grp = f["earnings_yield"].groupby(sec)
    f["earnings_yield_in_sector"] = grp.rank(pct=True).where(grp.transform("count") >= MIN_SECTOR_NAMES)
    f["value_composite"] = f["earnings_yield"].rank(pct=True)
    q = f[["roa", "net_margin", "low_leverage"]].rank(pct=True)
    f["quality_composite"] = q.mean(axis=1, skipna=False)
    return f


def run(prices: dict, hist: dict, keep_frames: bool = False) -> dict:
    """Per-date ICs; with keep_frames, also each date's factor table + label
    (used by scripts/propose_factor_weights.py for the walk-forward)."""
    raw, px, vol = _frames(prices)
    entries = hist["tickers"]
    excluded = sorted(tk for tk, e in entries.items() if tv_history.units_suspect(e))
    tables = {tk: tv_history.quarterly_table(e) for tk, e in entries.items()
              if tk not in excluded}
    tables = {tk: t for tk, t in tables.items() if t}

    per_date: list[dict] = []
    for i in range(WARMUP, len(px) - HORIZON - 1, HORIZON):
        d = px.index[i]
        feats = _features_at(i, raw, px, vol, tables, d.date())
        fwd = px.iloc[i + 1 + HORIZON] / px.iloc[i + 1] - 1
        fwd = fwd.reindex(feats.index)
        if fwd.notna().sum() < MIN_NAMES:
            continue
        excess = fwd - fwd.median()
        mkt_6m = float(feats["mom_6m"].median())
        mkt_60d = float((px.iloc[i] / px.iloc[i - 60] - 1).reindex(feats.index).median())
        rec = {"date": d.strftime("%Y-%m-%d"), "n_names": int(fwd.notna().sum()),
               "n_fund": int(feats["earnings_yield"].notna().sum()),
               "market_6m": mkt_6m, "market_60d": mkt_60d, "ic": {}, "spread": {}}
        if keep_frames:
            rec["frame"] = feats[ALL_FACTORS].assign(excess=excess)
        for fac in ALL_FACTORS:
            rec["ic"][fac] = _spearman(feats[fac], excess)
            ok = feats[fac].notna() & excess.notna()
            if ok.sum() >= MIN_NAMES:
                qs = pd.qcut(feats[fac][ok].rank(method="first"), 5, labels=False)
                e = excess[ok]
                rec["spread"][fac] = float(e[qs == 4].median() - e[qs == 0].median())
        per_date.append(rec)
    return {"per_date": per_date, "excluded_units_suspect": excluded,
            "n_tables": len(tables), "calendar": [str(px.index[0].date()), str(px.index[-1].date())]}


def _stats(vals: list[float]) -> dict:
    v = [x for x in vals if x is not None and not math.isnan(x)]
    if len(v) < 3:
        return {"n": len(v)}
    m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    return {"n": len(v), "mean": m, "t": m / (sd / math.sqrt(len(v))) if sd > 0 else None,
            "pct_pos": 100 * sum(x > 0 for x in v) / len(v)}


def summarize(res: dict) -> tuple[dict, list[str]]:
    pdates = res["per_date"]
    years = sorted({r["date"][:4] for r in pdates})
    summ: dict = {"overall": {}, "by_year": {}, "by_regime": {}, "spread": {}}
    L = ["=" * 78, "EGX POINT-IN-TIME FACTOR STUDY  (21-session horizon, vs cross-sectional median)",
         "=" * 78,
         f"price calendar: {res['calendar'][0]} .. {res['calendar'][1]}   "
         f"rebalance dates: {len(pdates)} (non-overlapping)",
         f"names with TradingView history: {res['n_tables']}   "
         f"excluded (units mismatch vs TV P/E): {len(res['excluded_units_suspect'])} "
         f"{res['excluded_units_suspect'][:12]}",
         f"fundamentals visible {tv_history.REPORTING_LAG_DAYS} days after quarter end. "
         "Residual biases: restated figures; survivors only.",
         "", "IC > 0: higher value -> better next-month relative return.  |t| > 2 ~ significant.",
         "", f"{'factor':18} {'dates':>5} {'meanIC':>7} {'t':>6} {'%IC>0':>6} {'Q5-Q1':>7}   by year (mean IC)"]
    for fac in ALL_FACTORS:
        s = _stats([r["ic"].get(fac) for r in pdates])
        sp = [r["spread"].get(fac) for r in pdates if r["spread"].get(fac) is not None]
        summ["overall"][fac] = s
        summ["spread"][fac] = float(np.median(sp)) * 100 if sp else None
        by = {}
        for y in years:
            ys = _stats([r["ic"].get(fac) for r in pdates if r["date"].startswith(y)])
            by[y] = ys.get("mean")
        summ["by_year"][fac] = by
        ups = _stats([r["ic"].get(fac) for r in pdates if r["market_6m"] > 0])
        dns = _stats([r["ic"].get(fac) for r in pdates if r["market_6m"] <= 0])
        summ["by_regime"][fac] = {"market_up": ups, "market_down": dns}
        if s.get("n", 0) < 3:
            L.append(f"{fac:18} {s.get('n', 0):>5}  (too few dates)")
            continue
        yr = " ".join(f"{y[2:]}:{by[y]:+.2f}" if by[y] is not None else f"{y[2:]}:  n/a"
                      for y in years)
        spread = summ["spread"][fac]
        L.append(f"{fac:18} {s['n']:>5} {s['mean']:+7.3f} {s['t'] or 0:+6.2f} {s['pct_pos']:6.0f} "
                 f"{(spread if spread is not None else float('nan')):+6.2f}%   {yr}")
    L += ["", "By market regime (median 6m return of the universe at the rebalance date):",
          f"{'factor':18} {'up: n':>6} {'IC':>7} {'t':>6}   {'down: n':>7} {'IC':>7} {'t':>6}"]
    for fac in ALL_FACTORS:
        u, d = summ["by_regime"][fac]["market_up"], summ["by_regime"][fac]["market_down"]
        def f(s):
            return (f"{s['n']:>6} {s['mean']:+7.3f} {s['t'] or 0:+6.2f}" if s.get("mean") is not None
                    else f"{s.get('n', 0):>6} {'n/a':>7} {'':>6}")
        L.append(f"{fac:18} {f(u)}   {f(d)}")
    L += ["", "Reading guide: the live model scores cheap (valuation) and high-quality names",
          "higher. If value_composite / quality_composite show a negative mean IC with",
          "|t| > 2 across most years, the live finding is structural, not one regime.",
          "If the sign flips by year or regime, the fix is regime-conditional weights."]
    return summ, L


def main() -> int:
    ap = argparse.ArgumentParser(description="Point-in-time EGX factor study.")
    ap.add_argument("--refresh-tv", action="store_true",
                    help="Refetch TradingView history before the study.")
    args = ap.parse_args()

    if args.refresh_tv or not tv_history.HISTORY_PATH.exists():
        print("Fetching TradingView quarterly history ...")
        payload = tv_history.fetch()
        tv_history.save(payload)
        print(f"  {payload['n_tickers']} names; errors: {payload['errors'] or 'none'}")
    hist = tv_history.load()
    prices, source = _load_prices()
    print(f"Price source: {source} ({len(prices)} series)")

    res = run(prices, hist)
    if not res["per_date"]:
        print("No rebalance dates with enough names — nothing to report.")
        return 1
    summ, lines = summarize(res)
    text = "\n".join(lines)
    print(text)
    _OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    _OUT_TXT.write_text(text + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps({"summary": summ, **res}, default=str), encoding="utf-8")
    print(f"\nWrote {_OUT_TXT} and {_OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
