"""Build the learning loop's training sample from market data.

The old loop learned only from verdicts it had already emitted and graded —
~20 usable rows at the 21-session claim horizon after two months of running.
This rebuilds the same sample from price history instead: at every weekly
rebalance date, every universe name is scored and labelled, so the learner
sees thousands of observations on day one.

Point-in-time contract
----------------------
Each name is scored with THE SAME four sub-scorers production uses
(`scoring._score_valuation/_quality/_momentum/_risk`), fed inputs rebuilt from
data available at the as-of date. Learning against a reimplementation would
tune weights for a model that isn't the one deployed.

  momentum, risk  — fully point-in-time (price history <= t only)
  valuation, quality — NOT point-in-time. EPS, book value, ROE, margin and
      D/E come from a single current snapshot (mubasher_fundamentals_cache),
      so these two sub-scores carry look-ahead bias by construction. The
      price half of the leak is removed (P/E and P/B are recomputed at the
      as-of price, exactly as fundamentals.py does live); the earnings half
      is not. Rows are stamped `fundamentals_asof` and the learner reports
      the affected levers separately. Deliberate, documented choice.

Label: forward 21-session return, entered at the next session's close, minus
the equal-weight basket of the same scored universe over the same window —
the synthetic-basket benchmark grade_briefings uses.

    python -m scripts.build_panel                 # use cached prices
    python -m scripts.build_panel --refresh       # refetch history first
    python -m scripts.build_panel --lookback 900  # deeper history
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import pandas as pd

# In-place reconfigure, not a fresh TextIOWrapper — see export_fundamentals_csv.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_fundamentals_csv import ALIASES, _tv_fill  # noqa: E402
from egx_mcp.data import investing, model_params, price_sanity, regime, scoring  # noqa: E402

ROOT = Path(__file__).parent.parent
_PRICES = ROOT / "logs" / "panel_prices.json"
_PANEL = ROOT / "logs" / "panel.jsonl"
_PRICE_CACHE = ROOT / "egx_mcp" / "data" / "price_cache.json"
_FUND_CACHE = ROOT / "egx_mcp" / "data" / "mubasher_fundamentals_cache.json"

_HORIZONS = (21, 5)        # 21 = the model's claim horizon; 5 = context only
_MOM_WINDOW = 126          # production scores on history_period="6mo"
_MIN_BARS = 210            # need SMA200 + a little runway before scoring a name
_REBALANCE_DOW = 3         # Thursday — matches the weekly walk-forward cutoffs


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def _yahoo_history(tk: str, lookback_days: int) -> list[dict] | None:
    """Yahoo fallback for names investing.com refuses.

    investing.com 403s every request from the GitHub runner IPs — on
    2026-08-23 that turned a working 18k-row panel into `Fetched 0 series
    (250 failed)` and the whole market-data learning arm produced nothing.
    A second source means one blocked vendor degrades the panel instead of
    zeroing it.
    """
    import yfinance as yf

    from egx_mcp.data.agentic_backtest import _benchmark_series
    from egx_mcp.data.universe import resolve_ticker

    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        if tk == "EGX30":
            s = _benchmark_series(start, pd.Timestamp.today().strftime("%Y-%m-%d"))
            if s is None or s.empty:
                return None
            return [{"date": d.strftime("%Y-%m-%d"), "close": float(v), "volume": None}
                    for d, v in s.items() if v == v and v > 0]
        _, yahoo, _ = resolve_ticker(tk)
        h = yf.Ticker(yahoo).history(start=start, interval="1d", auto_adjust=True)
        if h is None or h.empty:
            return None
        return [{"date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                 "close": float(row["Close"]),
                 "volume": (float(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"]
                            else None)}
                for d, row in h.iterrows() if float(row["Close"]) > 0]
    except Exception:  # noqa: BLE001 — a fallback that raises is no fallback
        return None


def _refresh_prices(tickers: list[str], lookback_days: int, throttle_s: float = 0.4) -> dict:
    """Fetch daily history for the universe + EGX30: investing.com, then Yahoo.

    The result is MERGED into whatever the cache already holds. An outage at
    one vendor must never delete history that was already fetched — the old
    behaviour overwrote the file unconditionally, so a bad fetch day wiped the
    learner's entire training substrate.
    """
    prev: dict[str, list[dict]] = {}
    if _PRICES.exists():
        try:
            prev = json.loads(_PRICES.read_text(encoding="utf-8")).get("prices", {}) or {}
        except Exception:  # noqa: BLE001
            prev = {}

    out: dict[str, list[dict]] = {}
    failed: list[str] = []
    from_yahoo: list[str] = []
    for i, tk in enumerate(tickers + ["EGX30"], 1):
        try:
            rows = investing.fetch_history(tk, lookback_days=lookback_days)
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: {type(e).__name__} {e}")
            rows = None
        if not rows:
            rows = _yahoo_history(tk, lookback_days)
            if rows:
                from_yahoo.append(tk)
        if rows:
            out[tk] = [{"date": r["date"], "close": r["close"], "volume": r.get("volume")}
                       for r in rows]
        else:
            failed.append(tk)
        if i % 20 == 0:
            print(f"  fetched {i}/{len(tickers) + 1} ...")
        time.sleep(throttle_s)

    merged = {**prev, **out}          # fresh series win; stale ones survive an outage
    kept_stale = sorted(set(prev) - set(out))
    payload = {"fetched_at": pd.Timestamp.now("UTC").isoformat(),
               "lookback_days": lookback_days,
               "n_tickers": len(merged), "failed": failed,
               "from_yahoo": from_yahoo, "kept_from_previous": kept_stale,
               "prices": merged}
    _PRICES.parent.mkdir(parents=True, exist_ok=True)
    _PRICES.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Fetched {len(out)} series this run "
          f"({len(out) - len(from_yahoo)} investing.com, {len(from_yahoo)} Yahoo fallback, "
          f"{len(failed)} failed)")
    if kept_stale:
        print(f"  kept {len(kept_stale)} previously-cached series that no source served today")
    print(f"Panel cache now holds {len(merged)} series -> {_PRICES}")
    return payload


def _load_prices() -> tuple[dict[str, list[dict]], str]:
    """Panel prices if present, else fall back to the committed price_cache."""
    if _PRICES.exists():
        d = json.loads(_PRICES.read_text(encoding="utf-8"))
        return d["prices"], f"panel_prices.json (fetched {d.get('fetched_at', '?')[:10]})"
    if _PRICE_CACHE.exists():
        d = json.loads(_PRICE_CACHE.read_text(encoding="utf-8"))
        prices = {tk: [{"date": r["date"], "close": r["close"], "volume": r.get("volume")}
                       for r in rows] for tk, rows in d["prices"].items()}
        egx = (d.get("factors") or {}).get("egx30")
        if egx:
            prices["EGX30"] = [{"date": r["date"], "close": r["close"]} for r in egx]
        return prices, f"price_cache.json (refreshed {d.get('refreshed_at', '?')[:10]})"
    raise SystemExit("No price data. Run with --refresh first.")


# ---------------------------------------------------------------------------
# Point-in-time feature reconstruction
# ---------------------------------------------------------------------------

def _closes(rows: list[dict]) -> pd.Series:
    s = pd.Series({r["date"]: r["close"] for r in rows if r.get("close")}, dtype="float64")
    s.index = pd.to_datetime(s.index)
    # Non-positive ticks are vendor errors, never prices (see price_sanity).
    return price_sanity.clean_series(s.sort_index())


def _volumes(rows: list[dict]) -> pd.Series:
    s = pd.Series({r["date"]: r.get("volume") for r in rows}, dtype="float64")
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _candidate_factors(closes: pd.Series, volumes: pd.Series) -> dict:
    """Point-in-time research features the production score does NOT use.

    Recorded so learn_panel can rank them by out-of-sample IC before anyone
    proposes wiring one in. The live composite's momentum leg measures ~0.00
    IC, so the useful question is which alternative definition, if any, carries
    signal on this market — reversal, longer-horizon momentum, low-vol, or
    liquidity. Nothing here changes a verdict; these are diagnostics.
    """
    out: dict[str, float | None] = {}

    def _ret(lookback: int, skip: int = 0) -> float | None:
        need = lookback + skip + 1
        if len(closes) < need:
            return None
        end = float(closes.iloc[-1 - skip])
        start = float(closes.iloc[-need])
        return round((end / start - 1) * 100, 4) if start > 0 else None

    out["cand_rev_5d"] = (-_ret(5)) if _ret(5) is not None else None
    out["cand_mom_63d"] = _ret(63)
    out["cand_mom_12_1"] = _ret(231, skip=21)      # 12 months, last month skipped
    rets = closes.tail(63).pct_change().dropna()
    out["cand_lowvol_63d"] = (round(-float(rets.std() * (252 ** 0.5) * 100), 4)
                              if len(rets) > 5 else None)
    high_252 = float(closes.tail(252).max())
    out["cand_dist_52w_high"] = (round(float(closes.iloc[-1]) / high_252 * 100, 4)
                                 if high_252 > 0 else None)
    v = volumes.loc[:closes.index[-1]].tail(20)
    if len(v) >= 10:
        turnover = float((v * closes.reindex(v.index).ffill()).median())
        out["cand_turnover_egp"] = round(turnover, 2) if turnover > 0 else None
    else:
        out["cand_turnover_egp"] = None
    return out


def _history_summary(closes: pd.Series) -> dict:
    """Mirror market.get_history()'s summary over the trailing 6M window."""
    win = closes.tail(_MOM_WINDOW)
    if len(win) < 20:
        return {}
    ret = (float(win.iloc[-1]) / float(win.iloc[0]) - 1) * 100
    dd = float((win / win.cummax() - 1).min() * 100)
    rets = win.pct_change().dropna()
    vol = float(rets.std() * (252 ** 0.5) * 100) if len(rets) > 1 else None
    return {"return_pct": round(ret, 2),
            "max_drawdown_pct": round(dd, 2),
            "annualized_volatility_pct": round(vol, 2) if vol else None}


def _indicators(closes: pd.Series) -> dict:
    """Mirror technicals.compute()'s formulas exactly (ewm adjust=False)."""
    if len(closes) < 60:
        return {}
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()

    def _last(s):
        if s is None or len(s) == 0:
            return None
        v = s.iloc[-1]
        return None if pd.isna(v) else round(float(v), 4)

    return {"rsi_14": _last(rsi), "macd": _last(macd_line), "macd_signal": _last(macd_sig),
            "sma_50": _last(closes.rolling(50).mean()),
            "sma_200": _last(closes.rolling(200).mean()) if len(closes) >= 200 else None}


def _regime_at(egx: pd.Series) -> tuple[str, dict]:
    """Reconstruct regime.classify()'s decision tree from EGX30 history <= t.

    Fully point-in-time: the live classifier is itself only a function of the
    index price series, so this reproduces it rather than approximating it."""
    if len(egx) < 60:
        return "UNKNOWN", {}
    last = float(egx.iloc[-1])
    last60 = egx.tail(60)
    ret60 = (float(last60.iloc[-1]) / float(last60.iloc[0]) - 1) * 100
    rets = last60.pct_change().dropna()
    vol = float(rets.std() * (252 ** 0.5) * 100) if len(rets) > 1 else 0.0
    tail252 = egx.tail(252)
    sma200 = float(tail252.tail(200).mean()) if len(tail252) >= 200 else float(tail252.mean())
    dd1y = (last / float(tail252.max()) - 1) * 100

    if vol >= 40:
        r = "HIGH_VOL"
    elif ret60 < -5 or dd1y < -15:
        r = "BEAR"
    elif ret60 > 5 and vol < 30 and last > sma200:
        r = "BULL"
    else:
        r = "SIDEWAYS"
    return r, regime._REGIME_WEIGHTS[r]


def _load_fundamentals() -> tuple[dict, dict, str]:
    """(current merged snapshot, point-in-time history, snapshot date).

    Current = the same merge production uses: Mubasher cache + TradingView fill
    for the ratios Mubasher dropped + the EGX-code aliases. History comes from
    scripts.snapshot_fundamentals and is empty until that has run for a while."""
    from scripts.snapshot_fundamentals import load_history

    cache = json.loads(_FUND_CACHE.read_text(encoding="utf-8"))
    _tv_fill(cache)
    for canonical, code in ALIASES.items():
        if canonical not in cache and code in cache:
            cache[canonical] = {**cache[code], "ticker": canonical}
    fetched = [v.get("fetched_at") for v in cache.values() if v.get("fetched_at")]
    asof = (pd.Timestamp.fromtimestamp(max(fetched), "UTC").strftime("%Y-%m-%d")
            if fetched else "unknown")
    return cache, load_history(), asof


def _fundamentals_at(tk: str, on: str, price: float,
                     current: dict, history: dict) -> tuple[dict, str, bool]:
    """Fundamentals as they stood on `on`, plus whether that read is honest.

    Prefers the most recent recorded snapshot at or before `on`. Falls back to
    the current snapshot only when history doesn't reach back that far — and
    says so, so the row is flagged contaminated rather than quietly trusted.

    P/E and P/B are always recomputed at the as-of price (fundamentals.py does
    the same live, because vendor P/E is stale). On the fallback path that
    strips the price half of the look-ahead; EPS/BVPS/ROE/margin/D-E remain
    as-of the snapshot date, which is the half that stays contaminated."""
    from scripts.snapshot_fundamentals import as_of

    pit = as_of(history, tk, on)
    if pit is not None:
        f, asof, contaminated = dict(pit), pit["snapshot_date"], False
    else:
        f, asof, contaminated = dict(current.get(tk) or {}), "current", True

    eps = f.get("trailing_eps")
    bvps = f.get("book_value_per_share")
    f["pe_ratio"] = round(price / eps, 2) if eps and eps > 0 else None
    f["pb_ratio"] = round(price / bvps, 2) if bvps and bvps > 0 else None
    return f, asof, contaminated


def _sector_medians(rows: list[dict]) -> dict[str, dict]:
    """Cross-sectional sector medians at t, from the as-of-price ratios."""
    by_sector: dict[str, list[dict]] = {}
    for r in rows:
        by_sector.setdefault(r["sector"], []).append(r)
    out = {}
    for sec, members in by_sector.items():
        def _med(key):
            vals = [m["f"][key] for m in members
                    if isinstance(m["f"].get(key), (int, float)) and m["f"][key] > 0]
            return round(st.median(vals), 2) if len(vals) >= 3 else None
        out[sec] = {"median_pe": _med("pe_ratio"), "median_pb": _med("pb_ratio"),
                    "median_roe_pct": _med("roe_pct"), "n": len(members)}
    return out


# ---------------------------------------------------------------------------
# Panel build
# ---------------------------------------------------------------------------

def _sector_of(tk: str) -> str:
    from egx_mcp.data.behavior import _SECTOR_MAP
    from egx_mcp.data.universe import EGX_UNIVERSE
    return _SECTOR_MAP.get(tk) or (EGX_UNIVERSE.get(tk, {}) or {}).get("sector") or "Unknown"


def build(prices: dict[str, list[dict]], source: str) -> list[dict]:
    fund_snap, fund_hist, fund_asof = _load_fundamentals()

    egx_all = _closes(prices["EGX30"]) if "EGX30" in prices else pd.Series(dtype="float64")
    series = {tk: _closes(rows) for tk, rows in prices.items() if tk != "EGX30"}
    vols = {tk: _volumes(rows) for tk, rows in prices.items() if tk != "EGX30"}
    series = {tk: s for tk, s in series.items() if len(s) >= _MIN_BARS and tk in fund_snap}
    print(f"Universe: {len(series)} names with >= {_MIN_BARS} bars and a fundamentals row")
    hist_start = min((r[0]["snapshot_date"] for r in fund_hist.values() if r), default=None)
    if hist_start:
        print(f"Point-in-time fundamentals from {hist_start} "
              f"({len(fund_hist)} tickers); earlier rows fall back to the "
              f"{fund_asof} snapshot and are flagged contaminated.")
    else:
        print(f"No fundamentals history yet — all rows fall back to the {fund_asof} "
              f"snapshot (valuation/quality look-ahead). Run "
              f"scripts.snapshot_fundamentals daily to start building it.")

    all_dates = sorted({d for s in series.values() for d in s.index})
    rebal = [d for d in all_dates if d.dayofweek == _REBALANCE_DOW]
    base_w = model_params.DEFAULTS["score_weights"]
    max_h = max(_HORIZONS)
    panel: list[dict] = []

    for t in rebal:
        # --- score every name with data at t ---
        staged = []
        day = t.strftime("%Y-%m-%d")
        for tk, s in series.items():
            hist = s.loc[:t]
            if len(hist) < _MIN_BARS or hist.index[-1] != t:
                continue                      # no bar on t — don't fabricate one
            price = float(hist.iloc[-1])
            f, f_asof, dirty = _fundamentals_at(tk, day, price, fund_snap, fund_hist)
            staged.append({"ticker": tk, "sector": _sector_of(tk), "price": price,
                           "hist": hist, "f": f, "f_asof": f_asof, "dirty": dirty})
        if len(staged) < 10:
            continue
        med = _sector_medians(staged)
        reg, reg_w = _regime_at(egx_all.loc[:t])

        # --- label: forward excess vs the equal-weight basket of the same set ---
        fwd: dict[str, dict[int, float]] = {}
        for m in staged:
            s = series[m["ticker"]]
            i = s.index.get_loc(t)
            fwd[m["ticker"]] = {}
            for h in _HORIZONS:
                # enter at the next session's close, hold h sessions
                if i + 1 + h < len(s):
                    entry, exit_ = float(s.iloc[i + 1]), float(s.iloc[i + 1 + h])
                    # A session outside the EGX daily band inside the holding
                    # window is a split or a bad tick, not a return — labelling
                    # on it teaches the learner a corporate action.
                    if price_sanity.find_break(s.iloc[i + 1: i + 2 + h]) is not None:
                        continue
                    if entry > 0:
                        fwd[m["ticker"]][h] = (exit_ / entry - 1) * 100
        basket = {h: st.mean([v[h] for v in fwd.values() if h in v])
                  for h in _HORIZONS
                  if any(h in v for v in fwd.values())}
        if max_h not in basket:
            continue                          # window not closed yet — leave unlabelled

        for m in staged:
            sub_v = scoring._score_valuation(m["f"], med.get(m["sector"], {}))["score"]
            sub_q = scoring._score_quality(m["f"], med.get(m["sector"], {}))["score"]
            sub_m = scoring._score_momentum({"summary": _history_summary(m["hist"])},
                                            {"indicators": _indicators(m["hist"])})["score"]
            sub_r = scoring._score_risk({"summary": _history_summary(m["hist"])})["score"]
            subs = {"valuation": sub_v, "quality": sub_q, "momentum": sub_m, "risk": sub_r}

            # composite as production forms it: base weights x regime bias, renormalized.
            # The macro +-5 sector tilt is NOT reconstructible point-in-time (it reads
            # current CBE policy), so it is held at 0 and excluded from threshold learning.
            w = {k: base_w[k] * (reg_w.get(k, 1.0) if reg_w else 1.0) for k in base_w}
            tot = sum(w.values())
            w = {k: v / tot for k, v in w.items()}
            composite = round(sum(subs[k] * w[k] for k in subs), 1)

            row = {"date": t.strftime("%Y-%m-%d"), "ticker": m["ticker"],
                   "sector": m["sector"], "price": round(m["price"], 4),
                   "regime": reg, "composite": composite,
                   "sub_valuation": sub_v, "sub_quality": sub_q,
                   "sub_momentum": sub_m, "sub_risk": sub_r,
                   "fundamentals_asof": m["f_asof"],
                   "pit_clean": ["momentum", "risk"] if m["dirty"] else list(subs),
                   "pit_contaminated": ["valuation", "quality"] if m["dirty"] else [],
                   **_candidate_factors(m["hist"],
                                        vols.get(m["ticker"], pd.Series(dtype="float64")))}
            for h in _HORIZONS:
                r = fwd[m["ticker"]].get(h)
                row[f"fwd_{h}d_pct"] = round(r, 4) if r is not None else None
                row[f"bench_{h}d_pct"] = round(basket[h], 4) if h in basket else None
                row[f"excess_{h}d_pct"] = (round(r - basket[h], 4)
                                           if r is not None and h in basket else None)
            panel.append(row)

    if not panel:
        print("Panel: empty")
        return panel
    clean = sum(1 for r in panel if not r["pit_contaminated"])
    print(f"Panel: {len(panel)} rows across {len({r['date'] for r in panel})} rebalance dates "
          f"({panel[0]['date']} -> {panel[-1]['date']})")
    print(f"  fully point-in-time rows: {clean}/{len(panel)} ({100 * clean / len(panel):.1f}%) "
          f"— the rest carry valuation/quality look-ahead")
    return panel


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the market-data training panel.")
    ap.add_argument("--refresh", action="store_true", help="Refetch price history first.")
    ap.add_argument("--lookback", type=int, default=900, help="Days of history to fetch.")
    args = ap.parse_args()

    if args.refresh:
        from egx_mcp.data.egx_listing import get_full_universe
        tickers = get_full_universe()
        print(f"Refreshing {len(tickers)} names, {args.lookback}d lookback ...")
        _refresh_prices(tickers, args.lookback)

    prices, source = _load_prices()
    print(f"Price source: {source}")
    panel = build(prices, source)
    if not panel:
        print("No labelled rows produced.")
        return 1
    _PANEL.parent.mkdir(parents=True, exist_ok=True)
    _PANEL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in panel) + "\n",
                      encoding="utf-8")
    print(f"Wrote {_PANEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
