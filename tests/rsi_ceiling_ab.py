"""W1 A/B backtest — OLD config vs NEW (RSI ceiling penalty + MACD-slope exemption).

Validates that the RSI ceiling change from the 2026-05 cohort review does
not regress the documented alpha numbers. Mirrors the methodology of
weekly_model_search.py (same buy-hold benchmark, same rebalance cadence,
same quality filter).

Run:
    python -m tests.rsi_ceiling_ab

Requires working yfinance EGX data access. If yfinance returns empty for
.CA tickers from your network, run this from a different one.
"""
from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from egx_mcp.data import egx_listing, risk_free
from egx_mcp.data.universe import resolve_ticker

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

START = "2024-01-01"
END = "2026-04-30"
HOLDOUT_START = "2025-07-01"
LAST_6MO_START = "2025-10-30"


@dataclass
class W1Config:
    w_mom5: float = 1.5
    w_mom20: float = 0.0
    w_mr1: float = 0.5
    w_vol_conf: float = 0.0
    w_breakout: float = 3.0
    w_trend: float = 3.0
    w_vol_pen: float = 0.0
    stretched_5d_thresh: float = 10.0
    stretched_5d_slope: float = 0.3
    dip_thresh_5d: float = -5.0
    dip_bonus: float = 5.0
    min_volume_ratio: float = 0.5
    min_roe_pct: float = 10.0
    top_n: int = 5
    rebal_days: int = 5
    # NEW knobs — old config sets w_rsi_ceiling=0.0 to disable
    rsi_ceiling: float = 80.0
    w_rsi_ceiling: float = 1.0
    macd_slope_exempt: bool = True


def _norm(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _features(cl: pd.Series, vp: pd.Series, asof: pd.Timestamp):
    c = cl.loc[:asof].dropna()
    if len(c) < 60:
        return None
    p_now = float(c.iloc[-1])
    p_1d = float(c.iloc[-2])
    p_5d = float(c.iloc[-6])
    p_20d = float(c.iloc[-21])
    if min(p_now, p_1d, p_5d, p_20d) <= 0:
        return None
    high_20 = float(c.tail(20).max())
    low_20 = float(c.tail(20).min())
    ma20 = float(c.tail(20).mean())
    v = vp.loc[:asof].dropna()
    today_vol = float(v.iloc[-1]) if len(v) else 0
    avg_vol = float(v.tail(20).mean()) if len(v) >= 20 else max(today_vol, 1)
    vol_ratio = today_vol / max(avg_vol, 1)

    if p_now >= high_20 * 0.999:
        brk = 1.0
    elif p_now <= low_20 * 1.001:
        brk = -1.0
    else:
        brk = 0.0

    # RSI(14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    g, l = float(gain.iloc[-1] or 0), float(loss.iloc[-1] or 0)
    rsi = 100.0 if l == 0 and g > 0 else (100 - 100 / (1 + g / l)) if l > 0 else None

    # MACD slope
    macd_slope_pos = False
    if len(c) >= 35:
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        sig = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - sig
        if not pd.isna(hist.iloc[-1]) and not pd.isna(hist.iloc[-2]):
            macd_slope_pos = float(hist.iloc[-1]) > float(hist.iloc[-2])

    return {
        "p_now": p_now,
        "mom_1d": (p_now / p_1d - 1) * 100,
        "mom_5d": (p_now / p_5d - 1) * 100,
        "mom_20d": (p_now / p_20d - 1) * 100,
        "above_ma20": p_now > ma20,
        "vol_ratio": vol_ratio,
        "breakout": brk,
        "rsi_14": rsi,
        "macd_slope_pos": macd_slope_pos,
    }


def _score(f, c: W1Config):
    s = (f["mom_5d"] * c.w_mom5
         + f["mom_20d"] * c.w_mom20
         + (-f["mom_1d"]) * c.w_mr1
         + c.w_breakout * f["breakout"]
         + (c.w_trend if f["above_ma20"] else -c.w_trend))
    if f["mom_5d"] > c.stretched_5d_thresh:
        s -= (f["mom_5d"] - c.stretched_5d_thresh) * c.stretched_5d_slope
    if f["mom_5d"] < c.dip_thresh_5d and f["above_ma20"]:
        s += c.dip_bonus
    # NEW: RSI ceiling penalty (set w_rsi_ceiling=0 to disable for OLD config)
    rsi = f.get("rsi_14")
    if rsi is not None and rsi > c.rsi_ceiling and c.w_rsi_ceiling > 0:
        if not (c.macd_slope_exempt and f.get("macd_slope_pos")):
            s -= (rsi - c.rsi_ceiling) * c.w_rsi_ceiling
    return s


def _load_quality(min_roe_pct: float):
    cache = json.loads(
        (Path(__file__).parent.parent / "egx_mcp" / "data" / "mubasher_fundamentals_cache.json"
         ).read_text(encoding="utf-8")
    )
    return {tk for tk, d in cache.items()
            if d.get("roe_pct") is not None and d["roe_pct"] >= min_roe_pct}


def run(cp, vp, cfg: W1Config, start_date, quality):
    panel = cp.loc[start_date:]
    rb = panel.index[::cfg.rebal_days]
    if len(rb) < 3:
        return {}
    eq, rets = [1.0], []
    rf = risk_free.get_rate()["rate_pct"]
    rf_p = (1 + rf / 100) ** (cfg.rebal_days / 252) - 1
    eligible = [t for t in cp.columns if t in quality]
    for i in range(len(rb) - 1):
        d0, d1 = rb[i], rb[i + 1]
        scores = {}
        for tk in eligible:
            if tk not in cp.columns or tk not in vp.columns:
                continue
            f = _features(cp[tk], vp[tk], d0)
            if f is None or f["vol_ratio"] < cfg.min_volume_ratio:
                continue
            scores[tk] = _score(f, cfg)
        if not scores:
            rets.append(0); eq.append(eq[-1]); continue
        picks = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:cfg.top_n]]
        sub = cp.loc[d0:d1, picks].dropna(how="all")
        if len(sub) < 2:
            rets.append(0); eq.append(eq[-1]); continue
        r = (sub.iloc[-1] / sub.iloc[0] - 1).dropna()
        per = float(r.mean()) if not r.empty else 0
        rets.append(per); eq.append(eq[-1] * (1 + per))

    eq, rets = np.array(eq), np.array(rets)
    if len(rets) < 2:
        return {}
    years = len(rets) * cfg.rebal_days / 252
    cagr = ((eq[-1]) ** (1 / max(years, 0.01)) - 1) * 100
    vol = rets.std() * ((252 / cfg.rebal_days) ** 0.5) * 100
    excess = rets - rf_p
    sharpe = excess.mean() / excess.std() * ((252 / cfg.rebal_days) ** 0.5) if excess.std() > 0 else 0
    rmax = np.maximum.accumulate(eq); dd = (eq / rmax - 1)
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_dd": dd.min() * 100, "hit": (rets > 0).mean() * 100,
            "n": len(rets)}


def buy_hold(cp, start_date, rebal_days=5):
    p = cp.loc[start_date:].dropna(how="all")
    rb = p.index[::rebal_days]
    if len(rb) < 3:
        return {}
    rets = []
    for i in range(len(rb) - 1):
        sub = p.loc[rb[i]:rb[i + 1]].dropna(how="all")
        if len(sub) < 2:
            rets.append(0); continue
        r = (sub.iloc[-1] / sub.iloc[0] - 1).dropna()
        rets.append(float(r.mean()) if not r.empty else 0)
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    eq, rets = np.array(eq), np.array(rets)
    years = len(rets) * rebal_days / 252
    cagr = (eq[-1] ** (1 / max(years, 0.01)) - 1) * 100
    rf = risk_free.get_rate()["rate_pct"]
    rf_p = (1 + rf / 100) ** (rebal_days / 252) - 1
    excess = rets - rf_p
    sharpe = excess.mean() / excess.std() * ((252 / rebal_days) ** 0.5) if excess.std() > 0 else 0
    rmax = np.maximum.accumulate(eq); dd = (eq / rmax - 1)
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": dd.min() * 100,
            "hit": (rets > 0).mean() * 100, "n": len(rets)}


def main():
    print(f"Loading prices/volumes ({START} → {END})...")
    universe = egx_listing.get_full_universe()
    if not universe:
        print("ERROR: empty universe — egx_listing.get_full_universe() returned 0 names. "
              "yfinance EGX data is unreachable from this network. Aborting.")
        return 1
    cp_d, vp_d = {}, {}
    for tk in universe:
        _, yh, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yh).history(start="2023-01-01", end=END, interval="1d")
            if h is None or h.empty:
                continue
            cp_d[tk] = _norm(h["Close"])
            vp_d[tk] = _norm(h["Volume"])
        except Exception:
            continue
    cp = pd.DataFrame(cp_d).sort_index()
    vp = pd.DataFrame(vp_d).sort_index()
    if cp.empty:
        print("ERROR: no price data returned from yfinance. Aborting.")
        return 1
    quality = _load_quality(10.0)
    print(f"  {len(cp.columns)} names with data, {len(quality)} pass ROE>=10%\n")

    OLD = W1Config(w_rsi_ceiling=0.0)  # disable RSI penalty
    NEW = W1Config()                    # default: w_rsi_ceiling=1.0

    windows = [
        ("Full (2024-01 → 2026-04)", START),
        ("Holdout (2025-07 → 2026-04)", HOLDOUT_START),
        ("Last 6mo (2025-10 → 2026-04)", LAST_6MO_START),
    ]
    print(f"{'Window':<32}{'Config':<6}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Hit%':>8}{'Alpha':>10}")
    print("-" * 83)
    for label, start in windows:
        mkt = buy_hold(cp, start_date=start)
        for name, cfg in [("OLD", OLD), ("NEW", NEW)]:
            r = run(cp, vp, cfg, start_date=start, quality=quality)
            if not r or not mkt:
                continue
            alpha = r["cagr"] - mkt["cagr"]
            print(f"{label:<32}{name:<6}{r['cagr']:>+8.1f}%{r['sharpe']:>9.2f}"
                  f"{r['max_dd']:>+8.1f}%{r['hit']:>+7.1f}%{alpha:>+8.1f}pp")
        print(f"{'  market buy-hold':<32}{'MKT':<6}{mkt['cagr']:>+8.1f}%{mkt['sharpe']:>9.2f}"
              f"{mkt['max_dd']:>+8.1f}%{mkt['hit']:>+7.1f}%{'':>10}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
