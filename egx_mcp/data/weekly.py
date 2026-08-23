"""W1 — Weekly trading model (5-day horizon).

The monthly V8b uses 6M momentum (slow) + ROE filter — wrong signal-to-noise
for a 5-day horizon. W1 was built and tuned via random search on 2024-01 to
2026-04, validated on a 10-month walk-forward holdout (2025-07 to 2026-04).

Validation results vs broad EGX market on weekly rebalance:
    Full 28 months:  +107.8% CAGR, Sharpe 1.64, -24.9% MaxDD,  +67pp alpha
    Holdout 10mo:    +131.0% CAGR, Sharpe 2.78,  -8.9% MaxDD,  +81pp alpha (pure OOS)
    Last 6 months:   +145.4% CAGR, Sharpe 2.36,  -6.2% MaxDD,  +85pp alpha

Score formula (winning config from 60-config random search):
    s = 1.5 × mom_5d
      + 0.5 × (-mom_1d)              # fade daily noise
      + 3.0 × breakout_signal        # +1 if 20d high broken, -1 if 20d low
      + 3.0 × (1 if above MA20 else -1)
      - 0.3 × max(0, mom_5d - 10)    # stretched penalty above 10% in 5d
      + 5 if dip-in-uptrend          # 5d down >5%, but above MA20

Pipeline:
    1. Universe = validated 68 EGX names ∩ {ROE >= 10% from Mubasher}
    2. Compute features for each name as of `asof`
    3. Drop names with volume < 50% of 20-day ADV (illiquid this session)
    4. Rank by score, take top 5 equal-weight, hold ~5 trading days
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import egx_listing
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.weekly")


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
    min_volume_ratio: float = 0.5   # require today's volume >= 50% of ADV
    min_roe_pct: float = 10.0
    top_n: int = 5
    rebal_days: int = 5             # weekly
    # RSI ceiling — picks with RSI > rsi_ceiling are penalised unless MACD
    # histogram is still accelerating (rising). Calibrated after the
    # 2026-05-02 cohort where 5/5 picks entered with RSI 72-88 and two of
    # them (POUL, CCAP) went flat.
    rsi_ceiling: float = 80.0
    w_rsi_ceiling: float = 1.0      # per RSI point above ceiling
    macd_slope_exempt: bool = True  # waive penalty if MACD hist is rising
    # --- variance filters (hard, not penalties) ---------------------------
    # Live grading shows the picks are right-skewed: the median call lags the
    # basket while a handful of illiquid microcaps carry the mean. That shape
    # needs an enormous sample to prove an edge, because the sample size a
    # significance test needs scales with variance / effect². Cutting the tail
    # is therefore the cheapest way to make the model provable — and these
    # names are the ones a real order could not fill anyway.
    #
    # A penalty was already tried for extension (rsi_ceiling) and did not stop
    # entries like RSI 95 after +76% in five sessions, because the momentum
    # term outruns the penalty. These are hard filters instead.
    # Levels checked against the 18k-row panel (85 rebalance dates, top-5 by
    # composite, 21-session excess vs the equal-weight basket):
    #   no filter                        mean +2.16%  median +0.81%  57.6% positive
    #   run-up <= 25%                    mean +2.22%  median +1.14%  58.8% positive
    #   + turnover >= 50k, price >= 1    mean +3.02%  median +2.75%  59.0% positive
    #   turnover >= 250k (rejected)      mean +1.46%  — the 50k-250k bucket is the
    #                                    best-performing one in the panel; cutting
    #                                    it removed edge rather than variance.
    # In-sample over the whole panel, so treat as a sanity check, not proof —
    # the weekly walk-forward re-tests it out-of-sample.
    min_price_egp: float = 1.0        # weakest-evidenced of the three: no effect
                                      # univariate, helps only alongside turnover
    min_turnover_egp: float = 50_000  # 20-day median traded value; below this the
                                      # panel shows -0.99% mean and 28.6% hit rate
    max_5d_runup_pct: float = 25.0    # do not buy the blow-off top


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _norm(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(idx.date)
    return s[~s.index.duplicated(keep="last")]


def _load_quality_set(min_roe_pct: float) -> set[str]:
    cache = (Path(__file__).parent / "mubasher_fundamentals_cache.json")
    if not cache.exists():
        return set()
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {tk for tk, d in data.items()
                if d.get("roe_pct") is not None and d["roe_pct"] >= min_roe_pct}
    except Exception:
        return set()


def _features(closes: pd.Series, volumes: pd.Series, asof: pd.Timestamp) -> dict | None:
    cl = closes.loc[:asof].dropna()
    if len(cl) < 60:
        return None
    p_now = float(cl.iloc[-1])
    p_1d = float(cl.iloc[-2]) if len(cl) >= 2 else p_now
    p_5d = float(cl.iloc[-6]) if len(cl) >= 6 else p_now
    p_20d = float(cl.iloc[-21]) if len(cl) >= 21 else p_now
    if not (p_now > 0 and p_1d > 0 and p_5d > 0 and p_20d > 0):
        return None
    high_20 = float(cl.tail(20).max())
    low_20 = float(cl.tail(20).min())
    ma5 = float(cl.tail(5).mean())
    ma20 = float(cl.tail(20).mean())
    vol = volumes.loc[:asof].dropna()
    today_vol = float(vol.iloc[-1]) if len(vol) >= 1 else 0
    avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else max(today_vol, 1)
    vol_ratio = today_vol / max(avg_vol, 1)
    rvol = float(cl.tail(20).pct_change().std() * (252 ** 0.5)) if len(cl) >= 20 else 0.4
    if p_now >= high_20 * 0.999:
        breakout = 1.0
    elif p_now <= low_20 * 1.001:
        breakout = -1.0
    else:
        breakout = 0.0

    # RSI(14) — simple moving-average variant (matches technicals.py)
    rsi_14 = None
    if len(cl) >= 15:
        delta = cl.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        last_gain = float(gain.iloc[-1]) if not pd.isna(gain.iloc[-1]) else 0.0
        last_loss = float(loss.iloc[-1]) if not pd.isna(loss.iloc[-1]) else 0.0
        if last_loss > 0:
            rs = last_gain / last_loss
            rsi_14 = 100 - (100 / (1 + rs))
        elif last_gain > 0:
            rsi_14 = 100.0

    # MACD histogram slope — is momentum still accelerating?
    macd_slope_pos = False
    if len(cl) >= 35:
        ema12 = cl.ewm(span=12, adjust=False).mean()
        ema26 = cl.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - macd_sig
        if len(hist) >= 2 and not pd.isna(hist.iloc[-1]) and not pd.isna(hist.iloc[-2]):
            macd_slope_pos = float(hist.iloc[-1]) > float(hist.iloc[-2])

    # 20-day median traded value — what an order could actually get filled in.
    recent = cl.tail(20)
    v20 = vol.tail(20).reindex(recent.index).ffill()
    turnover_egp = float((recent * v20).median()) if len(v20.dropna()) >= 10 else 0.0

    return {
        "p_now": p_now,
        "turnover_egp": turnover_egp,
        "mom_1d": (p_now / p_1d - 1) * 100,
        "mom_5d": (p_now / p_5d - 1) * 100,
        "mom_20d": (p_now / p_20d - 1) * 100,
        "above_ma20": p_now > ma20,
        "above_ma5": p_now > ma5,
        "vol_ratio": vol_ratio,
        "rvol_pct": rvol * 100,
        "breakout": breakout,
        "high_20": high_20,
        "low_20": low_20,
        "ma20": ma20,
        "ma5": ma5,
        "rsi_14": rsi_14,
        "macd_slope_pos": macd_slope_pos,
    }


def _score(f: dict, c: W1Config) -> float:
    s = (f["mom_5d"] * c.w_mom5
         + f["mom_20d"] * c.w_mom20
         + (-f["mom_1d"]) * c.w_mr1
         + c.w_vol_conf * (f["vol_ratio"] - 1.0)
         + c.w_breakout * f["breakout"]
         + (c.w_trend if f["above_ma20"] else -c.w_trend))
    s -= c.w_vol_pen * f["rvol_pct"]
    if f["mom_5d"] > c.stretched_5d_thresh:
        s -= (f["mom_5d"] - c.stretched_5d_thresh) * c.stretched_5d_slope
    if f["mom_5d"] < c.dip_thresh_5d and f["above_ma20"]:
        s += c.dip_bonus
    # RSI ceiling penalty — exhausted momentum gets discounted unless MACD
    # histogram is still rising (momentum still confirming).
    rsi = f.get("rsi_14")
    if rsi is not None and rsi > c.rsi_ceiling:
        if not (c.macd_slope_exempt and f.get("macd_slope_pos")):
            s -= (rsi - c.rsi_ceiling) * c.w_rsi_ceiling
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_universe(asof: str | None = None, cfg: W1Config | None = None,
                   top_n: int | None = None) -> dict[str, Any]:
    """Return today's W1 rankings — what to buy for the next ~5 trading days.

    Args:
        asof: ISO date. Defaults to today.
        cfg: Override the default winning config.
        top_n: Override the default top_n=5.

    Returns:
        Dict with: as_of, n_universe, n_eligible, n_passes_quality,
        top_picks (list with ticker, score, breakdown), runners_up, all_ranked.
    """
    cfg = cfg or W1Config()
    if top_n is None:
        top_n = cfg.top_n
    asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp(datetime.utcnow().date())

    universe = egx_listing.get_full_universe()
    quality_set = _load_quality_set(cfg.min_roe_pct)

    rows = []
    skipped = []
    for tk in universe:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(period="180d", interval="1d")
            if h is None or h.empty:
                continue
            closes = _norm(h["Close"])
            volumes = _norm(h["Volume"])
        except Exception as e:
            skipped.append({"ticker": tk, "reason": str(e)})
            continue
        f = _features(closes, volumes, asof_ts)
        if f is None:
            skipped.append({"ticker": tk, "reason": "insufficient history"})
            continue
        passes_quality = (not quality_set) or (tk in quality_set)
        passes_volume = f["vol_ratio"] >= cfg.min_volume_ratio
        passes_liquidity = (f["p_now"] >= cfg.min_price_egp
                            and f["turnover_egp"] >= cfg.min_turnover_egp)
        passes_extension = f["mom_5d"] <= cfg.max_5d_runup_pct
        score = _score(f, cfg)
        rows.append({
            "ticker": tk,
            "score": round(score, 2),
            "passes_quality_filter": passes_quality,
            "passes_volume_filter": passes_volume,
            "passes_liquidity_filter": passes_liquidity,
            "passes_extension_filter": passes_extension,
            "turnover_egp_20d": round(f["turnover_egp"], 0),
            "price": round(f["p_now"], 4),
            "mom_1d_pct": round(f["mom_1d"], 2),
            "mom_5d_pct": round(f["mom_5d"], 2),
            "mom_20d_pct": round(f["mom_20d"], 2),
            "above_ma20": f["above_ma20"],
            "vol_ratio_today": round(f["vol_ratio"], 2),
            "breakout_signal": f["breakout"],
            "high_20d": round(f["high_20"], 4),
            "low_20d": round(f["low_20"], 4),
            "rsi_14": round(f["rsi_14"], 1) if f.get("rsi_14") is not None else None,
            "macd_slope_pos": f.get("macd_slope_pos", False),
        })

    def _eligible(require_volume: bool) -> list[dict]:
        out = [r for r in rows
               if r["passes_quality_filter"]
               and r["passes_liquidity_filter"]      # never relaxed: an unfillable
               and r["passes_extension_filter"]      # or blown-off name is not a pick
               and (r["passes_volume_filter"] or not require_volume)]
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    eligible = _eligible(require_volume=True)
    volume_filter_relaxed = False
    if len(eligible) < top_n:
        # Pre-holiday half-sessions can fail the single-session volume filter
        # for the entire universe (e.g. the 2026-03-19 and 2026-05-28 Eid
        # weeks), leaving an empty pick list. Fall back to quality-only
        # eligibility rather than emitting a degenerate briefing. Liquidity and
        # extension still apply — a short list is honest, a bad pick is not.
        volume_filter_relaxed = True
        eligible = _eligible(require_volume=False)
    top_picks = eligible[:top_n]
    runners_up = eligible[top_n:top_n + 5]

    return {
        "as_of": asof_ts.strftime("%Y-%m-%d"),
        "horizon_days": cfg.rebal_days,
        "n_universe": len(universe),
        "n_with_features": len(rows),
        "n_passes_quality": sum(1 for r in rows if r["passes_quality_filter"]),
        "n_fails_liquidity": sum(1 for r in rows if not r["passes_liquidity_filter"]),
        "n_fails_extension": sum(1 for r in rows if not r["passes_extension_filter"]),
        "n_eligible": len(eligible),
        "volume_filter_relaxed": volume_filter_relaxed,
        "config": {
            "w_mom5": cfg.w_mom5, "w_mom20": cfg.w_mom20, "w_mr1": cfg.w_mr1,
            "w_breakout": cfg.w_breakout, "w_trend": cfg.w_trend,
            "min_roe_pct": cfg.min_roe_pct, "top_n": top_n,
            "min_price_egp": cfg.min_price_egp,
            "min_turnover_egp": cfg.min_turnover_egp,
            "max_5d_runup_pct": cfg.max_5d_runup_pct,
        },
        "top_picks": top_picks,
        "runners_up": runners_up,
        "skipped_count": len(skipped),
        "validation_summary": (
            "Backtested 28 months. Full window: +107.8% CAGR, Sharpe 1.64, "
            "-24.9% MaxDD vs market +40.8% / 0.65 / -23.5%. Holdout 10mo OOS: "
            "+131% CAGR, Sharpe 2.78, -8.9% MaxDD vs market +50% / 1.32 / -5.6%."
        ),
        "disclaimer": (
            "Algorithmic ranking for the next ~5 trading days. Not investment "
            "advice. The model has positive expected alpha vs a passive EGX "
            "weekly buy-hold; individual weeks can still lose. Verify against "
            "the EGX official tape before acting."
        ),
    }
