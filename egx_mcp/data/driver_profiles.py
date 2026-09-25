"""What moves each EGX stock — a per-name "context card".

For every listing with enough history, weekly returns over the last three
years are regressed on four drivers:

    market  equal-weight EGX universe       usdegp  USD/EGP (+ = pound weaker)
    brent   Brent crude (BZ=F)              gold    gold (GC=F)

Weekly (Thursday) returns rather than daily: thin EGX names skip sessions,
and daily regressions on them are mostly noise. A driver is reported only
when its t-stat clears 2, with its impact sized as "a typical (1 sd) weekly
move in the driver shifts this stock by X%", so a big beta on a quiet
driver does not outrank a small beta on a volatile one.

Context only: profiles appear next to each pick in the briefing and do not
feed the score until a study shows yesterday's driver moves predict
tomorrow's relative returns. Built weekly by scripts/build_driver_profiles.py
into driver_profiles.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROFILE_PATH = Path(__file__).parent / "driver_profiles.json"
DRIVERS = ("market", "usdegp", "brent", "gold")
LABELS = {"market": "EGX market", "usdegp": "USD/EGP", "brent": "Brent oil", "gold": "gold"}
LOOKBACK_WEEKS = 156
MIN_WEEKS = 80
T_SIG = 2.0


def weekly_returns(series: pd.Series) -> pd.Series:
    s = series.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("W-THU").last().pct_change(fill_method=None)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Betas (incl. intercept first), their t-stats, and R^2."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()) if y.std() > 0 else 0.0
    return beta, beta / se, r2


def build(stock_closes: dict[str, pd.Series], driver_levels: dict[str, pd.Series],
          sectors: dict[str, str] | None = None) -> dict[str, Any]:
    """Profiles for every stock with >= MIN_WEEKS of aligned weekly returns."""
    stock_w = {tk: weekly_returns(s) for tk, s in stock_closes.items()}
    market = pd.DataFrame(stock_w).clip(-0.5, 0.5).mean(axis=1)
    fac = pd.DataFrame({"market": market,
                        **{k: weekly_returns(v) for k, v in driver_levels.items() if k != "market"}})
    fac = fac[list(DRIVERS)].dropna().iloc[-LOOKBACK_WEEKS:]
    sd = fac.std()
    out: dict[str, Any] = {}
    for tk, r in stock_w.items():
        df = pd.concat([r.rename("y"), fac], axis=1, join="inner").dropna()
        df = df[df["y"].abs() < 0.5]                      # drop bad prints
        if len(df) < MIN_WEEKS:
            continue
        beta, t, r2 = _ols(df["y"].to_numpy(), df[list(DRIVERS)].to_numpy())
        rec = {"n_weeks": len(df), "r2": round(r2, 3),
               "vol_ann_pct": round(float(df["y"].std() * np.sqrt(52) * 100), 1),
               "sector": (sectors or {}).get(tk),
               "beta": {d: round(float(b), 3) for d, b in zip(DRIVERS, beta[1:])},
               "t": {d: round(float(x), 2) for d, x in zip(DRIVERS, t[1:])},
               "impact_1sd_pct": {d: round(float(b * sd[d] * 100), 2)
                                  for d, b in zip(DRIVERS, beta[1:])}}
        rec["drivers"] = sorted(
            ({"driver": d, "sign": "+" if rec["beta"][d] > 0 else "-",
              "impact_1sd_pct": rec["impact_1sd_pct"][d], "t": rec["t"][d]}
             for d in DRIVERS if abs(rec["t"][d]) >= T_SIG),
            key=lambda x: -abs(x["impact_1sd_pct"]))
        out[tk] = rec
    return {"lookback_weeks": LOOKBACK_WEEKS, "min_weeks": MIN_WEEKS,
            "driver_weekly_sd_pct": {d: round(float(sd[d] * 100), 2) for d in DRIVERS},
            "as_of": str(fac.index[-1].date()) if len(fac) else None,
            "profiles": out}


def load(path: Path = PROFILE_PATH) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def describe(ticker: str, macro_today: dict[str, float | None] | None = None,
             data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Context card for one name: what drives it, and today's tailwind/headwind
    from the drivers' latest moves (same sign convention as the betas)."""
    data = data if data is not None else load()
    p = (data.get("profiles") or {}).get(ticker)
    if not p:
        return {"ticker": ticker, "available": False,
                "note": "no driver profile (too little weekly history)"}
    lines = []
    for d in p["drivers"]:
        lines.append(f"moves {'with' if d['sign'] == '+' else 'against'} {LABELS[d['driver']]}: "
                     f"a typical weekly move shifts it {abs(d['impact_1sd_pct']):.1f}% (t={d['t']:+.1f})")
    if not p["drivers"]:
        lines.append("no significant link to market, USD/EGP, oil or gold: stock-specific news drives it")
    idio = 1 - p["r2"]
    lines.append(f"{idio:.0%} of its weekly moves are stock-specific (R² {p['r2']:.2f})")
    today = []
    for d in p["drivers"]:
        mv = (macro_today or {}).get(d["driver"])
        if d["driver"] == "market" or mv is None or abs(mv) < 0.5:
            continue
        effect = (1 if d["sign"] == "+" else -1) * (1 if mv > 0 else -1)
        today.append(f"{LABELS[d['driver']]} {mv:+.1f}% today: "
                     f"{'tailwind' if effect > 0 else 'headwind'}")
    return {"ticker": ticker, "available": True, "sector": p.get("sector"),
            "vol_ann_pct": p["vol_ann_pct"], "drivers": p["drivers"],
            "lines": lines, "today": today}
