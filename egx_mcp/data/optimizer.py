"""Portfolio optimizer — mean-variance, min-vol, risk-parity, equal-weight.

Three solvers, all closed-form or simple iterative — no scipy needed:

    min_variance      w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
    tangency          w ∝ Σ⁻¹ μ_excess  (then normalize)
    risk_parity       iterative: equalize marginal risk contributions
    equal_weight      sanity baseline

All long-only by default (negative weights are clipped to zero and
the result re-normalized — a heuristic, not the true constrained
optimum, but adequate for the EGX universe sizes we work with).

Optional constraints:
    max_weight        per-name cap (e.g. 0.10 = no more than 10% in one name)
    min_weight        floor — set to 0 for "drop the name" behavior

Inputs:
    tickers       list of symbols
    method        'min_variance' | 'tangency' | 'risk_parity' | 'equal_weight'
    expected_returns_pct  optional dict; required for 'tangency'
    target_vol    optional cap on annualized portfolio vol; scales toward cash
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from . import risk_free
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.optimizer")


def _build_cov_and_count(tickers: list[str], lookback_days: int) -> tuple[pd.DataFrame, list[str], int]:
    """Build the daily-return covariance matrix on a common time index.

    Returns (cov_matrix, kept_tickers, n_days_used).
    """
    rets = {}
    for tk in tickers:
        _, yahoo, _ = resolve_ticker(tk)
        try:
            h = yf.Ticker(yahoo).history(period=f"{lookback_days * 2}d", interval="1d")
            if h is None or h.empty:
                continue
            rets[tk] = h["Close"].pct_change()
        except Exception as e:
            log.warning(f"history fetch failed for {tk}: {e}")
    if not rets:
        return pd.DataFrame(), [], 0
    df = pd.DataFrame(rets).dropna().tail(lookback_days)
    if df.empty or df.shape[1] < 1:
        return pd.DataFrame(), [], 0
    cov = df.cov() * 252  # annualize
    return cov, list(df.columns), int(df.shape[0])


# Keep the legacy name as a thin wrapper for any external users
def _build_cov(tickers: list[str], lookback_days: int) -> tuple[pd.DataFrame, list[str]]:
    cov, kept, _ = _build_cov_and_count(tickers, lookback_days)
    return cov, kept


def _clip_long_only(w: np.ndarray, max_weight: float | None, min_weight: float = 0.0) -> np.ndarray:
    """Long-only projection with per-name cap and renormalization."""
    w = np.maximum(w, min_weight)
    if max_weight is not None:
        w = np.minimum(w, max_weight)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    raw = inv @ ones
    s = raw.sum()
    return raw / s if s != 0 else np.ones(n) / n


def _tangency(cov: np.ndarray, mu_excess: np.ndarray) -> np.ndarray:
    inv = np.linalg.pinv(cov)
    raw = inv @ mu_excess
    s = abs(raw).sum()
    return raw / s if s != 0 else np.ones(len(mu_excess)) / len(mu_excess)


def _risk_parity(cov: np.ndarray, n_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Iterative risk-parity: equalize marginal risk contributions."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(n_iter):
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol == 0:
            break
        mrc = (cov @ w) / port_vol           # marginal risk contribution
        rc = w * mrc                          # actual risk contribution per name
        target = port_vol / n                 # equal target
        # Update: nudge weights toward equal RC
        new_w = w * (target / np.maximum(rc, 1e-10))
        new_w = new_w / new_w.sum()
        if float(np.abs(new_w - w).max()) < tol:
            w = new_w
            break
        w = new_w
    return w


def optimize(
    tickers: list[str],
    method: str = "min_variance",
    expected_returns_pct: dict[str, float] | None = None,
    lookback_days: int = 252,
    max_weight: float | None = 0.20,
    min_weight: float = 0.0,
    target_vol_pct: float | None = None,
    nav_egp: float | None = None,
) -> dict[str, Any]:
    """Compute optimal weights and the resulting portfolio profile."""
    if not tickers:
        return {"error": "no tickers"}

    cov, kept, days_used = _build_cov_and_count(tickers, lookback_days)
    if cov.empty:
        return {"error": "could not build covariance — no overlapping history"}
    cov_arr = cov.values

    if method == "equal_weight":
        w = np.ones(len(kept)) / len(kept)
    elif method == "min_variance":
        w = _min_variance(cov_arr)
    elif method == "risk_parity":
        w = _risk_parity(cov_arr)
    elif method == "tangency":
        if not expected_returns_pct:
            return {"error": "tangency method requires expected_returns_pct"}
        rf = risk_free.get_rate()["rate_pct"]
        # Convert annual % to daily decimal then back to annual decimal
        mu_annual = np.array([
            expected_returns_pct.get(tk, 0.0) / 100 - rf / 100
            for tk in kept
        ])
        w = _tangency(cov_arr, mu_annual)
    else:
        return {"error": f"unknown method: {method}"}

    w = _clip_long_only(w, max_weight=max_weight, min_weight=min_weight)

    port_vol = float(np.sqrt(w @ cov_arr @ w))

    # If a target vol is requested, scale toward cash (de-lever)
    cash_weight = 0.0
    if target_vol_pct is not None and port_vol > 0:
        target = target_vol_pct / 100
        if port_vol > target:
            scale = target / port_vol
            w = w * scale
            cash_weight = 1 - w.sum()
            port_vol = target

    # Risk contributions
    if port_vol > 0:
        mrc = (cov_arr @ w) / port_vol
        rc = w * mrc
        rc_pct = (rc / port_vol).tolist()
    else:
        rc_pct = [1 / len(kept)] * len(kept)

    weights_named = [
        {
            "ticker": kept[i],
            "weight": round(float(w[i]), 4),
            "weight_pct": round(float(w[i]) * 100, 2),
            "risk_contribution_pct": round(float(rc_pct[i]) * 100, 2),
        }
        for i in range(len(kept))
    ]
    weights_named.sort(key=lambda r: r["weight"], reverse=True)

    payload = {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "method": method,
        "n_assets": len(kept),
        "tickers_used": kept,
        "tickers_dropped": [t for t in tickers if t not in kept],
        "weights": weights_named,
        "cash_weight_pct": round(cash_weight * 100, 2),
        "portfolio_volatility_annualized_pct": round(port_vol * 100, 2),
        "max_weight_cap_pct": (max_weight or 1) * 100,
        "lookback_days_used": days_used,
        "method_note": _method_note(method),
    }

    if expected_returns_pct:
        port_return = sum(w[i] * expected_returns_pct.get(kept[i], 0) / 100 for i in range(len(kept)))
        payload["expected_portfolio_return_pct"] = round(port_return * 100, 2)
        rf_pct = risk_free.get_rate()["rate_pct"] / 100
        if port_vol > 0:
            payload["expected_sharpe"] = round((port_return - rf_pct) / port_vol, 2)

    if nav_egp:
        payload["allocations_egp"] = [
            {**row, "egp_allocation": round(row["weight"] * nav_egp, 0)}
            for row in weights_named
        ]
        if cash_weight > 0:
            payload["cash_egp"] = round(cash_weight * nav_egp, 0)

    return payload


def _method_note(method: str) -> str:
    return {
        "equal_weight":  "Equal weight across all names. Sanity baseline.",
        "min_variance":  "Minimum variance — concentrates in low-vol, low-correlation names.",
        "risk_parity":   "Equal risk contribution — each name contributes the same to portfolio vol.",
        "tangency":      "Maximum Sharpe — uses provided expected returns and the EGP T-bill rate.",
    }.get(method, "")
