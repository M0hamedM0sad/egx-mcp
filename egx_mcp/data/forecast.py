"""B3 — Probabilistic price forecast as a decision feature.

A zero-shot time-series foundation model (Chronos by default) takes a raw
price series and returns a forecast DISTRIBUTION — not a point estimate. We
turn that into two signals the scoring/decision layer can consume:

    expected_return_pct   median forecast over the horizon vs last close
    uncertainty_pct       q90-q10 spread — a model-implied volatility / how
                          confident the forecast is

Use it as ONE input among many, not an oracle. EGX is thin and FX-driven;
these models are trained on broad global corpora and may not transfer. ALWAYS
backtest the signal (tests/backtest_accuracy.py) before weighting it in
decide(). Treating a wide uncertainty band as a reason to size down is often
more robust than trading the point forecast.

Optional dep, lazy-loaded once, graceful fallback to a naive drift+vol
estimate if the model isn't installed — so the function always returns
something the caller can branch on:

    pip install 'egx-mcp[forecast]'      # chronos-forecasting + torch

Override the model with EGX_FORECAST_MODEL (e.g. amazon/chronos-bolt-small
for speed, or amazon/chronos-t5-base for accuracy).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from . import market
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.forecast")

_MODEL = os.environ.get("EGX_FORECAST_MODEL", "amazon/chronos-t5-small")
_CONTEXT_LEN = 512  # trailing closes fed as context

_pipeline: Any = None
_load_failed = False


def _get_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None:
        return _pipeline
    if _load_failed:
        return None
    try:
        import torch
        from chronos import BaseChronosPipeline
    except Exception as e:  # noqa: BLE001
        log.warning("chronos not installed (%s); using naive drift fallback. "
                    "Install with: pip install 'egx-mcp[forecast]'", e)
        _load_failed = True
        return None
    try:
        _pipeline = BaseChronosPipeline.from_pretrained(
            _MODEL, device_map="cpu", torch_dtype=torch.float32,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load %s: %s; naive drift fallback", _MODEL, e)
        _load_failed = True
        return None
    return _pipeline


def available() -> bool:
    return _get_pipeline() is not None


def _closes(user_ticker: str, period: str) -> tuple[list[float], str | None]:
    hist = market.get_history(user_ticker, period=period)
    rows = hist.get("rows") or []
    closes = [r["close"] for r in rows if r.get("close")]
    return closes, hist.get("ticker")


def _naive_forecast(closes: list[float], horizon: int) -> dict[str, Any]:
    """Fallback: extrapolate trailing daily drift, band by trailing vol."""
    import statistics as st
    if len(closes) < 30:
        return {"error": "insufficient history for a naive forecast"}
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    tail = rets[-60:]
    mu = sum(tail) / len(tail)
    sd = st.pstdev(tail) if len(tail) > 1 else 0.0
    exp_ret = (1 + mu) ** horizon - 1
    band = sd * (horizon ** 0.5) * 1.2816 * 2  # ~q10..q90 width (z=1.2816)
    return {
        "expected_return_pct": round(exp_ret * 100, 2),
        "uncertainty_pct": round(band * 100, 2),
        "direction": "up" if exp_ret > 0 else "down" if exp_ret < 0 else "flat",
        "method": "naive drift+vol (model unavailable)",
    }


def forecast_return(user_ticker: str, horizon_days: int = 21,
                    period: str = "2y") -> dict[str, Any]:
    """Forecast the horizon-ahead return distribution for one EGX name.

    Returns {ticker, horizon_days, expected_return_pct, uncertainty_pct,
    direction, last_close, method}. Falls back to a naive drift estimate
    when the model isn't installed.
    """
    canonical, _, _ = resolve_ticker(user_ticker)
    closes, resolved = _closes(user_ticker, period)
    if len(closes) < 30:
        return {"ticker": resolved or canonical, "error": "insufficient price history"}
    last_close = closes[-1]

    pipe = _get_pipeline()
    if pipe is None:
        base = _naive_forecast(closes, horizon_days)
    else:
        try:
            import numpy as np
            import torch
            ctx = torch.tensor(closes[-_CONTEXT_LEN:], dtype=torch.float32)
            # quantile API differs across chronos versions; predict_quantiles
            # is the stable path on BaseChronosPipeline.
            q, _mean = pipe.predict_quantiles(
                context=ctx, prediction_length=horizon_days,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            arr = q[0].numpy()  # shape [horizon, 3]
            q10, q50, q90 = arr[-1, 0], arr[-1, 1], arr[-1, 2]
            exp_ret = float(q50 / last_close - 1)
            band = float((q90 - q10) / last_close)
            base = {
                "expected_return_pct": round(exp_ret * 100, 2),
                "uncertainty_pct": round(band * 100, 2),
                "direction": "up" if exp_ret > 0 else "down" if exp_ret < 0 else "flat",
                "method": f"zero-shot forecast ({_MODEL})",
            }
        except Exception as e:  # noqa: BLE001
            log.warning("forecast inference failed: %s; naive fallback", e)
            base = _naive_forecast(closes, horizon_days)

    return {
        "ticker": resolved or canonical,
        "horizon_days": horizon_days,
        "last_close": round(last_close, 4),
        **base,
        "disclaimer": (
            "One input among many. EGX is thin and FX-driven; a global "
            "foundation model may not transfer. Backtest before weighting it "
            "in a verdict. A wide uncertainty band is a reason to size down."
        ),
    }
