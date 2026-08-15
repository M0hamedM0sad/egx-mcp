"""Composite scoring engine — turns raw signals into a 0-100 score.

Four sub-scores, each 0-100, then weighted into the composite:

    Valuation (30%) : P/E and P/B vs. sector median, dividend yield
    Quality   (25%) : ROE, profit margin, debt/equity
    Momentum  (25%) : 3M / 6M return, RSI regime, MA50 vs MA200
    Risk      (20%) : annualized volatility, max drawdown

Every sub-score logs the inputs it used and the points it awarded so
the final verdict is auditable. No black boxes.
"""
from __future__ import annotations

import logging
from typing import Any

from . import fundamentals, market, technicals, regime, risk_free, model_params
from .macro import sector_macro_bias

log = logging.getLogger("egx-mcp.scoring")


# ---------------------------------------------------------------------------
# SUB-SCORERS
# ---------------------------------------------------------------------------

class _Ledger:
    """Scoring notes paired with the point delta each one awarded.

    Consumers need to know whether a note argued for or against the name.
    Inferring that from wording is unreliable: the momentum "stretched"
    penalty — the single largest deduction the model can make — reads as
    neutral prose and was being reported to users as a positive driver. The
    delta is recorded where the score is actually adjusted, so the sign is a
    fact rather than a guess about vocabulary.

    `notes` stays a plain list of strings for existing consumers; the parallel
    `note_deltas` list is additive.
    """

    __slots__ = ("points", "notes", "deltas")

    def __init__(self, start: float = 50.0) -> None:
        self.points = start          # neutral baseline
        self.notes: list[str] = []
        self.deltas: list[float] = []

    def add(self, delta: float, note: str) -> None:
        """Award `delta` points and record why. delta=0 for purely informational notes."""
        self.points += delta
        self.notes.append(note)
        self.deltas.append(delta)

    def result(self) -> dict[str, Any]:
        return {"score": round(max(0, min(100, self.points)), 1),
                "notes": self.notes, "note_deltas": self.deltas}


def _score_valuation(f: dict, sector_med: dict) -> dict:
    """Lower P/E and P/B vs. sector → higher score. Yield is a bonus."""
    led = _Ledger()

    pe = f.get("pe_ratio")
    med_pe = sector_med.get("median_pe")
    if pe is not None and med_pe:
        ratio = pe / med_pe
        if ratio <= 0.7:
            led.add(+25, f"P/E {pe} ≤ 70% of sector median {med_pe} (cheap)")
        elif ratio <= 0.9:
            led.add(+15, f"P/E {pe} below sector median {med_pe}")
        elif ratio >= 1.5:
            led.add(-20, f"P/E {pe} ≥ 150% of sector median {med_pe} (expensive)")
        elif ratio >= 1.2:
            led.add(-10, f"P/E {pe} above sector median {med_pe}")
        else:
            led.add(0, f"P/E {pe} near sector median {med_pe}")
    elif pe is None:
        led.add(0, "P/E unavailable — valuation read is incomplete")

    pb = f.get("pb_ratio")
    med_pb = sector_med.get("median_pb")
    if pb is not None and med_pb:
        if pb <= 0.8 * med_pb:
            led.add(+10, f"P/B {pb} below sector median {med_pb}")
        elif pb >= 1.5 * med_pb:
            led.add(-10, f"P/B {pb} above sector median {med_pb}")

    dy = f.get("dividend_yield_pct") or 0
    if dy >= 8:
        led.add(+8, f"Dividend yield {dy}% — strong income")
    elif dy >= 4:
        led.add(+4, f"Dividend yield {dy}% — supportive")

    return led.result()


def _score_quality(f: dict, sector_med: dict) -> dict:
    led = _Ledger()

    roe = f.get("roe_pct")
    med_roe = sector_med.get("median_roe_pct")
    if roe is not None:
        if roe >= 20:
            led.add(+25, f"ROE {roe}% — excellent")
        elif roe >= 12:
            led.add(+12, f"ROE {roe}% — solid")
        elif roe < 5:
            led.add(-15, f"ROE {roe}% — weak")
        if med_roe and roe > med_roe * 1.3:
            led.add(+5, f"ROE beats sector median {med_roe}%")

    margin = f.get("profit_margin_pct")
    if margin is not None:
        if margin >= 20:
            led.add(+12, f"Profit margin {margin}% — premium")
        elif margin >= 10:
            led.add(+6, f"Profit margin {margin}%")
        elif margin < 0:
            led.add(-25, f"Negative margin {margin}% — unprofitable")
        elif margin < 3:
            led.add(-10, f"Margin {margin}% — thin")

    dte = f.get("debt_to_equity")
    if dte is not None:
        # Yahoo reports D/E in percent (e.g. 120 = 1.2x)
        dte_x = dte / 100 if dte > 5 else dte
        if dte_x > 2:
            led.add(-12, f"D/E {dte_x:.2f}x — highly levered")
        elif dte_x < 0.3:
            led.add(+5, f"D/E {dte_x:.2f}x — conservative balance sheet")

    return led.result()


def _score_momentum(history: dict, tech: dict) -> dict:
    led = _Ledger()

    summary = history.get("summary") or {}
    ret = summary.get("return_pct")
    if ret is not None:
        if ret >= 30:
            led.add(+25, f"6M return {ret}% — strong")
        elif ret >= 10:
            led.add(+12, f"6M return {ret}%")
        elif ret <= -20:
            led.add(-20, f"6M return {ret}% — broken")
        elif ret < 0:
            led.add(-8, f"6M return {ret}% — soft")

        # Stretched penalty — V3 backtest validation: names already up >50%
        # over 6m systematically underperform their score predicts. Cuts max
        # DD from -7.9% to -3.8% with no Sharpe loss.
        if ret > 50:
            stretch_pen = (ret - 50) * 0.4
            led.add(-stretch_pen, f"6M +{ret}% — stretched (-{stretch_pen:.1f} pts)")

    ind = (tech or {}).get("indicators") or {}
    rsi = ind.get("rsi_14")
    if rsi is not None:
        if rsi >= 70:
            led.add(-10, f"RSI {rsi:.1f} — overbought (mean-revert risk)")
        elif rsi <= 30:
            led.add(+10, f"RSI {rsi:.1f} — oversold (bounce candidate)")
        elif 50 <= rsi < 65:
            led.add(+5, f"RSI {rsi:.1f} — healthy uptrend zone")

    macd = ind.get("macd")
    macd_sig = ind.get("macd_signal")
    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            led.add(+5, "MACD bullish cross")
        else:
            led.add(-5, "MACD bearish cross")

    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")
    if sma50 and sma200:
        if sma50 > sma200:
            led.add(+10, "SMA50 > SMA200 (golden-cross territory)")
        else:
            led.add(-10, "SMA50 < SMA200 (death-cross territory)")

    return led.result()


def _score_risk(history: dict) -> dict:
    """Lower vol/drawdown → higher score (risk is *inversely* desirable)."""
    led = _Ledger()

    summary = history.get("summary") or {}
    vol = summary.get("annualized_volatility_pct")
    if vol is not None:
        if vol < 25:
            led.add(+20, f"Vol {vol}% — calm")
        elif vol < 40:
            led.add(+5, f"Vol {vol}% — typical EGX")
        elif vol >= 60:
            led.add(-25, f"Vol {vol}% — extreme")
        elif vol >= 50:
            led.add(-15, f"Vol {vol}% — elevated")

    dd = summary.get("max_drawdown_pct")
    if dd is not None:
        if dd > -10:
            led.add(+15, f"Max drawdown {dd}% — shallow")
        elif dd < -35:
            led.add(-20, f"Max drawdown {dd}% — deep")
        elif dd < -25:
            led.add(-10, f"Max drawdown {dd}% — meaningful")

    return led.result()


# ---------------------------------------------------------------------------
# COMPOSITE
# ---------------------------------------------------------------------------

# Baseline composite weights. The ACTIVE weights are read at score time from
# model_params (learnable + human-approved); this dict is only the fallback the
# loop starts from and what DEFAULTS in model_params mirrors.
_WEIGHTS = {"valuation": 0.30, "quality": 0.25, "momentum": 0.25, "risk": 0.20}


def score_stock(user_ticker: str, history_period: str = "6mo") -> dict[str, Any]:
    """Return a structured 0-100 score with full audit trail."""
    f = fundamentals.get_fundamentals(user_ticker)
    if "error" in f:
        return {"ticker": user_ticker, "error": f["error"]}

    sector = f.get("sector") or "Unknown"
    try:
        sector_med = fundamentals.sector_medians(sector) if sector != "Unknown" else {}
    except Exception as e:
        log.warning(f"sector medians failed for {sector}: {e}")
        sector_med = {}

    try:
        history = market.get_history(user_ticker, period=history_period)
    except Exception as e:
        history = {"summary": None, "error": str(e)}

    try:
        tech = technicals.compute(user_ticker, period=history_period)
    except Exception as e:
        tech = {"indicators": {}, "error": str(e)}

    val = _score_valuation(f, sector_med)
    qual = _score_quality(f, sector_med)
    mom = _score_momentum(history, tech)
    risk = _score_risk(history)

    # Regime-aware weights: multiply base weights by regime bias and renormalize
    try:
        reg = regime.classify()
        bias = reg.get("weight_override", {})
    except Exception:
        reg = {"regime": "UNKNOWN", "weight_override": {}}
        bias = {}
    base_weights = model_params.score_weights()
    weights = {k: base_weights[k] * bias.get(k, 1.0) for k in base_weights}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    composite = (
        val["score"] * weights["valuation"]
        + qual["score"] * weights["quality"]
        + mom["score"] * weights["momentum"]
        + risk["score"] * weights["risk"]
    )

    macro = sector_macro_bias(sector) if sector != "Unknown" else {"macro_bias": 0, "reasons": []}
    macro_adj = composite + macro["macro_bias"] * 5  # ±5 pts max
    composite_final = round(max(0, min(100, macro_adj)), 1)

    # Excess-return context: take the 6M return and convert to excess over T-bills
    summary = (history.get("summary") or {})
    period_ret = summary.get("return_pct")
    excess_6m = None
    if period_ret is not None:
        # 6M ≈ 126 trading days
        excess_6m = risk_free.excess_return_pct(period_ret, horizon_days=126)

    return {
        "ticker": f["ticker"],
        "name": f["name"],
        "sector": sector,
        "price": f.get("price"),
        "composite_score": composite_final,
        "raw_composite": round(composite, 1),
        "macro_adjustment": round(macro["macro_bias"] * 5, 2),
        "subscores": {
            "valuation": val,
            "quality": qual,
            "momentum": mom,
            "risk": risk,
        },
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
        "weights_base": base_weights,
        "regime": reg.get("regime"),
        "regime_description": reg.get("description"),
        "sector_medians": sector_med,
        "macro_context": macro,
        "fundamentals_snapshot": {
            "pe": f.get("pe_ratio"),
            "pb": f.get("pb_ratio"),
            "roe_pct": f.get("roe_pct"),
            "margin_pct": f.get("profit_margin_pct"),
            "div_yield_pct": f.get("dividend_yield_pct"),
            "pe_was_corrected": f.get("pe_was_corrected"),
            "fundamentals_source": f.get("fundamentals_source"),
        },
        "trailing_6m_return_pct": period_ret,
        "trailing_6m_excess_return_pct": excess_6m,
    }
