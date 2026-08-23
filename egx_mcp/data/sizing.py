"""Position sizing — turns volatility into a concrete share count.

Uses ATR(14) as the per-share dollar risk estimate. The sizing rule:

    risk_egp = portfolio_value * (risk_pct / 100)
    stop_distance = atr_multiple * ATR_14
    shares = risk_egp / stop_distance
    cost = shares * price
    stop_loss = price - stop_distance

A `risk_pct` of 1% with `atr_multiple` of 2 means: "if my stop is hit
two ATRs below entry, I lose no more than 1% of portfolio value." This
is the standard Van Tharp / turtle convention.

`trader_plan` extends `position_size` with a tranched entry plan
(spec → entry levels) suited to thin EGX names where filling at one
print would move the tape.
"""
from __future__ import annotations

from typing import Any

from . import market, technicals, liquidity, risk_free

# Hard ceiling on the ATR-derived stop distance, as a share of entry price.
# Beyond this the "stop" stops being a risk control: a 58%-of-price ATR is a
# broken price series, not a volatility estimate.
MAX_STOP_PCT_OF_PRICE = 25.0


def position_size(
    user_ticker: str,
    portfolio_value_egp: float,
    risk_pct: float = 1.0,
    atr_multiple: float = 2.0,
    max_position_pct: float = 10.0,
) -> dict[str, Any]:
    """Return a sized position with explicit stop, target, and rationale."""
    if portfolio_value_egp <= 0:
        return {"error": "portfolio_value_egp must be positive"}
    if not (0 < risk_pct <= 5):
        return {"error": "risk_pct must be between 0 and 5"}

    quote = market.get_quote(user_ticker)
    price = quote.get("price")
    if not price:
        return {"error": f"No live price for {user_ticker}"}

    tech = technicals.compute(user_ticker, period="6mo")
    atr = (tech.get("indicators") or {}).get("atr_14")
    if not atr:
        return {"error": f"ATR(14) unavailable for {user_ticker} — not enough history"}

    risk_egp = portfolio_value_egp * (risk_pct / 100)
    stop_distance = atr_multiple * atr
    # A stop cannot sit at or below zero. When ATR is a large fraction of price
    # — usually because the series carries an unadjusted corporate action, e.g.
    # GTWL priced 29.84 with ATR 17.30 giving a -4.92 "stop" — cap the distance
    # and say so, rather than printing a negative price as a trade level.
    max_distance = price * MAX_STOP_PCT_OF_PRICE / 100
    stop_capped = stop_distance > max_distance
    if stop_capped:
        stop_distance = max_distance
    raw_shares = risk_egp / stop_distance

    # Cap by max position weight
    max_position_egp = portfolio_value_egp * (max_position_pct / 100)
    max_shares_by_cap = max_position_egp / price

    shares = int(min(raw_shares, max_shares_by_cap))
    if shares <= 0:
        return {
            "ticker": quote.get("ticker"),
            "error": "Computed share count is zero. Increase risk_pct or portfolio_value, or pick a less volatile name.",
            "atr_14": atr,
            "stop_distance_egp": round(stop_distance, 4),
        }

    cost = shares * price
    stop_loss = round(price - stop_distance, 2)
    # Scale-out at 1R (half the clip), full target at 1.5R on the rest.
    # Calibrated against May-2026 W1 cohort where 5/5 picks closed in profit
    # but 0/5 reached the old 2R target inside the 5-day horizon.
    scale_out_price = round(price + stop_distance, 2)
    target_price = round(price + 1.5 * stop_distance, 2)
    scale_out_shares = shares // 2
    runner_shares = shares - scale_out_shares

    # Liquidity gate — final cap by ADV
    liq = liquidity.check_capacity(quote.get("ticker"), shares, max_participation_pct=15.0)
    if liq.get("max_safe_shares") is not None and shares > liq["max_safe_shares"]:
        shares = liq["max_safe_shares"]
        cost = shares * price
        liq_capped = True
    else:
        liq_capped = False

    # Excess return context: target return vs T-bill yield over expected hold
    target_ret_pct = (target_price / price - 1) * 100
    expected_hold_days = 21  # roughly one month if target hit
    excess_return_pct = risk_free.excess_return_pct(target_ret_pct, horizon_days=expected_hold_days)

    return {
        "ticker": quote.get("ticker"),
        "name": quote.get("name"),
        "price": round(price, 4),
        "atr_14": round(atr, 4),
        "atr_pct_of_price": round(atr / price * 100, 2),
        "shares": shares,
        "position_cost_egp": round(cost, 2),
        "position_weight_pct": round(cost / portfolio_value_egp * 100, 2),
        "stop_loss_price": stop_loss,
        "stop_distance_egp": round(stop_distance, 2),
        "stop_distance_capped": stop_capped,
        "stop_distance_capped_note": (
            f"ATR({atr:.2f}) x {atr_multiple} exceeded {MAX_STOP_PCT_OF_PRICE}% of the "
            f"{price:.2f} price — stop distance capped. Check the price series for an "
            "unadjusted split or capital change before trading this."
            if stop_capped else None),
        "scale_out_price": scale_out_price,
        "scale_out_shares": scale_out_shares,
        "target_price": target_price,
        "runner_shares": runner_shares,
        "risk_egp": round(risk_egp, 2),
        "risk_pct_of_portfolio": risk_pct,
        "reward_to_risk": 1.5,
        "capped_by_max_position": raw_shares > max_shares_by_cap,
        "capped_by_liquidity": liq_capped,
        "liquidity_check": liq,
        "target_return_pct": round(target_ret_pct, 2),
        "target_excess_return_over_tbills_pct": excess_return_pct,
        "method": (
            f"shares = {risk_pct}% × portfolio / ({atr_multiple} × ATR), "
            f"capped at {max_position_pct}% of portfolio AND 15% of ADV"
        ),
    }


# ---------------------------------------------------------------------------
# Trader plan — tranched entry / scale-out for thin EGX names
# ---------------------------------------------------------------------------

def trader_plan(
    user_ticker: str,
    portfolio_value_egp: float,
    risk_pct: float = 1.0,
    atr_multiple: float = 2.0,
    max_position_pct: float = 10.0,
    tranches: int = 3,
) -> dict[str, Any]:
    """Return a multi-tranche entry / scale-out plan around the sized position.

    EGX names are thin. Filling a full clip at one print pays slippage and
    leaves no ammunition if the trade goes against you first. A tranched
    plan reduces both problems:

        Entry 1   1/N at market — confirm the thesis
        Entry 2   1/N on a pullback to MA20 (or -1 ATR from entry, whichever
                  is closer) — averages down only if structure holds
        Entry 3   1/N on confirmation — trade above entry by +0.5 ATR with
                  RSI > 50 — adds only on strength

        Stop      common stop = entry_avg - atr_multiple × ATR
        Target 1  entry_avg + 1 ATR — book 1/3, move stop to BE
        Target 2  entry_avg + 2 ATR — book 1/3
        Target 3  trail with chandelier (high − 3 × ATR)

    Args:
        user_ticker: EGX code or nickname.
        portfolio_value_egp: Total NAV.
        risk_pct: Per-trade risk. Default 1%.
        atr_multiple: Stop in ATRs. Default 2.
        max_position_pct: Cap. Default 10%.
        tranches: Number of entry slices. 1, 2, or 3. Default 3.

    Returns:
        Dict with: base sizing, entry_plan (list), exit_plan (list), notes.
    """
    if tranches not in (1, 2, 3):
        return {"error": "tranches must be 1, 2, or 3"}

    base = position_size(
        user_ticker,
        portfolio_value_egp=portfolio_value_egp,
        risk_pct=risk_pct,
        atr_multiple=atr_multiple,
        max_position_pct=max_position_pct,
    )
    if "error" in base:
        return base

    price = base["price"]
    atr = base["atr_14"]
    total_shares = base["shares"]

    # Reference levels for tranche placement
    tech = technicals.compute(user_ticker, period="6mo")
    ind = tech.get("indicators") or {}
    ma20 = ind.get("sma_20")
    rsi = ind.get("rsi_14")

    per_tranche = total_shares // tranches
    remainder = total_shares - per_tranche * tranches  # tail goes to first tranche
    sizes = [per_tranche] * tranches
    sizes[0] += remainder

    pullback_ref = ma20 if (ma20 and ma20 < price) else round(price - atr, 4)

    entry_plan = []
    if tranches >= 1:
        entry_plan.append({
            "tranche": 1,
            "shares": sizes[0],
            "trigger": "market",
            "limit_price": round(price, 4),
            "rationale": "Confirm thesis with a starter clip at current print.",
        })
    if tranches >= 2:
        entry_plan.append({
            "tranche": 2,
            "shares": sizes[1],
            "trigger": "pullback",
            "limit_price": round(pullback_ref, 4),
            "rationale": (
                f"Add on retest of {'MA20' if ma20 and ma20 < price else 'entry−1ATR'} "
                f"({pullback_ref}). Cancel if price closes below stop before fill."
            ),
        })
    if tranches >= 3:
        confirm_price = round(price + 0.5 * atr, 4)
        entry_plan.append({
            "tranche": 3,
            "shares": sizes[2],
            "trigger": "breakout_confirmation",
            "limit_price": confirm_price,
            "rationale": (
                f"Add on strength: print > {confirm_price} (entry + 0.5 ATR) with "
                f"RSI > 50. Current RSI: {rsi if rsi is not None else 'n/a'}."
            ),
        })

    avg_entry = price  # MC of fills approximates spot if all tranches fire
    common_stop = round(avg_entry - atr_multiple * atr, 4)

    exit_plan = [
        {
            "level": 1,
            "shares": total_shares // 3,
            "trigger_price": round(avg_entry + atr, 4),
            "action": "book 1/3, move stop to breakeven",
        },
        {
            "level": 2,
            "shares": total_shares // 3,
            "trigger_price": round(avg_entry + 2 * atr, 4),
            "action": "book 1/3",
        },
        {
            "level": 3,
            "shares": total_shares - 2 * (total_shares // 3),
            "trigger_price": "chandelier_trail",
            "action": (
                f"trail remainder using running_high − 3 × ATR (≈ {round(3 * atr, 2)} EGP "
                "wide). Exit on stop hit."
            ),
        },
    ]

    return {
        "ticker": base["ticker"],
        "name": base["name"],
        "price": price,
        "atr_14": atr,
        "total_shares": total_shares,
        "stop_loss_price": common_stop,
        "stop_distance_egp": round(atr_multiple * atr, 2),
        "tranches": tranches,
        "entry_plan": entry_plan,
        "exit_plan": exit_plan,
        "base_sizing": {
            "position_cost_egp": base["position_cost_egp"],
            "position_weight_pct": base["position_weight_pct"],
            "risk_egp": base["risk_egp"],
            "reward_to_risk": base["reward_to_risk"],
            "capped_by_liquidity": base["capped_by_liquidity"],
        },
        "notes": (
            "Tranched plan suited to thin EGX names. If the candidate has "
            "ADV-binding liquidity (capped_by_liquidity=true), tranches help; "
            "otherwise feel free to consolidate."
        ),
        "method": (
            "Total shares from position_size, then split equally across N "
            "tranches. Entries: market / pullback (MA20 or −1 ATR) / +0.5 "
            "ATR breakout. Exits: 1 ATR (1/3) / 2 ATR (1/3) / chandelier "
            "trail (1/3)."
        ),
    }
