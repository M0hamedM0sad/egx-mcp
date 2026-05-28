"""Portfolio-aware veto layer.

Sits between `decide()` and the trader. Takes a candidate verdict and
the current portfolio, then either:

  - PASSES the verdict through unchanged
  - DOWNGRADES it (BUY → ACCUMULATE → HOLD) when adding the position
    would breach a portfolio constraint
  - VETOES it entirely when the constraint is hard

This mirrors the Risk Management tier in TradingAgents but enforces
deterministic rules — concentration, sector, drawdown, beta, cash —
rather than LLM judgment. Each downgrade comes with a tagged reason so
the host model can explain the chain of cuts.

Default rules (overridable via constraints arg):

  max_single_name_pct       10%  hard cap on any one position
  max_sector_pct            30%  combined weight in any sector
  max_position_count        20   no more than N names
  min_cash_pct              5%   keep at least N% in cash
  drawdown_circuit_breaker  -8%  if portfolio rolling-20d DD breaches this,
                                 veto all new BUYs
  max_portfolio_beta        1.4  if adding the position would push beta
                                 above this, downgrade

Beta is computed against EGX 30 over a 90-day window via a thin call to
the existing factors module. If the candidate is illiquid or has no
history, beta is treated as 1.0.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from . import factors, portfolio as port_mod, risk as risk_mod
from .universe import EGX_UNIVERSE, resolve_ticker

log = logging.getLogger("egx-mcp.risk_gate")


# ---------------------------------------------------------------------------
# Default constraints
# ---------------------------------------------------------------------------

DEFAULT_CONSTRAINTS: dict[str, float] = {
    "max_single_name_pct":      10.0,
    "max_sector_pct":           30.0,
    "max_position_count":       20,
    "min_cash_pct":             5.0,
    "drawdown_circuit_breaker": -8.0,
    "max_portfolio_beta":       1.4,
}

_VERDICT_RANK = {
    "AVOID": 0, "REDUCE": 1, "HOLD": 2, "WAIT": 2,
    "ACCUMULATE": 3, "BUY": 4,
}
_RANK_VERDICT = {v: k for k, v in _VERDICT_RANK.items()}


def _downgrade(verdict: str, steps: int = 1) -> str:
    rank = _VERDICT_RANK.get(verdict, 2)
    new_rank = max(2, rank - steps)  # never below HOLD; HOLD is the floor
    if rank == 4 and new_rank == 3:
        return "ACCUMULATE"
    if new_rank <= 2:
        return "HOLD"
    return _RANK_VERDICT.get(new_rank, "HOLD")


def _sector_for(canonical: str) -> str | None:
    entry = EGX_UNIVERSE.get(canonical)
    return entry.get("sector") if entry else None


def _candidate_weight_pct(
    canonical: str,
    suggested_levels: dict | None,
    nav: float,
) -> float:
    """How much of NAV would the candidate eat if filled at suggested size?"""
    if suggested_levels and suggested_levels.get("position_weight_pct") is not None:
        return float(suggested_levels["position_weight_pct"])
    # Fallback: assume a default 5% if sizing wasn't run
    return 5.0


def _name_beta(canonical: str, lookback_days: int = 90) -> float | None:
    try:
        exp = factors.ticker_factor_exposure(canonical, lookback_days=lookback_days)
        beta = (exp.get("factor_betas") or {}).get("egx30")
        return float(beta) if beta is not None else None
    except Exception:
        return None


def _portfolio_beta(positions: list[dict], lookback_days: int = 90) -> float | None:
    """Mean of single-name betas weighted by current market value."""
    rows = []
    total_w = 0.0
    for p in positions:
        if not p.get("market_value"):
            continue
        w = p["market_value"]
        beta = _name_beta(p["ticker"].upper(), lookback_days=lookback_days)
        if beta is None:
            continue
        rows.append((w, beta))
        total_w += w
    if total_w <= 0 or not rows:
        return None
    return sum(w * b for w, b in rows) / total_w


def _candidate_beta(canonical: str, lookback_days: int = 90) -> float:
    b = _name_beta(canonical, lookback_days=lookback_days)
    return b if b is not None else 1.0


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def risk_gate(
    user_ticker: str,
    proposed_verdict: str,
    suggested_levels: dict | None = None,
    portfolio_csv: str | None = None,
    constraints: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run portfolio-aware checks on a proposed verdict.

    Args:
        user_ticker: EGX code or nickname.
        proposed_verdict: BUY / ACCUMULATE / HOLD / REDUCE / AVOID / WAIT.
        suggested_levels: From decide().suggested_levels — used to size the
            hypothetical add. If None, falls back to a 5% NAV assumption.
        portfolio_csv: Path to portfolio CSV. Defaults to env / home.
        constraints: Override any of DEFAULT_CONSTRAINTS keys.

    Returns:
        Dict with: original_verdict, final_verdict, downgrades (list of
        triggered rules), portfolio_snapshot, breaches (raw checks), as_of.
    """
    canonical, _, _ = resolve_ticker(user_ticker)
    rules = {**DEFAULT_CONSTRAINTS, **(constraints or {})}

    summary = port_mod.summary(csv_path=portfolio_csv)
    if "error" in summary:
        return {
            "original_verdict": proposed_verdict,
            "final_verdict": proposed_verdict,
            "downgrades": [],
            "warning": f"Portfolio not loaded — gate skipped. {summary['error']}",
        }

    nav = summary.get("total_value_egp") or 0
    positions = summary.get("positions") or []
    breaches: list[dict[str, Any]] = []
    downgrades: list[str] = []

    # If verdict isn't a buy-side action, gate is essentially a no-op
    is_buy_side = proposed_verdict in ("BUY", "ACCUMULATE")
    final = proposed_verdict

    cand_weight = _candidate_weight_pct(canonical, suggested_levels, nav) if is_buy_side else 0.0
    cand_sector = _sector_for(canonical)

    # 1. Single-name concentration (post-add)
    existing = next((p for p in positions if p["ticker"].upper() == canonical), None)
    existing_w = (existing or {}).get("weight_pct") or 0.0
    post_add_name_w = existing_w + cand_weight
    if is_buy_side and post_add_name_w > rules["max_single_name_pct"]:
        breaches.append({
            "rule": "max_single_name_pct",
            "limit": rules["max_single_name_pct"],
            "post_add_value": round(post_add_name_w, 2),
            "severity": "hard",
        })
        downgrades.append(
            f"Single-name cap: {canonical} would be {post_add_name_w:.1f}% NAV "
            f"(limit {rules['max_single_name_pct']:.0f}%). Verdict downgraded."
        )
        final = _downgrade(final, steps=2)  # BUY → HOLD

    # 2. Sector concentration (post-add)
    if cand_sector and is_buy_side:
        sector_w = sum(
            (p.get("weight_pct") or 0)
            for p in positions
            if _sector_for(p["ticker"].upper()) == cand_sector
        )
        post_add_sector_w = sector_w + cand_weight
        if post_add_sector_w > rules["max_sector_pct"]:
            breaches.append({
                "rule": "max_sector_pct",
                "sector": cand_sector,
                "limit": rules["max_sector_pct"],
                "post_add_value": round(post_add_sector_w, 2),
                "severity": "hard",
            })
            downgrades.append(
                f"Sector cap: '{cand_sector}' would be {post_add_sector_w:.1f}% "
                f"(limit {rules['max_sector_pct']:.0f}%). Verdict downgraded."
            )
            if final in ("BUY", "ACCUMULATE"):
                final = _downgrade(final, steps=1)

    # 3. Position count
    if is_buy_side and not existing and len(positions) >= rules["max_position_count"]:
        breaches.append({
            "rule": "max_position_count",
            "limit": rules["max_position_count"],
            "current": len(positions),
            "severity": "soft",
        })
        downgrades.append(
            f"Position-count cap: portfolio already holds {len(positions)} names "
            f"(limit {rules['max_position_count']}). Add only as a swap, not net new."
        )
        if final == "BUY":
            final = "ACCUMULATE"

    # 4. Drawdown circuit breaker (uses risk module on existing book)
    if is_buy_side and len(positions) >= 2:
        try:
            tickers = [p["ticker"] for p in positions]
            wts = [p.get("weight_pct") or 0 for p in positions]
            if sum(wts) > 0:
                rk = risk_mod.portfolio_risk(
                    tickers=tickers, weights=wts, lookback_days=120,
                )
                roll_dd = rk.get("rolling_20d_drawdown_pct")
                if roll_dd is not None and roll_dd <= rules["drawdown_circuit_breaker"]:
                    breaches.append({
                        "rule": "drawdown_circuit_breaker",
                        "rolling_20d_dd_pct": roll_dd,
                        "limit_pct": rules["drawdown_circuit_breaker"],
                        "severity": "hard",
                    })
                    downgrades.append(
                        f"Circuit breaker: portfolio rolling-20d DD = {roll_dd:.1f}% "
                        f"(limit {rules['drawdown_circuit_breaker']:.0f}%). All new BUYs vetoed."
                    )
                    final = "HOLD"
        except Exception as e:
            log.warning(f"DD breaker check failed: {e}")

    # 5. Portfolio beta (post-add)
    if is_buy_side and len(positions) >= 2:
        try:
            current_beta = _portfolio_beta(positions, lookback_days=90)
            cand_beta = _candidate_beta(canonical, lookback_days=90)
            if current_beta is not None:
                # Approximate post-add beta as weighted blend
                cand_w_frac = cand_weight / 100.0
                post_beta = current_beta * (1 - cand_w_frac) + cand_beta * cand_w_frac
                if post_beta > rules["max_portfolio_beta"]:
                    breaches.append({
                        "rule": "max_portfolio_beta",
                        "post_add_beta": round(post_beta, 2),
                        "current_beta": round(current_beta, 2),
                        "candidate_beta": round(cand_beta, 2),
                        "limit": rules["max_portfolio_beta"],
                        "severity": "soft",
                    })
                    downgrades.append(
                        f"Beta cap: portfolio β would rise to {post_beta:.2f} "
                        f"(limit {rules['max_portfolio_beta']}). Verdict softened."
                    )
                    if final == "BUY":
                        final = "ACCUMULATE"
        except Exception as e:
            log.warning(f"Beta check failed: {e}")

    # 6. Min cash (informational only — needs cash row in CSV which the
    # current adapter doesn't model). Recorded so the host can warn.
    if is_buy_side and cand_weight > 0:
        breaches.append({
            "rule": "min_cash_pct",
            "candidate_weight_pct": round(cand_weight, 2),
            "limit": rules["min_cash_pct"],
            "severity": "informational",
            "note": "Cash buffer not tracked in CSV — confirm manually before sizing up.",
        })

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "ticker": canonical,
        "original_verdict": proposed_verdict,
        "final_verdict": final,
        "downgrades": downgrades,
        "breaches": breaches,
        "constraints_applied": rules,
        "portfolio_snapshot": {
            "nav_egp": round(nav, 2),
            "position_count": len(positions),
            "candidate_weight_pct": round(cand_weight, 2) if is_buy_side else None,
            "candidate_sector": cand_sector,
            "existing_weight_pct": round(existing_w, 2) if existing else 0,
        },
        "method": (
            "Deterministic portfolio-aware gate. Checks single-name cap, "
            "sector cap, position count, drawdown circuit breaker, and "
            "post-add portfolio beta. Returns the most conservative verdict "
            "consistent with the constraint set."
        ),
    }
