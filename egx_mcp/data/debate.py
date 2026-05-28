"""Structured bull / bear / chairman debate.

Inspired by TradingAgents' researcher debate and the llm-council
three-stage pattern, but executed deterministically inside the MCP:

    Stage 1  bull_case   — top reasons to be long, sourced from
                           score, peers, momentum, catalysts, sentiment
    Stage 2  bear_case   — top reasons to avoid, sourced from
                           score, peers, valuation, risk, sentiment, catalysts
    Stage 3  chairman    — weighs bull vs bear, returns a synthesis
                           verdict, conviction, and the deciding factors

The chairman is *not* an LLM call — it's a rules-based synthesizer that
the host model (Claude in Claude Desktop) can override or extend in its
own reasoning. The point is to give the model a clean dossier with
both sides of the trade structured the way a real research desk would.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from . import calendar as cal_mod
from . import peers, scoring, sentiment as sent_mod
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.debate")


# ---------------------------------------------------------------------------
# Bull researcher
# ---------------------------------------------------------------------------

def _bull_case(score: dict, peer: dict, cal: dict, sent: dict) -> list[dict[str, Any]]:
    """Extract the strongest long-side arguments with weight + evidence."""
    args: list[dict[str, Any]] = []
    subs = score.get("subscores") or {}

    val = (subs.get("valuation") or {})
    if val.get("score", 0) >= 60:
        args.append({
            "thesis": "Valuation is attractive vs sector",
            "evidence": val.get("notes", []),
            "weight": min(0.30, val["score"] / 200),
            "source": "scoring.valuation",
        })

    qual = (subs.get("quality") or {})
    if qual.get("score", 0) >= 65:
        args.append({
            "thesis": "Quality fundamentals (ROE / margin / leverage)",
            "evidence": qual.get("notes", []),
            "weight": min(0.25, qual["score"] / 250),
            "source": "scoring.quality",
        })

    mom = (subs.get("momentum") or {})
    if mom.get("score", 0) >= 60:
        args.append({
            "thesis": "Momentum and trend alignment",
            "evidence": mom.get("notes", []),
            "weight": min(0.25, mom["score"] / 250),
            "source": "scoring.momentum",
        })

    pr = peer.get("target_relative_to_peers")
    if pr in ("best_in_sector", "above_sector_average"):
        rank = peer.get("target_rank_in_sector")
        total = peer.get("peer_count")
        args.append({
            "thesis": f"Ranked {rank} of {total} in sector — top quartile",
            "evidence": [f"peer_relative={pr}"],
            "weight": 0.20 if pr == "best_in_sector" else 0.12,
            "source": "peers.compare",
        })

    macro = score.get("macro_context") or {}
    if (macro.get("macro_bias") or 0) > 0:
        args.append({
            "thesis": "Macro regime is a tailwind for this sector",
            "evidence": macro.get("reasons", []),
            "weight": 0.10,
            "source": "macro.get_context",
        })

    if sent.get("aggregate_score", 0) >= 0.15:
        args.append({
            "thesis": f"Recent headlines lean {sent.get('label')}",
            "evidence": sent.get("bull_signals", [])[:3],
            "weight": 0.08,
            "source": "sentiment",
        })

    cat_flags = cal.get("catalyst_flags") or []
    upcoming_div = cal.get("ex_dividend_date")
    if upcoming_div:
        args.append({
            "thesis": f"Ex-dividend approaching ({upcoming_div})",
            "evidence": [f"ex_dividend_date={upcoming_div}"],
            "weight": 0.05,
            "source": "calendar",
        })

    args.sort(key=lambda a: a["weight"], reverse=True)
    return args


# ---------------------------------------------------------------------------
# Bear researcher
# ---------------------------------------------------------------------------

def _bear_case(score: dict, peer: dict, cal: dict, sent: dict) -> list[dict[str, Any]]:
    """Extract the strongest short-side / avoid arguments."""
    args: list[dict[str, Any]] = []
    subs = score.get("subscores") or {}

    val = (subs.get("valuation") or {})
    if val.get("score", 100) <= 45:
        args.append({
            "thesis": "Valuation is rich vs sector",
            "evidence": val.get("notes", []),
            "weight": min(0.30, (60 - val["score"]) / 100),
            "source": "scoring.valuation",
        })

    qual = (subs.get("quality") or {})
    if qual.get("score", 100) <= 50:
        args.append({
            "thesis": "Quality is weak (low ROE, thin margin, or high leverage)",
            "evidence": qual.get("notes", []),
            "weight": min(0.25, (60 - qual["score"]) / 120),
            "source": "scoring.quality",
        })

    risk_sub = (subs.get("risk") or {})
    if risk_sub.get("score", 100) <= 50:
        args.append({
            "thesis": "Risk profile is elevated (vol / drawdown)",
            "evidence": risk_sub.get("notes", []),
            "weight": min(0.20, (60 - risk_sub["score"]) / 120),
            "source": "scoring.risk",
        })

    mom = (subs.get("momentum") or {})
    if mom.get("score", 100) <= 45:
        args.append({
            "thesis": "Momentum is broken or trend down",
            "evidence": mom.get("notes", []),
            "weight": min(0.20, (60 - mom["score"]) / 120),
            "source": "scoring.momentum",
        })

    pr = peer.get("target_relative_to_peers")
    if pr in ("worst_in_sector", "below_sector_average"):
        rank = peer.get("target_rank_in_sector")
        total = peer.get("peer_count")
        args.append({
            "thesis": f"Bottom of sector — ranked {rank} of {total}",
            "evidence": [f"peer_relative={pr}"],
            "weight": 0.20 if pr == "worst_in_sector" else 0.12,
            "source": "peers.compare",
        })

    macro = score.get("macro_context") or {}
    if (macro.get("macro_bias") or 0) < 0:
        args.append({
            "thesis": "Macro regime is a headwind for this sector",
            "evidence": macro.get("reasons", []),
            "weight": 0.10,
            "source": "macro.get_context",
        })

    if sent.get("aggregate_score", 0) <= -0.15:
        args.append({
            "thesis": f"Recent headlines lean {sent.get('label')}",
            "evidence": sent.get("bear_signals", [])[:3],
            "weight": 0.08,
            "source": "sentiment",
        })

    if cal.get("blocking"):
        msgs = [f.get("message") for f in (cal.get("catalyst_flags") or [])
                if f.get("severity") in ("high", "medium")]
        args.append({
            "thesis": "Blocking catalyst within window (likely earnings)",
            "evidence": msgs or [f"days_to_earnings={cal.get('days_to_earnings')}"],
            "weight": 0.25,
            "source": "calendar.blocking",
        })

    args.sort(key=lambda a: a["weight"], reverse=True)
    return args


# ---------------------------------------------------------------------------
# Chairman synthesis
# ---------------------------------------------------------------------------

def _chairman_synthesis(
    score: float,
    bull: list[dict],
    bear: list[dict],
    blocking: bool,
    peer_relative: str | None,
) -> dict[str, Any]:
    """Weigh bull vs bear and produce the final read."""
    bull_weight = sum(a["weight"] for a in bull)
    bear_weight = sum(a["weight"] for a in bear)
    edge = bull_weight - bear_weight

    if blocking:
        verdict = "WAIT"
        rationale = (
            "Blocking catalyst within the window overrides directional view. "
            "Wait until the event prints before re-evaluating."
        )
        conviction = "low"
    elif edge >= 0.30 and score >= 70:
        verdict = "BUY"
        conviction = "high"
        rationale = "Bull case clearly outweighs bear case AND composite score is strong."
    elif edge >= 0.15 and score >= 60:
        verdict = "ACCUMULATE"
        conviction = "medium"
        rationale = "Bull case modestly stronger; build position gradually."
    elif edge <= -0.30 and score < 50:
        verdict = "AVOID"
        conviction = "high"
        rationale = (
            "Bear case dominates on monthly view — informational only. "
            "Does not override W1 short-horizon entries; treat as context."
        )
    elif edge <= -0.15:
        verdict = "REDUCE"
        conviction = "medium"
        rationale = (
            "Bear case has the edge on monthly view — informational only. "
            "Does not override W1 short-horizon entries; treat as context."
        )
    else:
        verdict = "HOLD"
        conviction = "low"
        rationale = "Bull and bear cases are roughly balanced."

    if peer_relative == "best_in_sector" and verdict in ("ACCUMULATE", "HOLD"):
        conviction = "high" if conviction == "medium" else "medium"
        rationale += " Best-in-sector relative position adds conviction."

    deciders = []
    for a in (bull[:2] + bear[:2]):
        deciders.append(f"{a['source']}: {a['thesis']} (w={a['weight']:.2f})")

    return {
        "verdict": verdict,
        "conviction": conviction,
        "bull_weight": round(bull_weight, 3),
        "bear_weight": round(bear_weight, 3),
        "edge": round(edge, 3),
        "rationale": rationale,
        "deciding_factors": deciders,
    }


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def debate(user_ticker: str, include_sentiment: bool = True) -> dict[str, Any]:
    """Run the bull/bear/chairman debate for one EGX name.

    Args:
        user_ticker: EGX code or nickname.
        include_sentiment: Pull headline sentiment too. Default True.

    Returns:
        Dict with: ticker, score, bull_case, bear_case, chairman,
        sentiment_summary, as_of.
    """
    canonical, _, name = resolve_ticker(user_ticker)

    score = scoring.score_stock(canonical)
    if "error" in score:
        return {"ticker": canonical, "error": score["error"]}

    try:
        peer = peers.compare(canonical, max_peers=8)
    except Exception as e:
        peer = {"error": str(e)}

    try:
        cal = cal_mod.get_calendar(canonical)
    except Exception as e:
        cal = {"catalyst_flags": [], "blocking": False, "error": str(e)}

    if include_sentiment:
        try:
            sent = sent_mod.analyze_sentiment(canonical, lang="both", limit=10)
        except Exception as e:
            sent = {"error": str(e), "aggregate_score": 0, "label": "unavailable",
                    "bull_signals": [], "bear_signals": []}
    else:
        sent = {"aggregate_score": 0, "label": "skipped",
                "bull_signals": [], "bear_signals": []}

    bull = _bull_case(score, peer, cal, sent)
    bear = _bear_case(score, peer, cal, sent)
    chair = _chairman_synthesis(
        score=score.get("composite_score", 0),
        bull=bull,
        bear=bear,
        blocking=cal.get("blocking", False),
        peer_relative=peer.get("target_relative_to_peers"),
    )

    return {
        "ticker": canonical,
        "name": name,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "composite_score": score.get("composite_score"),
        "bull_case": bull,
        "bear_case": bear,
        "chairman": chair,
        "sentiment_summary": {
            "label": sent.get("label"),
            "aggregate_score": sent.get("aggregate_score"),
            "coverage_pct": sent.get("coverage_pct"),
            "headline_count": sent.get("headline_count"),
        },
        "method": (
            "Three-stage debate: bull researcher and bear researcher each pull "
            "weighted theses from scoring, peers, calendar, sentiment. Chairman "
            "synthesizes net edge and produces verdict + conviction. The host "
            "LLM is encouraged to extend or override the chairman's read."
        ),
    }
