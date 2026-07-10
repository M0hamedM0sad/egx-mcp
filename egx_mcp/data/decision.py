"""Decision synthesizer — the headline tool.

Combines scoring, peer comparison, calendar, sizing, and macro into a
single auditable research verdict. Buy-side output is actionable only after
the live-evidence reliability gate passes:

    BUY        score ≥ 75 + macro tailwind + no blocking catalyst
    ACCUMULATE score 65–74 OR (score ≥ 75 with mild headwind)
    HOLD       score 50–64
    REDUCE     score 35–49
    AVOID      score < 35 OR blocking catalyst (e.g. earnings within 7d)

Output is structured: verdict, conviction, score, fair value estimate
(P/E-based), entry/stop/target, suggested position size, and the full
reasoning trail. Every number is sourced — no hidden inputs.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from . import calendar as cal_mod
from . import model_params, peers, reliability, scoring, sizing
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.decision")


def _verdict_from_score(score: float, blocking: bool, peer_relative: str | None) -> tuple[str, str]:
    """Map composite score → verdict + conviction.

    Blocking catalysts (earnings within 7 days) downgrade BUY → HOLD.
    Best-in-sector adds conviction; worst-in-sector removes it.
    """
    th = model_params.thresholds()  # learnable, human-approved (see scripts/learn.py)

    if blocking and score >= th["HOLD"]:
        return "HOLD", "low"

    if score >= th["BUY"]:
        verdict = "BUY"
        conviction = "high" if peer_relative in ("best_in_sector", "above_sector_average") else "medium"
    elif score >= th["ACCUMULATE"]:
        verdict = "ACCUMULATE"
        conviction = "medium"
    elif score >= th["HOLD"]:
        verdict = "HOLD"
        conviction = "low" if peer_relative == "worst_in_sector" else "medium"
    elif score >= th["REDUCE"]:
        verdict = "REDUCE"
        conviction = "medium"
    else:
        verdict = "AVOID"
        conviction = "high"

    return verdict, conviction


def _fair_value(price: float | None, pe: float | None, sector_med_pe: float | None,
                eps: float | None) -> dict[str, Any]:
    """Simple comparable-multiple fair value: EPS × sector median P/E.

    Capped at ±50% from current price to avoid garbage-in-garbage-out from
    bad EPS or extreme medians.
    """
    if not (price and pe and sector_med_pe and eps and eps > 0):
        return {"fair_value": None, "upside_pct": None, "method": "insufficient inputs"}

    fv_raw = eps * sector_med_pe
    # Cap to a sane range
    fv = max(price * 0.5, min(price * 1.5, fv_raw))
    upside = (fv / price - 1) * 100
    return {
        "fair_value": round(fv, 2),
        "upside_pct": round(upside, 1),
        "method": f"EPS {eps} × sector median P/E {sector_med_pe} = {round(fv_raw, 2)} (capped ±50%)",
        "uncapped_fair_value": round(fv_raw, 2),
        "was_capped": fv != fv_raw,
    }


# ---------------------------------------------------------------------------
# Confidence, cost, and actionability layer
#
# Turns the raw score→verdict mapping into a decision you can rely on:
#   #1 data confidence — how much of the verdict rests on audited vs scraped
#      vs missing fundamentals;
#   #2 conviction that means something — capped so it can't exceed what the
#      data supports, and flagged when signals conflict or the score sits on
#      a band boundary;
#   #3 abstention — an explicit "no clear edge" when inputs can't be trusted;
#   #4 net-of-cost edge — upside after an assumed round-trip cost, so a
#      verdict reflects net edge, not gross.
# ---------------------------------------------------------------------------

_CORE_FUND_FIELDS = ["pe_ratio", "pb_ratio", "roe_pct", "profit_margin_pct", "debt_to_equity"]
_CONV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_RANK_CONV = {0: "none", 1: "low", 2: "medium", 3: "high"}
_BAND_EDGES = (35, 50, 65, 75)   # verdict thresholds in _verdict_from_score
_BORDERLINE = 2.0                # score within this of an edge = borderline
_CONFLICT_SPREAD = 45            # sub-score range above this = split signal


def _data_confidence(f_raw: dict[str, Any]) -> dict[str, Any]:
    """How much can we trust the fundamentals feeding this verdict?

    high   audited override + most core fields present
    medium Yahoo with enough core fields, or audited but sparse
    low    fundamentals errored, missing, or unverified-and-sparse
    """
    if "error" in f_raw:
        return {"level": "low", "fundamentals_source": None, "fields_present": "0/5",
                "missing_fields": list(_CORE_FUND_FIELDS),
                "notes": [f"Fundamentals unavailable: {f_raw.get('error', 'unknown')}"]}

    present = [k for k in _CORE_FUND_FIELDS if f_raw.get(k) is not None]
    missing = [k for k in _CORE_FUND_FIELDS if f_raw.get(k) is None]
    src = f_raw.get("fundamentals_source")
    n = len(present)
    notes: list[str] = []

    if f_raw.get("pe_was_corrected"):
        notes.append("Yahoo P/E looked bogus and was recomputed from price ÷ EPS.")

    if src == "audited_override":
        notes.append("Fundamentals from audited override CSV.")
        level = "high" if n >= 4 else "medium"
    elif src == "yahoo":
        notes.append("Fundamentals from Yahoo — unverified for EGX, cross-check before sizing up.")
        level = "medium" if n >= 3 else "low"
    else:
        level = "low"
    if n <= 1:
        level = "low"

    return {"level": level, "fundamentals_source": src,
            "fields_present": f"{n}/{len(_CORE_FUND_FIELDS)}",
            "missing_fields": missing, "notes": notes}


def _assess(verdict: str, conviction: str, score_val: float,
            subscores: dict[str, float], upside_pct: float | None,
            dconf: dict[str, Any], reliability_gate: dict[str, Any], round_trip_cost_pct: float,
            min_net_edge_pct: float) -> dict[str, Any]:
    """Apply confidence cap, cost-adjusted edge, and abstention rules.

    Returns the final verdict/conviction plus the reasoning for any change.
    Cascade:
      A. Low data confidence  -> ABSTAIN (can't trust the inputs).
      B. Buy-side but net upside below the cost threshold -> downgrade to HOLD
         (the honest call is "don't buy", not "buy a sub-cost edge").
      C. Borderline score / split sub-scores -> keep verdict, cap conviction,
         flag the fragility.
    Conviction is always capped to what the data confidence supports.
    """
    cautions: list[str] = []
    abstain_reasons: list[str] = []
    final_verdict, final_conv = verdict, conviction

    net_upside = round(upside_pct - round_trip_cost_pct, 1) if upside_pct is not None else None

    # Cap conviction by data confidence — can't be "high" on "low" data.
    cap_rank = {"high": 3, "medium": 2, "low": 1}[dconf["level"]]
    if _CONV_RANK[final_conv] > cap_rank:
        new = _RANK_CONV[cap_rank]
        cautions.append(f"Conviction capped {final_conv}→{new}: {dconf['level']} data confidence.")
        final_conv = new

    # Borderline score — small input changes could flip the verdict.
    near = [e for e in _BAND_EDGES if abs(score_val - e) <= _BORDERLINE]
    if near:
        cautions.append(f"Score {score_val} is within {_BORDERLINE} of a verdict boundary "
                        f"{near} — fragile, small input changes could flip it.")
        if _CONV_RANK[final_conv] > 2:
            final_conv = "medium"

    # Split signal — composite hides disagreement across sub-scores.
    if subscores:
        vals = list(subscores.values())
        if max(vals) - min(vals) >= _CONFLICT_SPREAD:
            cautions.append(f"Sub-scores disagree sharply (range {min(vals)}–{max(vals)}); "
                            "the composite averages a split signal.")
            if _CONV_RANK[final_conv] > 2:
                final_conv = "medium"

    # A. Low data confidence -> abstain.
    if dconf["level"] == "low":
        abstain_reasons.append("Low data confidence — fundamentals missing or unverified; "
                               "no reliable valuation possible.")
        final_verdict, final_conv = "ABSTAIN", "none"

    # B. The model must have demonstrated live, independent edge before it can
    # issue an actionable buy-side call.  A good-looking single-name score is
    # not a substitute for validation of the decision process itself.
    elif verdict in ("BUY", "ACCUMULATE") and not reliability_gate.get("passed", False):
        abstain_reasons.append(
            "Model reliability gate is not passed; buy-side output is research-only until "
            "live 21-session evidence, independent dates, signed edge, and conviction "
            "calibration all pass."
        )
        final_verdict, final_conv = "ABSTAIN", "none"

    # C. Buy-side edge must clear costs.
    elif verdict in ("BUY", "ACCUMULATE"):
        if net_upside is not None and net_upside < min_net_edge_pct:
            cautions.append(f"Cost-adjusted upside {net_upside}% < required {min_net_edge_pct}% "
                            f"(gross {upside_pct}% − {round_trip_cost_pct}% round-trip cost) "
                            f"→ downgraded {verdict}→HOLD.")
            final_verdict, final_conv = "HOLD", "low"

    return {
        "verdict": final_verdict,
        "conviction": final_conv,
        "provisional_verdict": verdict if final_verdict != verdict else None,
        "actionable": final_verdict not in ("ABSTAIN",),
        "abstain_reasons": abstain_reasons,
        "cautions": cautions,
        "gross_upside_pct": upside_pct,
        "round_trip_cost_pct_assumed": round_trip_cost_pct,
        "expected_net_upside_pct": net_upside,
    }


def decide(
    user_ticker: str,
    portfolio_value_egp: float | None = None,
    risk_pct: float = 1.0,
    round_trip_cost_pct: float = 1.0,
    min_net_edge_pct: float = 0.0,
) -> dict[str, Any]:
    """Return a complete decision package: verdict, levels, sizing, rationale.

    round_trip_cost_pct: assumed total cost of a round trip (commission +
        fees + tax), used to compute net-of-cost edge. Default 1.0% — set it
        to your broker's actual rate. A buy-side verdict whose cost-adjusted
        upside falls below `min_net_edge_pct` is downgraded to HOLD.
    min_net_edge_pct: minimum net upside (after cost) required to keep a
        buy-side verdict. Default 0.0 (edge must merely clear costs).
    """
    canonical, _, _ = resolve_ticker(user_ticker)

    # 1. Composite score
    score = scoring.score_stock(canonical)
    if "error" in score:
        return {"ticker": canonical, "error": score["error"]}

    # 2. Calendar / catalysts
    try:
        cal = cal_mod.get_calendar(canonical)
    except Exception as e:
        cal = {"catalyst_flags": [], "blocking": False, "error": str(e)}

    # 3. Peer relative
    try:
        peer = peers.compare(canonical, max_peers=8)
        peer_relative = peer.get("target_relative_to_peers")
    except Exception as e:
        peer = {"error": str(e)}
        peer_relative = None

    # 4. Base verdict from score
    base_verdict, base_conviction = _verdict_from_score(
        score["composite_score"], cal.get("blocking", False), peer_relative
    )

    # 5. Fair value — needs EPS, pull from cached fundamentals
    from .fundamentals import get_fundamentals
    f_raw = get_fundamentals(canonical)
    snap = score.get("fundamentals_snapshot") or {}
    fv = _fair_value(
        price=score.get("price"),
        pe=snap.get("pe"),
        sector_med_pe=(score.get("sector_medians") or {}).get("median_pe"),
        eps=f_raw.get("trailing_eps"),
    )

    # 5b. Confidence / cost / abstention layer — the reliability gate.
    subscores_flat = {k: v["score"] for k, v in (score.get("subscores") or {}).items()}
    dconf = _data_confidence(f_raw)
    reliability_gate = reliability.status()
    assessment = _assess(
        base_verdict, base_conviction, score["composite_score"], subscores_flat,
        fv.get("upside_pct"), dconf, reliability_gate, round_trip_cost_pct, min_net_edge_pct,
    )
    verdict = assessment["verdict"]
    conviction = assessment["conviction"]

    # 6. Position sizing (only if portfolio value provided AND verdict is buy-side)
    sizing_payload = None
    if portfolio_value_egp and verdict in ("BUY", "ACCUMULATE"):
        try:
            sizing_payload = sizing.position_size(
                canonical,
                portfolio_value_egp=portfolio_value_egp,
                risk_pct=risk_pct,
            )
        except Exception as e:
            sizing_payload = {"error": str(e)}

    # 7. Key drivers and risks — extract from sub-scores
    drivers = []
    risks = []
    for category, sub in (score.get("subscores") or {}).items():
        for note in sub.get("notes", []):
            # Heuristic: notes that imply weakness go to risks, others to drivers
            lower = note.lower()
            if any(k in lower for k in ("weak", "expensive", "extreme", "elevated",
                                         "overbought", "deep", "negative", "broken",
                                         "highly levered", "thin", "death-cross",
                                         "above sector median", "≥ 150%", "≥ 1.5", "soft")):
                risks.append(f"[{category}] {note}")
            else:
                drivers.append(f"[{category}] {note}")

    # Add macro and catalyst notes
    macro = score.get("macro_context") or {}
    for r in macro.get("reasons", []):
        if macro.get("macro_bias", 0) >= 0:
            drivers.append(f"[macro] {r}")
        else:
            risks.append(f"[macro] {r}")

    blocking_catalysts = [f["message"] for f in cal.get("catalyst_flags", [])
                          if f.get("severity") in ("high", "medium")]

    return {
        "ticker": canonical,
        "name": score.get("name"),
        "sector": score.get("sector"),
        "as_of": datetime.utcnow().isoformat() + "Z",
        "price": score.get("price"),

        "verdict": verdict,
        "conviction": conviction,
        "actionable": assessment["actionable"],
        "provisional_verdict": assessment["provisional_verdict"],
        "abstain_reasons": assessment["abstain_reasons"],
        "assessment_cautions": assessment["cautions"],
        "composite_score": score["composite_score"],

        "data_confidence": dconf["level"],
        "data_confidence_detail": dconf,
        "reliability_gate": reliability_gate,
        "conviction_note": (
            "Conviction is capped by data confidence and flagged when signals "
            "conflict or the score is borderline. Validate that conviction "
            "tracks realized accuracy with tests/calibration_report.py before "
            "sizing against it."
        ),

        "fair_value_estimate": fv.get("fair_value"),
        "upside_pct": fv.get("upside_pct"),
        "gross_upside_pct": assessment["gross_upside_pct"],
        "expected_net_upside_pct": assessment["expected_net_upside_pct"],
        "round_trip_cost_pct_assumed": assessment["round_trip_cost_pct_assumed"],
        "fair_value_method": fv.get("method"),

        "suggested_levels": sizing_payload and {
            "entry": sizing_payload.get("price"),
            "stop_loss": sizing_payload.get("stop_loss_price"),
            "target": sizing_payload.get("target_price"),
            "shares": sizing_payload.get("shares"),
            "position_cost_egp": sizing_payload.get("position_cost_egp"),
            "position_weight_pct": sizing_payload.get("position_weight_pct"),
            "reward_to_risk": sizing_payload.get("reward_to_risk"),
        } or None,

        "key_drivers": drivers,
        "key_risks": risks,
        "blocking_catalysts": blocking_catalysts,

        "subscores": {
            "valuation": score["subscores"]["valuation"]["score"],
            "quality": score["subscores"]["quality"]["score"],
            "momentum": score["subscores"]["momentum"]["score"],
            "risk": score["subscores"]["risk"]["score"],
        },
        "peer_relative": peer_relative,
        "peer_rank_in_sector": peer.get("target_rank_in_sector"),
        "macro_bias": macro.get("macro_bias"),

        "next_earnings_date": cal.get("next_earnings_date"),
        "days_to_earnings": cal.get("days_to_earnings"),
        "ex_dividend_date": cal.get("ex_dividend_date"),

        "data_quality_notes": [
            n for n in [
                *dconf.get("notes", []),
                f"Fundamental coverage: {dconf['fields_present']} core fields"
                + (f" (missing: {', '.join(dconf['missing_fields'])})" if dconf.get("missing_fields") else ""),
                "Quote is ~15-min delayed (Yahoo).",
                f"Sector medians built from {(score.get('sector_medians') or {}).get('peer_count', 0)} peers in curated universe.",
            ] if n
        ],
        "disclaimer": (
            "Algorithmic verdict for research only. Not investment advice. "
            "Verify against EGX official tape, audited disclosures, and your "
            "own risk parameters before acting."
        ),
    }
