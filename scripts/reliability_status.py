"""Print the fail-closed EGX decision-readiness scorecard.

This report and ``decision.decide`` use the same live V8b evidence gate, so
the status shown to Claude cannot disagree with the status that authorizes a
buy-side response.  Fundamentals coverage remains a separate data-quality
check because it is assessed per issuer at decision time.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import fundamentals as f_mod, reliability
from egx_mcp.data.universe import EGX_UNIVERSE


_CORE = ["pe_ratio", "pb_ratio", "roe_pct", "profit_margin_pct", "debt_to_equity"]


def _confidence(present: int, covered: bool) -> str:
    if not covered or present <= 1:
        return "low"
    return "high" if present >= 4 else "medium"


def _evidence_gate(gate: dict) -> tuple[str, list[str]]:
    checks = gate["checks"]
    lines = [
        f"graded V8b {gate['primary_horizon_days']}d directional calls: "
        f"{gate['directional_calls']} (need >= {reliability.MIN_DIRECTIONAL_CALLS})",
        f"independent briefing dates: {gate['independent_briefing_dates']} "
        f"(need >= {reliability.MIN_INDEPENDENT_DATES})",
        f"directional accuracy vs benchmark: "
        f"{gate['directional_accuracy_pct'] if gate['directional_accuracy_pct'] is not None else 'n/a'}% "
        f"(need >= {reliability.MIN_DIRECTIONAL_ACCURACY_PCT:.0f}%)",
        f"date-weighted signed edge: "
        f"{gate['mean_date_signed_edge_pct'] if gate['mean_date_signed_edge_pct'] is not None else 'n/a'}% "
        f"CI95 {gate['signed_edge_ci95'] or 'n/a'} (CI lower bound must be > 0)",
        f"latest evidence: {gate['latest_evidence_date'] or 'n/a'} "
        f"(age {gate['evidence_age_days'] if gate['evidence_age_days'] is not None else 'n/a'} days; "
        f"max {reliability.MAX_EVIDENCE_AGE_DAYS})",
    ]
    checks_used = ("sample_size", "independent_dates", "directional_accuracy", "positive_signed_edge", "evidence_freshness")
    passed = all(checks[name] for name in checks_used)
    if not passed:
        lines.append("-> buy-side output remains research-only until every evidence check passes.")
    return ("PASS" if passed else "OPEN"), lines


def _cross_section_gate(gate: dict) -> tuple[str, list[str]]:
    """Tier 1 — the whole scored cross-section, not just the directional calls.

    Reaches significance far sooner than per-name hit-rate because it uses
    every scored name per date and judges the basket, which is the unit that
    actually gets traded.
    """
    ic, pf = gate["rank_ic"], gate["portfolio_edge"]

    def _line(label: str, blk: dict, unit: str) -> str:
        if blk["mean"] is None:
            return f"{label}: n/a ({blk['n_dates']} dates — {blk['reason']})"
        return (f"{label}: {blk['mean']:+.4f}{unit} "
                f"CI95 {blk['ci95']} over {blk['n_dates']} dates "
                f"-> {'PASS' if blk['positive'] else 'not yet'}")

    lines = [
        _line("date-wise rank IC (score vs 21d excess)", ic, ""),
        _line(f"top-{reliability.PORTFOLIO_TOP_N} basket excess per date", pf, "%"),
        f"needs >= {reliability.MIN_RANKED_DATES} dates with >= "
        f"{reliability.MIN_NAMES_PER_DATE} scored names, CI lower bound > 0",
    ]
    passed = all(gate["tier1_checks"].values())
    if not passed:
        lines.append("-> satellite sizing stays disabled until both intervals clear zero.")
    return ("PASS" if passed else "OPEN"), lines


def _data_gate() -> tuple[str, list[str]]:
    overrides = f_mod._load_overrides()
    universe = [ticker for ticker, meta in EGX_UNIVERSE.items() if meta.get("sector") != "Index"]
    counts = {"high": 0, "medium": 0, "low": 0}
    for ticker in universe:
        override = overrides.get(ticker)
        present = {field for field in _CORE if override and override.get(field) is not None}
        if override and override.get("trailing_eps") is not None:
            present.add("pe_ratio")
        if override and override.get("book_value_per_share") is not None:
            present.add("pb_ratio")
        counts[_confidence(len(present), override is not None)] += 1
    lines = [f"high={counts['high']}  medium={counts['medium']}  low/ABSTAIN={counts['low']} "
             f"({len(universe)} names)"]
    passed = counts["low"] == 0 and counts["high"] >= counts["medium"]
    if counts["low"]:
        lines.append(f"-> {counts['low']} names will ABSTAIN until audited fundamentals are added.")
    if counts["medium"]:
        lines.append("-> add profit margin and debt/equity fields to promote medium-confidence names.")
    return ("PASS" if passed else "OPEN"), lines


def _calibration_gate(gate: dict) -> tuple[str, list[str]]:
    buckets = gate["calibration"]
    lines = [
        "  ".join(
            f"{name}={info['accuracy_pct'] if info['accuracy_pct'] is not None else 'n/a'}% "
            f"(n={info['n']})" for name, info in buckets.items()
        ),
        f"conviction tracks accuracy (monotone): {gate['checks']['conviction_calibration']}",
    ]
    if not gate["checks"]["conviction_calibration"]:
        lines.append(f"-> need >= {reliability.MIN_BUCKET_CALLS} V8b calls in at least two conviction buckets.")
    return ("PASS" if gate["checks"]["conviction_calibration"] else "OPEN"), lines


def main() -> int:
    gate = reliability.status()
    gates = [
        ("1  EVIDENCE   (proven live edge)", _evidence_gate(gate)),
        ("2  DATA       (fundamentals coverage)", _data_gate()),
        ("3  CALIBRATION (conviction = accuracy)", _calibration_gate(gate)),
        ("4  CROSS-SECTION (tier-1 satellite gate)", _cross_section_gate(gate)),
    ]
    print("=" * 70)
    print("EGX MODEL RELIABILITY SCORECARD")
    print("=" * 70)
    scope = ("this version only" if gate["version_filtered"]
             else "ALL versions — no stamped rows yet")
    print(f"\nmodel version: {gate['model_version']}  (evidence scope: {scope})")
    for title, (status, lines) in gates:
        print(f"\n[{'PASS' if status == 'PASS' else 'OPEN'}]  Gate {title}")
        for line in lines:
            print(f"        {line}")
    open_gates = [title for title, (status, _) in gates if status != "PASS"]
    print("\n" + "=" * 70)
    if open_gates:
        print(f"VERDICT: NOT YET RELIABLE — {len(open_gates)} gate(s) open.")
        print(f"Tier {gate['tier']} — {gate['tier_name']}.")
        if gate["tier"] >= 1:
            print("Cross-sectional edge is established: capped satellite sizing is")
            print("authorized. Full buy-side sizing still needs the directional gates.")
        else:
            print("Research-only: keep the human gate and do not use buy-side sizing.")
    else:
        print("VERDICT: live reliability gates pass. Still use human supervision and re-check after regime change.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
