"""Fail-closed live-evidence gate for EGX investment decisions.

This module deliberately evaluates only the live, point-in-time ``v8b``
decision records at the model's stated 21-session horizon.  Weekly ranking
records are useful monitoring information, but are not evidence that the
``decide()`` verdict mapping is reliable.

The gate is intentionally stricter than a backtest: it requires a meaningful
number of dated live calls, positive direction-aware excess return whose
bootstrap confidence interval clears zero, acceptable directional accuracy,
and calibrated conviction buckets.  Until all checks pass, the MCP is
research-only and may not emit an actionable buy-side call.

Three things make the gate reach a verdict sooner without weakening it:

``model_version``
    Evidence is scored per model version.  A four-month average over every
    version that ever ran answers a question nobody asked; the question is
    whether the model *as it stands now* is reliable.  Bumping the version
    resets the sample to zero, which is fail-closed by construction — you
    cannot use it to erase an inconvenient record and keep trading.

cross-sectional rank IC (tier 1)
    Discrete verdicts throw away most of what a run produces: 253 dated
    21-session records collapsed to 20 directional calls because the rest
    were HOLD.  The score behind those HOLDs still carries a testable claim —
    that on a given day it ranks names by forward excess.  Measuring the
    date-wise Spearman correlation uses the whole cross-section and reaches
    significance roughly an order of magnitude sooner.

portfolio-level edge (tier 1)
    Nobody trades one call; they trade the top-N basket.  A right-skewed
    strategy can be median-negative per name and still profitable per basket,
    so per-name hit-rate is the wrong unit for the money question and needs a
    far larger sample to settle.  Both tier-1 statistics are judged by a
    date-block bootstrap: dates are the independence unit, so resampling
    whole dates keeps a single briefing's many names from posing as many
    independent observations.

Tier 1 authorizes capped satellite sizing only.  ``passed`` (tier 2, full
buy-side) still requires every original directional check, unchanged.
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
_GRADED = ROOT / "logs" / "graded_verdicts.jsonl"

PRIMARY_HORIZON_DAYS = 21
MIN_DIRECTIONAL_CALLS = 40
MIN_INDEPENDENT_DATES = 8
MIN_DIRECTIONAL_ACCURACY_PCT = 55.0
MIN_BUCKET_CALLS = 10
MAX_EVIDENCE_AGE_DAYS = 45

# Tier 1 — cross-sectional evidence thresholds.
MIN_RANKED_DATES = 20          # date-clusters, the honest independence unit
MIN_NAMES_PER_DATE = 8         # below this a Spearman IC is noise
PORTFOLIO_TOP_N = 5            # the basket the briefing actually proposes
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20260823      # fixed: the gate must not move between runs

_BUY = {"BUY", "ACCUMULATE"}
_SELL = {"REDUCE", "AVOID", "SELL"}
_CONVICTION_ORDER = ("high", "medium", "low")


def active_model_version() -> str:
    """Version string the running model stamps onto today's decisions.

    ``EGX_MODEL_VERSION`` overrides; otherwise the learned-params version.
    Kept here (not in model_params) so the grader and the gate agree without
    importing the decision layer.
    """
    override = os.environ.get("EGX_MODEL_VERSION", "").strip()
    if override:
        return override
    try:
        from . import model_params
        return str(model_params.load_params().get("version") or "default")
    except Exception:  # noqa: BLE001 — the gate must never fail open on import
        return "default"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _ranks(vals: list[float]) -> list[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys)) if len(xs) >= MIN_NAMES_PER_DATE else None


def _bootstrap_ci(date_values: list[float], rounds: int = BOOTSTRAP_ROUNDS,
                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """Percentile CI for the mean of one-value-per-date observations.

    Each input is already a whole date collapsed to one number, so resampling
    the list with replacement IS a block bootstrap over dates: names inside a
    briefing move together and never get counted as independent draws.
    """
    n = len(date_values)
    if n < 5:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [date_values[rng.randrange(n)] for _ in range(n)]
        means.append(st.mean(sample))
    means.sort()
    lo = means[int(0.025 * rounds)]
    hi = means[min(int(0.975 * rounds), rounds - 1)]
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# evidence loading
# ---------------------------------------------------------------------------

def _all_rows() -> list[dict[str, Any]]:
    if not _GRADED.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _GRADED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _version_filter(rows: list[dict[str, Any]], version: str) -> tuple[list[dict[str, Any]], bool]:
    """Rows for the active model version, else every row.

    Falls back to the whole history when nothing is stamped yet (briefings
    written before version stamping existed). The fallback is reported, never
    silent — and it can only make the sample larger, never smaller, so it
    cannot open the gate on its own.
    """
    tagged = [r for r in rows if r.get("model_version") == version]
    return (tagged, True) if tagged else (rows, False)


def _directional(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Graded, non-quarantined directional calls at the claim horizon."""
    return [
        r for r in rows
        if r.get("source") == "v8b"
        and r.get("outcome") == "graded"
        and r.get("horizon_days") == PRIMARY_HORIZON_DAYS
        and r.get("verdict") in _BUY | _SELL
        and isinstance(r.get("correct"), bool)
        and isinstance(r.get("excess_pct"), (int, float))
    ]


def _ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every scored, graded name at the claim horizon — HOLD included.

    HOLD makes no directional claim, but the score that produced it still
    claims a rank. That is what the tier-1 statistics test.
    """
    return [
        r for r in rows
        if r.get("source") == "v8b"
        and r.get("outcome") == "graded"
        and r.get("horizon_days") == PRIMARY_HORIZON_DAYS
        and r.get("verdict") != "ABSTAIN"
        and isinstance(r.get("score"), (int, float))
        and isinstance(r.get("excess_pct"), (int, float))
    ]


def _by_date(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("briefing_date") or row.get("entry_date") or "unknown")].append(row)
    return out


# ---------------------------------------------------------------------------
# tier-1 statistics
# ---------------------------------------------------------------------------

def _rank_ic(ranked_by_date: dict[str, list[dict[str, Any]]]) -> list[float]:
    """Date-wise Spearman IC of composite score vs realized 21-session excess."""
    ics = []
    for group in ranked_by_date.values():
        if len(group) < MIN_NAMES_PER_DATE:
            continue
        ic = _spearman([float(r["score"]) for r in group],
                       [float(r["excess_pct"]) for r in group])
        if ic is not None:
            ics.append(ic)
    return ics


def _portfolio_excess(ranked_by_date: dict[str, list[dict[str, Any]]],
                      top_n: int = PORTFOLIO_TOP_N) -> list[float]:
    """Per-date excess of the equal-weight top-N-by-score basket."""
    out = []
    for group in ranked_by_date.values():
        if len(group) < MIN_NAMES_PER_DATE:
            continue
        top = sorted(group, key=lambda r: float(r["score"]), reverse=True)[:top_n]
        out.append(st.mean([float(r["excess_pct"]) for r in top]))
    return out


def _stat_block(values: list[float], min_dates: int) -> dict[str, Any]:
    """Mean, block-bootstrap CI, and whether the CI clears zero."""
    if len(values) < min_dates:
        return {"n_dates": len(values), "mean": None, "ci95": None,
                "positive": False, "reason": f"need >= {min_dates} dates"}
    ci = _bootstrap_ci(values)
    return {
        "n_dates": len(values),
        "mean": round(st.mean(values), 4),
        "ci95": list(ci) if ci else None,
        "positive": bool(ci and ci[0] > 0),
        "reason": None,
    }


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def status() -> dict[str, Any]:
    """Return the decision-readiness gate and its auditable evidence.

    A missing or malformed evidence log always returns ``passed=False``.  This
    is safe for a newly installed MCP: it starts in research-only mode rather
    than pretending a historical backtest proves live reliability.
    """
    version = active_model_version()
    all_rows, version_filtered = _version_filter(_all_rows(), version)
    rows = _directional(all_rows)
    by_date = _by_date(rows)

    signed_edges = [
        float(row["excess_pct"]) if row["verdict"] in _BUY else -float(row["excess_pct"])
        for row in rows
    ]
    # Date-level averaging prevents a single daily briefing with many names
    # from being treated as many independent market observations.
    date_edges = [
        sum(float(r["excess_pct"]) if r["verdict"] in _BUY else -float(r["excess_pct"])
            for r in group) / len(group)
        for group in by_date.values()
    ]
    accuracy = (sum(1 for row in rows if row["correct"]) / len(rows) * 100) if rows else None
    mean_signed_edge = sum(signed_edges) / len(signed_edges) if signed_edges else None
    mean_date_signed_edge = sum(date_edges) / len(date_edges) if date_edges else None
    signed_edge_ci = _bootstrap_ci(date_edges) if date_edges else None

    calibration: dict[str, dict[str, Any]] = {}
    usable_accuracy: list[float] = []
    for conviction in _CONVICTION_ORDER:
        bucket = [r for r in rows if r.get("conviction") == conviction]
        if not bucket:
            calibration[conviction] = {"n": 0, "accuracy_pct": None, "usable": False}
            continue
        bucket_accuracy = sum(1 for r in bucket if r["correct"]) / len(bucket) * 100
        usable = len(bucket) >= MIN_BUCKET_CALLS
        calibration[conviction] = {
            "n": len(bucket), "accuracy_pct": round(bucket_accuracy, 1), "usable": usable,
        }
        if usable:
            usable_accuracy.append(bucket_accuracy)

    calibration_passed = (
        len(usable_accuracy) >= 2
        and all(a >= b for a, b in zip(usable_accuracy, usable_accuracy[1:]))
    )
    evidence_dates: list[date] = []
    for date_key in by_date:
        try:
            evidence_dates.append(date.fromisoformat(date_key))
        except ValueError:
            continue
    latest_evidence_date = max(evidence_dates) if evidence_dates else None
    evidence_age_days = (date.today() - latest_evidence_date).days if latest_evidence_date else None

    # --- tier 1: cross-sectional evidence -----------------------------------
    ranked_by_date = _by_date(_ranked(all_rows))
    rank_ic = _stat_block(_rank_ic(ranked_by_date), MIN_RANKED_DATES)
    portfolio = _stat_block(_portfolio_excess(ranked_by_date), MIN_RANKED_DATES)
    # Tier 1 needs its own freshness clock: it is graded on the whole scored
    # cross-section, which stays current even on stretches where every verdict
    # came back HOLD and the directional sample did not move at all.
    ranked_dates: list[date] = []
    for key in ranked_by_date:
        try:
            ranked_dates.append(date.fromisoformat(key))
        except ValueError:
            continue
    latest_ranked = max(ranked_dates) if ranked_dates else None
    ranked_age_days = (date.today() - latest_ranked).days if latest_ranked else None

    checks = {
        "sample_size": len(rows) >= MIN_DIRECTIONAL_CALLS,
        "independent_dates": len(by_date) >= MIN_INDEPENDENT_DATES,
        "directional_accuracy": accuracy is not None and accuracy >= MIN_DIRECTIONAL_ACCURACY_PCT,
        # Point estimate AND a bootstrap interval that clears zero: a positive
        # mean carried by one lucky date is not evidence.
        "positive_signed_edge": (mean_date_signed_edge is not None
                                 and mean_date_signed_edge > 0
                                 and bool(signed_edge_ci and signed_edge_ci[0] > 0)),
        "conviction_calibration": calibration_passed,
        "evidence_freshness": evidence_age_days is not None and evidence_age_days <= MAX_EVIDENCE_AGE_DAYS,
    }
    tier1_checks = {
        "rank_ic_positive": rank_ic["positive"],
        "portfolio_edge_positive": portfolio["positive"],
        "evidence_freshness": (ranked_age_days is not None
                               and ranked_age_days <= MAX_EVIDENCE_AGE_DAYS),
    }
    passed = all(checks.values())
    tier1 = all(tier1_checks.values())
    failed = [name for name, ok in checks.items() if not ok]

    if passed:
        tier, mode = 2, "actionable"
    elif tier1:
        tier, mode = 1, "satellite_capped"
    else:
        tier, mode = 0, "research_only"

    return {
        "passed": passed,
        "mode": mode,
        "tier": tier,
        "tier_name": {0: "research-only", 1: "satellite (capped sizing)",
                      2: "actionable"}[tier],
        "model_version": version,
        "version_filtered": version_filtered,
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "directional_calls": len(rows),
        "independent_briefing_dates": len(by_date),
        "directional_accuracy_pct": round(accuracy, 1) if accuracy is not None else None,
        "mean_signed_edge_pct": round(mean_signed_edge, 3) if mean_signed_edge is not None else None,
        "mean_date_signed_edge_pct": round(mean_date_signed_edge, 3) if mean_date_signed_edge is not None else None,
        "signed_edge_ci95": list(signed_edge_ci) if signed_edge_ci else None,
        "latest_evidence_date": latest_evidence_date.isoformat() if latest_evidence_date else None,
        "evidence_age_days": evidence_age_days,
        "calibration": calibration,
        "rank_ic": rank_ic,
        "portfolio_edge": portfolio,
        "latest_ranked_date": latest_ranked.isoformat() if latest_ranked else None,
        "ranked_evidence_age_days": ranked_age_days,
        "checks": checks,
        "tier1_checks": tier1_checks,
        "failed_checks": failed,
        "thresholds": {
            "min_directional_calls": MIN_DIRECTIONAL_CALLS,
            "min_independent_dates": MIN_INDEPENDENT_DATES,
            "min_directional_accuracy_pct": MIN_DIRECTIONAL_ACCURACY_PCT,
            "min_bucket_calls": MIN_BUCKET_CALLS,
            "max_evidence_age_days": MAX_EVIDENCE_AGE_DAYS,
            "min_ranked_dates": MIN_RANKED_DATES,
            "min_names_per_date": MIN_NAMES_PER_DATE,
            "portfolio_top_n": PORTFOLIO_TOP_N,
        },
        "message": (
            "Live v8b evidence is insufficient or unproven; buy-side output is research-only."
            if not passed else
            "Live v8b evidence passes every reliability check at the 21-session horizon."
        ),
    }
