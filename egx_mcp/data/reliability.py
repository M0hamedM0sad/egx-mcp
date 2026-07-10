"""Fail-closed live-evidence gate for EGX investment decisions.

This module deliberately evaluates only the live, point-in-time ``v8b``
decision records at the model's stated 21-session horizon.  Weekly ranking
records are useful monitoring information, but are not evidence that the
``decide()`` verdict mapping is reliable.

The gate is intentionally stricter than a backtest: it requires a meaningful
number of dated live calls, positive direction-aware excess return, acceptable
directional accuracy, and calibrated conviction buckets.  Until all checks
pass, the MCP is research-only and may not emit an actionable buy-side call.
"""
from __future__ import annotations

import json
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
_BUY = {"BUY", "ACCUMULATE"}
_SELL = {"REDUCE", "AVOID", "SELL"}
_CONVICTION_ORDER = ("high", "medium", "low")


def _load_rows() -> list[dict[str, Any]]:
    if not _GRADED.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _GRADED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("source") == "v8b"
            and row.get("outcome") == "graded"
            and row.get("horizon_days") == PRIMARY_HORIZON_DAYS
            and row.get("verdict") in _BUY | _SELL
            and isinstance(row.get("correct"), bool)
            and isinstance(row.get("excess_pct"), (int, float))
        ):
            rows.append(row)
    return rows


def status() -> dict[str, Any]:
    """Return the decision-readiness gate and its auditable evidence.

    A missing or malformed evidence log always returns ``passed=False``.  This
    is safe for a newly installed MCP: it starts in research-only mode rather
    than pretending a historical backtest proves live reliability.
    """
    rows = _load_rows()
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row.get("briefing_date") or row.get("entry_date") or "unknown")].append(row)

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
    checks = {
        "sample_size": len(rows) >= MIN_DIRECTIONAL_CALLS,
        "independent_dates": len(by_date) >= MIN_INDEPENDENT_DATES,
        "directional_accuracy": accuracy is not None and accuracy >= MIN_DIRECTIONAL_ACCURACY_PCT,
        "positive_signed_edge": mean_date_signed_edge is not None and mean_date_signed_edge > 0,
        "conviction_calibration": calibration_passed,
        "evidence_freshness": evidence_age_days is not None and evidence_age_days <= MAX_EVIDENCE_AGE_DAYS,
    }
    passed = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "passed": passed,
        "mode": "actionable" if passed else "research_only",
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "directional_calls": len(rows),
        "independent_briefing_dates": len(by_date),
        "directional_accuracy_pct": round(accuracy, 1) if accuracy is not None else None,
        "mean_signed_edge_pct": round(mean_signed_edge, 3) if mean_signed_edge is not None else None,
        "mean_date_signed_edge_pct": round(mean_date_signed_edge, 3) if mean_date_signed_edge is not None else None,
        "latest_evidence_date": latest_evidence_date.isoformat() if latest_evidence_date else None,
        "evidence_age_days": evidence_age_days,
        "calibration": calibration,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": {
            "min_directional_calls": MIN_DIRECTIONAL_CALLS,
            "min_independent_dates": MIN_INDEPENDENT_DATES,
            "min_directional_accuracy_pct": MIN_DIRECTIONAL_ACCURACY_PCT,
            "min_bucket_calls": MIN_BUCKET_CALLS,
            "max_evidence_age_days": MAX_EVIDENCE_AGE_DAYS,
        },
        "message": (
            "Live v8b evidence passes all reliability checks."
            if passed else
            "Live v8b evidence is insufficient or unproven; buy-side output is research-only."
        ),
    }
