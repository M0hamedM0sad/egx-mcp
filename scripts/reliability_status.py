"""Reliability scorecard — are we there yet? One command, three gates.

Reads what the daily pipeline produces (no network) and reports how close the
model is to being relied on:

  Gate 1  EVIDENCE     — enough graded verdicts, and do they beat the index?
                         (from logs/graded_verdicts.jsonl, grown by the daily
                          briefing via tests/grade_briefings)
  Gate 2  DATA         — fundamentals coverage / confidence across the universe
                         (from egx_fundamentals_audited.csv)
  Gate 3  CALIBRATION  — does higher conviction mean higher accuracy?
                         (needs enough graded calls per conviction bucket)

    python -m scripts.reliability_status

This does not make the model reliable — it tells you, honestly, which gates
are still open and by how much. The gates close by RUNNING the system
(accumulating briefings) and FILLING data, not by more code.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import fundamentals as f_mod
from egx_mcp.data.universe import EGX_UNIVERSE

ROOT = Path(__file__).parent.parent
_GRADED = ROOT / "logs" / "graded_verdicts.jsonl"
_AUDITED = ROOT / "egx_fundamentals_audited.csv"
_CORE = ["pe_ratio", "pb_ratio", "roe_pct", "profit_margin_pct", "debt_to_equity"]

_MIN_CALLS = 30          # graded directional calls before stats are meaningful
_MIN_ACC = 55.0          # directional accuracy vs benchmark to claim edge
_MIN_BUCKET = 10         # graded calls per conviction bucket for calibration
_BUY = {"BUY", "ACCUMULATE", "WEEKLY_BUY"}
_SELL = {"REDUCE", "AVOID", "SELL"}


def _load_graded() -> list[dict]:
    if not _GRADED.exists():
        return []
    rows = []
    for line in _GRADED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _gate1(rows: list[dict]) -> tuple[str, list[str]]:
    graded = [r for r in rows if r.get("outcome") == "graded" and r.get("correct") is not None]
    pending = [r for r in rows if r.get("outcome") == "pending"]
    lines = [f"graded directional calls: {len(graded)} (need {_MIN_CALLS})  |  pending: {len(pending)}"]
    if not graded:
        return "OPEN", lines + ["No graded calls yet — run the briefing over time to accumulate."]
    acc = sum(1 for r in graded if r["correct"]) / len(graded) * 100
    exc = [r["excess_pct"] for r in graded if r.get("excess_pct") is not None]
    mean_exc = sum(exc) / len(exc) if exc else None
    lines.append(f"directional accuracy vs benchmark: {acc:.0f}% (need >={_MIN_ACC:.0f}%)")
    if mean_exc is not None:
        lines.append(f"mean excess vs benchmark: {mean_exc:+.2f}%")
    else:
        lines.append("(!) no excess returns recorded — calls were graded vs 0%, "
                     "re-run tests/grade_briefings to grade vs the basket benchmark")
    status = "PASS" if (len(graded) >= _MIN_CALLS and acc >= _MIN_ACC) else (
        "EMERGING" if len(graded) >= _MIN_CALLS else "OPEN")
    if len(graded) < _MIN_CALLS:
        lines.append(f"-> need {_MIN_CALLS - len(graded)} more graded calls before this means anything.")
    return status, lines


def _confidence(present: int, covered: bool) -> str:
    if not covered or present <= 1:
        return "low"
    return "high" if present >= 4 else "medium"


def _gate2() -> tuple[str, list[str]]:
    overrides = f_mod._load_overrides()
    universe = [t for t, m in EGX_UNIVERSE.items() if m.get("sector") != "Index"]
    counts = {"high": 0, "medium": 0, "low": 0}
    for tk in universe:
        ov = overrides.get(tk)
        have = {c for c in _CORE if ov and ov.get(c) is not None}
        if ov:
            if ov.get("trailing_eps") is not None:
                have.add("pe_ratio")
            if ov.get("book_value_per_share") is not None:
                have.add("pb_ratio")
        counts[_confidence(len(have), ov is not None)] += 1
    lines = [f"high={counts['high']}  medium={counts['medium']}  low/ABSTAIN={counts['low']}  "
             f"({len(universe)} names)"]
    status = "PASS" if counts["low"] == 0 and counts["high"] >= counts["medium"] else (
        "EMERGING" if counts["low"] <= 5 else "OPEN")
    if counts["low"]:
        lines.append(f"-> {counts['low']} names will ABSTAIN until fundamentals are added.")
    if counts["medium"]:
        lines.append(f"-> {counts['medium']} at medium; add profit_margin + debt_to_equity to reach high.")
    return status, lines


def _gate3(rows: list[dict]) -> tuple[str, list[str]]:
    graded = [r for r in rows if r.get("outcome") == "graded" and r.get("correct") is not None]
    buckets: dict[str, list[dict]] = {}
    for r in graded:
        buckets.setdefault(r.get("conviction") or "unknown", []).append(r)
    usable = {k: v for k, v in buckets.items() if len(v) >= _MIN_BUCKET
              and k in ("high", "medium", "low")}
    if len(usable) < 2:
        return "OPEN", [f"need >={_MIN_BUCKET} graded calls in >=2 conviction buckets "
                        f"(have: {{ {', '.join(f'{k}:{len(v)}' for k, v in buckets.items())} }})"]
    order = [c for c in ("high", "medium", "low") if c in usable]
    accs = [(c, sum(1 for r in usable[c] if r["correct"]) / len(usable[c]) * 100) for c in order]
    seq = [a for _, a in accs]
    monotone = all(x >= y for x, y in zip(seq, seq[1:]))
    lines = ["  ".join(f"{c}={a:.0f}%" for c, a in accs),
             f"conviction tracks accuracy (monotone): {monotone}"]
    return ("PASS" if monotone else "OPEN"), lines


def main() -> int:
    rows = _load_graded()
    gates = [
        ("1  EVIDENCE   (proven edge)", _gate1(rows)),
        ("2  DATA       (fundamentals coverage)", _gate2()),
        ("3  CALIBRATION(conviction = accuracy)", _gate3(rows)),
    ]
    print("=" * 70)
    print("EGX MODEL RELIABILITY SCORECARD")
    print("=" * 70)
    badge = {"PASS": "[PASS]", "EMERGING": "[~~~~]", "OPEN": "[OPEN]"}
    for title, (status, lines) in gates:
        print(f"\n{badge.get(status, '[????]')}  Gate {title}")
        for ln in lines:
            print(f"        {ln}")
    open_gates = [t for t, (s, _) in gates if s != "PASS"]
    print("\n" + "=" * 70)
    if open_gates:
        print(f"VERDICT: NOT YET RELIABLE — {len(open_gates)} gate(s) open.")
        print("Decision-support only: keep the human gate, stops, and size against conviction.")
    else:
        print("VERDICT: all gates PASS. Reliable WITHIN a risk-managed, human-supervised")
        print("process — never unattended. Re-check after any regime change.")
    print("=" * 70)
    print("\nClose the gates by: running the daily briefing (Gate 1), running the IR")
    print("pipeline to fill fundamentals (Gate 2), and accumulating graded calls (Gate 3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
