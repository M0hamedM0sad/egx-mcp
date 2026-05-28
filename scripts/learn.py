"""Learning loop — turn graded outcomes into a human-approved model update.

The SAFE form of "the model enhances over time." It does NOT retrain itself
silently. It:
  1. reads the accumulated evidence (logs/graded_verdicts.jsonl),
  2. learns a data-driven BUY score-threshold from the score→excess relationship
     on an IN-SAMPLE slice,
  3. VALIDATES it out-of-sample (does the learned threshold beat the current
     one on data it wasn't fit on?),
  4. writes a PROPOSAL (logs/learning_proposal.json) with the evidence —
     and changes nothing.

You review the proposal; only `--apply` writes the new threshold into
model_params.json, where the decision layer picks it up. Guardrails: needs a
minimum graded sample, only proposes if OOS-validated, keeps verdict
thresholds monotone, and refuses to apply without your flag.

    python -m scripts.learn              # analyze + write proposal (no change)
    python -m scripts.learn --apply       # apply the current proposal (after review)

Nothing meaningful happens until the daily grading has accumulated enough
graded calls — by design. With thin data it reports "insufficient evidence".
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import model_params

ROOT = Path(__file__).parent.parent
_GRADED = ROOT / "logs" / "graded_verdicts.jsonl"
_PROPOSAL = ROOT / "logs" / "learning_proposal.json"

_MIN_SAMPLE = 40        # total graded v8b calls before we dare learn anything
_MIN_TRADES = 8         # min selected names for a threshold's stat to count
_CANDIDATES = list(range(65, 91, 5))   # BUY-threshold candidates (>=ACCUMULATE, keeps order)


def _load_v8b() -> list[dict]:
    """Graded decide()-sourced rows with a usable score and excess."""
    if not _GRADED.exists():
        return []
    rows = []
    for line in _GRADED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if (r.get("source") == "v8b" and r.get("outcome") == "graded"
                and isinstance(r.get("score"), (int, float))
                and isinstance(r.get("excess_pct"), (int, float))):
            rows.append(r)
    return rows


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _sel_stats(rows: list[dict], thr: float) -> dict:
    sel = [r for r in rows if r["score"] >= thr]
    exc = [r["excess_pct"] for r in sel]
    return {"threshold": thr, "n": len(sel),
            "mean_excess_pct": round(_mean(exc), 2) if exc else None,
            "beat_bench_pct": round(sum(1 for e in exc if e > 0) / len(exc) * 100, 1) if exc else None}


def _corr(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, no numpy dependency."""
    n = len(xs)
    if n < 8:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return round(cov / (vx ** 0.5 * vy ** 0.5), 3)


def _subscore_signal(rows: list[dict]) -> dict:
    """Correlation of each sub-score with realized forward excess.

    Informational groundwork for weight-learning: a factor that correlates
    positively with excess deserves more weight, one near zero or negative
    less. NOT auto-applied — the scoring weights aren't tunable yet, and
    reweighting needs its own OOS validation to avoid overfitting."""
    fields = ["sub_valuation", "sub_quality", "sub_momentum", "sub_risk"]
    usable = [r for r in rows if all(isinstance(r.get(f), (int, float)) for f in fields)
              and isinstance(r.get("excess_pct"), (int, float))]
    if len(usable) < 8:
        return {"n": len(usable), "note": "need >=8 decisions with recorded sub-scores"}
    exc = [r["excess_pct"] for r in usable]
    return {"n": len(usable),
            "corr_with_excess": {f.replace("sub_", ""): _corr([r[f] for r in usable], exc)
                                 for f in fields},
            "note": ("Higher positive correlation = the factor predicted out-performance "
                     "and arguably deserves more weight. Groundwork only — weights are "
                     "not auto-tuned.")}


def _conviction_reliability(rows: list[dict]) -> dict:
    out = {}
    for c in ("high", "medium", "low", "weekly"):
        sub = [r for r in rows if (r.get("conviction") or "") == c and r.get("correct") is not None]
        if sub:
            out[c] = {"n": len(sub),
                      "accuracy_pct": round(sum(1 for r in sub if r["correct"]) / len(sub) * 100, 1)}
    return out


def _build_proposal() -> dict:
    rows = _load_v8b()
    current = model_params.load_params()
    cur_buy = current["verdict_thresholds"]["BUY"]
    n = len(rows)

    if n < _MIN_SAMPLE:
        return {"status": "insufficient_evidence", "graded_v8b_calls": n,
                "need": _MIN_SAMPLE,
                "message": (f"Only {n} graded decide() calls (need {_MIN_SAMPLE}). "
                            "Let the daily grading accumulate — no proposal made.")}

    rows.sort(key=lambda r: r.get("entry_date") or "")
    cut = int(n * 0.6)
    in_s, oos = rows[:cut], rows[cut:]
    stamp = rows[-1].get("entry_date")

    # Learn on IS: candidate BUY threshold maximizing in-sample mean excess.
    is_scored = [(c, _sel_stats(in_s, c)) for c in _CANDIDATES]
    viable = [(c, s) for c, s in is_scored if s["n"] >= _MIN_TRADES and s["mean_excess_pct"] is not None]
    if not viable:
        return {"status": "insufficient_evidence", "graded_v8b_calls": n,
                "message": "Not enough names clear the candidate thresholds in-sample."}
    cand, _ = max(viable, key=lambda x: x[1]["mean_excess_pct"])

    # Validate OOS: learned threshold vs current, on data it wasn't fit on.
    oos_cand = _sel_stats(oos, cand)
    oos_cur = _sel_stats(oos, cur_buy)
    improves = (oos_cand["mean_excess_pct"] is not None and oos_cur["mean_excess_pct"] is not None
                and oos_cand["n"] >= _MIN_TRADES
                and oos_cand["mean_excess_pct"] >= oos_cur["mean_excess_pct"]
                and cand != cur_buy)

    proposed = json.loads(json.dumps(current))  # deep copy
    proposed["verdict_thresholds"]["BUY"] = cand
    proposed["version"] = f"learned-{stamp}"
    proposed["learned_at"] = stamp
    proposed["provenance"] = (f"learned from {n} graded v8b calls; BUY {cur_buy}->{cand}, "
                              f"OOS-validated on {len(oos)} held-out calls")

    return {
        "status": "proposal" if improves else "no_change",
        "graded_v8b_calls": n,
        "in_sample": len(in_s), "out_of_sample": len(oos),
        "current_buy_threshold": cur_buy,
        "candidate_buy_threshold": cand,
        "oos_current": oos_cur,
        "oos_candidate": oos_cand,
        "conviction_reliability": _conviction_reliability(rows),
        "subscore_signal": _subscore_signal(rows),
        "recommendation": "APPLY" if improves else "KEEP_CURRENT",
        "proposed_params": proposed if improves else None,
        "message": (
            f"Learned BUY threshold {cand} beats current {cur_buy} out-of-sample "
            f"({oos_cand['mean_excess_pct']}% vs {oos_cur['mean_excess_pct']}% mean excess "
            f"on {oos_cand['n']} held-out names). Review, then apply."
            if improves else
            f"Current BUY threshold {cur_buy} holds up — learned candidate {cand} did not "
            "beat it out-of-sample. No change recommended."),
    }


def cmd_analyze():
    prop = _build_proposal()
    _PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    _PROPOSAL.write_text(json.dumps(prop, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in prop.items() if k != "proposed_params"},
                     ensure_ascii=False, indent=2))
    print(f"\nProposal written -> {_PROPOSAL}")
    if prop["status"] == "proposal":
        print("Review it, then apply with:  python -m scripts.learn --apply")
    return 0


def cmd_apply():
    if not _PROPOSAL.exists():
        print("No proposal found. Run `python -m scripts.learn` first.")
        return 1
    prop = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    if prop.get("status") != "proposal" or not prop.get("proposed_params"):
        print(f"Nothing to apply (status: {prop.get('status')}). "
              f"{prop.get('message', '')}")
        return 1
    model_params.save_params(prop["proposed_params"])
    print(f"Applied: BUY threshold -> {prop['candidate_buy_threshold']} "
          f"(was {prop['current_buy_threshold']}).")
    print(f"Written to {model_params._PARAMS_FILE}. The decision layer now uses it.")
    print("Re-run grading/backtests to confirm the change holds going forward.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Human-gated learning loop.")
    ap.add_argument("--apply", action="store_true",
                    help="Apply the current proposal to model_params.json (after review).")
    args = ap.parse_args()
    return cmd_apply() if args.apply else cmd_analyze()


if __name__ == "__main__":
    sys.exit(main())
