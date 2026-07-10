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

from egx_mcp.data import model_params, reliability

ROOT = Path(__file__).parent.parent
_GRADED = ROOT / "logs" / "graded_verdicts.jsonl"
_PROPOSAL = ROOT / "logs" / "learning_proposal.json"

_MIN_SAMPLE = 40        # total graded v8b calls before we dare learn anything
_MIN_TRADES = 8         # min selected names for a threshold's stat to count
_CANDIDATES = list(range(65, 91, 5))   # BUY-threshold candidates (>=ACCUMULATE, keeps order)

# --- score-weight learning ---
_WEIGHT_KEYS = ("valuation", "quality", "momentum", "risk")
_SUB_FIELDS = {k: f"sub_{k}" for k in _WEIGHT_KEYS}
_MIN_SUBSCORE_SAMPLE = 40   # graded v8b rows carrying ALL four sub-scores
_WEIGHT_TILT = 0.5          # how hard to tilt toward positive-corr factors
_WEIGHT_BOUND = (0.5, 1.5)  # candidate weight stays within ±50% of base, per factor
_WEIGHT_MIN_GAIN = 0.05     # OOS ranking-corr improvement required to propose


_PRIMARY_HORIZON = 21   # decide() is a monthly model — learn on its native claim.
                        # The 5d slice is live-inverted (34% hit, p<0.01); learning
                        # from it would tune thresholds against noise.


def _load_v8b() -> list[dict]:
    """Graded decide()-sourced rows with a usable score and excess, at the
    model's claim horizon only."""
    if not _GRADED.exists():
        return []
    rows = []
    for line in _GRADED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if (r.get("source") == "v8b" and r.get("outcome") == "graded"
                and r.get("horizon_days") == _PRIMARY_HORIZON
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


def _subscore_complete(rows: list[dict]) -> list[dict]:
    """Graded rows carrying all four sub-scores AND an excess — usable for reweighting."""
    return [r for r in rows
            if all(isinstance(r.get(f), (int, float)) for f in _SUB_FIELDS.values())
            and isinstance(r.get("excess_pct"), (int, float))]


def _weighted_composite(r: dict, w: dict) -> float:
    return sum(r[_SUB_FIELDS[k]] * w[k] for k in _WEIGHT_KEYS)


def _normalize(w: dict) -> dict:
    s = sum(w.values())
    return {k: round(v / s, 4) for k, v in w.items()} if s > 0 else dict(w)


def _rank_corr(rows: list[dict], w: dict) -> float | None:
    """How well the w-weighted composite ranks realized forward excess."""
    return _corr([_weighted_composite(r, w) for r in rows],
                 [r["excess_pct"] for r in rows])


def _candidate_weights(in_rows: list[dict], base: dict) -> dict:
    """Tilt the base weights toward factors that correlated positively with
    excess in-sample, capped to ±50% per factor and renormalized. One
    conservative candidate (not a free fit) to keep overfitting in check."""
    exc = [r["excess_pct"] for r in in_rows]
    cand = {}
    for k in _WEIGHT_KEYS:
        c = _corr([r[_SUB_FIELDS[k]] for r in in_rows], exc) or 0.0
        factor = max(_WEIGHT_BOUND[0], min(_WEIGHT_BOUND[1], 1 + _WEIGHT_TILT * c))
        cand[k] = base[k] * factor
    return _normalize(cand)


def _learn_weights(rows: list[dict], base: dict) -> dict:
    """Propose an OOS-validated composite reweight, or explain why not.

    Mirrors the threshold loop: fit on a time-ordered IS slice, validate on a
    held-out OOS slice. The metric is rank correlation of the weighted composite
    with realized excess — higher means the score orders winners better."""
    usable = _subscore_complete(rows)
    n = len(usable)
    if n < _MIN_SUBSCORE_SAMPLE:
        return {"status": "insufficient_evidence", "n_with_subscores": n,
                "need": _MIN_SUBSCORE_SAMPLE, "current_weights": base,
                "note": (f"{n} graded calls carry all four sub-scores "
                         f"(need {_MIN_SUBSCORE_SAMPLE}). Newer briefings record them; "
                         "let the daily loop accumulate.")}
    usable.sort(key=lambda r: r.get("entry_date") or "")
    cut = int(n * 0.6)
    in_s, oos = usable[:cut], usable[cut:]
    cand = _candidate_weights(in_s, base)
    oos_base = _rank_corr(oos, base)
    oos_cand = _rank_corr(oos, cand)
    improves = (oos_cand is not None and oos_base is not None
                and len(oos) >= _MIN_TRADES
                and oos_cand >= oos_base + _WEIGHT_MIN_GAIN
                and oos_cand > 0
                and cand != base)
    return {
        "status": "proposal" if improves else "no_change",
        "n_with_subscores": n, "in_sample": len(in_s), "out_of_sample": len(oos),
        "current_weights": base, "candidate_weights": cand,
        "oos_rankcorr_current": oos_base, "oos_rankcorr_candidate": oos_cand,
        "improves": improves,
        "message": (
            f"Reweight {base} -> {cand} ranks held-out excess better "
            f"(corr {oos_cand} vs {oos_base} on {len(oos)} calls)."
            if improves else
            "Current weights hold up — the tilted candidate did not rank "
            "out-of-sample excess meaningfully better."),
    }


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

    gate = reliability.status()
    if not gate["passed"]:
        return {
            "status": "blocked_by_reliability",
            "graded_v8b_calls": n,
            "reliability_gate": gate,
            "recommendation": "KEEP_CURRENT",
            "message": (
                "No learning proposal: the live decision-reliability gate is not passed. "
                "Accumulate independent, directionally positive, calibrated 21-session "
                "v8b evidence before changing model parameters."
            ),
        }

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
    # No viable candidate (scores clustered below the candidate range) is NOT
    # a dead end — the weight lever below learns from the same rows regardless.
    is_scored = [(c, _sel_stats(in_s, c)) for c in _CANDIDATES]
    viable = [(c, s) for c, s in is_scored if s["n"] >= _MIN_TRADES and s["mean_excess_pct"] is not None]
    oos_cur = _sel_stats(oos, cur_buy)
    if viable:
        cand, _ = max(viable, key=lambda x: x[1]["mean_excess_pct"])
        # Validate OOS: learned threshold vs current, on data it wasn't fit on.
        oos_cand = _sel_stats(oos, cand)
        thr_improves = (oos_cand["mean_excess_pct"] is not None and oos_cur["mean_excess_pct"] is not None
                        and oos_cand["n"] >= _MIN_TRADES
                        and oos_cand["mean_excess_pct"] >= oos_cur["mean_excess_pct"]
                        and cand != cur_buy)
    else:
        cand, oos_cand, thr_improves = None, None, False

    # Second lever: composite reweight, learned + OOS-validated independently.
    cur_weights = current.get("score_weights", model_params.DEFAULTS["score_weights"])
    weights_prop = _learn_weights(rows, cur_weights)
    w_improves = weights_prop.get("improves", False)

    any_change = thr_improves or w_improves
    proposed = json.loads(json.dumps(current))  # deep copy
    if thr_improves:
        proposed["verdict_thresholds"]["BUY"] = cand
    if w_improves:
        proposed["score_weights"] = weights_prop["candidate_weights"]
    proposed["version"] = f"learned-{stamp}"
    proposed["learned_at"] = stamp
    changes = []
    if thr_improves:
        changes.append(f"BUY {cur_buy}->{cand}")
    if w_improves:
        changes.append("score_weights tilted")
    proposed["provenance"] = (f"learned from {n} graded v8b calls; "
                              f"{', '.join(changes) or 'no change'}; "
                              f"OOS-validated on {len(oos)} held-out calls")

    msgs = []
    if thr_improves:
        msgs.append(f"BUY threshold {cur_buy}->{cand} beats current out-of-sample "
                    f"({oos_cand['mean_excess_pct']}% vs {oos_cur['mean_excess_pct']}% mean excess "
                    f"on {oos_cand['n']} held-out names).")
    if w_improves:
        msgs.append(weights_prop["message"])
    if not any_change:
        thr_note = (f"candidate {cand} didn't beat it OOS" if cand is not None
                    else "no candidate threshold viable in-sample")
        msgs.append(f"BUY threshold {cur_buy} holds ({thr_note}); "
                    + weights_prop.get("message", "weights unchanged."))

    prop = {
        "status": "proposal" if any_change else "no_change",
        "graded_v8b_calls": n,
        "in_sample": len(in_s), "out_of_sample": len(oos),
        "current_buy_threshold": cur_buy,
        "candidate_buy_threshold": cand,
        "threshold_improves": thr_improves,
        "oos_current": oos_cur,
        "oos_candidate": oos_cand,
        "weights_proposal": weights_prop,
        "conviction_reliability": _conviction_reliability(rows),
        "subscore_signal": _subscore_signal(rows),
        "recommendation": "APPLY" if any_change else "KEEP_CURRENT",
        "proposed_params": proposed if any_change else None,
        "change_tag": _change_tag(thr_improves, cand, w_improves,
                                  weights_prop.get("candidate_weights")),
        "message": " ".join(msgs),
    }
    prop["pr_title"], prop["pr_body"] = _pr_text(prop)
    return prop


def _change_tag(thr_improves: bool, cand: int, w_improves: bool,
                cand_w: dict | None) -> str:
    """Stable branch-name tag keyed on WHAT changes, so the daily workflow
    reuses one PR per distinct candidate and opens a fresh one when it shifts."""
    parts = []
    if thr_improves:
        parts.append(f"buy-{cand}")
    if w_improves and cand_w:
        parts.append("w-" + "-".join(str(int(round(cand_w[k] * 100))) for k in _WEIGHT_KEYS))
    return "_".join(parts) or "none"


def _pr_text(p: dict) -> tuple[str, str]:
    """Precompute the PR title + markdown body so the workflow stays simple."""
    bits = []
    if p["threshold_improves"]:
        bits.append(f"BUY {p['current_buy_threshold']}→{p['candidate_buy_threshold']}")
    if p["weights_proposal"].get("improves"):
        bits.append("reweight")
    title = "Learning loop: " + (" + ".join(bits) or "update") + " (OOS-validated)"

    lines = ["## Learning loop — OOS-validated model update", "", p["message"], ""]
    if p["threshold_improves"]:
        oc, oa = p["oos_current"], p["oos_candidate"]
        lines += [
            "### BUY threshold",
            "| | current | candidate |", "|---|---|---|",
            f"| BUY threshold | {p['current_buy_threshold']} | {p['candidate_buy_threshold']} |",
            f"| OOS mean excess | {oc['mean_excess_pct']}% | {oa['mean_excess_pct']}% |",
            f"| OOS names | {oc['n']} | {oa['n']} |", "",
        ]
    wp = p["weights_proposal"]
    if wp.get("improves"):
        cw, nw = wp["current_weights"], wp["candidate_weights"]
        lines += [
            "### Composite weights",
            "| factor | current | candidate |", "|---|---|---|",
            *[f"| {k} | {cw[k]} | {nw[k]} |" for k in _WEIGHT_KEYS],
            f"| OOS rank-corr w/ excess | {wp['oos_rankcorr_current']} | "
            f"{wp['oos_rankcorr_candidate']} |", "",
        ]
    lines += [
        f"Learned from {p['graded_v8b_calls']} graded v8b calls "
        f"({p['in_sample']} in-sample / {p['out_of_sample']} out-of-sample).", "",
        "**Merging this PR is the human approval gate** — it applies the update to "
        "`model_params.json`, which the decision layer reads on next run. "
        "Close without merging to reject.",
    ]
    return title, "\n".join(lines)


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
    if prop.get("threshold_improves"):
        print(f"Applied: BUY threshold -> {prop['candidate_buy_threshold']} "
              f"(was {prop['current_buy_threshold']}).")
    if prop.get("weights_proposal", {}).get("improves"):
        print(f"Applied: score_weights -> {prop['weights_proposal']['candidate_weights']} "
              f"(was {prop['weights_proposal']['current_weights']}).")
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
