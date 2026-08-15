"""Learn model parameters from the market-data panel — two fitters, compared.

Replaces the evidence source of the old loop (verdicts it had already emitted
and graded, ~20 usable rows) with the reconstructed panel from
`scripts.build_panel`. Same output contract as `scripts.learn`: an OOS-validated
PROPOSAL that changes nothing until a human applies it.

Two fitters run on the identical folds so their numbers are comparable:

  tilt  — the incumbent. One conservative candidate: tilt each weight by its
          in-sample correlation with excess, capped at +-50%, renormalize.
  cv    — grid search over the weight simplex, scored by purged, embargoed
          walk-forward cross-validation. Uses the panel's statistical power;
          the embargo is what stops a 21-session forward label from leaking
          across the train/test boundary.

Metric is the date-wise information coefficient: within each rebalance date,
Spearman correlation between composite score and realized 21-session excess,
averaged over dates. That measures what the model is actually for — ranking
names on a given day — instead of pooling across dates where a market-wide
move would dominate.

LOOK-AHEAD: valuation and quality are computed from a fundamentals snapshot
that post-dates most panel rows (see build_panel's docstring). Their weights
are therefore fitted on contaminated features. Every report block states this,
and a price-only reference fit (momentum + risk, fully point-in-time) is
reported alongside so the clean signal is always visible.

    python -m scripts.learn_panel            # analyze + write proposal
    python -m scripts.learn_panel --apply    # apply it (after review)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from itertools import product
from pathlib import Path

# In-place reconfigure, not a fresh TextIOWrapper: several scripts in this repo
# wrap sys.stdout.buffer at import time, and a second wrapper orphans the first,
# which closes the shared buffer when collected. See export_fundamentals_csv.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import model_params, reliability  # noqa: E402

ROOT = Path(__file__).parent.parent
_PANEL = ROOT / "logs" / "panel.jsonl"
_PROPOSAL = ROOT / "logs" / "learning_proposal_panel.json"

_KEYS = ("valuation", "quality", "momentum", "risk")
_SUB = {k: f"sub_{k}" for k in _KEYS}
_CLEAN = ("momentum", "risk")            # point-in-time; safe to trust
_DIRTY = ("valuation", "quality")        # fitted on post-dated fundamentals

_HORIZON = 21                            # the model's claim horizon
_EMBARGO_SESSIONS = 22                   # label window (21) + 1 entry session
_MIN_ROWS = 400
_MIN_DATES = 20
_MIN_NAMES_PER_DATE = 10
_N_FOLDS = 5                             # last fold is the untouched holdout
_GRID_STEP = 0.05
_GRID_BOUNDS = (0.05, 0.60)
_TILT = 0.5                              # incumbent's tilt strength
_TILT_BOUND = (0.5, 1.5)
_MIN_IC_GAIN = 0.01                      # OOS IC improvement required to propose
_MIN_FOLD_WINS = 3                       # ... in at least this many selection folds
_BUY_CANDIDATES = list(range(50, 91, 5))
_MIN_TRADES = 30


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


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


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 5 else None


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def _load_panel() -> list[dict]:
    if not _PANEL.exists():
        raise SystemExit(f"No panel at {_PANEL}. Run `python -m scripts.build_panel` first.")
    rows = []
    for line in _PANEL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (isinstance(r.get(f"excess_{_HORIZON}d_pct"), (int, float))
                and all(isinstance(r.get(f), (int, float)) for f in _SUB.values())):
            rows.append(r)
    return rows


def _by_date(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["date"], []).append(r)
    return {d: v for d, v in out.items() if len(v) >= _MIN_NAMES_PER_DATE}


def _composite(r: dict, w: dict) -> float:
    return sum(r[_SUB[k]] * w[k] for k in _KEYS)


def _normalize(w: dict) -> dict:
    s = sum(w.values())
    return {k: round(v / s, 4) for k, v in w.items()} if s > 0 else dict(w)


def _ic(dates: dict[str, list[dict]], w: dict) -> float | None:
    """Mean date-wise Spearman IC of the w-weighted composite vs excess."""
    ics = []
    for _, group in dates.items():
        c = [_composite(r, w) for r in group]
        e = [r[f"excess_{_HORIZON}d_pct"] for r in group]
        s = _spearman(c, e)
        if s is not None:
            ics.append(s)
    return st.mean(ics) if ics else None


# ---------------------------------------------------------------------------
# folds — purged + embargoed walk-forward
# ---------------------------------------------------------------------------

def _folds(sorted_dates: list[str]) -> list[tuple[list[str], list[str]]]:
    """Expanding-window walk-forward. Training dates within the embargo of the
    test start are DROPPED: their 21-session label overlaps the test window, so
    keeping them would leak realized returns the model shouldn't have seen.
    Rebalances are weekly, so the embargo is ceil(22/5) = 5 dates."""
    embargo = -(-_EMBARGO_SESSIONS // 5)
    n = len(sorted_dates)
    first = max(_MIN_DATES, n // 2)
    if first >= n:
        return []
    test_size = max(1, (n - first) // _N_FOLDS)
    out = []
    for k in range(_N_FOLDS):
        start = first + k * test_size
        end = n if k == _N_FOLDS - 1 else min(n, start + test_size)
        if start >= n:
            break
        train = sorted_dates[:max(0, start - embargo)]
        test = sorted_dates[start:end]
        if len(train) >= _MIN_DATES and test:
            out.append((train, test))
    return out


def _cv_score(dates: dict[str, list[dict]], folds, w: dict) -> list[float]:
    """OOS IC per fold for one weight vector."""
    out = []
    for _, test in folds:
        ic = _ic({d: dates[d] for d in test}, w)
        if ic is not None:
            out.append(ic)
    return out


# ---------------------------------------------------------------------------
# fitters
# ---------------------------------------------------------------------------

def _fit_tilt(train: dict[str, list[dict]], base: dict) -> dict:
    """Incumbent fitter: correlation tilt, capped, renormalized."""
    rows = [r for g in train.values() for r in g]
    exc = [r[f"excess_{_HORIZON}d_pct"] for r in rows]
    cand = {}
    for k in _KEYS:
        c = _pearson([r[_SUB[k]] for r in rows], exc) or 0.0
        cand[k] = base[k] * max(_TILT_BOUND[0], min(_TILT_BOUND[1], 1 + _TILT * c))
    return _normalize(cand)


def _simplex_grid() -> list[dict]:
    """Weight vectors on a 0.05 grid, each factor in [0.05, 0.60], summing to 1."""
    lo, hi = _GRID_BOUNDS
    steps = [round(lo + i * _GRID_STEP, 4)
             for i in range(int(round((hi - lo) / _GRID_STEP)) + 1)]
    out = []
    for v, q, m in product(steps, repeat=3):
        r = round(1.0 - v - q - m, 4)
        if lo - 1e-9 <= r <= hi + 1e-9:
            out.append({"valuation": v, "quality": q, "momentum": m, "risk": r})
    return out


def _fit_cv(dates: dict[str, list[dict]], folds, base: dict) -> dict:
    """Grid search scored by mean OOS IC across purged walk-forward folds."""
    grid = _simplex_grid()
    base_folds = _cv_score(dates, folds, base)
    base_ic = st.mean(base_folds) if base_folds else None
    scored = []
    for w in grid:
        f = _cv_score(dates, folds, w)
        if len(f) == len(folds):
            scored.append((st.mean(f), w, f))
    if not scored or base_ic is None:
        return {"status": "no_fit", "n_grid": len(grid),
                "note": "no weight vector scored on every fold"}
    scored.sort(key=lambda x: -x[0])
    best_ic, best_w, best_folds = scored[0]
    wins = sum(1 for a, b in zip(best_folds, base_folds) if a > b)
    return {"status": "ok", "n_grid": len(grid),
            "candidate_weights": _normalize(best_w),
            "oos_ic_candidate": round(best_ic, 4),
            "oos_ic_base": round(base_ic, 4),
            "fold_ic_candidate": [round(x, 4) for x in best_folds],
            "fold_ic_base": [round(x, 4) for x in base_folds],
            "folds_won": wins, "n_folds": len(folds),
            "top5": [{"weights": _normalize(w), "oos_ic": round(ic, 4)}
                     for ic, w, _ in scored[:5]]}


# ---------------------------------------------------------------------------
# threshold + contamination reporting
# ---------------------------------------------------------------------------

def _threshold_stats(dates: dict[str, list[dict]], w: dict, thr: float) -> dict:
    sel = [r for g in dates.values() for r in g if _composite(r, w) >= thr]
    exc = [r[f"excess_{_HORIZON}d_pct"] for r in sel]
    return {"threshold": thr, "n": len(exc),
            "mean_excess_pct": round(st.mean(exc), 3) if exc else None,
            "beat_bench_pct": round(sum(1 for e in exc if e > 0) / len(exc) * 100, 1) if exc else None}


def _learn_threshold(dates: dict[str, list[dict]], folds, w: dict, current: float) -> dict:
    """Fit the BUY cut on training folds, score it on the held-out ones."""
    train = {d: dates[d] for f in folds for d in f[0]}
    test = {d: dates[d] for f in folds for d in f[1]}
    is_stats = [_threshold_stats(train, w, c) for c in _BUY_CANDIDATES]
    viable = [s for s in is_stats if s["n"] >= _MIN_TRADES and s["mean_excess_pct"] is not None]
    oos_cur = _threshold_stats(test, w, current)
    if not viable:
        return {"status": "no_candidate", "oos_current": oos_cur,
                "note": f"no candidate cut kept >= {_MIN_TRADES} names in-sample"}
    best = max(viable, key=lambda s: s["mean_excess_pct"])
    oos_cand = _threshold_stats(test, w, best["threshold"])
    improves = (oos_cand["mean_excess_pct"] is not None
                and oos_cur["mean_excess_pct"] is not None
                and oos_cand["n"] >= _MIN_TRADES
                and oos_cand["mean_excess_pct"] > oos_cur["mean_excess_pct"]
                and best["threshold"] != current)
    return {"status": "ok", "candidate": best["threshold"], "current": current,
            "in_sample": best, "oos_candidate": oos_cand, "oos_current": oos_cur,
            "improves": improves}


def _clean_row_pct(rows: list[dict]) -> float:
    """Share of rows with a genuine point-in-time fundamentals read.

    Measured from the panel, not assumed: as scripts.snapshot_fundamentals
    accumulates history, build_panel stops falling back to the current snapshot
    and this climbs toward 100%, at which point valuation/quality stop being
    contaminated and the warning below retires itself.

    Fails CLOSED: a row must carry the key and carry it empty to count as clean.
    A missing key means the row predates the flag, and guessing "clean" there
    would silently declare look-ahead-biased rows trustworthy."""
    if not rows:
        return 0.0
    clean = sum(1 for r in rows if "pit_contaminated" in r and not r["pit_contaminated"])
    return 100 * clean / len(rows)


def _contamination(dates: dict[str, list[dict]], folds, w: dict) -> dict:
    """What the clean half of the feature set says, on its own.

    valuation/quality are fitted on a fundamentals snapshot that post-dates
    most rows. This reports the point-in-time-clean subset (momentum + risk,
    renormalized) so the trustworthy signal is never buried."""
    all_rows = [r for g in dates.values() for r in g]
    clean_pct = _clean_row_pct(all_rows)
    fully_clean = clean_pct >= 99.9

    clean_w = _normalize({k: (w[k] if k in _CLEAN else 0.0) for k in _KEYS}
                         if sum(w[k] for k in _CLEAN) > 0 else
                         {k: (1.0 if k in _CLEAN else 0.0) for k in _KEYS})
    per_factor = {}
    for k in _KEYS:
        solo = {kk: (1.0 if kk == k else 0.0) for kk in _KEYS}
        f = _cv_score(dates, folds, solo)
        per_factor[k] = {"oos_ic": round(st.mean(f), 4) if f else None,
                         "point_in_time": fully_clean or k in _CLEAN}
    clean_folds = _cv_score(dates, folds, clean_w)
    return {
        "clean_row_pct": round(clean_pct, 1),
        "warning": ("all panel rows carry a point-in-time fundamentals read — "
                    "valuation and quality are no longer contaminated."
                    if fully_clean else
                    f"only {clean_pct:.1f}% of panel rows have a point-in-time "
                    "fundamentals read; the rest fall back to a snapshot that "
                    "post-dates them, so valuation/quality weights are fitted on "
                    "look-ahead-contaminated features and must NOT be read as "
                    "out-of-sample evidence. Run scripts.snapshot_fundamentals "
                    "daily — this share grows as history accumulates."),
        "point_in_time_clean": list(_KEYS) if fully_clean else list(_CLEAN),
        "contaminated": [] if fully_clean else list(_DIRTY),
        "per_factor_oos_ic": per_factor,
        "clean_only_weights": clean_w,
        "clean_only_oos_ic": round(st.mean(clean_folds), 4) if clean_folds else None,
    }


# ---------------------------------------------------------------------------
# proposal
# ---------------------------------------------------------------------------

def _build() -> dict:
    rows = _load_panel()
    dates = _by_date(rows)
    sorted_dates = sorted(dates)
    current = model_params.load_params()
    base = current.get("score_weights", model_params.DEFAULTS["score_weights"])
    cur_buy = current["verdict_thresholds"]["BUY"]

    meta = {"panel_rows": len(rows), "panel_dates": len(sorted_dates),
            "date_range": [sorted_dates[0], sorted_dates[-1]] if sorted_dates else None,
            "horizon_days": _HORIZON,
            "fundamentals_asof": rows[0].get("fundamentals_asof") if rows else None}

    if len(rows) < _MIN_ROWS or len(sorted_dates) < _MIN_DATES:
        return {"status": "insufficient_panel", **meta,
                "need": {"rows": _MIN_ROWS, "dates": _MIN_DATES},
                "message": (f"Panel has {len(rows)} rows across {len(sorted_dates)} dates "
                            f"(need {_MIN_ROWS}/{_MIN_DATES}). Rebuild with deeper history: "
                            "python -m scripts.build_panel --refresh --lookback 1000")}

    all_folds = _folds(sorted_dates)
    if len(all_folds) < 3:
        return {"status": "insufficient_panel", **meta,
                "message": (f"Only {len(all_folds)} usable walk-forward fold(s) after "
                            "embargo; need >=3 (2 to select on, 1 untouched holdout).")}

    # The last fold is a HOLDOUT neither fitter may see. The cv fitter picks the
    # best of ~800 grid points by mean OOS IC, so those fold scores are a
    # selection statistic, not an unbiased estimate — a candidate must also beat
    # the incumbent on data that played no part in choosing it.
    folds, holdout = all_folds[:-1], all_folds[-1:]

    train_all = {d: dates[d] for f in folds for d in f[0]}
    tilt_w = _fit_tilt(train_all, base)
    tilt_folds = _cv_score(dates, folds, tilt_w)
    base_folds = _cv_score(dates, folds, base)
    tilt = {"candidate_weights": tilt_w,
            "oos_ic_candidate": round(st.mean(tilt_folds), 4) if tilt_folds else None,
            "oos_ic_base": round(st.mean(base_folds), 4) if base_folds else None,
            "fold_ic_candidate": [round(x, 4) for x in tilt_folds],
            "folds_won": sum(1 for a, b in zip(tilt_folds, base_folds) if a > b),
            "n_folds": len(folds)}
    cv = _fit_cv(dates, folds, base)

    base_hold = _cv_score(dates, holdout, base)
    hold_base_ic = st.mean(base_hold) if base_hold else None
    for f in (tilt, cv):
        if f.get("candidate_weights"):
            h = _cv_score(dates, holdout, f["candidate_weights"])
            f["holdout_ic_candidate"] = round(st.mean(h), 4) if h else None
            f["holdout_ic_base"] = round(hold_base_ic, 4) if hold_base_ic is not None else None
            f["holdout_dates"] = [holdout[0][1][0], holdout[0][1][-1]]

    def _qualifies(f: dict) -> bool:
        return bool(f.get("oos_ic_candidate") is not None
                    and f.get("oos_ic_base") is not None
                    and f["oos_ic_candidate"] >= f["oos_ic_base"] + _MIN_IC_GAIN
                    and f["oos_ic_candidate"] > 0
                    and f.get("folds_won", 0) >= _MIN_FOLD_WINS
                    and f["candidate_weights"] != base
                    # must survive the fold it had no hand in choosing
                    and f.get("holdout_ic_candidate") is not None
                    and f.get("holdout_ic_base") is not None
                    and f["holdout_ic_candidate"] > f["holdout_ic_base"])

    tilt["qualifies"] = _qualifies(tilt)
    if cv.get("status") == "ok":
        cv["qualifies"] = _qualifies(cv)

    # Pick the winner on OOS IC among fitters that clear the guardrails.
    contenders = [("tilt", tilt)] + ([("cv", cv)] if cv.get("status") == "ok" else [])
    qualified = [(n, f) for n, f in contenders if f.get("qualifies")]
    winner, wfit = (max(qualified, key=lambda x: x[1]["oos_ic_candidate"])
                    if qualified else (None, None))

    chosen_w = wfit["candidate_weights"] if wfit else base
    thr = _learn_threshold(dates, folds, chosen_w, cur_buy)
    contam = _contamination(dates, folds, chosen_w)

    any_change = bool(winner) or thr.get("improves")
    proposed = json.loads(json.dumps(current))
    if winner:
        proposed["score_weights"] = chosen_w
    if thr.get("improves"):
        proposed["verdict_thresholds"]["BUY"] = thr["candidate"]
    proposed["version"] = f"panel-{sorted_dates[-1]}"
    proposed["learned_at"] = sorted_dates[-1]
    proposed["provenance"] = (
        f"learned from a {len(rows)}-row market-data panel "
        f"({len(sorted_dates)} rebalance dates, {sorted_dates[0]}..{sorted_dates[-1]}), "
        f"fitter={winner or 'none'}, purged walk-forward CV over {len(folds)} folds; "
        f"valuation/quality contaminated by fundamentals_asof={meta['fundamentals_asof']}")

    msg = []
    if winner:
        msg.append(f"{winner} fitter wins: weights {base} -> {chosen_w} "
                   f"(OOS IC {wfit['oos_ic_candidate']} vs {wfit['oos_ic_base']} base, "
                   f"{wfit['folds_won']}/{wfit['n_folds']} folds).")
    else:
        msg.append("Neither fitter cleared the guardrails — current weights hold.")
    if thr.get("improves"):
        msg.append(f"BUY cut {cur_buy}->{thr['candidate']} "
                   f"({thr['oos_candidate']['mean_excess_pct']}% vs "
                   f"{thr['oos_current']['mean_excess_pct']}% OOS mean excess).")
    msg.append("Valuation/quality weights are fitted on look-ahead-contaminated "
               "features — see contamination block before approving.")

    # The daily loop refuses to propose at all while the live gate is open.
    # This loop learns from historical market data rather than live calls, so it
    # still runs — but applying it changes live verdicts, so the gate's state
    # travels with the proposal instead of being silently bypassed.
    try:
        gate = reliability.status()
    except Exception as e:  # noqa: BLE001
        gate = {"passed": False, "error": f"{type(e).__name__}: {e}"}
    if not gate.get("passed"):
        msg.append("Live reliability gate is NOT passed — the model is research-only. "
                   "Merging still changes live verdicts.")

    prop = {
        "status": "proposal" if any_change else "no_change",
        **meta,
        "n_selection_folds": len(folds),
        "holdout_dates": [holdout[0][1][0], holdout[0][1][-1]],
        "embargo_sessions": _EMBARGO_SESSIONS,
        "current_weights": base,
        "current_buy_threshold": cur_buy,
        "fitters": {"tilt": tilt, "cv": cv},
        "winning_fitter": winner,
        "threshold": thr,
        "contamination": contam,
        "live_reliability_gate": {"passed": bool(gate.get("passed")),
                                  "mode": gate.get("mode"),
                                  "failed_checks": gate.get("failed_checks")},
        "recommendation": "APPLY" if any_change else "KEEP_CURRENT",
        "proposed_params": proposed if any_change else None,
        "change_tag": _change_tag(winner, chosen_w, thr),
        "message": " ".join(msg),
    }
    prop["pr_title"], prop["pr_body"] = _pr_text(prop)
    return prop


def _change_tag(winner: str | None, w: dict, thr: dict) -> str:
    parts = []
    if winner:
        parts.append(winner + "-w-" + "-".join(str(int(round(w[k] * 100))) for k in _KEYS))
    if thr.get("improves"):
        parts.append(f"buy-{thr['candidate']}")
    return "_".join(parts) or "none"


def _pr_text(p: dict) -> tuple[str, str]:
    bits = []
    if p["winning_fitter"]:
        bits.append(f"reweight ({p['winning_fitter']})")
    if p["threshold"].get("improves"):
        bits.append(f"BUY {p['current_buy_threshold']}→{p['threshold']['candidate']}")
    title = "Panel learning loop: " + (" + ".join(bits) or "no change")

    t, c = p["fitters"]["tilt"], p["fitters"]["cv"]
    lines = [
        "## Learning loop — market-data panel", "", p["message"], "",
        f"Panel: **{p['panel_rows']} rows** over {p['panel_dates']} rebalance dates "
        f"({p['date_range'][0]} → {p['date_range'][1]}), {p['horizon_days']}-session "
        f"forward excess vs the equal-weight basket. Purged walk-forward CV, "
        f"{p['n_selection_folds']} selection folds, {p['embargo_sessions']}-session "
        f"embargo, holdout {p['holdout_dates'][0]} → {p['holdout_dates'][1]} "
        f"(never seen by either fitter).", "",
        "### Fitters", "",
        "| fitter | sel. IC | base | folds won | holdout IC | holdout base | qualifies |",
        "|---|---|---|---|---|---|---|",
        f"| tilt | {t.get('oos_ic_candidate')} | {t.get('oos_ic_base')} | "
        f"{t.get('folds_won')}/{t.get('n_folds')} | {t.get('holdout_ic_candidate')} | "
        f"{t.get('holdout_ic_base')} | {t.get('qualifies')} |",
    ]
    if c.get("status") == "ok":
        lines.append(f"| cv | {c.get('oos_ic_candidate')} | {c.get('oos_ic_base')} | "
                     f"{c.get('folds_won')}/{c.get('n_folds')} | "
                     f"{c.get('holdout_ic_candidate')} | {c.get('holdout_ic_base')} | "
                     f"{c.get('qualifies')} |")
    else:
        lines.append(f"| cv | — | — | — | — | — | {c.get('status')} |")

    lines += ["", "### Weights", "", "| factor | current | tilt | cv |", "|---|---|---|---|"]
    for k in _KEYS:
        cvw = c.get("candidate_weights", {}).get(k, "—") if c.get("status") == "ok" else "—"
        flag = " ⚠️" if k in _DIRTY else ""
        lines.append(f"| {k}{flag} | {p['current_weights'][k]} | "
                     f"{t['candidate_weights'][k]} | {cvw} |")

    cont = p["contamination"]
    lines += [
        "", "### ⚠️ Look-ahead", "", cont["warning"], "",
        "| factor | solo OOS IC | point-in-time |", "|---|---|---|",
        *[f"| {k} | {v['oos_ic']} | {'yes' if v['point_in_time'] else '**no**'} |"
          for k, v in cont["per_factor_oos_ic"].items()],
        "",
        f"Price-only reference (momentum + risk, fully point-in-time): "
        f"**OOS IC {cont['clean_only_oos_ic']}**.", "",
    ]
    g = p["live_reliability_gate"]
    if not g["passed"]:
        lines += [
            "### ⚠️ Live reliability gate not passed", "",
            f"`model_reliability()` reports **{g['mode']}** "
            f"(failing: {', '.join(g['failed_checks'] or []) or 'n/a'}). This "
            "proposal is fitted on historical market data, not on live graded "
            "calls — but merging it changes live verdicts all the same.", "",
        ]
    lines += [
        "**Merging this PR is the human approval gate** — it writes "
        "`model_params.json`, which the decision layer reads on next run. "
        "Close without merging to reject.",
    ]
    return title, "\n".join(lines)


def cmd_analyze() -> int:
    p = _build()
    _PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    _PROPOSAL.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    skip = {"proposed_params", "pr_body"}
    print(json.dumps({k: v for k, v in p.items() if k not in skip},
                     ensure_ascii=False, indent=2))
    print(f"\nProposal written -> {_PROPOSAL}")
    if p["status"] == "proposal":
        print("Review it, then apply with:  python -m scripts.learn_panel --apply")
    return 0


def cmd_apply() -> int:
    if not _PROPOSAL.exists():
        print("No proposal. Run `python -m scripts.learn_panel` first.")
        return 1
    p = json.loads(_PROPOSAL.read_text(encoding="utf-8"))
    if p.get("status") != "proposal" or not p.get("proposed_params"):
        print(f"Nothing to apply (status: {p.get('status')}). {p.get('message', '')}")
        return 1
    model_params.save_params(p["proposed_params"])
    print(f"Applied -> {model_params._PARAMS_FILE}")
    print(f"  weights:   {p['proposed_params']['score_weights']}")
    print(f"  BUY cut:   {p['proposed_params']['verdict_thresholds']['BUY']}")
    print("NOTE: valuation/quality weights carry documented look-ahead bias.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Learn model params from the market-data panel.")
    ap.add_argument("--apply", action="store_true", help="Apply the proposal (after review).")
    args = ap.parse_args()
    return cmd_apply() if args.apply else cmd_analyze()


if __name__ == "__main__":
    sys.exit(main())
