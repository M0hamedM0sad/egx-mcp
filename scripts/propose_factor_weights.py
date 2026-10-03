"""Walk-forward PROPOSAL for the composite's weights and two scoring switches.

Built on scripts/factor_study.py's point-in-time panel (TradingView quarterly
history + prices, fundamentals visible 60 days after quarter end, label =
next-21-session return vs the cross-sectional median). Each sub-score is
stood in for by its point-in-time proxy, as a cross-sectional percentile:

    valuation -> earnings_yield     quality -> quality_composite
    momentum  -> mom_6m             risk    -> low_vol_3m

Candidates, all scored on the SAME out-of-sample dates (2021 onward):

    current            live weights, WITH the regime weight overrides
    current_no_regime  live weights, overrides off
    walk_forward       for each test year, weights refit on all earlier years
                       only (grid over the simplex, each weight >= 0.05) —
                       the honest estimate of what a fitted set would earn

The proposal changes nothing until a human runs --apply (which writes
model_params.json). It recommends:
  - regime_weight_overrides=False  if current_no_regime beats current OOS;
  - score_weights=<full-sample fit> if walk_forward beats both OOS with t>=2;
  - valuation_market_fallback=True if earnings_yield has OOS IC t>=2 (the
    signal the fallback lets the live model see for ~220 more names).

Proxies are not the live sub-scorers (ROE / P/B history does not exist), so
applied weights must still clear the live reliability gate before any
buy-side output becomes actionable.

    python -m scripts.propose_factor_weights            # analyze + write proposal
    python -m scripts.propose_factor_weights --apply    # apply it (after review)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import model_params, tv_history  # noqa: E402
from egx_mcp.data.regime import _REGIME_WEIGHTS  # noqa: E402
from scripts import factor_study  # noqa: E402
from scripts.build_panel import _load_prices  # noqa: E402

ROOT = Path(__file__).parent.parent
_OUT_JSON = ROOT / "logs" / "learning_proposal_factor.json"
_OUT_TXT = ROOT / "logs" / "learning_proposal_factor_latest.txt"

PROXY = {"valuation": "earnings_yield", "quality": "quality_composite",
         "momentum": "mom_6m", "risk": "low_vol_3m"}
KEYS = list(PROXY)
FIRST_OOS_YEAR = 2021
MIN_TRAIN_DATES = 12
GRID_STEP = 0.05
MIN_WEIGHT = 0.05
MIN_T = 2.0


def _regime_of(mkt_60d: float) -> str:
    """Proxy for regime.classify() from the universe's median 60d return."""
    if mkt_60d > 0.05:
        return "BULL"
    if mkt_60d < -0.05:
        return "BEAR"
    return "SIDEWAYS"


def prepare(per_date: list[dict]) -> list[dict]:
    """Per date: percentile ranks of the four proxies (missing -> 0.5, the
    live scorer's neutral) and the label, as aligned numpy arrays."""
    out = []
    for r in per_date:
        fr = r["frame"]
        fr = fr[fr["excess"].notna()]
        if len(fr) < factor_study.MIN_NAMES or fr[PROXY["valuation"]].notna().sum() < factor_study.MIN_NAMES:
            continue
        ranks = np.column_stack([fr[PROXY[k]].rank(pct=True).fillna(0.5).to_numpy()
                                 for k in KEYS])
        out.append({"date": r["date"], "year": int(r["date"][:4]),
                    "regime": _regime_of(r.get("market_60d", 0.0)),
                    "ranks": ranks, "excess": fr["excess"].to_numpy()})
    return out


def _spearman(score: np.ndarray, excess: np.ndarray) -> float:
    a = pd.Series(score).rank().to_numpy()
    b = pd.Series(excess).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _weights_for(base: dict[str, float], regime: str, use_regime: bool) -> np.ndarray:
    w = np.array([base[k] for k in KEYS], dtype=float)
    if use_regime:
        w = w * np.array([_REGIME_WEIGHTS.get(regime, {}).get(k, 1.0) for k in KEYS])
    return w / w.sum()


def evaluate(dates: list[dict], base: dict[str, float], use_regime: bool) -> dict:
    """OOS metrics of one weight set: mean date IC with t, and the buy/sell
    analogues of the live gate — share of the top quintile beating the
    median, share of the bottom quintile lagging it."""
    ics, top_hit, bot_hit, top_exc = [], [], [], []
    for d in dates:
        score = d["ranks"] @ _weights_for(base, d["regime"], use_regime)
        ics.append(_spearman(score, d["excess"]))
        q = pd.qcut(pd.Series(score).rank(method="first"), 5, labels=False).to_numpy()
        top, bot = d["excess"][q == 4], d["excess"][q == 0]
        top_hit.append(float(np.mean(top > 0)))
        bot_hit.append(float(np.mean(bot < 0)))
        top_exc.append(float(np.median(top)))
    ics = [x for x in ics if not math.isnan(x)]
    n = len(ics)
    m = float(np.mean(ics)) if ics else float("nan")
    sd = float(np.std(ics, ddof=1)) if n > 1 else float("nan")
    return {"n_dates": n, "mean_ic": m,
            "t": m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None,
            "top_quintile_beat_median_pct": 100 * float(np.mean(top_hit)) if top_hit else None,
            "bottom_quintile_lag_median_pct": 100 * float(np.mean(bot_hit)) if bot_hit else None,
            "top_quintile_median_excess_pct": 100 * float(np.mean(top_exc)) if top_exc else None}


def grid() -> list[dict[str, float]]:
    steps = round((1 - MIN_WEIGHT * len(KEYS)) / GRID_STEP)
    out = []
    for combo in itertools.product(range(steps + 1), repeat=len(KEYS) - 1):
        if sum(combo) > steps:
            continue
        units = [*combo, steps - sum(combo)]
        out.append({k: round(MIN_WEIGHT + u * GRID_STEP, 4) for k, u in zip(KEYS, units)})
    return out


def fit(dates: list[dict], candidates: list[dict[str, float]]) -> dict[str, float]:
    """Weights with the highest mean date IC on `dates` (overrides off)."""
    R = [d["ranks"] for d in dates]
    E = [pd.Series(d["excess"]).rank().to_numpy() for d in dates]
    best, best_ic = candidates[0], -np.inf
    for w in candidates:
        wv = np.array([w[k] for k in KEYS])
        ics = []
        for r, e in zip(R, E):
            s = pd.Series(r @ wv).rank().to_numpy()
            if s.std() > 0 and e.std() > 0:
                ics.append(np.corrcoef(s, e)[0, 1])
        m = float(np.mean(ics)) if ics else -np.inf
        if m > best_ic:
            best, best_ic = w, m
    return best


def walk_forward(dates: list[dict], candidates: list[dict[str, float]]) -> tuple[list[dict], list[dict]]:
    """Refit on all years before each test year; return OOS dates tagged with
    the weights used, and the per-fold weights."""
    oos, folds = [], []
    for year in sorted({d["year"] for d in dates if d["year"] >= FIRST_OOS_YEAR}):
        train = [d for d in dates if d["year"] < year]
        test = [d for d in dates if d["year"] == year]
        if len(train) < MIN_TRAIN_DATES or not test:
            continue
        w = fit(train, candidates)
        folds.append({"test_year": year, "n_train": len(train), "n_test": len(test), "weights": w})
        oos += [{**d, "fold_weights": w} for d in test]
    return oos, folds


def evaluate_walk_forward(oos: list[dict]) -> dict:
    """Pool OOS dates, each scored with the weights its own fold fitted."""
    per = [evaluate([d], d["fold_weights"], use_regime=False) for d in oos]
    ics = [r["mean_ic"] for r in per if r["n_dates"]]
    n = len(ics)
    m = float(np.mean(ics)) if ics else float("nan")
    sd = float(np.std(ics, ddof=1)) if n > 1 else float("nan")

    def avg(key):
        vals = [r[key] for r in per if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    return {"n_dates": n, "mean_ic": m,
            "t": m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None,
            "top_quintile_beat_median_pct": avg("top_quintile_beat_median_pct"),
            "bottom_quintile_lag_median_pct": avg("bottom_quintile_lag_median_pct"),
            "top_quintile_median_excess_pct": avg("top_quintile_median_excess_pct")}


def build_proposal(per_date: list[dict], factor_summary: dict | None = None) -> dict:
    dates = prepare(per_date)
    current = model_params.score_weights()
    oos_dates = [d for d in dates if d["year"] >= FIRST_OOS_YEAR]
    candidates = grid()
    wf_dates, folds = walk_forward(dates, candidates)
    res = {
        "current": evaluate(oos_dates, current, use_regime=True),
        "current_no_regime": evaluate(oos_dates, current, use_regime=False),
        "walk_forward": evaluate_walk_forward(wf_dates) if wf_dates else {"n_dates": 0},
    }
    full_fit = fit(dates, candidates) if dates else None

    def ic(k):
        v = res[k].get("mean_ic")
        return v if v is not None and not math.isnan(v) else -np.inf

    changes: dict = {}
    reasons: list[str] = []
    if ic("current_no_regime") > ic("current"):
        changes["regime_weight_overrides"] = False
        reasons.append(f"Regime overrides lower OOS IC ({ic('current'):+.3f} with vs "
                       f"{ic('current_no_regime'):+.3f} without): turn them off.")
    else:
        reasons.append("Regime overrides do not hurt OOS IC: keep them.")
    wf = res["walk_forward"]
    if (full_fit and ic("walk_forward") > max(ic("current"), ic("current_no_regime"))
            and (wf.get("t") or 0) >= MIN_T):
        changes["score_weights"] = full_fit
        reasons.append(f"Walk-forward refit beats current OOS ({ic('walk_forward'):+.3f}, "
                       f"t={wf['t']:+.2f}): adopt the full-sample fit {full_fit}.")
    else:
        reasons.append("Walk-forward refit does not beat current weights with t>=2: keep weights.")
    ey = (factor_summary or {}).get("overall", {}).get("earnings_yield", {})
    if (ey.get("t") or 0) >= MIN_T and (ey.get("mean") or 0) > 0:
        changes["valuation_market_fallback"] = True
        reasons.append(f"Earnings yield predicts relative return (IC {ey['mean']:+.3f}, "
                       f"t={ey['t']:+.2f}): let the ~220 unsectored names be valued "
                       "against the market median instead of a flat 50.")
    return {
        "status": "proposal" if changes else "no_change",
        "recommendation": "APPLY" if changes else "KEEP_CURRENT",
        "changes": changes, "reasons": reasons,
        "current_weights": current, "oos_from_year": FIRST_OOS_YEAR,
        "results": res, "folds": folds, "full_sample_fit": full_fit,
        "proxies": PROXY,
        "caveats": [
            "Proxies, not the live sub-scorers: no ROE / P/B history exists.",
            "Survivors only; TradingView serves restated figures.",
            "Applying does not make buy-side output actionable: the live "
            "reliability gate still has to pass on new calls.",
        ],
    }


def render(p: dict) -> list[str]:
    L = ["=" * 78, "FACTOR-WEIGHT PROPOSAL  (walk-forward, point-in-time proxies)", "=" * 78,
         f"out-of-sample dates: {p['oos_from_year']} onward   current weights: {p['current_weights']}",
         "", f"{'candidate':20} {'dates':>5} {'meanIC':>7} {'t':>6} {'top5 beat%':>10} "
             f"{'bot5 lag%':>9} {'top5 exc':>9}"]
    for k, r in p["results"].items():
        if not r.get("n_dates"):
            L.append(f"{k:20} {'0':>5}")
            continue
        L.append(f"{k:20} {r['n_dates']:>5} {r['mean_ic']:+7.3f} {r['t'] or 0:+6.2f} "
                 f"{r['top_quintile_beat_median_pct']:10.1f} {r['bottom_quintile_lag_median_pct']:9.1f} "
                 f"{r['top_quintile_median_excess_pct']:+8.2f}%")
    L += ["", "walk-forward folds:"]
    L += [f"  test {f['test_year']}: trained on {f['n_train']} dates -> {f['weights']}"
          for f in p["folds"]]
    L += ["", f"RECOMMENDATION: {p['recommendation']}"] + [f"  - {r}" for r in p["reasons"]]
    L += ["", f"proposed changes: {json.dumps(p['changes'])}", "", "caveats:"]
    L += [f"  - {c}" for c in p["caveats"]]
    L += ["", "Nothing is applied. Review, then: python -m scripts.propose_factor_weights --apply"]
    return L


def apply(proposal: dict) -> None:
    if not proposal.get("changes"):
        print("Proposal has no changes; nothing applied.")
        return
    params = dict(model_params.load_params())
    params.update(proposal["changes"])
    params["version"] = f"factor-proposal-{proposal.get('generated_at', '')[:10]}"
    params["provenance"] = "scripts/propose_factor_weights.py (walk-forward, human-applied)"
    model_params.save_params(params)
    print(f"Applied: {proposal['changes']} -> model_params.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward factor-weight proposal.")
    ap.add_argument("--apply", action="store_true", help="Apply the last written proposal.")
    args = ap.parse_args()
    if args.apply:
        apply(json.loads(_OUT_JSON.read_text(encoding="utf-8")))
        return 0

    if not tv_history.HISTORY_PATH.exists():
        tv_history.save(tv_history.fetch())
    prices, source = _load_prices()
    print(f"Price source: {source} ({len(prices)} series)")
    study = factor_study.run(prices, tv_history.load(), keep_frames=True)
    summ, _ = factor_study.summarize(study)
    proposal = build_proposal(study["per_date"], summ)
    proposal["generated_at"] = pd.Timestamp.now("UTC").isoformat()
    text = "\n".join(render(proposal))
    print(text)
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(proposal, indent=2, default=str), encoding="utf-8")
    _OUT_TXT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
