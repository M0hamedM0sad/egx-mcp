"""A1 — Grade the FULL live model forward against realized prices.

This is the only lookahead-free way to measure the overall model (fundamentals
+ macro + catalysts + sentiment included), because your daily briefings are
point-in-time snapshots: each one recorded what the model said on that date,
before the future was known.

It reads briefings/*.json, extracts every verdict (the v8b_verdicts block and
the w1 weekly BUY picks), pulls realized prices from each briefing date
forward, and grades each call against the EGX 30 benchmark over 5- and 21-day
horizons. Output is one row per (briefing_date, ticker, horizon) written to
BOTH JSONL and CSV in a schema ready to push to a Hugging Face Dataset (see
the snippet printed at the end) so the evidence base is versioned and grows
every day you run the briefing.

"Correct" = beat the index for buy-side calls (BUY/ACCUMULATE/weekly-pick),
lag the index for sell-side (REDUCE/AVOID). HOLD makes no directional claim.

    python -m tests.grade_briefings
    python -m tests.grade_briefings --briefings-dir briefings --horizons 5,21

Rows whose horizon hasn't elapsed yet are written with outcome="pending" and
excluded from accuracy — they get graded automatically once prices exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Same workaround as oos_last_week.py: curl_cffi with impersonate=chrome uses
# BoringSSL which ignores cert env vars; without this every yfinance fetch
# fails ("possibly delisted") and grading silently produces nothing.
# curl_cffi ships with recent yfinance but is not our declared dependency —
# skip the patch rather than die if it's absent.
try:
    import curl_cffi.requests as _curl_requests

    _orig_session_init = _curl_requests.Session.__init__

    def _patched_session_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_session_init(self, *args, **kwargs)

    _curl_requests.Session.__init__ = _patched_session_init
except ImportError:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

from egx_mcp.data import backtest as bt_mod
from egx_mcp.data import egx_listing, price_sanity
from egx_mcp.data.agentic_backtest import _benchmark_series


def _synthetic_basket(start: str, end: str) -> pd.Series | None:
    """Equal-weight basket of the full validated universe, as an index series.

    Fallback benchmark: every EGX30 symbol Yahoo once served (^CASE30,
    EGS69491M015.CA, EGX30.CA) now returns no data, which left bench=None and
    every excess_pct null — grading silently degraded to "vs zero" while the
    scorecard reported it as "vs EGX30". This basket is the same benchmark
    notion tests/oos_last_week.py already uses.
    """
    universe = egx_listing.get_full_universe()
    panel = bt_mod._price_panel(universe, start=start, end=end)
    if panel.empty:
        return None
    daily = panel.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    return (1.0 + daily).cumprod()

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_BUY_SIDE = {"BUY", "ACCUMULATE", "WEEKLY_BUY"}
_SELL_SIDE = {"REDUCE", "AVOID", "SELL"}

_SUBSCORES = ["sub_valuation", "sub_quality", "sub_momentum", "sub_risk"]
_FIELDS = ["briefing_date", "ticker", "source", "verdict", "conviction", "score",
           "model_version", "horizon_days", "entry_date", "entry_price", "exit_date",
           "exit_price", "fwd_return_pct", "bench_return_pct", "excess_pct", "outcome",
           "correct", "quarantine_reason", *_SUBSCORES]


def _briefing_date(path: Path, payload: dict) -> str | None:
    m = _DATE_RE.search(path.name)
    if m:
        return m.group(1)
    asof = payload.get("as_of_utc") or payload.get("cairo_local") or ""
    m = _DATE_RE.search(asof)
    return m.group(1) if m else None


def _extract_verdicts(payload: dict) -> list[dict]:
    """Pull (ticker, source, verdict, conviction, score) tuples from a briefing."""
    out: list[dict] = []
    for v in payload.get("v8b_verdicts", []) or []:
        sub = v.get("v8b_subscores") or {}
        out.append({
            "ticker": v.get("ticker"),
            "source": "v8b",
            "verdict": (v.get("v8b_verdict") or "").upper(),
            "conviction": v.get("v8b_conviction"),
            "score": v.get("v8b_score"),
            # Flatten sub-scores into evidence columns (None when not recorded,
            # e.g. older briefings predating this field).
            "sub_valuation": sub.get("valuation"),
            "sub_quality": sub.get("quality"),
            "sub_momentum": sub.get("momentum"),
            "sub_risk": sub.get("risk"),
        })
    # Weekly picks are an explicit BUY list (5-day horizon model).
    for p in (payload.get("w1_picks", {}) or {}).get("top_picks", []) or []:
        out.append({
            "ticker": p.get("ticker"),
            "source": "w1",
            "verdict": "WEEKLY_BUY",
            "conviction": "weekly",
            "score": p.get("score"),
        })
    return [r for r in out if r["ticker"] and r["verdict"]]


def _fwd_prices(ser: pd.Series, entry_date: str, horizon: int):
    """Entry = first close on/after entry_date; exit = `horizon` rows later.

    Returns (entry_date, entry_px, exit_date, exit_px) or None if no entry,
    or (..., None, None) if the horizon hasn't elapsed yet (pending).
    """
    # Non-positive / non-finite ticks are dropped before indexing: a vendor
    # zero or negative close would otherwise become an entry price and
    # manufacture a several-hundred-percent return (see price_sanity).
    ser = price_sanity.clean_series(ser)
    if ser.empty:
        return None
    pos = ser.index.searchsorted(pd.Timestamp(entry_date))
    if pos >= len(ser):
        return None
    entry_i = pos
    exit_i = entry_i + horizon
    e_date = ser.index[entry_i].strftime("%Y-%m-%d")
    e_px = float(ser.iloc[entry_i])
    if exit_i >= len(ser):
        return (e_date, e_px, None, None)  # pending
    return (e_date, e_px, ser.index[exit_i].strftime("%Y-%m-%d"), float(ser.iloc[exit_i]))


def _bench_return(bench: pd.Series | None, entry_date: str, exit_date: str | None) -> float | None:
    if bench is None or exit_date is None:
        return None
    b = bench.dropna()
    if b.empty:
        return None
    ei = b.index.searchsorted(pd.Timestamp(entry_date))
    xi = b.index.searchsorted(pd.Timestamp(exit_date))
    if ei >= len(b) or xi >= len(b) or b.iloc[ei] <= 0:
        return None
    return float(b.iloc[xi] / b.iloc[ei] - 1)


def _grade(rows: list[dict], panel: pd.DataFrame, bench: pd.Series | None,
           horizons: list[int]) -> list[dict]:
    graded: list[dict] = []
    for r in rows:
        tk = r["ticker"]
        if tk not in panel.columns:
            continue
        ser = panel[tk]
        for h in horizons:
            fp = _fwd_prices(ser, r["briefing_date"], h)
            if fp is None:
                continue
            e_date, e_px, x_date, x_px = fp
            rec = {**r, "horizon_days": h, "entry_date": e_date,
                   "entry_price": round(e_px, 4), "quarantine_reason": None}
            if x_px is None:
                rec.update({"exit_date": None, "exit_price": None, "fwd_return_pct": None,
                            "bench_return_pct": None, "excess_pct": None,
                            "outcome": "pending", "correct": None})
                graded.append(rec)
                continue
            fwd = x_px / e_px - 1
            bret = _bench_return(bench, e_date, x_date)
            excess = (fwd - bret) if bret is not None else None

            # A session outside the EGX daily band inside the holding window is
            # a corporate action, not a return. Keep the numbers for audit but
            # quarantine the row so no gate or learner consumes it.
            brk = price_sanity.find_break(ser, e_date, x_date)
            if brk is not None:
                rec.update({
                    "exit_date": x_date, "exit_price": round(x_px, 4),
                    "fwd_return_pct": round(fwd * 100, 2),
                    "bench_return_pct": round(bret * 100, 2) if bret is not None else None,
                    "excess_pct": None, "outcome": "quarantined", "correct": None,
                    "quarantine_reason": (f"suspected corporate action: {brk['pct']:+.1f}% "
                                          f"on {brk['date']} ({brk['from']} -> {brk['to']})"),
                })
                graded.append(rec)
                continue

            side = r["verdict"]
            # Correct against benchmark when we have it, else against zero.
            ref = excess if excess is not None else fwd
            if side in _BUY_SIDE:
                correct = ref > 0
            elif side in _SELL_SIDE:
                correct = ref < 0
            else:  # HOLD — no directional claim
                correct = None
            rec.update({
                "exit_date": x_date, "exit_price": round(x_px, 4),
                "fwd_return_pct": round(fwd * 100, 2),
                "bench_return_pct": round(bret * 100, 2) if bret is not None else None,
                "excess_pct": round(excess * 100, 2) if excess is not None else None,
                "outcome": "graded", "correct": correct,
            })
            graded.append(rec)
    return graded


def _summary(graded: list[dict], bench_name: str = "EGX30") -> None:
    done = [g for g in graded if g["outcome"] == "graded" and g["correct"] is not None]
    pending = [g for g in graded if g["outcome"] == "pending"]
    quarantined = [g for g in graded if g["outcome"] == "quarantined"]
    print(f"\n{'=' * 64}")
    print(f"GRADED {len(done)} directional calls   ({len(pending)} pending horizon)")
    print('=' * 64)
    if quarantined:
        print(f"  QUARANTINED {len(quarantined)} row(s) — suspected corporate action or bad tick:")
        for g in quarantined[:8]:
            print(f"    {g['briefing_date']} {g['ticker']:6s} H={g['horizon_days']:>3}d  "
                  f"{g['quarantine_reason']}")
        if len(quarantined) > 8:
            print(f"    ... and {len(quarantined) - 8} more")
    if not done:
        print("No elapsed calls yet — re-run after more time passes since the briefings.")
        return
    for h in sorted({g["horizon_days"] for g in done}):
        sub = [g for g in done if g["horizon_days"] == h]
        acc = sum(1 for g in sub if g["correct"]) / len(sub)
        buy = [g for g in sub if g["verdict"] in _BUY_SIDE]
        buy_hit = (sum(1 for g in buy if g["correct"]) / len(buy)) if buy else None
        mean_exc = [g["excess_pct"] for g in sub if g["excess_pct"] is not None]
        avg_exc = round(sum(mean_exc) / len(mean_exc), 2) if mean_exc else None
        print(f"  H={h:>3}d  accuracy={acc:.0%}  ({len(sub)} calls)  "
              f"buy-side hit={buy_hit:.0%} of {len(buy)}" if buy_hit is not None
              else f"  H={h:>3}d  accuracy={acc:.0%}  ({len(sub)} calls)")
        if avg_exc is not None:
            print(f"         mean excess vs {bench_name} = {avg_exc:+.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade briefing verdicts forward (A1).")
    ap.add_argument("--briefings-dir", default=str(Path(__file__).parent.parent / "briefings"))
    ap.add_argument("--horizons", default="5,21")
    ap.add_argument("--out-jsonl", default=str(Path(__file__).parent.parent / "logs" / "graded_verdicts.jsonl"))
    ap.add_argument("--out-csv", default=str(Path(__file__).parent.parent / "logs" / "graded_verdicts.csv"))
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    bdir = Path(args.briefings_dir)
    files = sorted(bdir.glob("briefing_*.json"))
    if not files:
        print(f"No briefings found in {bdir}.")
        return 1

    rows: list[dict] = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip {f.name}: {e}")
            continue
        bd = _briefing_date(f, payload)
        if not bd:
            print(f"  ! skip {f.name}: no date")
            continue
        # Stamp the model version the briefing ran under so the reliability
        # gate can score the CURRENT model instead of a four-month average of
        # every version that ever ran. Older briefings carry no stamp.
        mv = payload.get("model_version")
        for v in _extract_verdicts(payload):
            rows.append({"briefing_date": bd, "model_version": mv, **v})

    if not rows:
        print("No verdicts extracted from briefings.")
        return 1

    tickers = sorted({r["ticker"] for r in rows})
    start = min(r["briefing_date"] for r in rows)
    end = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"Grading {len(rows)} verdicts across {len(files)} briefings "
          f"({len(tickers)} tickers, {start}..{end}, horizons={horizons})")
    print("Fetching realized prices...")

    panel = bt_mod._price_panel(tickers, start=start, end=end)
    if panel.empty:
        print("No realized prices fetched (network / SSL issue?). Cannot grade.")
        return 1
    bench = _benchmark_series(start, end)
    bench_name = "EGX30"
    if bench is None:
        print("EGX30 index unavailable on Yahoo — using synthetic equal-weight basket.")
        bench = _synthetic_basket(start, end)
        bench_name = "synthetic equal-weight basket"
    if bench is None:
        print("WARNING: no benchmark available — grading vs absolute return (0%).")
        bench_name = "absolute (no benchmark!)"

    graded = _grade(rows, panel, bench, horizons)

    out_jsonl, out_csv = Path(args.out_jsonl), Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for g in graded:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(graded)

    _summary(graded, bench_name)
    print(f"\nWrote {len(graded)} rows -> {out_jsonl}")
    print(f"                      -> {out_csv}")
    print("\nThis is the evidence base. Re-run after each briefing — it grows and")
    print("pending rows get graded automatically. Feed it to tests/calibration_report.py.")
    print("\nTo version it on Hugging Face (once you have a meaningful sample):")
    print("    from datasets import load_dataset")
    print(f"    ds = load_dataset('json', data_files='{out_jsonl.name}')")
    print("    ds.push_to_hub('m0hamedm0sad/egx-verdict-outcomes')   # needs `huggingface_hub` login")
    return 0


if __name__ == "__main__":
    sys.exit(main())
