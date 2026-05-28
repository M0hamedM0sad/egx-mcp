"""Decision memory + realized-return reflection.

Logs every `decide()` call to a JSONL file, then later replays the log
against actual market data to grade hits, measure slippage, and surface
patterns ("BUYs with imminent earnings underperformed by 4%").

This is the only mechanism that lets the system learn from itself.
TradingAgents calls this "reflection." Mechanically:

  1. log_decision(...)        — append a row to logs/decisions.jsonl
  2. reflect(window_days=90)  — replay the log, fetch realized prices
                                from yfinance for each (ticker, date),
                                compute hit rate, average return, and
                                cluster failures by tag

The log path is configurable via $EGX_DECISION_LOG, defaulting to
logs/decisions.jsonl in the project tree.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf

from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.reflection")


def _log_path() -> Path:
    env = os.environ.get("EGX_DECISION_LOG")
    if env:
        return Path(env).expanduser().resolve()
    # Default to logs/decisions.jsonl alongside the package
    here = Path(__file__).resolve().parents[2]  # egx-mcp/
    return here / "logs" / "decisions.jsonl"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_decision(
    ticker: str,
    verdict: str,
    composite_score: float | None = None,
    fair_value: float | None = None,
    target_price: float | None = None,
    stop_loss: float | None = None,
    entry_price: float | None = None,
    conviction: str | None = None,
    blocking_catalysts: list[str] | None = None,
    tags: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Append a decision record to the log. Idempotent, append-only."""
    canonical, _, _ = resolve_ticker(ticker)
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "ticker": canonical,
        "verdict": verdict,
        "conviction": conviction,
        "composite_score": composite_score,
        "entry_price": entry_price,
        "fair_value": fair_value,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "blocking_catalysts": blocking_catalysts or [],
        "tags": tags or [],
        "note": note,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"logged": True, "path": str(path), "row": row}


def _load_log() -> list[dict[str, Any]]:
    path = _log_path()
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Realized-return replay
# ---------------------------------------------------------------------------

def _realized_return(ticker_yahoo: str, ts_iso: str, hold_days: int) -> dict[str, Any]:
    """Fetch close on the decision date and `hold_days` later. Returns pct change."""
    try:
        decision_dt = datetime.fromisoformat(ts_iso.replace("Z", ""))
    except Exception:
        return {"error": f"bad timestamp {ts_iso}"}

    end_dt = decision_dt + timedelta(days=hold_days + 7)
    if end_dt > datetime.utcnow():
        end_dt = datetime.utcnow()

    try:
        h = yf.Ticker(ticker_yahoo).history(
            start=decision_dt.date().isoformat(),
            end=end_dt.date().isoformat(),
            interval="1d",
        )
    except Exception as e:
        return {"error": f"yfinance fetch failed: {e}"}

    if h is None or h.empty or len(h) < 2:
        return {"error": "insufficient history"}

    entry = float(h["Close"].iloc[0])
    # Find the bar closest to decision_dt + hold_days
    target_dt = decision_dt + timedelta(days=hold_days)
    h_after = h[h.index.date >= target_dt.date()]
    if h_after.empty:
        exit_price = float(h["Close"].iloc[-1])
        actual_days = (h.index[-1].to_pydatetime() - decision_dt).days
    else:
        exit_price = float(h_after["Close"].iloc[0])
        actual_days = (h_after.index[0].to_pydatetime() - decision_dt).days

    return {
        "entry_price": round(entry, 4),
        "exit_price": round(exit_price, 4),
        "return_pct": round((exit_price / entry - 1) * 100, 2),
        "hold_days_actual": actual_days,
    }


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

_BUY_SIDE = {"BUY", "ACCUMULATE"}
_SELL_SIDE = {"REDUCE", "AVOID"}


def reflect(window_days: int = 90, hold_days: int = 21) -> dict[str, Any]:
    """Replay logged decisions and grade them against actual returns.

    Args:
        window_days: Lookback for which decisions to grade. Default 90.
        hold_days: Holding period applied to each call. Default 21 (~1mo).

    Returns:
        Dict with: total_decisions, graded, hit_rate, mean_return_pct,
        by_verdict, by_tag, recent_misses, lessons.
    """
    rows = _load_log()
    if not rows:
        return {"error": "decision log is empty", "path": str(_log_path())}

    cutoff = datetime.utcnow() - timedelta(days=window_days)
    in_window = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["ts"].replace("Z", ""))
        except Exception:
            continue
        if ts >= cutoff:
            in_window.append(r)

    graded: list[dict[str, Any]] = []
    for r in in_window:
        _, yahoo, _ = resolve_ticker(r["ticker"])
        actual = _realized_return(yahoo, r["ts"], hold_days)
        if "error" in actual:
            graded.append({**r, "grade": None, "actual": actual})
            continue

        verdict = r.get("verdict", "")
        ret = actual["return_pct"]
        if verdict in _BUY_SIDE:
            hit = ret > 0
        elif verdict in _SELL_SIDE:
            hit = ret < 0
        else:  # HOLD / WAIT
            hit = abs(ret) < 5.0  # within ±5% counts as a sensible hold

        graded.append({
            **r,
            "actual_return_pct": ret,
            "actual_hold_days": actual["hold_days_actual"],
            "hit": hit,
        })

    grade_clean = [g for g in graded if g.get("hit") is not None]
    n_graded = len(grade_clean)

    if n_graded == 0:
        return {
            "as_of": datetime.utcnow().isoformat() + "Z",
            "log_path": str(_log_path()),
            "window_days": window_days,
            "hold_days_per_call": hold_days,
            "total_decisions_in_window": len(in_window),
            "graded": 0,
            "warning": (
                "No decision in the window has enough realized history yet. "
                f"Need at least {hold_days} trading days past the decision."
            ),
        }

    hits = sum(1 for g in grade_clean if g["hit"])
    mean_ret = sum(g["actual_return_pct"] for g in grade_clean) / n_graded

    by_verdict: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "hits": 0, "mean_return": 0.0, "returns": []}
    )
    for g in grade_clean:
        v = g["verdict"]
        by_verdict[v]["n"] += 1
        by_verdict[v]["hits"] += int(g["hit"])
        by_verdict[v]["returns"].append(g["actual_return_pct"])
    for v, b in by_verdict.items():
        b["mean_return"] = round(sum(b["returns"]) / b["n"], 2) if b["n"] else 0
        b["hit_rate"] = round(b["hits"] / b["n"] * 100, 1) if b["n"] else 0
        del b["returns"]

    by_tag: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "hits": 0, "mean_return": 0.0, "returns": []}
    )
    for g in grade_clean:
        for tag in g.get("tags") or []:
            by_tag[tag]["n"] += 1
            by_tag[tag]["hits"] += int(g["hit"])
            by_tag[tag]["returns"].append(g["actual_return_pct"])
        if g.get("blocking_catalysts"):
            t = "blocking_catalyst"
            by_tag[t]["n"] += 1
            by_tag[t]["hits"] += int(g["hit"])
            by_tag[t]["returns"].append(g["actual_return_pct"])
    for t, b in by_tag.items():
        b["mean_return"] = round(sum(b["returns"]) / b["n"], 2) if b["n"] else 0
        b["hit_rate"] = round(b["hits"] / b["n"] * 100, 1) if b["n"] else 0
        del b["returns"]

    misses = sorted(
        (g for g in grade_clean if not g["hit"]),
        key=lambda g: g["actual_return_pct"]
            if g["verdict"] in _BUY_SIDE
            else -g["actual_return_pct"],
    )[:5]

    lessons: list[str] = []
    for tag, b in by_tag.items():
        if b["n"] >= 3 and b["hit_rate"] < 40:
            lessons.append(
                f"Decisions tagged '{tag}' have {b['hit_rate']:.0f}% hit rate "
                f"over {b['n']} samples — consider a stricter filter."
            )
    for v, b in by_verdict.items():
        if v in _BUY_SIDE and b["n"] >= 3 and b["hit_rate"] < 40:
            lessons.append(
                f"{v} verdicts hit {b['hit_rate']:.0f}% over {b['n']} samples — "
                f"raise the score threshold or layer in stricter filters."
            )

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "log_path": str(_log_path()),
        "window_days": window_days,
        "hold_days_per_call": hold_days,
        "total_decisions_in_window": len(in_window),
        "graded": n_graded,
        "hit_rate_pct": round(hits / n_graded * 100, 1),
        "mean_return_pct": round(mean_ret, 2),
        "by_verdict": dict(by_verdict),
        "by_tag": dict(by_tag),
        "recent_misses": misses,
        "lessons": lessons,
        "method": (
            "Replays each decision in window against yfinance close-to-close "
            "returns over `hold_days`. Hit definition: BUY/ACCUMULATE → "
            "positive return; REDUCE/AVOID → negative; HOLD/WAIT → |return|<5%."
        ),
    }
