"""Simulate EGX 30 performance over the next trading week.

Yahoo serves only 1 bar for ^CASE30, so we reconstruct the index daily-return
series from its constituents (cap-weighted via the fundamentals CSV), anchor it
to the real current ^CASE30 level, and bootstrap a 5-day Monte Carlo.

Data path: Playwright request context (Chromium trusts the Windows cert store;
the network's TLS-intercepting proxy breaks plain Python SSL / yfinance).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# EGX 30 constituents we have in the curated universe (Yahoo .CA symbols).
CONSTITUENTS = {
    "COMI": "COMI.CA", "HRHO": "HRHO.CA", "EFIH": "EFIH.CA", "TMGH": "TMGH.CA",
    "SWDY": "SWDY.CA", "ABUK": "ABUK.CA", "ETEL": "ETEL.CA", "ESRS": "ESRS.CA",
    "FWRY": "FWRY.CA", "EAST": "EAST.CA", "ADIB": "ADIB.CA", "CIEB": "CIEB.CA",
    "HDBK": "HDBK.CA", "MFPC": "MFPC.CA", "EFID": "EFID.CA", "JUFO": "JUFO.CA",
    "CLHO": "CLHO.CA", "IDHC": "IDHC.CA", "PHDC": "PHDC.CA", "MNHD": "MNHD.CA",
    "ORHD": "ORHD.CA", "EMFD": "EMFD.CA", "HELI": "HELI.CA", "EGTS": "EGTS.CA",
    "CIRA": "CIRA.CA", "ORWE": "ORWE.CA", "CCAP": "CCAP.CA", "MTIE": "MTIE.CA",
    "SODIC": "OCDI.CA", "ABUK2": "MFPC.CA",
}

HORIZON = 5
N_PATHS = 20000
LOOKBACK = 250  # ~1 trading year


def load_caps() -> dict[str, float]:
    caps: dict[str, float] = {}
    with open(ROOT / "egx_fundamentals_audited.csv", newline="") as f:
        for row in csv.DictReader(f):
            try:
                caps[row["ticker"]] = float(row["market_cap"])
            except (ValueError, KeyError, TypeError):
                pass
    return caps


def fetch_series(req, yahoo_sym: str) -> dict[int, float]:
    """Return {unix_day: close} for ~1y of daily bars, or {} on failure."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}"
           f"?range=1y&interval=1d")
    try:
        r = req.get(url, timeout=25000)
        j = r.json()
        res = j.get("chart", {}).get("result")
        if not res:
            return {}
        ts = res[0].get("timestamp") or []
        cl = res[0]["indicators"]["quote"][0].get("close") or []
        out = {}
        for t, c in zip(ts, cl):
            if c is not None:
                out[int(t)] = float(c)
        return out
    except Exception:
        return {}


def fetch_index_level(req) -> float | None:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%5ECASE30"
           "?range=5d&interval=1d")
    try:
        j = req.get(url, timeout=25000).json()
        res = j.get("chart", {}).get("result")
        cl = [x for x in res[0]["indicators"]["quote"][0].get("close") or [] if x is not None]
        return float(cl[-1]) if cl else None
    except Exception:
        return None


def main() -> None:
    caps = load_caps()
    series: dict[str, dict[int, float]] = {}
    used_caps: dict[str, float] = {}

    with sync_playwright() as p:
        req = p.request.new_context(extra_http_headers={"User-Agent": UA})
        index_level = fetch_index_level(req)
        for tk, ysym in CONSTITUENTS.items():
            s = fetch_series(req, ysym)
            if len(s) >= 60:
                base = "MFPC" if tk == "ABUK2" else tk
                cap = caps.get(base)
                if cap and base not in used_caps:
                    series[base] = s
                    used_caps[base] = cap
        req.dispose()

    if not series:
        print(json.dumps({"error": "no constituent data fetched"}))
        return

    # Common trading days across all fetched constituents.
    common = set.intersection(*(set(s.keys()) for s in series.values()))
    days = sorted(common)[-(LOOKBACK + 1):]
    if len(days) < 40:
        print(json.dumps({"error": f"insufficient overlap ({len(days)} days)"}))
        return

    total_cap = sum(used_caps.values())
    weights = {tk: used_caps[tk] / total_cap for tk in used_caps}

    # Cap-weighted daily index return series.
    index_rets: list[float] = []
    for i in range(1, len(days)):
        d0, d1 = days[i - 1], days[i]
        r = 0.0
        for tk, w in weights.items():
            p0, p1 = series[tk][d0], series[tk][d1]
            if p0 > 0:
                r += w * (p1 / p0 - 1)
        index_rets.append(r)

    mean_d = statistics.mean(index_rets)
    std_d = statistics.pstdev(index_rets)
    ann_vol = std_d * math.sqrt(250) * 100
    recent20 = index_rets[-20:]

    if index_level is None:
        index_level = 52658.80  # last known ^CASE30 anchor

    # Bootstrap Monte Carlo: sample HORIZON daily returns with replacement.
    rng = random.Random(42)
    terminals = []
    for _ in range(N_PATHS):
        lvl = index_level
        for _ in range(HORIZON):
            lvl *= (1 + rng.choice(index_rets))
        terminals.append(lvl)

    terminals.sort()

    def prob_ge(thr_pct):
        target = index_level * (1 + thr_pct / 100)
        return round(sum(1 for t in terminals if t >= target) / N_PATHS * 100, 1)

    def prob_le(thr_pct):
        target = index_level * (1 + thr_pct / 100)
        return round(sum(1 for t in terminals if t <= target) / N_PATHS * 100, 1)

    def pct(q):
        k = (len(terminals) - 1) * q
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return terminals[int(k)]
        return terminals[f] + (terminals[c] - terminals[f]) * (k - f)

    def ret(level):
        return round((level / index_level - 1) * 100, 2)

    mean_term = statistics.mean(terminals)
    prob_up = sum(1 for t in terminals if t > index_level) / N_PATHS

    out = {
        "as_of": "2026-05-28",
        "forecast_week": "2026-05-31 (Sun) to 2026-06-04 (Thu)",
        "current_index_level": round(index_level, 2),
        "constituents_used": len(weights),
        "lookback_days": len(index_rets),
        "top_weights": sorted(
            ({"ticker": k, "weight_pct": round(v * 100, 1)} for k, v in weights.items()),
            key=lambda x: -x["weight_pct"])[:8],
        "hist_daily_mean_pct": round(mean_d * 100, 3),
        "hist_daily_std_pct": round(std_d * 100, 3),
        "annualized_vol_pct": round(ann_vol, 1),
        "recent_20d_cum_pct": round((math.prod(1 + r for r in recent20) - 1) * 100, 2),
        "mc": {
            "n_paths": N_PATHS,
            "horizon_trading_days": HORIZON,
            "expected_level": round(mean_term, 1),
            "expected_return_pct": ret(mean_term),
            "prob_week_up_pct": round(prob_up * 100, 1),
            "prob_up_ge_1pct": prob_ge(1),
            "prob_up_ge_2pct": prob_ge(2),
            "prob_up_ge_3pct": prob_ge(3),
            "prob_down_ge_1pct": prob_le(-1),
            "prob_down_ge_2pct": prob_le(-2),
            "prob_down_ge_3pct": prob_le(-3),
            "p05_level": round(pct(0.05), 1), "p05_return_pct": ret(pct(0.05)),
            "p25_level": round(pct(0.25), 1), "p25_return_pct": ret(pct(0.25)),
            "p50_level": round(pct(0.50), 1), "p50_return_pct": ret(pct(0.50)),
            "p75_level": round(pct(0.75), 1), "p75_return_pct": ret(pct(0.75)),
            "p95_level": round(pct(0.95), 1), "p95_return_pct": ret(pct(0.95)),
            "var_95_1week_pct": ret(pct(0.05)),
        },
    }
    print(json.dumps(out, indent=2))
    (ROOT / "logs" / "egx30_sim.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
