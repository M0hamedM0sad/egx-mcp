"""Point-in-time historical fundamentals from TradingView's scanner history.

The live fundamentals (Mubasher + TradingView fill) are a current snapshot,
with a daily history only from 2026-08-15, so valuation and quality could not
be backtested without look-ahead. TradingView's scanner serves per-company
history arrays for the whole Egyptian market in one POST per field:

    *_fq_h   last ~32 fiscal quarters (8 years), newest first
    fiscal_period_end_fq   period-end of the newest quarter in those arrays

Probed 2026-09-25: EPS (diluted), net income, revenue, total debt and total
assets are populated for ~242 names. Equity, book value, ROE and per-quarter
publication dates are NOT served, so ROE / P/B cannot be rebuilt; ROA, net
margin, debt/assets and trailing E/P can.

Point-in-time rule: a quarter becomes visible REPORTING_LAG_DAYS after its
period end. EGX listing rules give companies 45 days after a quarter (longer
after the fiscal year), so 60 days errs late rather than early. Two residual
biases, stated wherever results are reported:
  - restatements: TradingView serves today's (possibly restated) figures;
  - survivorship: only currently listed names are covered.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("egx-mcp.tv_history")

ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = ROOT / "logs" / "tv_fundamentals_history.json"

REPORTING_LAG_DAYS = 60

_SCAN_URL = "https://scanner.tradingview.com/egypt/scan"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0 Safari/537.36",
            "Origin": "https://www.tradingview.com"}

# history field -> short name
QUARTERLY_FIELDS = {
    "earnings_per_share_diluted_fq_h": "eps",
    "net_income_fq_h": "net_income",
    "total_revenue_fq_h": "revenue",
    "total_debt_fq_h": "total_debt",
    "total_assets_fq_h": "total_assets",
}
# Current values, used only to sanity-check units (e.g. USD reporters).
_CHECK_FIELDS = ["fiscal_period_end_fq", "price_earnings_ttm", "close", "currency",
                 "fundamental_currency_code"]


def _scan(columns: list[str]) -> dict[str, list]:
    import httpx
    from ._certs import ensure_ca_bundle

    ensure_ca_bundle()
    body = {"filter": [], "options": {"lang": "en"}, "markets": ["egypt"],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": ["name", *columns], "range": [0, 1000]}
    r = httpx.post(_SCAN_URL, json=body, timeout=45, headers=_HEADERS)
    r.raise_for_status()
    return {row["d"][0]: row["d"][1:] for row in r.json().get("data", [])}


def fetch(throttle_s: float = 0.5) -> dict[str, Any]:
    """Pull every history field for the whole market. One POST per field so an
    unknown/removed field fails alone instead of taking the whole pull down."""
    per_ticker: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for field, short in QUARTERLY_FIELDS.items():
        try:
            for tk, (vals,) in _scan([field]).items():
                if isinstance(vals, list) and vals:
                    per_ticker.setdefault(tk, {"q": {}})["q"][short] = vals
        except Exception as e:  # noqa: BLE001
            errors[field] = f"{type(e).__name__}: {e}"
        time.sleep(throttle_s)
    for field in _CHECK_FIELDS:
        try:
            for tk, (val,) in _scan([field]).items():
                if tk in per_ticker and val not in (None, ""):
                    per_ticker[tk][field] = val
        except Exception as e:  # noqa: BLE001
            errors[field] = f"{type(e).__name__}: {e}"
        time.sleep(throttle_s)
    return {"fetched_at": datetime.now(timezone.utc).isoformat(),
            "reporting_lag_days": REPORTING_LAG_DAYS,
            "n_tickers": len(per_ticker), "errors": errors, "tickers": per_ticker}


def save(payload: dict[str, Any], path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load(path: Path = HISTORY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _month_end(y: int, m: int) -> date:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return nxt - timedelta(days=1)


def quarter_ends(latest_end: date, n: int) -> list[date]:
    """Period-end dates for n quarters, newest first, stepping back 3 months."""
    out, y, m = [], latest_end.year, latest_end.month
    for _ in range(n):
        out.append(_month_end(y, m))
        m -= 3
        if m <= 0:
            m += 12
            y -= 1
    return out


def _as_date(v: Any) -> date | None:
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc).date()
    if isinstance(v, str):
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def quarterly_table(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of {period_end, visible_from, eps, net_income, ...}, oldest first."""
    latest = _as_date(entry.get("fiscal_period_end_fq"))
    q = entry.get("q") or {}
    if latest is None or not q:
        return []
    n = max(len(v) for v in q.values())
    ends = quarter_ends(latest, n)
    rows = []
    for i, end in enumerate(ends):
        row: dict[str, Any] = {"period_end": end,
                               "visible_from": end + timedelta(days=REPORTING_LAG_DAYS)}
        for short, vals in q.items():
            row[short] = vals[i] if i < len(vals) else None
        rows.append(row)
    return rows[::-1]


def pit(table: list[dict[str, Any]], asof: date) -> dict[str, Any] | None:
    """Fundamentals as they could have been known on `asof`.

    TTM flows (EPS, net income, revenue) need the last four visible quarters
    all present; stocks (assets, debt) use the latest visible quarter.
    """
    seen = [r for r in table if r["visible_from"] <= asof]
    if not seen:
        return None
    last = seen[-1]
    out: dict[str, Any] = {"period_end": last["period_end"].isoformat(),
                           "total_assets": last.get("total_assets"),
                           "total_debt": last.get("total_debt")}
    last4 = seen[-4:]
    for k in ("eps", "net_income", "revenue"):
        vals = [r.get(k) for r in last4]
        out[f"ttm_{k}"] = (float(sum(vals)) if len(last4) == 4
                           and all(isinstance(v, (int, float)) for v in vals) else None)
    return out


def ratios(f: dict[str, Any] | None, price: float | None) -> dict[str, float | None]:
    """Point-in-time proxies for the live valuation / quality inputs."""
    nan: dict[str, float | None] = {"earnings_yield": None, "roa": None,
                                    "net_margin": None, "debt_to_assets": None}
    if not f:
        return nan
    out = dict(nan)
    eps, ni, rev = f.get("ttm_eps"), f.get("ttm_net_income"), f.get("ttm_revenue")
    assets, debt = f.get("total_assets"), f.get("total_debt")
    if eps is not None and price and price > 0:
        out["earnings_yield"] = eps / price          # E/P: defined for losses too
    if ni is not None and assets and assets > 0:
        out["roa"] = ni / assets
    if ni is not None and rev and rev > 0:
        out["net_margin"] = ni / rev
    if debt is not None and assets and assets > 0:
        out["debt_to_assets"] = debt / assets
    return out


def units_suspect(entry: dict[str, Any]) -> bool:
    """True when our TTM P/E at TradingView's own close disagrees with its P/E
    by more than 2x — the signature of statements in another currency than the
    quote, or of misaligned quarter arrays. Such names are excluded."""
    pe, latest_price = entry.get("price_earnings_ttm"), entry.get("close")
    table = quarterly_table(entry)
    if not table or not pe or pe <= 0 or not latest_price:
        return False
    f = pit(table, date.max)
    if not f or not f.get("ttm_eps") or f["ttm_eps"] <= 0:
        return False
    ours = latest_price / f["ttm_eps"]
    return not (0.5 <= ours / pe <= 2.0)
