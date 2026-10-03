"""Probe which sources can supply POINT-IN-TIME historical fundamentals for EGX.

The live model's valuation and quality sub-scores (both negatively related to
21-session excess in live grading) cannot be backtested honestly: the only
fundamentals on file are a current snapshot plus a daily history that starts
2026-08-15. This probe answers, per source, before anything is built on it:

  - does it return MORE THAN ONE period (history, not just "latest")?
  - how far back does it go, and at what frequency (quarterly / annual)?
  - does it carry a PUBLICATION date, or only a period-end date (which then
    needs a conservative reporting lag to be point-in-time)?
  - how many EGX names does it cover?

Sources probed: TradingView scanner history columns, Yahoo statements,
Mubasher statement pages, the EGX disclosures portal. Read-only, low volume,
never fails the run: every probe records its own error.

    python -m scripts.probe_fundamentals_history
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
_OUT = ROOT / "logs" / "probe_fundamentals_history.json"

# Large caps the model favours plus small caps it has been ranking low.
_SAMPLE = ["COMI", "SWDY", "ABUK", "TMGH", "EFIH", "ETEL", "AMES", "LUTS", "GTWL", "MPCO"]
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
       "Accept-Language": "en-US,en;q=0.9,ar;q=0.8"}
_YEAR = re.compile(r"\b(20[0-2]\d)\b")
_QTR = re.compile(r"\b(Q[1-4]\s*[-/ ]?\s*20[0-2]\d|20[0-2]\d\s*[-/ ]?\s*Q[1-4])\b", re.I)
_FIN_TITLE = re.compile(r"financial|statement|results|قوائم|نتائج|مالية", re.I)


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:200]}"


def _years(values) -> list[int]:
    ys = set()
    for v in values:
        for m in _YEAR.findall(str(v)):
            ys.add(int(m))
    return sorted(ys)


# ---------------------------------------------------------------------------
# TradingView scanner
# ---------------------------------------------------------------------------

_TV_HISTORY_COLS = [
    # history arrays (_h) — one value per past period, newest first
    "earnings_per_share_diluted_fq_h", "earnings_per_share_basic_fq_h",
    "earnings_per_share_diluted_fy_h", "net_income_fq_h", "net_income_fy_h",
    "total_revenue_fq_h", "total_revenue_fy_h", "total_equity_fq_h",
    "total_debt_fq_h", "return_on_equity_fq_h", "book_value_per_share_fq_h",
    "fiscal_period_end_fq_h", "fiscal_period_end_fy_h",
    "fiscal_period_fq_h", "fiscal_period_fy_h",
    # equity / balance-sheet history under alternative field names
    "total_equity_fy_h", "total_shareholders_equity_fq_h", "total_shareholders_equity_fy_h",
    "shareholders_equity_fq_h", "common_equity_total_fq_h", "total_assets_fq_h",
    "total_assets_fy_h", "total_liabilities_fq_h", "total_liabilities_fy_h",
    "book_value_per_share_fy_h", "return_on_equity_fy_h", "total_shares_outstanding_fq_h",
    "dividends_per_share_fy_h", "fiscal_period_end_fy", "fiscal_period_end_fq",
    # publication / release dates
    "earnings_release_date", "earnings_release_date_fq_h",
    "earnings_release_next_date", "earnings_publication_type_fq_h",
]


def probe_tradingview() -> dict:
    import httpx
    from egx_mcp.data._certs import ensure_ca_bundle

    ensure_ca_bundle()
    url = "https://scanner.tradingview.com/egypt/scan"
    hdr = {**_UA, "Origin": "https://www.tradingview.com"}
    out: dict = {"source": url, "columns": {}}
    # One column per request: an unknown field makes the scanner reject the
    # whole body, which would hide the columns that do exist.
    for col in _TV_HISTORY_COLS:
        body = {"filter": [], "options": {"lang": "en"}, "markets": ["egypt"],
                "symbols": {"query": {"types": []}, "tickers": []},
                "columns": ["name", col], "range": [0, 400]}
        rec: dict = {}
        try:
            r = httpx.post(url, json=body, timeout=30, headers=hdr)
            rec["status"] = r.status_code
            if r.status_code != 200:
                rec["body"] = r.text[:200]
            else:
                rows = r.json().get("data", [])
                vals = {row["d"][0]: row["d"][1] for row in rows}
                nonnull = {k: v for k, v in vals.items() if v not in (None, [], "")}
                rec["names_total"] = len(vals)
                rec["names_with_value"] = len(nonnull)
                lens = [len(v) for v in nonnull.values() if isinstance(v, list)]
                if lens:
                    rec["history_len_median"] = sorted(lens)[len(lens) // 2]
                    rec["history_len_max"] = max(lens)
                rec["samples"] = {tk: nonnull.get(tk) for tk in _SAMPLE[:3]}
        except Exception as e:  # noqa: BLE001
            rec["error"] = _err(e)
        out["columns"][col] = rec
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# Yahoo statements
# ---------------------------------------------------------------------------

def probe_yahoo() -> dict:
    import yfinance as yf
    from egx_mcp.data.universe import resolve_ticker

    out: dict = {"tickers": {}}
    for tk in _SAMPLE:
        _, yahoo, _ = resolve_ticker(tk)
        rec: dict = {"symbol": yahoo}
        try:
            t = yf.Ticker(yahoo)
            for name, freq, kind in (("income_q", "quarterly", "income"),
                                     ("income_a", "yearly", "income"),
                                     ("balance_q", "quarterly", "balance"),
                                     ("balance_a", "yearly", "balance")):
                try:
                    df = (t.get_income_stmt(freq=freq) if kind == "income"
                          else t.get_balance_sheet(freq=freq))
                    if df is None or df.empty:
                        rec[name] = {"periods": 0}
                        continue
                    cols = [str(c)[:10] for c in df.columns]
                    rec[name] = {
                        "periods": len(cols), "oldest": min(cols), "newest": max(cols),
                        "has_net_income": any("NetIncome" in str(i) for i in df.index),
                        "has_equity": any("StockholdersEquity" in str(i) for i in df.index),
                        "has_eps": any("EPS" in str(i) for i in df.index),
                    }
                except Exception as e:  # noqa: BLE001
                    rec[name] = {"error": _err(e)}
            try:
                ed = t.get_earnings_dates(limit=24)
                rec["earnings_dates"] = (
                    {"n": len(ed), "oldest": str(ed.index.min())[:10],
                     "newest": str(ed.index.max())[:10]}
                    if ed is not None and not ed.empty else {"n": 0})
            except Exception as e:  # noqa: BLE001
                rec["earnings_dates"] = {"error": _err(e)}
        except Exception as e:  # noqa: BLE001
            rec["error"] = _err(e)
        out["tickers"][tk] = rec
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# Mubasher statement pages
# ---------------------------------------------------------------------------

_MUBASHER_PATHS = ["ratios", "financial-statements", "financials", "income-statement",
                   "balance-sheet", "financial-results", "results"]


def probe_mubasher() -> dict:
    import httpx
    from bs4 import BeautifulSoup

    out: dict = {"tickers": {}}
    for tk in _SAMPLE[:5]:
        base = f"https://english.mubasher.info/markets/EGX/stocks/{tk}"
        rec: dict = {}
        with httpx.Client(timeout=20, headers=_UA, follow_redirects=True) as c:
            for path in _MUBASHER_PATHS:
                url = f"{base}/{path}"
                try:
                    r = c.get(url)
                    info: dict = {"status": r.status_code, "final_url": str(r.url)[:120],
                                  "bytes": len(r.text)}
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, "html.parser")
                        heads = [th.get_text(" ", strip=True)
                                 for th in soup.find_all(["th"])][:60]
                        info["tables"] = len(soup.find_all("table"))
                        info["header_years"] = _years(heads)
                        info["quarter_tokens"] = sorted(set(_QTR.findall(r.text)))[:12]
                        info["page_years"] = _years([r.text])[-10:]
                    rec[path] = info
                except Exception as e:  # noqa: BLE001
                    rec[path] = {"error": _err(e)}
                time.sleep(0.4)
        out["tickers"][tk] = rec
    return out


# ---------------------------------------------------------------------------
# EGX disclosures portal
# ---------------------------------------------------------------------------

_EGX_URLS = [
    "https://www.egx.com.eg/en/Disclosure.aspx",
    "https://www.egx.com.eg/ar/Disclosure.aspx",
    "https://www.egx.com.eg/en/CompanyDisclosure.aspx",
    "https://www.egx.com.eg/en/stocksdata.aspx",
]


def probe_egx() -> dict:
    import httpx
    from bs4 import BeautifulSoup
    from egx_mcp.data.disclosures import _try_parse_date

    out: dict = {"pages": {}}
    with httpx.Client(timeout=25, headers=_UA, follow_redirects=True) as c:
        for url in _EGX_URLS:
            info: dict = {}
            try:
                r = c.get(url)
                info.update({"status": r.status_code, "final_url": str(r.url)[:120],
                             "bytes": len(r.text)})
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    dates, fin = [], 0
                    for row in soup.find_all("tr"):
                        cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
                        for cell in cells[:2]:
                            d = _try_parse_date(cell)
                            if d:
                                dates.append(d)
                                if _FIN_TITLE.search(" ".join(cells)):
                                    fin += 1
                                break
                    info["dated_rows"] = len(dates)
                    info["financial_rows"] = fin
                    if dates:
                        info["oldest"] = min(dates).strftime("%Y-%m-%d")
                        info["newest"] = max(dates).strftime("%Y-%m-%d")
                    # ASP.NET postback paging => older pages are reachable.
                    info["aspnet_viewstate"] = bool(soup.find("input", {"name": "__VIEWSTATE"}))
                    info["date_inputs"] = [i.get("name") for i in soup.find_all("input")
                                           if re.search(r"date|from|to", i.get("name") or "", re.I)][:8]
                    info["pager_links"] = len([a for a in soup.find_all("a", href=True)
                                               if "__doPostBack" in a["href"] and "Page" in a["href"]])
            except Exception as e:  # noqa: BLE001
                info["error"] = _err(e)
            out["pages"][url] = info
            time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------

def _verdict(res: dict) -> list[str]:
    lines = []
    tv = res.get("tradingview", {}).get("columns", {})
    hist = {c: r for c, r in tv.items() if c.endswith("_h") and r.get("names_with_value")}
    if hist:
        best = max(hist.items(), key=lambda kv: (kv[1]["names_with_value"],
                                                  kv[1].get("history_len_median", 0)))
        lines.append(f"TradingView: {len(hist)} history column(s) populated; best "
                     f"{best[0]} -> {best[1]['names_with_value']} names, median "
                     f"{best[1].get('history_len_median')} periods.")
    else:
        lines.append("TradingView: no populated history columns.")
    rel = {c: r for c, r in tv.items() if "release" in c and r.get("names_with_value")}
    lines.append(f"TradingView release-date columns populated: {sorted(rel) or 'none'}")

    yh = res.get("yahoo", {}).get("tickers", {})
    q = [r.get("income_q", {}).get("periods", 0) for r in yh.values() if isinstance(r, dict)]
    a = [r.get("income_a", {}).get("periods", 0) for r in yh.values() if isinstance(r, dict)]
    if q:
        lines.append(f"Yahoo: quarterly income periods per name {q}; annual {a}.")

    mb = res.get("mubasher", {}).get("tickers", {})
    ok_paths = sorted({p for r in mb.values() for p, i in r.items()
                       if i.get("status") == 200 and len(i.get("header_years", [])) > 1})
    lines.append(f"Mubasher pages with multi-year tables: {ok_paths or 'none'}")

    egx = res.get("egx", {}).get("pages", {})
    for u, i in egx.items():
        if i.get("status") == 200:
            lines.append(f"EGX {u.split('/')[-1]}: {i.get('dated_rows', 0)} dated rows "
                         f"({i.get('oldest')}..{i.get('newest')}), "
                         f"{i.get('financial_rows', 0)} financial, "
                         f"paging={'yes' if i.get('pager_links') or i.get('aspnet_viewstate') else 'no'}")
        else:
            lines.append(f"EGX {u.split('/')[-1]}: {i.get('status') or i.get('error')}")
    return lines


def main() -> int:
    res: dict = {"probed_at": datetime.now(timezone.utc).isoformat(), "sample": _SAMPLE}
    for name, fn in (("tradingview", probe_tradingview), ("yahoo", probe_yahoo),
                     ("mubasher", probe_mubasher), ("egx", probe_egx)):
        print(f"probing {name} ...", flush=True)
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": _err(e)}
    res["verdict"] = _verdict(res)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== VERDICT ===")
    for line in res["verdict"]:
        print(" ", line)
    print("\n=== FULL RESULT ===")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
