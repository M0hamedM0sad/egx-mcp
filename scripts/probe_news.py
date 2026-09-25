"""Probe the live news sources: how many headlines now carry a publication
date, and what the CBE rates page serves (its scrape returned null in all
40 recent briefings). Read-only; never fails the run.

    python -m scripts.probe_news
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import news_scrapers as ns  # noqa: E402

_TICKERS = ["COMI", "SWDY", "TMGH", "MAAL", "ALCN", "EGTS"]
_CBE_URLS = [
    "https://www.cbe.org.eg/en/economic-research/statistics/cbe-rates",
    "https://www.cbe.org.eg/en/monetary-policy",
    "https://www.cbe.org.eg/ar/monetary-policy",
]


def _summ(items: list[dict]) -> dict:
    dated = [it for it in items if it.get("date")]
    return {"n": len(items), "dated": len(dated),
            "dates": sorted({it["date"] for it in dated})[-6:],
            "samples": [{k: it.get(k) for k in ("date", "published", "title", "url")}
                        for it in items[:3]]}


def main() -> int:
    res: dict = {"sources": {}, "cbe": {}}
    for name, fn in (("mubasher_market", lambda: ns.fetch_mubasher_market(limit=8)),
                     ("enterprise", lambda: ns.fetch_enterprise(limit=6)),
                     ("daily_news_egypt", lambda: ns.fetch_daily_news_egypt(limit=6)),
                     *[(f"mubasher_{t}", (lambda t=t: ns.fetch_mubasher_stock(t, limit=5)))
                       for t in _TICKERS]):
        try:
            res["sources"][name] = _summ(fn())
        except Exception as e:  # noqa: BLE001
            res["sources"][name] = {"error": f"{type(e).__name__}: {e}"}
    import httpx
    for url in _CBE_URLS:
        try:
            r = httpx.get(url, timeout=20, follow_redirects=True, headers=ns._HEADERS)
            text = re.sub(r"\s+", " ", r.text)
            pct = re.findall(r"(\d{1,2}\.\d{1,3})\s*%", text)
            ctx = [text[max(0, m.start() - 80):m.end() + 40]
                   for m in re.finditer(r"(?i)overnight|lending rate|deposit rate|الإيداع|الإقراض", text)][:4]
            res["cbe"][url] = {"status": r.status_code, "bytes": len(r.text),
                               "percentages": pct[:12], "context": ctx}
        except Exception as e:  # noqa: BLE001
            res["cbe"][url] = {"error": f"{type(e).__name__}: {e}"}

    # Where does a Mubasher article keep its date? Dump every candidate.
    arts = [it["url"] for v in res["sources"].values() for it in v.get("samples", [])
            if "mubasher.info/news/" in (it.get("url") or "")][:2]
    months = ("يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|"
              "اكتوبر|نوفمبر|ديسمبر")
    res["mubasher_article"] = {}
    for url in arts:
        try:
            r = httpx.get(url, timeout=20, follow_redirects=True, headers=ns._HEADERS)
            html = r.text
            flat = re.sub(r"\s+", " ", html)
            metas = re.findall(r"<meta[^>]+>", html)[:60]
            res["mubasher_article"][url[:80]] = {
                "status": r.status_code, "bytes": len(html),
                "metas_with_digits": [m for m in metas if re.search(r"\d{4}", m)][:15],
                "iso_dates": sorted(set(re.findall(r"20\d\d-\d\d-\d\d[T ]?[\d:]{0,8}", flat)))[:10],
                "slash_dates": sorted(set(re.findall(r"\b\d{1,2}/\d{1,2}/20\d\d\b", flat)))[:10],
                "arabic_dates": sorted(set(re.findall(rf"\d{{1,2}}\s+(?:{months})\s+20\d\d", flat)))[:10],
                "time_tags": re.findall(r"<time[^>]*>[^<]{0,40}", html)[:5],
                "date_classes": [flat[m.start():m.start() + 160] for m in
                                 re.finditer(r'class="[^"]*(?:date|time|publish)[^"]*"', flat)][:6],
                "json_ld": len(re.findall(r"application/ld\+json", html)),
            }
        except Exception as e:  # noqa: BLE001
            res["mubasher_article"][url[:80]] = {"error": f"{type(e).__name__}: {e}"}

    total = sum(v.get("n", 0) for v in res["sources"].values())
    dated = sum(v.get("dated", 0) for v in res["sources"].values())
    print(f"=== VERDICT ===\n  headlines: {total}, with a publication date: {dated}")
    for k, v in res["sources"].items():
        print(f"  {k:22} {v.get('dated', 0)}/{v.get('n', 0)} dated  {v.get('dates', v.get('error', ''))}")
    for u, v in res["cbe"].items():
        print(f"  CBE {u.split('/')[-1]:14} status={v.get('status')} pct={v.get('percentages', v.get('error'))}")
    print("\n=== FULL RESULT ===")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
