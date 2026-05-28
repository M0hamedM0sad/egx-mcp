"""Investor-relations document acquisition — per-company archive.

Builds and maintains an `ir_data/<TICKER>/` folder for each EGX name:

    ir_data/
      ir_registry.json        ticker -> discovered IR url + confidence + verified
      <TICKER>/
        manifest.json         fetched docs: source url, date, sha256, bytes, kind
        documents/            the raw downloaded files (PDFs) — source of truth

Three responsibilities:
  - scaffold_company()  create the folder + empty manifest
  - discover_ir_url()   best-effort find a company's IR page (auto-discovery,
                        always flagged low/medium/high confidence, verified=False
                        until a human confirms — NEVER trusted blindly)
  - fetch_documents()   download new IR docs from a (verified) IR url into the
                        company folder, deduped by content hash, manifest updated

Politeness: respects robots.txt, sets a contact User-Agent, caps file size,
and sleeps briefly between downloads. This fetches public IR pages for your own
research — keep it low-volume.

Extraction of figures from the downloaded docs lives in ir_extract.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from .universe import EGX_UNIVERSE

log = logging.getLogger("egx-mcp.ir_fetch")

IR_ROOT = Path(__file__).parent.parent.parent / "ir_data"
_REGISTRY = IR_ROOT / "ir_registry.json"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (egx-mcp IR research; mailto:m0hamedm0sad@gmail.com)",
    "Accept-Language": "ar,en;q=0.9",
}
_MAX_BYTES = 30 * 1024 * 1024   # 30 MB per file cap
_FETCH_DELAY = 1.5              # seconds between downloads — be polite

# Link text / href hints that mark a financial document worth archiving.
_DOC_HINTS = (
    "financial", "statement", "results", "report", "annual", "quarter",
    "earnings", "disclosure", "presentation", "balance", "income",
    "القوائم", "المالية", "نتائج", "تقرير", "سنوي", "ربع", "إفصاح", "عرض",
)
# Domains that are aggregators, not the company itself — demote in discovery.
_AGGREGATORS = ("mubasher", "reuters", "zawya", "wikipedia", "bloomberg",
                "investing.com", "marketwatch", "facebook", "linkedin",
                "youtube", "tradingview", "egx.com.eg")


# ---------------------------------------------------------------------------
# Folder scaffolding + registry
# ---------------------------------------------------------------------------

def company_dir(ticker: str) -> Path:
    return IR_ROOT / ticker.upper()


def scaffold_company(ticker: str) -> Path:
    """Create ir_data/<TICKER>/documents and an empty manifest if missing."""
    d = company_dir(ticker)
    (d / "documents").mkdir(parents=True, exist_ok=True)
    (d / "extracted").mkdir(parents=True, exist_ok=True)
    mpath = d / "manifest.json"
    if not mpath.exists():
        mpath.write_text(json.dumps({
            "ticker": ticker.upper(),
            "name": EGX_UNIVERSE.get(ticker.upper(), {}).get("name"),
            "ir_url": None,
            "documents": [],
            "last_fetch": None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def _load_manifest(ticker: str) -> dict[str, Any]:
    mpath = company_dir(ticker) / "manifest.json"
    if mpath.exists():
        return json.loads(mpath.read_text(encoding="utf-8"))
    return {"ticker": ticker.upper(), "documents": [], "ir_url": None, "last_fetch": None}


def _save_manifest(ticker: str, manifest: dict[str, Any]) -> None:
    (company_dir(ticker) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_registry() -> dict[str, Any]:
    if _REGISTRY.exists():
        return json.loads(_REGISTRY.read_text(encoding="utf-8"))
    return {}


def save_registry(reg: dict[str, Any]) -> None:
    IR_ROOT.mkdir(parents=True, exist_ok=True)
    _REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Auto-discovery (best-effort, always flagged)
# ---------------------------------------------------------------------------

def _score_candidate(url: str, title: str, name: str) -> float:
    """Heuristic score that a URL is a company's official IR page."""
    u = url.lower()
    host = urlparse(u).netloc
    score = 0.0
    if any(a in host for a in _AGGREGATORS):
        return -1.0  # not the company's own site
    name_tokens = [t for t in name.lower().replace("-", " ").split() if len(t) > 2]
    score += sum(1 for t in name_tokens if t in host) * 1.5      # name in domain
    score += sum(0.4 for t in name_tokens if t in (title or "").lower())
    for kw in ("investor", "ir", "relations", "financial", "shareholder"):
        if kw in u:
            score += 1.0
    if u.startswith("https://"):
        score += 0.3
    return score


def _confidence(score: float) -> str:
    return "high" if score >= 3.5 else "medium" if score >= 2.0 else "low"


def discover_ir_url(ticker: str, name: str | None = None,
                    timeout: float = 15.0) -> dict[str, Any]:
    """Best-effort search for a company's IR page. NEVER trusted blindly.

    Returns {ticker, name, ir_url, confidence, method, candidates, verified:False}.
    Uses DuckDuckGo HTML results; flags everything as unverified for human
    review. Returns confidence='none' on no usable result.
    """
    ticker = ticker.upper()
    name = name or EGX_UNIVERSE.get(ticker, {}).get("name") or ticker
    query = f"{name} Egypt investor relations financial statements"
    base = {"ticker": ticker, "name": name, "verified": False,
            "method": "duckduckgo-html", "discovered_at": datetime.utcnow().isoformat() + "Z"}
    try:
        r = httpx.get("https://html.duckduckgo.com/html/", params={"q": query},
                      headers=_HEADERS, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log.warning("discovery failed for %s: %s", ticker, e)
        return {**base, "ir_url": None, "confidence": "none", "candidates": [],
                "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    scored: list[tuple[float, str, str]] = []
    for a in soup.select("a.result__a, a.result__url, a[href]"):
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        title = a.get_text(" ", strip=True)
        s = _score_candidate(href, title, name)
        if s > 0:
            scored.append((s, href, title))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Dedup by host, keep best per host
    seen_hosts: set[str] = set()
    candidates = []
    for s, href, title in scored:
        host = urlparse(href).netloc
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        candidates.append({"url": href, "title": title, "score": round(s, 2)})
        if len(candidates) >= 5:
            break

    if not candidates:
        return {**base, "ir_url": None, "confidence": "none", "candidates": []}
    best = candidates[0]
    return {**base, "ir_url": best["url"], "confidence": _confidence(best["score"]),
            "candidates": candidates}


# ---------------------------------------------------------------------------
# Document fetching
# ---------------------------------------------------------------------------

def _robots_ok(url: str) -> bool:
    try:
        p = urlparse(url)
        rp = RobotFileParser()
        rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(_HEADERS["User-Agent"], url)
    except Exception:
        return True  # no robots / unreachable -> don't block research fetches


def _same_site(url: str, base: str) -> bool:
    """True if `url` shares the registrable domain of `base` (ir.x.com ~ www.x.com)."""
    def reg(host: str) -> str:
        parts = host.lower().split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()
    return reg(urlparse(url).netloc) == reg(urlparse(base).netloc)


def _doc_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Pull candidate financial-document links (PDFs + hinted links)."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        text = a.get_text(" ", strip=True)
        low = (href + " " + text).lower()
        is_pdf = href.lower().split("?")[0].endswith(".pdf")
        if not (is_pdf or any(h in low for h in _DOC_HINTS)):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append({"url": href, "text": text, "kind": "pdf" if is_pdf else "link"})
    return out


def _gather_pdf_links(html: str, page_url: str, crawl: bool,
                      max_subpages: int, timeout: float) -> tuple[list[dict[str, str]], int]:
    """Collect PDF links on a page, plus (one hop) PDFs inside same-site
    document index sub-pages (e.g. a "Results Center").

    EGX IR sites commonly list PDFs one level deep, so without this most
    fetches would find nothing on the landing page. Bounded to `max_subpages`
    same-site sub-pages, one hop only — never recursive.
    """
    links = _doc_links(html, page_url)
    pdfs = [l for l in links if l["kind"] == "pdf"]
    seen = {l["url"] for l in pdfs}
    crawled = 0
    if crawl:
        subpages = [l for l in links
                    if l["kind"] == "link" and _same_site(l["url"], page_url)][:max_subpages]
        for sp in subpages:
            if not _robots_ok(sp["url"]):
                continue
            try:
                sr = httpx.get(sp["url"], headers=_HEADERS, timeout=timeout, follow_redirects=True)
                sr.raise_for_status()
            except Exception as e:  # noqa: BLE001
                log.warning("subpage fetch failed %s: %s", sp["url"], e)
                continue
            crawled += 1
            for pl in _doc_links(sr.text, sp["url"]):
                if pl["kind"] == "pdf" and pl["url"] not in seen:
                    seen.add(pl["url"])
                    pdfs.append(pl)
            time.sleep(_FETCH_DELAY)
    return pdfs, crawled


def fetch_documents(ticker: str, ir_url: str | None = None,
                    max_docs: int = 25, timeout: float = 30.0,
                    crawl: bool = True, max_subpages: int = 8) -> dict[str, Any]:
    """Download new IR documents from `ir_url` into ir_data/<TICKER>/documents.

    Follows one hop into same-site document sub-pages (e.g. a "Results
    Center") so nested PDFs are reached — set crawl=False to fetch only the
    landing page. Deduped by content SHA-256, so re-running only fetches new
    files. Updates the manifest. Respects robots.txt and caps file size.
    """
    ticker = ticker.upper()
    scaffold_company(ticker)
    manifest = _load_manifest(ticker)
    ir_url = ir_url or manifest.get("ir_url") or load_registry().get(ticker, {}).get("ir_url")
    if not ir_url:
        return {"ticker": ticker, "error": "no IR url — discover or set one first"}

    if not _robots_ok(ir_url):
        return {"ticker": ticker, "ir_url": ir_url, "error": "blocked by robots.txt"}

    try:
        r = httpx.get(ir_url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker, "ir_url": ir_url, "error": f"fetch failed: {e}"}

    links, subpages_crawled = _gather_pdf_links(r.text, ir_url, crawl, max_subpages, timeout)
    have_hashes = {d["sha256"] for d in manifest.get("documents", [])}
    have_urls = {d["source_url"] for d in manifest.get("documents", [])}
    docs_dir = company_dir(ticker) / "documents"

    added, skipped = [], 0
    for link in links:
        if len(added) >= max_docs:
            break
        if link["url"] in have_urls:
            skipped += 1
            continue
        try:
            with httpx.stream("GET", link["url"], headers=_HEADERS, timeout=timeout,
                              follow_redirects=True) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                # Only archive PDFs / documents, not HTML pages.
                if "pdf" not in ctype and not link["url"].lower().split("?")[0].endswith(".pdf"):
                    skipped += 1
                    continue
                chunks, total = [], 0
                for c in resp.iter_bytes():
                    total += len(c)
                    if total > _MAX_BYTES:
                        raise ValueError(f"exceeds {_MAX_BYTES} byte cap")
                    chunks.append(c)
                blob = b"".join(chunks)
        except Exception as e:  # noqa: BLE001
            log.warning("download failed %s: %s", link["url"], e)
            skipped += 1
            continue

        sha = hashlib.sha256(blob).hexdigest()
        if sha in have_hashes:
            skipped += 1
            continue
        fname = f"{sha[:12]}_{Path(urlparse(link['url']).path).name or 'doc.pdf'}"[:120]
        (docs_dir / fname).write_bytes(blob)
        rec = {
            "filename": fname,
            "source_url": link["url"],
            "link_text": link["text"],
            "sha256": sha,
            "bytes": len(blob),
            "content_type": ctype,
            "kind": link["kind"],
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest.setdefault("documents", []).append(rec)
        have_hashes.add(sha)
        have_urls.add(link["url"])
        added.append(rec)
        time.sleep(_FETCH_DELAY)

    manifest["ir_url"] = ir_url
    manifest["last_fetch"] = datetime.utcnow().isoformat() + "Z"
    _save_manifest(ticker, manifest)

    return {
        "ticker": ticker,
        "ir_url": ir_url,
        "subpages_crawled": subpages_crawled,
        "pdf_links_found": len(links),
        "downloaded": len(added),
        "skipped": skipped,
        "total_documents": len(manifest.get("documents", [])),
        "new_files": [d["filename"] for d in added],
    }


def status(ticker: str | None = None) -> dict[str, Any]:
    """Report archive status for one ticker or the whole IR archive."""
    if ticker:
        m = _load_manifest(ticker)
        return {"ticker": ticker.upper(), "ir_url": m.get("ir_url"),
                "documents": len(m.get("documents", [])), "last_fetch": m.get("last_fetch")}
    out = []
    if IR_ROOT.exists():
        for d in sorted(IR_ROOT.iterdir()):
            mp = d / "manifest.json"
            if mp.is_file():
                m = json.loads(mp.read_text(encoding="utf-8"))
                out.append({"ticker": m.get("ticker"), "ir_url": m.get("ir_url"),
                            "documents": len(m.get("documents", [])),
                            "last_fetch": m.get("last_fetch")})
    return {"companies": len(out), "archive": out}
