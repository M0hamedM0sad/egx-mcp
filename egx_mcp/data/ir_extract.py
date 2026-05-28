"""Extract fundamentals from archived IR documents — PROVISIONAL only.

Reads the PDFs in ir_data/<TICKER>/documents/, attempts to pull the figures
the decision layer needs (EPS, ROE, net margin, debt/equity, book value), and
writes them to ir_data/<TICKER>/extracted/fundamentals_candidate.json with the
source file, page, and the matched text snippet for every number.

CRITICAL: these are PROVISIONAL. Parsing financial PDFs with regex is
unreliable — layouts vary per company, numbers sit in tables, and a wrong
figure feeding a verdict is worse than no figure. So extraction NEVER writes
into egx_fundamentals_audited.csv. Promotion is a separate, explicit,
human-reviewed step (scripts/ir_pipeline.py promote --verified). Every value
carries low confidence by default and the snippet so you can check it against
the source page in seconds.

PDF text extraction needs the optional `ir` extra:
    pip install 'egx-mcp[ir]'      # pypdf
Without it, this reports unavailable rather than guessing.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .ir_fetch import company_dir, _load_manifest

log = logging.getLogger("egx-mcp.ir_extract")

# Bilingual (EN + AR) cue → regex. Capture the first number after the cue.
# Crude by design; confidence stays low and the snippet is always kept.
_NUM = r"([-+]?\d[\d,]*\.?\d*)"
_PATTERNS: dict[str, list[str]] = {
    "trailing_eps": [
        r"(?:earnings per share|EPS|ربحية السهم|عائد السهم)\D{0,40}" + _NUM,
    ],
    "roe_pct": [
        r"(?:return on equity|ROE|العائد على حقوق (?:الملكية|المساهمين))\D{0,40}" + _NUM,
    ],
    "profit_margin_pct": [
        r"(?:net (?:profit|income) margin|هامش صافي الربح|هامش الربح)\D{0,40}" + _NUM,
    ],
    "debt_to_equity": [
        r"(?:debt[- ]?to[- ]?equity|debt\s*/\s*equity|D/E ratio|نسبة الد(?:ين|يون) إلى حقوق)\D{0,40}" + _NUM,
    ],
    "book_value_per_share": [
        r"(?:book value per share|BVPS|القيمة الدفترية للسهم)\D{0,40}" + _NUM,
    ],
}

# Sanity bounds — a parsed value outside these is almost certainly a misparse.
_BOUNDS = {
    "trailing_eps": (-1000, 1000),
    "roe_pct": (-100, 200),
    "profit_margin_pct": (-100, 100),
    "debt_to_equity": (0, 50),
    "book_value_per_share": (0, 100000),
}


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_from_text(text: str) -> dict[str, Any]:
    """Pure-function figure extraction from a block of text (testable offline).

    Returns {field: {value, snippet}} for every field matched within bounds.
    """
    found: dict[str, Any] = {}
    flat = re.sub(r"\s+", " ", text)
    for field, pats in _PATTERNS.items():
        for pat in pats:
            m = re.search(pat, flat, flags=re.IGNORECASE)
            if not m:
                continue
            val = _to_float(m.group(1))
            lo, hi = _BOUNDS[field]
            if val is None or not (lo <= val <= hi):
                continue
            start = max(0, m.start() - 30)
            found[field] = {"value": val, "snippet": flat[start:m.end() + 10].strip()}
            break
    return found


def company_context(ticker: str) -> dict[str, Any]:
    """IR material for a ticker as DECISION CONTEXT for the model to read.

    This is the qualitative path: it surfaces what's been archived plus the
    provisional figures (each with the source file, page, and the exact text
    snippet they came from) so the model can weigh them when forming a verdict
    — WITHOUT promoting anything into the scored fundamentals. Numbers here are
    unverified; treat them as evidence to reason about, not as ground truth.
    """
    ticker = ticker.upper()
    manifest = _load_manifest(ticker)
    docs = [{
        "filename": d.get("filename"),
        "title": d.get("link_text"),
        "source_url": d.get("source_url"),
        "fetched_at": d.get("fetched_at"),
        "kind": d.get("kind"),
    } for d in manifest.get("documents", [])]

    cand_path = company_dir(ticker) / "extracted" / "fundamentals_candidate.json"
    provisional = {}
    extracted_at = None
    if cand_path.exists():
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        provisional = cand.get("fields", {})
        extracted_at = cand.get("extracted_at")

    return {
        "ticker": ticker,
        "ir_url": manifest.get("ir_url"),
        "document_count": len(docs),
        "documents": docs,
        "provisional_fundamentals": provisional,
        "extracted_at": extracted_at,
        "usage": (
            "Decision context, not scored inputs. The model may read these "
            "figures and document titles to inform a verdict qualitatively. "
            "Each figure carries its source page + snippet — cite/verify before "
            "relying on it. To feed a number into the quantitative score, it "
            "must go through the explicit promote --verified gate."
        ),
    }


def _read_pdf_pages(path: Path) -> list[str] | None:
    try:
        from pypdf import PdfReader
    except Exception as e:  # noqa: BLE001
        log.warning("pypdf not installed (%s); install with: pip install 'egx-mcp[ir]'", e)
        return None
    try:
        reader = PdfReader(str(path))
        return [(pg.extract_text() or "") for pg in reader.pages]
    except Exception as e:  # noqa: BLE001
        log.warning("failed to read %s: %s", path.name, e)
        return []


def extract_company(ticker: str) -> dict[str, Any]:
    """Extract provisional fundamentals from every archived PDF for a ticker.

    Writes ir_data/<TICKER>/extracted/fundamentals_candidate.json. Keeps the
    most recent (last document wins) value per field, each with source +
    snippet. Does NOT touch the audited CSV.
    """
    ticker = ticker.upper()
    docs_dir = company_dir(ticker) / "documents"
    if not docs_dir.exists():
        return {"ticker": ticker, "error": "no documents — fetch first"}

    pdfs = sorted(docs_dir.glob("*.pdf")) + sorted(docs_dir.glob("*.PDF"))
    if not pdfs:
        return {"ticker": ticker, "error": "no PDFs archived"}

    fields: dict[str, Any] = {}
    parsed_any = False
    for pdf in pdfs:
        pages = _read_pdf_pages(pdf)
        if pages is None:
            return {"ticker": ticker, "error": "pypdf not installed — pip install 'egx-mcp[ir]'"}
        parsed_any = True
        for i, page_text in enumerate(pages):
            got = extract_from_text(page_text)
            for field, hit in got.items():
                fields[field] = {
                    "value": hit["value"],
                    "confidence": "low",          # regex-from-PDF — always review
                    "source_file": pdf.name,
                    "page": i + 1,
                    "snippet": hit["snippet"][:160],
                }

    candidate = {
        "ticker": ticker,
        "extracted_at": datetime.utcnow().isoformat() + "Z",
        "status": "PROVISIONAL — review against source before promoting",
        "fields": fields,
        "pdfs_scanned": len(pdfs),
    }
    out = company_dir(ticker) / "extracted" / "fundamentals_candidate.json"
    out.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ticker": ticker,
        "pdfs_scanned": len(pdfs),
        "fields_extracted": list(fields.keys()),
        "candidate_file": str(out),
        "note": ("Provisional. Verify each value against the cited page, then "
                 "promote with: python -m scripts.ir_pipeline promote " + ticker + " --verified"),
    }
