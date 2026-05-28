"""IR pipeline orchestrator — scaffold, discover, fetch, extract, promote.

    python -m scripts.ir_pipeline seed                 # folders + registry (offline)
    python -m scripts.ir_pipeline discover --all        # auto-find IR urls (online)
    python -m scripts.ir_pipeline discover COMI SWDY     # ...for specific names
    python -m scripts.ir_pipeline fetch COMI             # download docs (online)
    python -m scripts.ir_pipeline extract COMI           # provisional figures
    python -m scripts.ir_pipeline status                 # archive overview
    python -m scripts.ir_pipeline promote COMI --verified # candidate -> audited CSV

Flow: seed → discover → (VERIFY the discovered url, set verified=true in
ir_data/ir_registry.json) → fetch → extract → REVIEW the candidate JSON →
promote --verified. The two human gates (verify url, review figures) are the
point: auto-discovery and PDF parsing are fragile, so nothing reaches the
decision layer without a person confirming it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import ir_extract, ir_fetch
from egx_mcp.data.universe import EGX_UNIVERSE

_AUDITED_CSV = ir_fetch.IR_ROOT.parent / "egx_fundamentals_audited.csv"
_REVIEW_SHEET = ir_fetch.IR_ROOT / "ir_review_sheet.csv"
_SHEET_FIELDS = ["ticker", "name", "confidence", "verify", "ir_url",
                 "candidate_2", "candidate_3"]
_TRUTHY = {"y", "yes", "true", "1", "ok", "✓"}
_CSV_FIELDS = ["ticker", "trailing_eps", "book_value_per_share", "pe_ratio", "pb_ratio",
               "roe_pct", "profit_margin_pct", "debt_to_equity", "dividend_yield_pct", "market_cap"]
# Fields the extractor can contribute to the audited CSV.
_PROMOTABLE = ["trailing_eps", "book_value_per_share", "roe_pct",
               "profit_margin_pct", "debt_to_equity"]


def _universe() -> list[str]:
    return sorted(t for t, m in EGX_UNIVERSE.items() if m.get("sector") != "Index")


def _targets(args) -> list[str]:
    return _universe() if args.all else [t.upper() for t in args.tickers]


def cmd_seed(args):
    reg = ir_fetch.load_registry()
    for tk in _universe():
        ir_fetch.scaffold_company(tk)
        reg.setdefault(tk, {"ticker": tk, "name": EGX_UNIVERSE[tk].get("name"),
                            "ir_url": None, "confidence": "none", "verified": False})
    ir_fetch.save_registry(reg)
    print(f"Seeded {len(_universe())} company folders + registry at {ir_fetch.IR_ROOT}")


def cmd_discover(args):
    reg = ir_fetch.load_registry()
    for tk in _targets(args):
        res = ir_fetch.discover_ir_url(tk)
        reg[tk] = {**reg.get(tk, {}), **{
            "ticker": tk, "name": res["name"], "ir_url": res["ir_url"],
            "confidence": res["confidence"], "verified": False,
            "candidates": res.get("candidates", []), "method": res.get("method"),
        }}
        print(f"  {tk}: {res['confidence']:>6}  {res['ir_url']}")
    ir_fetch.save_registry(reg)
    print(f"\nDiscovered urls saved. VERIFY each (set \"verified\": true in "
          f"{ir_fetch._REGISTRY}) before fetching — auto-discovery is unverified.")


def cmd_fetch(args):
    reg = ir_fetch.load_registry()
    for tk in _targets(args):
        entry = reg.get(tk, {})
        if not args.force and not entry.get("verified"):
            print(f"  {tk}: SKIP — IR url not verified. Set verified:true in the registry "
                  f"or pass --force. (url={entry.get('ir_url')})")
            continue
        res = ir_fetch.fetch_documents(tk, ir_url=entry.get("ir_url"))
        if "error" in res:
            print(f"  {tk}: ERROR {res['error']}")
        else:
            print(f"  {tk}: +{res['downloaded']} new ({res['total_documents']} total) from {res['ir_url']}")


def cmd_extract(args):
    for tk in _targets(args):
        res = ir_extract.extract_company(tk)
        if "error" in res:
            print(f"  {tk}: {res['error']}")
        else:
            print(f"  {tk}: extracted {res['fields_extracted']} from {res['pdfs_scanned']} PDF(s)")


def cmd_review_sheet(args):
    """Export all discovered candidates to one CSV for one-pass verification."""
    reg = ir_fetch.load_registry()
    path = Path(args.path) if args.path else _REVIEW_SHEET
    rows = []
    for tk in sorted(reg):
        e = reg[tk]
        cands = e.get("candidates", []) or []
        alts = [c.get("url") for c in cands[1:3]]
        rows.append({
            "ticker": tk,
            "name": e.get("name") or "",
            "confidence": e.get("confidence") or "none",
            "verify": "Y" if e.get("verified") else "",  # fill Y to approve
            "ir_url": e.get("ir_url") or "",
            "candidate_2": alts[0] if len(alts) > 0 else "",
            "candidate_3": alts[1] if len(alts) > 1 else "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SHEET_FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_low = sum(1 for r in rows if r["confidence"] in ("low", "none"))
    print(f"Wrote {len(rows)} rows -> {path}")
    print(f"  {n_low} are low/none confidence — check those URLs especially.")
    print("In Excel: put Y in the `verify` column for each correct URL (edit the")
    print("`ir_url` cell, or paste from candidate_2/3, if the best guess is wrong).")
    print(f"Then: python -m scripts.ir_pipeline import-sheet")


def cmd_import_sheet(args):
    """Read a verified review sheet back into the registry."""
    path = Path(args.path) if args.path else _REVIEW_SHEET
    if not path.exists():
        print(f"Review sheet not found: {path}\nRun: python -m scripts.ir_pipeline review-sheet")
        return
    reg = ir_fetch.load_registry()
    verified = changed = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tk = (row.get("ticker") or "").strip().upper()
            if not tk or tk not in reg:
                continue
            is_v = (row.get("verify") or "").strip().lower() in _TRUTHY
            url = (row.get("ir_url") or "").strip() or None
            if reg[tk].get("ir_url") != url or bool(reg[tk].get("verified")) != is_v:
                changed += 1
            reg[tk]["ir_url"] = url
            reg[tk]["verified"] = is_v
            if is_v:
                verified += 1
    ir_fetch.save_registry(reg)
    print(f"Imported {path.name}: {verified} URLs marked verified, {changed} entries changed.")
    print("Verified names are now fetchable: python -m scripts.ir_pipeline fetch --all")


def cmd_status(args):
    tk = args.tickers[0] if args.tickers else None
    print(json.dumps(ir_fetch.status(tk), ensure_ascii=False, indent=2))


def _load_csv() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if _AUDITED_CSV.exists():
        with _AUDITED_CSV.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[(r.get("ticker") or "").upper()] = r
    return rows


def cmd_promote(args):
    tk = args.tickers[0].upper()
    if not args.verified:
        print(f"Refusing to promote {tk} without --verified. Review "
              f"ir_data/{tk}/extracted/fundamentals_candidate.json against the source "
              f"PDFs first, then re-run with --verified.")
        return
    cand_path = ir_fetch.company_dir(tk) / "extracted" / "fundamentals_candidate.json"
    if not cand_path.exists():
        print(f"No candidate for {tk}. Run: python -m scripts.ir_pipeline extract {tk}")
        return
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    fields = cand.get("fields", {})

    rows = _load_csv()
    row = rows.get(tk, {k: "" for k in _CSV_FIELDS})
    row["ticker"] = tk
    updated = []
    for f in _PROMOTABLE:
        if f in fields:
            row[f] = fields[f]["value"]
            updated.append(f)
    rows[tk] = row

    with _AUDITED_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for t in sorted(rows):
            w.writerow({k: rows[t].get(k, "") for k in _CSV_FIELDS})

    print(f"Promoted {tk}: updated {updated or '(nothing extracted)'} in {_AUDITED_CSV.name}")
    print("Re-run: python -m scripts.fundamentals_coverage  to see the new confidence level.")


def main() -> int:
    ap = argparse.ArgumentParser(description="IR document pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seed", "discover", "fetch", "extract", "status", "promote",
                 "review-sheet", "import-sheet"):
        p = sub.add_parser(name)
        p.add_argument("tickers", nargs="*")
        p.add_argument("--all", action="store_true", help="Apply to the whole universe.")
        if name == "fetch":
            p.add_argument("--force", action="store_true", help="Fetch even if url unverified.")
        if name == "promote":
            p.add_argument("--verified", action="store_true",
                           help="Confirm you reviewed the candidate against source PDFs.")
        if name in ("review-sheet", "import-sheet"):
            p.add_argument("--path", default=None, help="CSV path (default ir_data/ir_review_sheet.csv).")
    args = ap.parse_args()

    {"seed": cmd_seed, "discover": cmd_discover, "fetch": cmd_fetch,
     "extract": cmd_extract, "status": cmd_status, "promote": cmd_promote,
     "review-sheet": cmd_review_sheet, "import-sheet": cmd_import_sheet}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
