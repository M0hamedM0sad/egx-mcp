"""Audited-fundamentals coverage report.

Shows, for every name in the universe, whether audited fundamentals are
loaded and what data-confidence level decide() will assign it — so you can
see which names are actionable, which would ABSTAIN, and exactly which fields
to add to push a name from medium to high confidence.

    python -m scripts.fundamentals_coverage

This reads the same override CSV the live loader uses (repo-root
egx_fundamentals_audited.csv by default, or $EGX_FUNDAMENTALS_CSV). It does
NOT hit the network — it reports the audited layer only.

Confidence mirrors decision._data_confidence:
    high   audited + >=4 of 5 core fields
    medium audited + 2-3 core fields
    low    audited + <=1 core field, or not covered  -> decide() ABSTAINS
Core fields: pe_ratio, pb_ratio, roe_pct, profit_margin_pct, debt_to_equity.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import fundamentals as f_mod
from egx_mcp.data.universe import EGX_UNIVERSE

_CORE = ["pe_ratio", "pb_ratio", "roe_pct", "profit_margin_pct", "debt_to_equity"]


def _confidence(present: int, covered: bool) -> str:
    if not covered or present <= 1:
        return "low"
    return "high" if present >= 4 else "medium"


def main() -> int:
    overrides = f_mod._load_overrides()
    universe = sorted(t for t, m in EGX_UNIVERSE.items() if m.get("sector") != "Index")

    print(f"Audited override rows loaded: {len(overrides)}")
    print(f"Universe (non-index): {len(universe)}\n")

    rows = []
    for tk in universe:
        ov = overrides.get(tk)
        covered = ov is not None
        # A field counts if the audited CSV provides it (pe/pb may also be
        # derivable from eps/book at runtime, but we report the CSV as-is).
        have = [c for c in _CORE if ov and ov.get(c) is not None]
        # pe/pb are derivable from eps + book even if not stored directly.
        if ov:
            if "pe_ratio" not in have and ov.get("trailing_eps") is not None:
                have.append("pe_ratio*")
            if "pb_ratio" not in have and ov.get("book_value_per_share") is not None:
                have.append("pb_ratio*")
        n = len([h for h in have if not h.endswith("*")]) + len([h for h in have if h.endswith("*")])
        conf = _confidence(n, covered)
        missing = [c for c in _CORE if c not in [h.rstrip("*") for h in have]]
        rows.append((tk, covered, conf, n, missing))

    counts = {"high": 0, "medium": 0, "low": 0}
    for _, _, conf, _, _ in rows:
        counts[conf] += 1

    print(f"{'ticker':>8} | {'covered':>7} | {'confidence':>10} | core | missing core fields")
    print("-" * 78)
    for tk, covered, conf, n, missing in rows:
        flag = "ABSTAIN" if conf == "low" else ""
        print(f"{tk:>8} | {'yes' if covered else 'NO':>7} | {conf:>10} | {n}/5  | "
              f"{', '.join(missing) if missing else '—'} {flag}")

    print("\n" + "=" * 78)
    print(f"SUMMARY:  high={counts['high']}  medium={counts['medium']}  "
          f"low/ABSTAIN={counts['low']}   ({len(universe)} names)")
    uncovered = [tk for tk, cov, *_ in rows if not cov]
    if uncovered:
        print(f"\nNot covered (will ABSTAIN until added): {', '.join(uncovered)}")
    med = [tk for tk, _, conf, *_ in rows if conf == "medium"]
    if med:
        print(f"\nAt MEDIUM (actionable, but capped conviction). To reach HIGH, add the")
        print(f"missing core fields (usually profit_margin_pct + debt_to_equity) for:")
        print(f"  {', '.join(med)}")
    print("\nTo refresh the audited CSV from the Mubasher cache:")
    print("  python -m egx_mcp.data.mubasher_fundamentals   # rebuild cache (online)")
    print("  python -m scripts.export_fundamentals_csv       # cache -> egx_fundamentals_audited.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
