"""Pull TradingView's sector / industry for every EGX listing.

The model knows the sector of only the 30 curated names (EGX_UNIVERSE), so
~220 names get no sector medians and no sector macro bias. This prints
TradingView's classification for the whole Egyptian market (one POST per
field) so it can be mapped onto the model's sector names and committed as
egx_mcp/data/egx_sectors.json.

    python -m scripts.refresh_sectors            # print summary + JSON
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data import tv_history  # noqa: E402
from egx_mcp.data.universe import EGX_UNIVERSE  # noqa: E402


def fetch() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for col in ("sector", "industry", "description"):
        try:
            for tk, (val,) in tv_history._scan([col]).items():
                if val not in (None, ""):
                    out.setdefault(tk, {})[col] = val
        except Exception as e:  # noqa: BLE001
            print(f"{col}: {type(e).__name__} {e}")
    return out


def main() -> int:
    data = fetch()
    print(f"names with a classification: {len(data)}")
    print("\nTradingView sectors:")
    for s, n in Counter(d.get("sector") for d in data.values()).most_common():
        print(f"  {n:4}  {s}")
    print("\nTradingView industries (with the curated model sector where known):")
    by_ind: dict[str, Counter] = {}
    for tk, d in data.items():
        cur = EGX_UNIVERSE.get(tk, {}).get("sector")
        by_ind.setdefault(d.get("industry") or "?", Counter())[cur or "-"] += 1
    for ind, c in sorted(by_ind.items(), key=lambda kv: -sum(kv[1].values())):
        known = {k: v for k, v in c.items() if k != "-"}
        print(f"  {sum(c.values()):4}  {ind:45} curated: {known or ''}")
    print("\n=== JSON ===")
    print(json.dumps(data, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
