"""Extended EGX universe — beyond the 29 hand-curated names.

The hand-curated set in `universe.py` is for fundamentals work where we
need sector taxonomy and friendly names. For simulation, all we need is
the symbol — yfinance can pull price history for any `.CA`-suffixed
EGX ticker that's actually traded.

This module:
  1. Holds a broader candidate list of EGX 100 components (~75 names).
  2. Validates each against yfinance and keeps only the live ones.
  3. Caches the validated set to disk so we don't re-probe every run.

Run the validation once with:
    python -m egx_mcp.data.egx_listing
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yfinance as yf

log = logging.getLogger("egx-mcp.egx_listing")

_CACHE_PATH = Path(__file__).parent / "egx_symbols_cache.json"
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # weekly refresh

# Full-market universe validated against investing.com (the model's real fetch
# path), built by scripts.expand_universe. ~249 names vs the legacy ~68. Yahoo's
# .CA probe (below) is kept only as a fallback when this file is absent.
_INVESTING_UNIVERSE_PATH = Path(__file__).parent / "egx_universe_investing.json"


# Candidate EGX symbols beyond the curated 29 — drawn from EGX 70 / EGX 100
# components I'm confident are tradeable. yfinance probe drops the dead ones.
EXTENDED_CANDIDATES: list[str] = [
    # --- Already in curated set (kept here so the validator covers them too) ---
    "COMI", "HDBK", "CIEB", "ADIB", "FAIT",
    "HRHO", "EFIH", "CIRA", "MNHD",
    "TMGH", "PHDC", "ORHD", "EMFD", "OCDI", "HELI",
    "SWDY", "ESRS", "ORWE", "ABUK", "MFPC",
    "EFID", "JUFO", "DOMT", "CCAP",
    "ETEL", "EAST", "CLHO", "IDHC", "PHAR", "EGTS", "FWRY", "MTIE",

    # --- Banks / financials ---
    "BTFH",  # Beltone Financial Holding
    "EGBE",  # Egyptian Gulf Bank
    "SAUD",  # Suez Canal Bank
    "EGFI",  # EFG Hermes / financial group variants
    "PIOH",  # Pioneers Holding
    "PHAR",  # already mapped via EIPI

    # --- Real estate (additional) ---
    "AMER",  # Amer Group Holding
    "OBRI",  # Orascom Hotels & Development
    "UEGC",  # Upper Egypt Contracting
    "RREI",  # Reedy Group / real estate variants
    "IRAX",  # Industrial & Engineering Enterprises
    "ROTO",  # Rotopaco / packaging real-estate variants
    "MASR",  # Madinet Masr (formerly MNHD — sometimes listed separately)
    "ZMID",  # Zahraa Maadi Investment

    # --- Industrial / construction / cement ---
    "ARCC",  # Arabian Cement
    "SUCE",  # Suez Cement
    "MISR",  # Misr Beni Suef Cement
    "SCEM",  # Sinai Cement
    "MICH",  # Misr Cement Qena (variants)
    "IRON",  # Egyptian Iron & Steel
    "CERA",  # Lecico Egypt / ceramics
    "LCSW",  # Lecico
    "OFH",   # Orascom Financial Holding
    "ORAS",  # Orascom Construction
    "ELEC",  # El Sewedy Cables / cables variants
    "DSCW",  # El Sewedy Cables (some listings)
    "POUL",  # Cairo Poultry
    "AUTO",  # GB Auto (GB Corp)
    "GBCO",  # GB Auto alternate code
    "RAYA",  # Raya Holding

    # --- Petrochems / chemicals ---
    "AMOC",  # Alexandria Mineral Oils
    "SKPC",  # Sidi Kerir Petrochemicals
    "EFIC",  # Egyptian Financial & Industrial
    "PACH",  # Paints & Chemical Industries (Pachin)
    "PPCI",  # Pachin alt

    # --- Food / consumer ---
    "ADCM",  # Arab Dairy
    "OLFI",  # Olympic Group
    "ASCM",  # Arab Cotton Ginning
    "KZAR",  # Kabo
    "MOIL",  # Misr Oils & Soap
    "SUGR",  # Delta Sugar
    "EGCH",  # Egyptian Chemical Industries (Kima)

    # --- Tourism / leisure ---
    "PRDR",  # Pyramisa
    "OBRI",  # Orascom Hotels (already above; ok if duplicated, validator dedupes)

    # --- Tech / services ---
    "RACT",  # Raya Contact Center
    "AINH",  # ?

    # --- Healthcare ---
    "BIOC",  # Cairo Pharmaceuticals
    "MEFP",  # Memphis Pharmaceuticals
    "ACGC",  # Alexandria Pharmaceuticals
    "PHDC",  # already

    # --- Logistics / containers ---
    "ALCN",  # Alexandria Containers
    "CSAG",  # Canal Shipping Agencies

    # --- Diversified / misc ---
    "BINV",  # ?
    "ETRS",  # Egyptian Transport (El-Nasr)
    "EXPA",  # Export Development Bank
    "TPEG",  # Talaat Mostafa subsidiary?
]


def _is_alive(symbol: str) -> tuple[bool, dict]:
    """Probe yfinance — does this `.CA` symbol return real data?

    Returns (alive, sample_info). A symbol is alive if it returns:
      - at least 30 daily bars in the last 6 months, AND
      - a positive last close.
    """
    yahoo = symbol if symbol.endswith(".CA") or symbol.startswith("^") else f"{symbol}.CA"
    try:
        t = yf.Ticker(yahoo)
        df = t.history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 30:
            return False, {"reason": f"only {0 if df is None else len(df)} bars"}
        last_close = float(df["Close"].iloc[-1])
        if last_close <= 0:
            return False, {"reason": "non-positive last close"}
        return True, {
            "yahoo_symbol": yahoo,
            "last_close": round(last_close, 4),
            "bars_6mo": len(df),
        }
    except Exception as e:
        return False, {"reason": str(e)}


def validate_and_cache(force: bool = False) -> dict[str, Any]:
    """Probe every candidate, cache the live ones, return the validated set."""
    if not force and _CACHE_PATH.exists():
        age = time.time() - _CACHE_PATH.stat().st_mtime
        if age < _CACHE_TTL_SECONDS:
            try:
                return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

    seen: set[str] = set()
    validated: list[dict] = []
    dead: list[dict] = []

    for cand in EXTENDED_CANDIDATES:
        if cand in seen:
            continue
        seen.add(cand)
        alive, info = _is_alive(cand)
        if alive:
            validated.append({
                "ticker": cand,
                "yahoo_symbol": info["yahoo_symbol"],
                "last_close": info["last_close"],
                "bars_6mo": info["bars_6mo"],
            })
            log.info(f"  alive: {cand}")
        else:
            dead.append({"ticker": cand, "reason": info.get("reason")})
            log.info(f"  dead:  {cand} ({info.get('reason')})")

    payload = {
        "validated_at": time.time(),
        "count": len(validated),
        "validated": validated,
        "dead": dead,
    }
    try:
        _CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"cache write failed: {e}")
    return payload


def get_full_universe() -> list[str]:
    """Return the validated EGX symbol list.

    Prefers the investing.com-validated full market (egx_universe_investing.json,
    ~249 names from scripts.expand_universe). Falls back to the legacy
    yfinance-probed candidate set when that file is absent.
    """
    if _INVESTING_UNIVERSE_PATH.exists():
        try:
            data = json.loads(_INVESTING_UNIVERSE_PATH.read_text(encoding="utf-8"))
            tickers = [row["ticker"] for row in data.get("validated", [])]
            if tickers:
                return tickers
        except Exception as e:  # noqa: BLE001
            log.warning("could not read investing universe (%s); using yfinance probe", e)
    cache = validate_and_cache(force=False)
    return [row["ticker"] for row in cache.get("validated", [])]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Validating EGX candidate symbols against yfinance...\n")
    payload = validate_and_cache(force=True)
    print(f"\nValidated: {payload['count']}")
    print(f"Dead/skipped: {len(payload['dead'])}")
    print(f"Cache: {_CACHE_PATH}")
