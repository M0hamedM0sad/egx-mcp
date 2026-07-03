"""EGX ticker registry and resolution.

Yahoo Finance covers EGX with a `.CA` suffix on the ISIN-style code, e.g.
EGS65541C012.CA for CIRA Education. This module:

  1. Maintains a curated map of common nicknames -> Yahoo symbols for the
     most-traded EGX names (covers ~95% of retail trading volume).
  2. Provides resolve_ticker() to normalize any user input.
  3. Provides the underlying universe used by list_egx_stocks and screen_stocks.

The full EGX has ~240 listings; this curated set is the practical universe
for retail FP&A / equity research. Add more names here as needed.
"""
from __future__ import annotations

# Curated EGX universe — extend as needed.
# Sectors follow EGX's official taxonomy.
EGX_UNIVERSE: dict[str, dict[str, str]] = {
    # Banks
    "COMI":  {"yahoo": "COMI.CA",         "name": "Commercial International Bank",        "sector": "Banks"},
    "HDBK":  {"yahoo": "HDBK.CA",         "name": "Housing & Development Bank",            "sector": "Banks"},
    "CIEB":  {"yahoo": "CIEB.CA",         "name": "Crédit Agricole Egypt",                 "sector": "Banks"},
    "ADIB":  {"yahoo": "ADIB.CA",         "name": "Abu Dhabi Islamic Bank Egypt",          "sector": "Banks"},
    "FAIT":  {"yahoo": "FAIT.CA",         "name": "Faisal Islamic Bank of Egypt",          "sector": "Banks"},

    # Non-bank financials
    "HRHO":  {"yahoo": "HRHO.CA",         "name": "EFG Hermes Holding",                    "sector": "Financial Services"},
    "EFIH":  {"yahoo": "EFIH.CA",         "name": "EFG Holding",                           "sector": "Financial Services"},
    "CIRA":  {"yahoo": "CIRA.CA",         "name": "Cairo for Investment & Real Estate",    "sector": "Education"},
    "MNHD":  {"yahoo": "MNHD.CA",         "name": "Madinet Nasr Housing & Development",    "sector": "Real Estate"},

    # Real estate
    "TMGH":  {"yahoo": "TMGH.CA",         "name": "Talaat Moustafa Group",                 "sector": "Real Estate"},
    "PHDC":  {"yahoo": "PHDC.CA",         "name": "Palm Hills Developments",               "sector": "Real Estate"},
    "ORHD":  {"yahoo": "ORHD.CA",         "name": "Orascom Development Egypt",             "sector": "Real Estate"},
    "EMFD":  {"yahoo": "EMFD.CA",         "name": "Emaar Misr for Development",            "sector": "Real Estate"},
    "SODIC": {"yahoo": "OCDI.CA",         "name": "Sixth of October Development (SODIC)",  "sector": "Real Estate"},
    "HELI":  {"yahoo": "HELI.CA",         "name": "Heliopolis Housing",                    "sector": "Real Estate"},

    # Industrial / construction
    "SWDY":  {"yahoo": "SWDY.CA",         "name": "Elsewedy Electric",                     "sector": "Industrial Goods"},
    # "ESRS" (Ezz Steel) removed — delisted from EGX. Kept in news_filter and
    # historical sims; no longer part of the live watchlist.
    "ORWE":  {"yahoo": "ORWE.CA",         "name": "Oriental Weavers",                      "sector": "Personal & Household"},
    "ABUK":  {"yahoo": "ABUK.CA",         "name": "Abou Kir Fertilizers",                  "sector": "Chemicals"},
    "MFPC":  {"yahoo": "MFPC.CA",         "name": "Misr Fertilizers Production (MOPCO)",   "sector": "Chemicals"},
    "EFID":  {"yahoo": "EFID.CA",         "name": "Edita Food Industries",                 "sector": "Food & Beverage"},
    "JUFO":  {"yahoo": "JUFO.CA",         "name": "Juhayna Food Industries",               "sector": "Food & Beverage"},
    "DOMT":  {"yahoo": "DOMT.CA",         "name": "Domty",                                 "sector": "Food & Beverage"},
    "CCAP":  {"yahoo": "CCAP.CA",         "name": "Qalaa Holdings",                        "sector": "Diversified"},

    # Telecom
    "ETEL":  {"yahoo": "ETEL.CA",         "name": "Telecom Egypt",                         "sector": "Telecom"},
    "EAST":  {"yahoo": "EAST.CA",         "name": "Eastern Company",                       "sector": "Personal & Household"},

    # Healthcare / pharma
    "CLHO":  {"yahoo": "CLHO.CA",         "name": "Cleopatra Hospital",                    "sector": "Healthcare"},
    # "IDHC" (Integrated Diagnostics) removed — delisted from EGX (LSE-only).
    "EIPI":  {"yahoo": "PHAR.CA",         "name": "EIPICO",                                "sector": "Healthcare"},

    # Tourism / leisure
    "EGTS":  {"yahoo": "EGTS.CA",         "name": "Egyptian Resorts Company",              "sector": "Travel & Leisure"},

    # Tech / e-commerce
    "FWRY":  {"yahoo": "FWRY.CA",         "name": "Fawry for Banking Technology",          "sector": "Technology"},
    "MTIE":  {"yahoo": "MTIE.CA",         "name": "MM Group for Industry & Trade",         "sector": "Retail"},

    # Indices (used by get_index)
    "EGX30":    {"yahoo": "^CASE30",      "name": "EGX 30 Price Return Index",             "sector": "Index"},
    "EGX30TR":  {"yahoo": "EGX30.CA",     "name": "EGX 30 Total Return Index",             "sector": "Index"},
    "EGX70":    {"yahoo": "EGX70.CA",     "name": "EGX 70 EWI",                            "sector": "Index"},
    "EGX70EWI": {"yahoo": "EGX70.CA",     "name": "EGX 70 Equal-Weighted",                 "sector": "Index"},
    "EGX100":   {"yahoo": "EGX100.CA",    "name": "EGX 100 Price Return Index",            "sector": "Index"},
}


def resolve_ticker(user_input: str) -> tuple[str, str, str]:
    """Resolve any user-provided identifier to (canonical, yahoo_symbol, name).

    Accepts:
      - Common nicknames: 'CIRA', 'COMI', 'cira'
      - Yahoo symbols: 'CIRA.CA', '^CASE30'
      - ISIN-style codes: 'EGS65541C012' (auto-suffixed with .CA)
      - Index aliases: 'EGX30', 'egx 30', 'CASE30'
    """
    raw = user_input.strip().upper().replace(" ", "")

    # Index aliases + common equity nicknames
    aliases = {
        "CASE30": "EGX30",
        "^CASE30": "EGX30",
        "EGX-30": "EGX30",
        "EGX-70": "EGX70",
        "EGX-100": "EGX100",
        "TMG": "TMGH",
    }
    raw = aliases.get(raw, raw)

    # Direct hit in universe
    if raw in EGX_UNIVERSE:
        entry = EGX_UNIVERSE[raw]
        return raw, entry["yahoo"], entry["name"]

    # Strip .CA suffix and retry
    if raw.endswith(".CA"):
        bare = raw[:-3]
        if bare in EGX_UNIVERSE:
            entry = EGX_UNIVERSE[bare]
            return bare, entry["yahoo"], entry["name"]
        # Unknown but properly formatted — pass through
        return bare, raw, bare

    # ISIN-style code (EGS prefix, 12 chars) — pass through with .CA
    if raw.startswith("EGS") and len(raw) >= 12:
        return raw, f"{raw}.CA", raw

    # Last resort — try with .CA suffix
    return raw, f"{raw}.CA", raw


def list_stocks(sector: str | None = None) -> dict:
    """List equities (excludes indices)."""
    rows = []
    for ticker, meta in EGX_UNIVERSE.items():
        if meta["sector"] == "Index":
            continue
        if sector and sector.lower() not in meta["sector"].lower():
            continue
        rows.append({
            "ticker": ticker,
            "name": meta["name"],
            "sector": meta["sector"],
            "yahoo_symbol": meta["yahoo"],
        })
    rows.sort(key=lambda r: r["sector"] + r["ticker"])
    return {
        "count": len(rows),
        "sector_filter": sector,
        "stocks": rows,
        "note": "Curated universe of most-liquid EGX names. Full exchange has ~240 listings.",
    }


def screen(
    min_pe: float | None = None,
    max_pe: float | None = None,
    min_market_cap_egp_m: float | None = None,
    min_avg_volume: int | None = None,
    sector: str | None = None,
    sort_by: str = "market_cap",
) -> dict:
    """Run live screen across the universe.

    Note: This makes one yfinance call per stock and can be slow (~30s for
    full universe). For production use, add caching.
    """
    from . import market

    candidates = list_stocks(sector=sector)["stocks"]
    results = []

    for stock in candidates:
        try:
            quote = market.get_quote(stock["ticker"])
        except Exception:
            continue

        pe = quote.get("pe_ratio")
        mcap_m = (quote.get("market_cap") or 0) / 1_000_000
        avg_vol = quote.get("avg_volume") or 0

        if min_pe is not None and (pe is None or pe < min_pe):
            continue
        if max_pe is not None and (pe is None or pe > max_pe):
            continue
        if min_market_cap_egp_m is not None and mcap_m < min_market_cap_egp_m:
            continue
        if min_avg_volume is not None and avg_vol < min_avg_volume:
            continue

        results.append({
            "ticker": stock["ticker"],
            "name": stock["name"],
            "sector": stock["sector"],
            "price": quote.get("price"),
            "change_pct": quote.get("change_pct"),
            "pe_ratio": pe,
            "market_cap_egp_m": round(mcap_m, 1),
            "avg_volume": avg_vol,
        })

    sort_keys = {
        "market_cap": lambda r: r.get("market_cap_egp_m") or 0,
        "pe": lambda r: r.get("pe_ratio") or float("inf"),
        "volume": lambda r: r.get("avg_volume") or 0,
        "change_pct": lambda r: r.get("change_pct") or 0,
    }
    results.sort(key=sort_keys.get(sort_by, sort_keys["market_cap"]), reverse=True)

    return {
        "count": len(results),
        "filters": {
            "min_pe": min_pe, "max_pe": max_pe,
            "min_market_cap_egp_m": min_market_cap_egp_m,
            "min_avg_volume": min_avg_volume,
            "sector": sector,
        },
        "sort_by": sort_by,
        "results": results,
    }
