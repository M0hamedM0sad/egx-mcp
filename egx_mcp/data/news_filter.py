"""Portfolio-aware news filter.

Given a list of headlines and the user's current portfolio, return only
headlines that plausibly impact a held position. Filtering is keyword-
based across three layers:

  1. Ticker symbols           CIRA / COMI / SWDY / TMGH …  case-insensitive
  2. Company names            full English name from EGX_UNIVERSE,
                              plus a small Arabic alias table for the
                              names common in Mubasher/Al-Borsa headlines
  3. Sector keywords          "bank" / "real estate" / "education" / …
                              with Arabic equivalents

Each surviving headline is annotated with:
    matched_tickers   list of held tickers the headline directly names
    matched_sectors   list of sectors the headline references that
                      overlap with the portfolio
    impact_level      "direct" (ticker name match) > "sector" (sector
                      keyword match only) > "macro" (CBE / EGP /
                      Brent — affects everyone, currently not tagged)

Direct matches always rank higher than sector matches. Tied items are
ordered by source recency.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from .universe import EGX_UNIVERSE

log = logging.getLogger("egx-mcp.news_filter")


# ---------------------------------------------------------------------------
# Arabic aliases for common EGX names — only the ones that show up in
# Mubasher / Al-Borsa coverage. Add more as needed.
# ---------------------------------------------------------------------------

_AR_TICKER_ALIASES: dict[str, list[str]] = {
    "COMI":  ["البنك التجاري الدولي", "تجاري دولي", "CIB"],
    "HDBK":  ["بنك التعمير والإسكان", "التعمير والإسكان"],
    "CIEB":  ["كريدي أجريكول", "كريديه أجريكول"],
    "ADIB":  ["أبو ظبي الإسلامي", "أبوظبي الإسلامي"],
    "FAIT":  ["فيصل الإسلامي", "بنك فيصل"],
    "HRHO":  ["هيرميس", "إي إف جي", "EFG"],
    "EFIH":  ["إي إف جي القابضة", "EFG القابضة"],
    "CIRA":  ["كايرو للاستثمار", "كيرا", "سيرا", "للاستثمار والتنمية العقارية"],
    "MNHD":  ["مدينة نصر للإسكان", "مدينة نصر"],
    "TMGH":  ["طلعت مصطفى", "TMG", "مدينتي"],
    "PHDC":  ["بالم هيلز"],
    "ORHD":  ["أوراسكوم للتنمية", "أوراسكوم"],
    "EMFD":  ["إعمار مصر", "إعمار"],
    "SODIC": ["سوديك", "السادس من أكتوبر للتنمية"],
    "HELI":  ["هليوبوليس للإسكان", "مصر الجديدة"],
    "SWDY":  ["السويدي إليكتريك", "السويدي", "إلكتريك"],
    "ESRS":  ["عز للحديد", "حديد عز"],
    "ORWE":  ["النساجون الشرقيون", "نساجون"],
    "ABUK":  ["أبو قير للأسمدة", "أبو قير"],
    "MFPC":  ["موبكو", "مصر للأسمدة"],
    "EFID":  ["إديتا"],
    "JUFO":  ["جهينة"],
    "DOMT":  ["دومتي"],
    "CCAP":  ["القلعة القابضة", "القلعة"],
    "ETEL":  ["المصرية للاتصالات", "تليكوم مصر"],
    "EAST":  ["الشرقية للدخان", "الشرقية"],
    "CLHO":  ["كليوباترا للمستشفيات", "كليوباترا"],
    "IDHC":  ["IDH", "المتكاملة للخدمات التشخيصية"],
    "EIPI":  ["إيبيكو", "EIPICO"],
    "EGTS":  ["المصرية للمنتجعات", "المنتجعات"],
    "FWRY":  ["فوري"],
    "MTIE":  ["إم إم جروب", "MM Group"],
}


# Sector keyword maps — English + Arabic
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Banks": [
        "bank", "banking", "banks", "lender", "deposit",
        "بنك", "بنوك", "مصرف", "مصارف", "ودائع",
    ],
    "Financial Services": [
        "broker", "brokerage", "investment bank", "asset management",
        "وساطة", "إدارة أصول", "بنك استثمار",
    ],
    "Real Estate": [
        "real estate", "property", "developer", "housing", "land plot", "compound",
        "عقار", "عقارات", "إسكان", "وحدات سكنية", "مدن جديدة", "أراضي",
    ],
    "Education": [
        "education", "university", "school", "tuition",
        "تعليم", "جامعة", "مدارس",
    ],
    "Industrial Goods": [
        "industrial", "manufacturing", "factory", "infrastructure",
        "صناعة", "صناعي", "مصنع", "بنية تحتية",
    ],
    "Basic Resources": [
        "steel", "iron", "metals",
        "حديد", "صلب", "معادن",
    ],
    "Personal & Household": [
        "household", "tobacco", "consumer goods",
        "سلع استهلاكية", "تبغ",
    ],
    "Chemicals": [
        "fertilizer", "fertilizers", "petrochemical", "chemicals",
        "أسمدة", "بتروكيماويات", "كيماويات",
    ],
    "Food & Beverage": [
        "food", "beverage", "dairy",
        "أغذية", "ألبان", "مشروبات",
    ],
    "Telecom": [
        "telecom", "telecoms", "5g", "4g", "fiber",
        "اتصالات", "تليكوم", "خط أرضي", "نطاق ترددي",
    ],
    "Healthcare": [
        "healthcare", "hospital", "pharma", "pharmaceutical",
        "صحة", "مستشفى", "مستشفيات", "أدوية", "دوائي",
    ],
    "Travel & Leisure": [
        "tourism", "hotel", "resort", "leisure",
        "سياحة", "فندق", "فنادق", "منتجع",
    ],
    "Technology": [
        "technology", "fintech", "software", "digital payments",
        "تكنولوجيا", "تقنية", "مدفوعات رقمية",
    ],
    "Retail": [
        "retail", "consumer", "e-commerce",
        "تجزئة", "تسوق", "تجارة إلكترونية",
    ],
    "Diversified": [],  # No clean keyword set
}


# Macro-impact keywords that affect the whole market — currently logged
# but not used to filter; can be enabled if user wants macro tone too.
_MACRO_KEYWORDS = [
    "cbe", "central bank", "interest rate", "policy rate", "egp", "pound",
    "treasury bill", "tbill",
    "البنك المركزي", "سعر الفائدة", "الجنيه", "أذون الخزانة",
]


def _ar_matches(text: str, needles: Iterable[str]) -> bool:
    """Substring match for Arabic — Arabic doesn't separate words like
    English so we use plain substring; needles in our table are
    descriptive enough that false positives are rare."""
    if not text:
        return False
    for n in needles:
        if not n:
            continue
        if n in text:
            return True
    return False


def _en_word_match(text: str, needles: Iterable[str]) -> bool:
    """Whole-word match for English — avoid false positives like
    'bank' inside 'embankment'. Each needle is escaped, then wrapped
    with word boundaries."""
    if not text:
        return False
    lower = text.lower()
    for n in needles:
        if not n:
            continue
        nl = n.lower()
        # If the needle has spaces or special chars, fall back to substring
        if not re.match(r"^[\w\s.\-]+$", nl):
            if nl in lower:
                return True
            continue
        if re.search(rf"\b{re.escape(nl)}\b", lower):
            return True
    return False


def filter_for_portfolio(
    headlines: list[dict[str, Any]],
    portfolio_tickers: Iterable[str],
    include_sector_matches: bool = True,
    include_macro: bool = False,
) -> list[dict[str, Any]]:
    """Return only headlines that impact the held tickers.

    Args:
        headlines: Each dict should have keys: title, source, url,
            (optional) lang, date.
        portfolio_tickers: Iterable of canonical EGX tickers (e.g.
            ['CIRA', 'COMI', 'SWDY']).
        include_sector_matches: Keep headlines that match a held
            sector even if no specific ticker is named. Default True.
        include_macro: Keep headlines tagged as macro (CBE, EGP, rates).
            Default False — macro affects everyone, gets noisy.

    Returns:
        List of dicts with original keys plus:
          matched_tickers   list[str]
          matched_sectors   list[str]
          impact_level      "direct" | "sector" | "macro"
    """
    portfolio_tickers = [t.upper() for t in portfolio_tickers]
    held_set = set(portfolio_tickers)

    # Build per-ticker keyword set: ticker symbol + EN name + AR aliases
    ticker_keywords: dict[str, list[str]] = {}
    for tk in held_set:
        keys: list[str] = [tk]
        meta = EGX_UNIVERSE.get(tk, {})
        if meta.get("name"):
            keys.append(meta["name"])
            # Also key on the first significant word (e.g. "Talaat Moustafa Group" → "Talaat Moustafa")
            tokens = meta["name"].split()
            if len(tokens) >= 2:
                keys.append(" ".join(tokens[:2]))
        keys.extend(_AR_TICKER_ALIASES.get(tk, []))
        ticker_keywords[tk] = keys

    # Sectors held
    held_sectors = {EGX_UNIVERSE.get(tk, {}).get("sector") for tk in held_set}
    held_sectors.discard(None)
    held_sectors.discard("Index")

    out: list[dict[str, Any]] = []
    for art in headlines:
        title = (art.get("title") or "").strip()
        if not title:
            continue

        matched_tickers: list[str] = []
        for tk, kws in ticker_keywords.items():
            ar_kws = [k for k in kws if any(0x0600 <= ord(c) <= 0x06FF for c in k)]
            en_kws = [k for k in kws if k not in ar_kws]
            if _en_word_match(title, en_kws) or _ar_matches(title, ar_kws):
                matched_tickers.append(tk)

        matched_sectors: list[str] = []
        if include_sector_matches:
            for sector in held_sectors:
                kws = _SECTOR_KEYWORDS.get(sector, [])
                ar_kws = [k for k in kws if any(0x0600 <= ord(c) <= 0x06FF for c in k)]
                en_kws = [k for k in kws if k not in ar_kws]
                if _en_word_match(title, en_kws) or _ar_matches(title, ar_kws):
                    matched_sectors.append(sector)

        is_macro = False
        if include_macro:
            ar_kws = [k for k in _MACRO_KEYWORDS if any(0x0600 <= ord(c) <= 0x06FF for c in k)]
            en_kws = [k for k in _MACRO_KEYWORDS if k not in ar_kws]
            if _en_word_match(title, en_kws) or _ar_matches(title, ar_kws):
                is_macro = True

        if matched_tickers:
            impact = "direct"
        elif matched_sectors:
            impact = "sector"
        elif is_macro:
            impact = "macro"
        else:
            continue  # not relevant to the book — drop

        out.append({
            **art,
            "matched_tickers": matched_tickers,
            "matched_sectors": matched_sectors,
            "impact_level": impact,
        })

    # Sort: direct > sector > macro, then preserve original order within band
    rank = {"direct": 0, "sector": 1, "macro": 2}
    out.sort(key=lambda a: rank.get(a["impact_level"], 9))
    return out


def filter_for_portfolio_csv(
    headlines: list[dict[str, Any]],
    csv_path: str | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Convenience wrapper — reads tickers from the portfolio CSV."""
    from . import portfolio as port_mod
    summary = port_mod.summary(csv_path=csv_path)
    if "error" in summary:
        # No portfolio loaded — return unfiltered (caller decides what to do)
        log.warning(f"news_filter: portfolio not loaded ({summary.get('error')[:80]}); skipping filter")
        return headlines
    tickers = [p["ticker"].upper() for p in (summary.get("positions") or [])]
    return filter_for_portfolio(headlines, tickers, **kwargs)
