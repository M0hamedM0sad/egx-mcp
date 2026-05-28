"""Headline sentiment for EGX names — Arabic + English.

Lightweight lexicon-based scorer. No transformer dependency, no API key.
Scores each headline on a -1..+1 scale; aggregates across the recent
window into a tone reading the host LLM can use.

Why a lexicon, not a model:
  - EGX is retail-driven and the news flow is dominated by Mubasher
    (AR) and Yahoo (EN). Retail tone moves prices on thin names.
  - A small, finance-tuned lexicon scores in milliseconds with no
    network call and no GPU. Accuracy is "directionally right" — good
    enough for a regime indicator, not for individual headline calls.
  - The lexicon is auditable and editable. If a word doesn't fit the
    Egyptian financial press, fix it here in five seconds.

Headlines that match neither side score 0. The aggregate is mean of
non-zero scores, plus a coverage ratio so the LLM knows how much of the
sample actually had tonal content.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

from . import news
from . import transformer_sentiment
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.sentiment")

# Default scoring backend. "lexicon" keeps the zero-dependency, no-GPU
# behaviour; "transformer" uses FinBERT (EN) + CAMeLBERT-DA (AR) when the
# optional deps are installed; "auto" prefers the transformer per-language
# and silently falls back to the lexicon when it isn't available.
_DEFAULT_BACKEND = os.environ.get("EGX_SENTIMENT_BACKEND", "lexicon").lower()


# ---------------------------------------------------------------------------
# Lexicons — finance-tuned for EGX flow
# ---------------------------------------------------------------------------

_EN_POS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally",
    "rallies", "record", "upgrade", "upgraded", "outperform",
    "outperforms", "raised", "raises", "raise", "expands", "expansion",
    "growth", "growing", "profit", "profits", "approved", "approval",
    "wins", "won", "secure", "secured", "secures", "strong", "robust",
    "boost", "boosts", "exceed", "exceeds", "exceeded", "rebound",
    "rebounds", "recovery", "buy", "bullish", "dividend", "buyback",
    "acquires", "acquisition", "partnership", "deal", "expand",
    "favorable", "improve", "improves", "improved", "milestone",
    "tailwind", "accretive", "raise guidance", "beats estimates",
}
_EN_NEG = {
    "miss", "misses", "missed", "drop", "drops", "fall", "falls",
    "plunge", "plunges", "tumble", "tumbles", "downgrade", "downgraded",
    "underperform", "underperforms", "loss", "losses", "decline",
    "declines", "declined", "weak", "weakness", "warning", "warns",
    "concern", "concerns", "investigation", "probe", "fraud", "lawsuit",
    "sued", "fine", "fined", "penalty", "delisted", "delisting",
    "bearish", "sell", "sell-off", "sell off", "crash", "headwind",
    "headwinds", "cut", "cuts", "cutting", "layoff", "layoffs", "fired",
    "resign", "resigns", "resigned", "bankruptcy", "default", "defaulted",
    "downgraded", "shortfall", "guidance cut", "missed estimates",
    "deteriorat", "uncertainty", "risk", "risky", "delay", "delayed",
}

# Arabic finance lexicon — common Mubasher / EGX press vocabulary
_AR_POS = {
    "ارتفاع", "ارتفع", "ترتفع", "ترتفعت", "صعود", "صاعد", "قفزة",
    "تجاوز", "يتجاوز", "تخطى", "يتخطى", "زيادة", "زاد", "زادت",
    "نمو", "ينمو", "أرباح", "ربح", "أرباحاً", "ربحية", "تحسن",
    "إيجابي", "إيجابية", "موافقة", "وافق", "وافقت", "اعتماد", "تعتمد",
    "توزيع", "كوبون", "كوبونات", "صفقة", "صفقات", "استحواذ", "شراكة",
    "تعاون", "افتتاح", "افتتح", "افتتحت", "توسع", "تتوسع", "قياسي",
    "قياسية", "قياسياً", "قوي", "قوية", "ربح صافي", "صافي ربح", "إنجاز",
    "تطور", "تطورات إيجابية",
}
_AR_NEG = {
    "تراجع", "تراجعت", "هبوط", "هبط", "هبطت", "انخفاض", "انخفض",
    "انخفضت", "خسارة", "خسائر", "ضعف", "ضعيف", "ضعيفة", "تباطؤ",
    "تباطأ", "تحذير", "حذرت", "تحذر", "تحقيق", "تحقيقات", "تخفيض",
    "خفض", "خفضت", "تقاضي", "غرامة", "غرامات", "احتيال", "إفلاس",
    "تعثر", "تعثرت", "تعليق التداول", "وقف", "أوقف", "أوقفت",
    "استقالة", "استقال", "أقال", "تسريح", "تأخير", "تأخر", "تأخرت",
    "خطر", "مخاطر", "أزمة", "سلبي", "سلبية",
}

# Light negation handlers — both languages
_EN_NEGATORS = {"not", "no", "won't", "wouldn't", "didn't", "doesn't", "isn't", "aren't"}
_AR_NEGATORS = {"لا", "لن", "ليس", "ليست", "ولا", "بدون", "غير"}

_TOKEN_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def _score_text(text: str, lang: str) -> tuple[float, list[str]]:
    """Return (score in -1..+1, list of matched terms with sign)."""
    if not text:
        return 0.0, []

    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0, []

    pos_lex, neg_lex = (_EN_POS, _EN_NEG) if lang == "en" else (_AR_POS, _AR_NEG)
    negators = _EN_NEGATORS if lang == "en" else _AR_NEGATORS

    pos_hits = 0
    neg_hits = 0
    matched: list[str] = []

    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        flipped = prev in negators

        if tok in pos_lex:
            if flipped:
                neg_hits += 1
                matched.append(f"-{tok}")
            else:
                pos_hits += 1
                matched.append(f"+{tok}")
        elif tok in neg_lex:
            if flipped:
                pos_hits += 1
                matched.append(f"+{tok}")
            else:
                neg_hits += 1
                matched.append(f"-{tok}")

    total = pos_hits + neg_hits
    if total == 0:
        return 0.0, []

    score = (pos_hits - neg_hits) / total
    return score, matched


def _resolve_backend(backend: str, lang: str) -> str:
    """Decide the effective backend for one language.

    Returns "transformer" only when requested (or "auto") AND the model is
    actually loadable for that language; otherwise "lexicon".
    """
    if backend == "lexicon":
        return "lexicon"
    if backend in ("transformer", "auto"):
        if transformer_sentiment.available(lang):
            return "transformer"
        if backend == "transformer":
            log.warning(
                "backend='transformer' requested but model unavailable for "
                "lang=%s; falling back to lexicon", lang,
            )
        return "lexicon"
    log.warning("unknown sentiment backend %r; using lexicon", backend)
    return "lexicon"


def _score_headline(text: str, lang: str, backend: str) -> tuple[float, list[str]]:
    """Score one headline with the resolved backend for its language."""
    if backend == "transformer":
        return transformer_sentiment.score_text(text, lang)
    return _score_text(text, lang)


def _label(score: float) -> str:
    if score >= 0.4:
        return "bullish"
    if score >= 0.1:
        return "mildly_bullish"
    if score <= -0.4:
        return "bearish"
    if score <= -0.1:
        return "mildly_bearish"
    return "neutral"


def analyze_sentiment(
    user_ticker: str | None = None,
    lang: str = "both",
    limit: int = 15,
    backend: str | None = None,
) -> dict[str, Any]:
    """Score recent headlines for an EGX name (or the market).

    Args:
        user_ticker: EGX code or nickname. Omit for market-wide.
        lang: 'en', 'ar', or 'both' (default).
        limit: Max headlines per language. Default 15.
        backend: 'lexicon' (default, zero-dependency), 'transformer'
            (FinBERT EN + CAMeLBERT-DA AR), or 'auto' (transformer when
            available, else lexicon). Defaults to the EGX_SENTIMENT_BACKEND
            env var, or 'lexicon' if unset. Resolved per-language, so EN
            can use the model while AR falls back, or vice versa.

    Returns:
        Dict with: ticker, lang, headline_count, aggregate_score,
        coverage_pct, label, headlines (per-headline score + matches),
        bull_signals, bear_signals, backend (effective per language).
    """
    requested_backend = (backend or _DEFAULT_BACKEND).lower()

    canonical = None
    if user_ticker:
        canonical, _, _ = resolve_ticker(user_ticker)

    sources: list[tuple[str, dict[str, Any]]] = []
    if lang in ("en", "both"):
        sources.append(("en", news.fetch(ticker=canonical, lang="en", limit=limit)))
    if lang in ("ar", "both"):
        sources.append(("ar", news.fetch(ticker=canonical, lang="ar", limit=limit)))

    scored: list[dict[str, Any]] = []
    bull_signals: list[str] = []
    bear_signals: list[str] = []
    backend_used: dict[str, str] = {}

    for lng, payload in sources:
        eff_backend = _resolve_backend(requested_backend, lng)
        backend_used[lng] = eff_backend
        for art in payload.get("articles", []) or []:
            title = art.get("title") or ""
            score, matches = _score_headline(title, lng, eff_backend)
            entry = {
                "lang": lng,
                "date": art.get("date"),
                "source": art.get("source"),
                "title": title,
                "url": art.get("url"),
                "score": round(score, 3),
                "matches": matches,
            }
            scored.append(entry)
            if score >= 0.34 and title:
                bull_signals.append(title)
            elif score <= -0.34 and title:
                bear_signals.append(title)

    nonzero = [h for h in scored if h["score"] != 0]
    n = len(scored)
    if nonzero:
        agg = sum(h["score"] for h in nonzero) / len(nonzero)
    else:
        agg = 0.0
    coverage = (len(nonzero) / n * 100) if n else 0.0

    return {
        "ticker": canonical,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "lang_requested": lang,
        "headline_count": n,
        "scored_count": len(nonzero),
        "coverage_pct": round(coverage, 1),
        "aggregate_score": round(agg, 3),
        "label": _label(agg),
        "bull_signals": bull_signals[:5],
        "bear_signals": bear_signals[:5],
        "headlines": scored,
        "backend": backend_used,
        "method": (
            f"Backend per language: {backend_used or 'lexicon'}. Per-headline "
            "score in [-1, +1]. Aggregate is mean over non-zero headlines. "
            "Coverage = share of headlines with tonal content. Lexicon is "
            "directionally right; transformer (FinBERT EN / CAMeLBERT-DA AR) "
            "is headline-accurate at the cost of model load + inference."
        ),
        "disclaimer": (
            "Tone reading only. Does not predict price. Combine with "
            "fundamentals + technicals before acting."
        ),
    }
