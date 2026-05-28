"""B2 — Structured event extraction from EGX disclosures.

The catalyst layer currently leans on keyword matching, which is fragile and
language-brittle. This tags each disclosure into a fixed set of price-moving
event types using a zero-shot classifier (multilingual NLI), so AR and EN
disclosures map to the same labels with no per-language rule maintenance:

    dividend distribution | capital increase | acquisition or merger |
    profit warning or loss | earnings results | board or management change |
    trading suspension | other / routine

It returns a `material` flag the decision/catalyst layer can use to gate
verdicts (e.g. a profit warning or trading suspension should block a BUY),
turning a keyword guess into an auditable, scored signal.

Design mirrors transformer_sentiment.py: the model is an OPTIONAL dep, loaded
lazily once; if transformers isn't installed or the model can't load, it falls
back to a small AR+EN keyword classifier so the function always returns
something. Override the model with EGX_NLI_MODEL.

    pip install 'egx-mcp[sentiment]'     # provides transformers + torch
"""
from __future__ import annotations

import logging
import os
from typing import Any

from . import disclosures as disc_mod
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.events")

_NLI_MODEL = os.environ.get("EGX_NLI_MODEL", "joeddav/xlm-roberta-large-xnli")

# Candidate labels in English — the NLI model is multilingual, so Arabic
# disclosure text classifies against these fine.
EVENT_LABELS = [
    "dividend distribution",
    "capital increase",
    "acquisition or merger",
    "profit warning or loss",
    "earnings results",
    "board or management change",
    "trading suspension",
    "other routine announcement",
]

# Events that move price / should gate a BUY verdict.
_MATERIAL = {
    "dividend distribution", "capital increase", "acquisition or merger",
    "profit warning or loss", "earnings results", "trading suspension",
}

# Keyword fallback (AR + EN) when the model isn't available. Order matters:
# first match wins, so list the more specific/material events first.
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("trading suspension", ("suspend", "halt", "تعليق التداول", "وقف التداول", "إيقاف")),
    ("profit warning or loss", ("loss", "warning", "warns", "خسارة", "خسائر", "تحذير", "تراجع الأرباح")),
    ("capital increase", ("capital increase", "rights issue", "زيادة رأس المال", "رأس المال")),
    ("acquisition or merger", ("acquire", "acquisition", "merger", "stake", "استحواذ", "اندماج", "حصة")),
    ("dividend distribution", ("dividend", "coupon", "distribution", "توزيع", "كوبون", "أرباح نقدية")),
    ("earnings results", ("results", "earnings", "profit", "net income", "نتائج", "أرباح", "القوائم المالية")),
    ("board or management change", ("board", "ceo", "chairman", "resign", "appoint",
                                    "مجلس الإدارة", "استقالة", "تعيين", "الرئيس التنفيذي")),
]

_pipeline: Any = None
_load_failed = False


def _get_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None:
        return _pipeline
    if _load_failed:
        return None
    try:
        from transformers import pipeline
    except Exception as e:  # noqa: BLE001
        log.warning("transformers not installed (%s); using keyword fallback. "
                    "Install with: pip install 'egx-mcp[sentiment]'", e)
        _load_failed = True
        return None
    try:
        _pipeline = pipeline("zero-shot-classification", model=_NLI_MODEL)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to load %s: %s; using keyword fallback", _NLI_MODEL, e)
        _load_failed = True
        return None
    return _pipeline


def available() -> bool:
    return _get_pipeline() is not None


def _keyword_classify(text: str) -> tuple[str, float]:
    low = text.lower()
    for label, kws in _KEYWORDS:
        if any(k in low for k in kws):
            return label, 0.5  # fixed low confidence — it's a keyword guess
    return "other routine announcement", 0.0


def classify_event(text: str) -> dict[str, Any]:
    """Classify one disclosure title into an event type.

    Returns {event, confidence, material, method}.
    """
    if not text or not text.strip():
        return {"event": "other routine announcement", "confidence": 0.0,
                "material": False, "method": "empty"}

    pipe = _get_pipeline()
    if pipe is None:
        label, conf = _keyword_classify(text)
        method = "keyword"
    else:
        try:
            res = pipe(text, candidate_labels=EVENT_LABELS, multi_label=False)
            label, conf = res["labels"][0], float(res["scores"][0])
            method = "zero-shot-nli"
        except Exception as e:  # noqa: BLE001
            log.warning("NLI inference failed: %s; keyword fallback", e)
            label, conf = _keyword_classify(text)
            method = "keyword"

    return {
        "event": label,
        "confidence": round(conf, 3),
        "material": label in _MATERIAL,
        "method": method,
    }


def tag_disclosures(user_ticker: str | None = None, days: int = 14) -> dict[str, Any]:
    """Pull recent disclosures and tag each with an event type.

    Returns the disclosures enriched with event/confidence/material, plus a
    `material_events` shortlist the catalyst/decision layer can gate on.
    """
    canonical = None
    if user_ticker:
        canonical, _, _ = resolve_ticker(user_ticker)

    raw = disc_mod.fetch(ticker=user_ticker, days=days)
    tagged: list[dict[str, Any]] = []
    material: list[dict[str, Any]] = []
    for d in raw.get("disclosures", []) or []:
        ev = classify_event(d.get("title") or "")
        row = {**d, **ev}
        tagged.append(row)
        if ev["material"]:
            material.append(row)

    return {
        "ticker": canonical,
        "days": days,
        "count": len(tagged),
        "backend": "zero-shot-nli" if available() else "keyword",
        "disclosures": tagged,
        "material_events": material,
        "method": (
            "Each disclosure title classified into a fixed event taxonomy via "
            "multilingual zero-shot NLI (AR+EN), or a keyword fallback when the "
            "model isn't installed. `material` flags price-moving events the "
            "decision layer can use to gate a BUY."
        ),
        "disclaimer": (
            "Event tags are model inferences, not legal classifications. "
            "Verify material events against the official EGX disclosure before acting."
        ),
    }
