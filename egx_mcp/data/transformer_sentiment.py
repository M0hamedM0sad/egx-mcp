"""Transformer-backed headline sentiment — optional upgrade over the lexicon.

This is the model path behind `sentiment.analyze_sentiment(..., backend=...)`.
It loads two finance/Arabic-tuned classifiers from Hugging Face on first use:

  EN : ProsusAI/finbert
       — the canonical financial-sentiment model (pos/neg/neutral).
  AR : CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
       — dialectal Arabic sentiment; the `-da` variant fits the mixed
         MSA + Egyptian-dialect register of Mubasher / EGX press better
         than the fixed lexicon's exact-match + negation list.

Design constraints (mirroring the lexicon's contract in sentiment.py):
  - Returns a signed score in [-1, +1] and a list[str] of audit notes,
    exactly like the lexicon's `_score_text`, so the caller is agnostic.
  - A *neutral* prediction returns (0.0, []) — same semantics as "no
    tonal content" in the lexicon, so coverage_pct stays meaningful.
  - Heavy deps (transformers, torch) are optional. If they're missing or
    a model fails to load, `available()` returns False and the caller
    falls back to the lexicon. No hard dependency is introduced.
  - Models load lazily and once; per-text scores are memoised so repeated
    headlines (the common case across cached news pulls) cost nothing.

Override model ids with EGX_FINBERT_MODEL / EGX_ARABIC_SENTIMENT_MODEL.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("egx-mcp.sentiment.transformer")

_EN_MODEL = os.environ.get("EGX_FINBERT_MODEL", "ProsusAI/finbert")
_AR_MODEL = os.environ.get(
    "EGX_ARABIC_SENTIMENT_MODEL",
    "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
)

# Lazy singletons. None = not yet attempted; False = tried and failed.
_pipelines: dict[str, Any] = {}
_load_failed: set[str] = set()
_score_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}


def _get_pipeline(lang: str):
    """Lazily build and cache a text-classification pipeline for `lang`.

    Returns the pipeline, or None if transformers is unavailable or the
    model could not be loaded (logged once per lang).
    """
    if lang in _pipelines:
        return _pipelines[lang]
    if lang in _load_failed:
        return None

    model_id = _EN_MODEL if lang == "en" else _AR_MODEL
    try:
        from transformers import pipeline  # imported lazily — optional dep
    except Exception as e:  # noqa: BLE001 — any import error means "not available"
        log.warning(
            "transformers not installed (%s); falling back to lexicon. "
            "Install with: pip install 'egx-mcp[sentiment]'",
            e,
        )
        _load_failed.add(lang)
        return None

    try:
        pipe = pipeline(
            "text-classification",
            model=model_id,
            top_k=None,  # return scores for all labels
            truncation=True,
            max_length=256,
        )
    except Exception as e:  # noqa: BLE001 — download/load failure → fall back
        log.warning("failed to load %s for lang=%s: %s", model_id, lang, e)
        _load_failed.add(lang)
        return None

    _pipelines[lang] = pipe
    return pipe


def available(lang: str) -> bool:
    """True if a transformer pipeline can be used for this language."""
    return _get_pipeline(lang) is not None


# Label → sign. Models emit English or Arabic label strings; match on the
# lowercased label substring so we tolerate variants across model versions.
def _label_sign(label: str) -> int:
    lab = label.strip().lower()
    if lab.startswith("pos") or "positive" in lab or "إيجاب" in lab:
        return 1
    if lab.startswith("neg") or "negative" in lab or "سلب" in lab:
        return -1
    return 0  # neutral / unknown


def score_text(text: str, lang: str) -> tuple[float, list[str]]:
    """Score one headline. Returns (signed score in [-1, +1], audit notes).

    Signed score = P(positive) - P(negative); neutral-dominant headlines
    collapse to 0.0 with no notes, matching the lexicon's "no tonal
    content" semantics so the caller's coverage math is unchanged.
    """
    if not text:
        return 0.0, []

    key = (lang, text)
    cached = _score_cache.get(key)
    if cached is not None:
        return cached

    pipe = _get_pipeline(lang)
    if pipe is None:
        return 0.0, []  # caller should have checked available(); be safe

    try:
        raw = pipe(text)
    except Exception as e:  # noqa: BLE001 — runtime failure → neutral, don't crash
        log.warning("inference failed for lang=%s: %s", lang, e)
        return 0.0, []

    # pipeline(top_k=None) returns either [{label,score}, ...] or
    # [[{label,score}, ...]] depending on version — normalise to a flat list.
    scores = raw[0] if raw and isinstance(raw[0], list) else raw

    p_pos = p_neg = 0.0
    top_label, top_conf = "neutral", 0.0
    for item in scores:
        sign = _label_sign(item["label"])
        conf = float(item["score"])
        if sign > 0:
            p_pos = conf
        elif sign < 0:
            p_neg = conf
        if conf > top_conf:
            top_label, top_conf = item["label"], conf

    if _label_sign(top_label) == 0:
        result: tuple[float, list[str]] = (0.0, [])
    else:
        score = round(p_pos - p_neg, 3)
        note = f"model:{top_label.lower()}({top_conf:.2f})"
        result = (score, [note])

    _score_cache[key] = result
    return result
