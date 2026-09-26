"""Phrase rules for Arabic EGX headlines — a third sentiment backend.

The word lexicon in sentiment.py scores "أرباح" (profits) as positive on its
own, so "أرباح X تتراجع 69%" nets to zero, and it misses the verb forms the
EGX press actually uses (تتراجع، تهبط، تقفز). On 382 labeled briefing
headlines it caught 5 of 63 negatives; CAMeL-BERT (a general dialect model)
caught 17 and lost half the positives.

These rules read the structure of an EGX headline instead of single words:

  earnings + direction   أرباح … تتراجع / تهبط          -> negative
                         أرباح … تقفز / ترتفع / تنمو     -> positive
  losses + direction     خسائر … تقفز / ترتفع            -> negative
                         خسائر … تتراجع / تقلص           -> positive
  turnarounds            تتحول للربحية / للخسارة
  payouts                تقرر / تقر توزيع، أسهم مجانية   -> positive
                         عدم توزيع                        -> negative
  stakes                 مساهم يبيع / يخفض حصته          -> negative
                         يرفع حصته، شراء أسهم خزينة      -> positive
  governance / market    استقالة، تفتيش، الرقابة ترفض، إيقاف التداول،
                         ضغوط بيعية، تجاوز الخسائر نصف   -> negative
                         قمة تاريخية، يقفز X%             -> positive

Anything no rule covers falls back to the word lexicon with topic words
(أرباح، توزيع، صفقة …) removed, since they name the subject rather than
its direction. English headlines use the lexicon unchanged.

Selected with backend="rules" (or EGX_SENTIMENT_BACKEND=rules); the default
backend is unchanged.
"""
from __future__ import annotations

import re

_DIACRITICS = re.compile(r"[ً-ْـ​-‏﻿]")


def normalize(text: str) -> str:
    """Strip diacritics/tatweel/zero-width marks and unify alef, ya, ta marbuta."""
    t = _DIACRITICS.sub("", text or "")
    t = re.sub("[إأآ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("چ", "ج")
    return t


# Direction verbs/nouns (normalized spelling).
_UP = r"(?:ترتفع|يرتفع|ارتفاع|ارتفعت|تقفز|يقفز|قفزه|قفزت|تنمو|ينمو|نمو|تصعد|صعود|تزيد|زياده|تتضاعف|تضاعف|تقفز)"
_DOWN = r"(?:تتراجع|يتراجع|تراجع|تراجعت|تهبط|يهبط|هبوط|هبطت|تنخفض|ينخفض|انخفاض|انخفضت|تنكمش|تقلص|يقلص|تقلصت)"
_PROFIT = r"(?:ارباح|الارباح|ربحيه|صافي ربح|صافي الربح)"
_LOSS = r"(?:خسائر|الخسائر|خسائرها|خساره|الخساره)"
_GAP = r"[^؟!]{0,60}?"       # up to ~a clause between subject and verb

# (pattern, sign, tag). Order matters only for the audit trail.
_RULES: list[tuple[re.Pattern, int, str]] = [(re.compile(p), s, tag) for p, s, tag in [
    # Turnarounds first: "تتحول للربحية" must not also read as "ربحية ↑".
    # "للربحية" is ل + الربحية with the alef dropped, hence (?:لل|الي ال|ل).
    (r"(?:تتحول|يتحول|تحول|تحولها|تحوله)\s+" + _GAP + r"(?:لل|الي\s+ال|ل)ربحيه", +1, "turn_profit"),
    (r"(?:تتحول|يتحول|تحول|تحولها|تحوله)\s+" + _GAP + r"(?:لل|الي\s+ال|ل)خسا(?:ره|ئر)", -1, "turn_loss"),
    # Losses shrinking is good; losses growing is bad.
    (_LOSS + _GAP + _DOWN, +1, "loss_down"),
    (_DOWN + r"\s+" + _LOSS, +1, "loss_down"),
    (_LOSS + _GAP + _UP, -1, "loss_up"),
    (r"تجاوز\s+الخسائر", -1, "loss_breach"),
    # Earnings direction.
    (_PROFIT + _GAP + _DOWN, -1, "profit_down"),
    (_DOWN + r"\s+" + _PROFIT, -1, "profit_down"),
    (_PROFIT + _GAP + _UP, +1, "profit_up"),
    (r"ب?" + _UP + _GAP + _PROFIT, +1, "profit_up"),
    (r"(?:ترفع|يرفع)\s+توقعات\s+ارباح", +1, "guidance_up"),
    # Payout decisions (announced), not record-date notices.
    (r"عدم\s+توزيع", -1, "no_dividend"),
    (r"ترحيل\s+الارباح", -1, "no_dividend"),
    (r"للعاملين\s+دون\s+المساهمين", -1, "staff_not_holders"),
    (r"(?:تقر|تقرر|يقر|يقرر|تقترح|تعتمد|يقرون)\s+" + _GAP + r"توزيع", +1, "dividend_decision"),
    (r"(?:تقرر|يقرر|يقرون|تقر)\s+" + _GAP + r"اسهم\s+مجانيه", +1, "bonus_decision"),
    # Ownership flows.
    (r"مساهم\s+(?:يبيع|يخفض)", -1, "holder_sells"),
    (r"(?:يخفض|تخفض)\s+حصت", -1, "holder_sells"),
    (r"(?:يرفع|ترفع)\s+حصت", +1, "holder_buys"),
    (r"(?:تشتري|يشتري|شراء)\s+" + _GAP + r"(?:سهم|اسهم)\s+خزينه", +1, "buyback"),
    (r"عرضا?\s+للاستحواذ", +1, "takeover_bid"),
    # Governance / regulator / trading halts.
    (r"استقاله", -1, "resignation"),
    (r"تفتيش", -1, "inspection"),
    (r"الرقابه\s+الماليه\s+ترفض", -1, "regulator_rejects"),
    (r"(?:ايقاف|وقف)\s+(?:التداول|التعامل)", -1, "halt"),
    (r"تاخر\s+افصاحات", -1, "late_filing"),
    # Price action words the press uses.
    (r"ضغوطا?\s+بيعيه", -1, "selling_pressure"),
    (r"بانخفاض", -1, "price_down"),
    (r"قمه\s+تاريخيه|قمم\s+تاريخيه", +1, "record_high"),
    (r"يقفز\s+(?:اكثر\s+من\s+)?\d+", +1, "price_jump"),
]]

_DESPITE = re.compile(r"رغم[^.]*")

# Short halts for crossing a daily limit are routine, not bad news.
_ROUTINE = re.compile(r"لمده\s+10\s+دقائق")

# Lexicon words that name a topic rather than a direction in EGX headlines.
_TOPIC_WORDS = {
    "أرباح", "ربح", "أرباحاً", "ربحية", "ربح صافي", "صافي ربح", "توزيع",
    "كوبون", "كوبونات", "صفقة", "صفقات", "اعتماد", "تعتمد", "موافقة",
    "وافق", "وافقت", "زيادة", "زاد", "زادت", "تجاوز", "يتجاوز", "تخفيض",
    "خفض", "خفضت", "تحقيق", "وقف",
}


def score_text(text: str) -> tuple[float, list[str]]:
    """(score in [-1, +1], matched rule tags with sign) for one Arabic headline."""
    t = normalize(text)
    # "… despite revenue falling": the despite-clause carries the opposite
    # direction of the headline, so it is dropped before matching.
    t = _DESPITE.sub(" ", t)
    if _ROUTINE.search(t):
        return 0.0, ["0routine_halt"]
    pos = neg = 0
    tags: list[str] = []
    used: list[tuple[int, int]] = []
    for pat, sign, tag in _RULES:
        for m in pat.finditer(t):
            span = m.span()
            # One stretch of text counts once (turnaround beats profit_up).
            if any(span[0] < b and a < span[1] for a, b in used):
                continue
            used.append(span)
            if sign > 0:
                pos += 1
            else:
                neg += 1
            tags.append(("+" if sign > 0 else "-") + tag)
    if pos or neg:
        return (pos - neg) / (pos + neg), tags
    return _lexicon_fallback(text)


def _lexicon_fallback(text: str) -> tuple[float, list[str]]:
    from . import sentiment   # local import: sentiment imports this module

    pos_lex = sentiment._AR_POS - _TOPIC_WORDS
    neg_lex = sentiment._AR_NEG - _TOPIC_WORDS
    return sentiment._score_tokens(text, pos_lex, neg_lex, sentiment._AR_NEGATORS)
