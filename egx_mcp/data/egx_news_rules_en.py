"""Phrase rules for English EGX headlines — the English half of backend='rules'.

Same idea as egx_news_rules (Arabic): the word lexicon scores topic words
("profit", "dividend", "deal") as positive by themselves, so "net profit
falls 40%" nets to zero and "loss narrows" reads as negative. These rules
attach the direction to its subject:

  profit / earnings / net income + rises / jumps   -> positive
                                 + falls / slumps   -> negative
  loss + narrows / shrinks                          -> positive
  loss + widens / deepens / posts a loss            -> negative
  swings / returns to profit | to a loss
  beats / tops estimates | misses estimates
  raises | cuts guidance, upgrade | downgrade, dividend declared | cut,
  buyback, stake sale, probe / fraud / prosecutor / fine / resigns

"Despite ..." clauses are dropped before matching. Headlines no rule covers
fall back to the English lexicon with topic words removed.
"""
from __future__ import annotations

import re

_UP = r"\b(?:rises?|rose|jumps?|jumped|surges?|surged|soars?|soared|climbs?|climbed|grows?|grew|increases?|increased|up|doubles?|doubled|triples?|tripled|higher|gains?|leaps?|leapt|spikes?|rebounds?)\b"
_DOWN = r"\b(?:falls?|fell|drops?|dropped|declines?|declined|slumps?|slumped|plunges?|plunged|slides?|slid|dips?|dipped|down|shrinks?|shrank|lower|tumbles?|tumbled|sinks?|sank|decreases?|decreased|halves|halved|contracts?|contracted|eases?|eased|cools?|cooled|slows?|slowed)\b"
_PROFIT = r"(?:(?:net\s+)?profits?|earnings|net\s+income|ebitda|bottom\s+line|eps)"
_LOSS = r"(?:net\s+)?loss(?:es)?"
_GAP = r"[^.;:!?]{0,50}?"

_RULES: list[tuple[re.Pattern, int, str]] = [(re.compile(p, re.I), s, tag) for p, s, tag in [
    (r"(?:swings?|swung|returns?|returned|turns?|turned)\s+(?:to\s+(?:a\s+)?)?(?:profit|profitab)", +1, "turn_profit"),
    (r"(?:swings?|swung|turns?|turned|slips?|slipped)\s+(?:in)?to\s+(?:a\s+)?loss", -1, "turn_loss"),
    (_LOSS + _GAP + r"(?:narrows?|narrowed|shrinks?|shrank|eases?|eased|falls?|fell|declines?|declined|drops?|dropped|halves|halved)", +1, "loss_down"),
    (_LOSS + _GAP + r"(?:\b(?:widens?|widened|deepens?|deepened|swells?|swelled)\b|" + _UP + ")", -1, "loss_up"),
    (r"(?:posts?|posted|reports?|reported|books?|booked|records?|recorded)\s+(?:a\s+|an\s+)?(?:[\w-]+\s+){0,3}loss", -1, "posts_loss"),
    (_PROFIT + _GAP + _DOWN, -1, "profit_down"),
    (_PROFIT + _GAP + _UP, +1, "profit_up"),
    (r"(?:record|higher|stronger)\s+" + _PROFIT, +1, "profit_up"),
    (r"(?:lower|weaker)\s+" + _PROFIT, -1, "profit_down"),
    (r"(?:beats?|beat|tops?|topped|exceeds?|exceeded)\s+(?:[\w-]+\s+){0,2}(?:estimates|expectations|forecasts?|consensus)", +1, "beat"),
    (r"(?:miss(?:es|ed)?|falls?\s+short\s+of)\s+(?:[\w-]+\s+){0,2}(?:estimates|expectations|forecasts?|consensus|targets?)", -1, "miss"),
    (r"(?:raises?|raised|lifts?|lifted|ups?)\s+(?:[\w-]+\s+){0,2}(?:guidance|outlook|forecast|target)", +1, "guidance_up"),
    (r"(?:cuts?|lowers?|lowered|slashes|slashed|trims?|trimmed)\s+(?:[\w-]+\s+){0,2}(?:guidance|outlook|forecast|target)", -1, "guidance_down"),
    (r"upgrad(?:e|es|ed)\b", +1, "upgrade"),
    (r"downgrad(?:e|es|ed)\b", -1, "downgrade"),
    (r"(?:suspends?|suspended|scraps?|scrapped|skips?|skipped|cuts?|omits?)\s+(?:[\w-]+\s+){0,2}dividends?", -1, "dividend_cut"),
    (r"(?:approves?|approved|declares?|declared|proposes?|proposed|raises?|raised)\s+(?:[\w-]+\s+){0,3}(?:dividends?|payout|bonus\s+shares)", +1, "dividend_up"),
    (r"(?:buy-?backs?|treasury\s+shares|share\s+repurchase)", +1, "buyback"),
    (r"(?:sells?|sold|offloads?|offloaded|cuts?|reduces?|reduced|trims?)\s+(?:[\w-]+\s+){0,3}stake", -1, "stake_sale"),
    (r"(?:raises?|raised|increases?|increased|ups?|boosts?)\s+(?:[\w-]+\s+){0,3}stake", +1, "stake_buy"),
    (r"\b(?:probe|investigation|fraud|lawsuit|sued|prosecut\w*|fined?|penalt\w*|violations?|default(?:s|ed)?)\b", -1, "legal"),
    (r"\bresign(?:s|ed|ation)?\b|\bsteps?\s+down\b|\bousted\b", -1, "exit"),
    (r"\b(?:suspend(?:s|ed)?\s+trading|trading\s+halt(?:ed)?|delist\w*)\b", -1, "halt"),
    # Company wins: contracts, financing and strategic stakes taken in it.
    (r"\b(?:wins?|won|secures?|secured|signs?|signed|awarded)\s+(?:[\w-]+\s+){0,4}(?:contract|order|tender|financing|loan|facility|deal)\b", +1, "win"),
    (r"\b(?:equity\s+investment|takes?\s+(?:a\s+)?stake)\s+in\b", +1, "strategic_stake"),
    # Macro headwinds.
    (r"\btighten(?:s|ed|ing)?\s+(?:[\w-]+\s+){0,2}(?:rules|restrictions|requirements|controls|curbs)\b", -1, "tightening"),
    (r"\binflation\b" + _GAP + _UP, -1, "inflation_up"),
    (r"\binflation\b" + _GAP + _DOWN, +1, "inflation_down"),
    (r"\brecord\s+high\b|\ball-time\s+high\b|\brall(?:y|ies|ied)\b", +1, "price_up"),
    (r"\bsell-?off\b|\b(?:52-week|record)\s+low\b", -1, "price_down"),
]]

_DESPITE = re.compile(r"\b(?:despite|even\s+as|although|though)\b[^.;:,]*", re.I)

_TOPIC_WORDS = {
    "profit", "profits", "dividend", "deal", "acquisition", "acquires",
    "approval", "approved", "partnership", "raise", "raised", "raises",
    "cut", "cuts", "cutting", "risk", "fine", "buy", "sell", "record",
    "loss", "losses",
}


def score_text(text: str) -> tuple[float, list[str]]:
    """(score in [-1, +1], matched rule tags with sign) for one English headline."""
    t = _DESPITE.sub(" ", text or "")
    pos = neg = 0
    tags: list[str] = []
    used: list[tuple[int, int]] = []
    for pat, sign, tag in _RULES:
        for m in pat.finditer(t):
            a, b = m.span()
            if any(a < y and x < b for x, y in used):
                continue
            used.append((a, b))
            pos += sign > 0
            neg += sign < 0
            tags.append(("+" if sign > 0 else "-") + tag)
    if pos or neg:
        return (pos - neg) / (pos + neg), tags
    from . import sentiment   # local import: sentiment imports this module

    return sentiment._score_tokens(text, sentiment._EN_POS - _TOPIC_WORDS,
                                   sentiment._EN_NEG - _TOPIC_WORDS,
                                   sentiment._EN_NEGATORS)
