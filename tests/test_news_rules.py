"""EGX phrase rules for Arabic headlines (egx_news_rules / backend='rules').

Headlines here are written for the test, not copied from the labeled eval
set, so they check that the rules generalize rather than memorize.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from egx_mcp.data import egx_news_rules as rules
from egx_mcp.data import sentiment


def _sign(text: str) -> int:
    score, _ = rules.score_text(text)
    return (score > 0.1) - (score < -0.1)


class RuleTests(unittest.TestCase):
    def test_earnings_direction_beats_the_word_profits(self) -> None:
        self.assertEqual(_sign("أرباح \"النيل للأسمنت\" تتراجع 41% خلال النصف الأول"), -1)
        self.assertEqual(_sign("أرباح بنك القاهرة تهبط إلى 2.1 مليار جنيه"), -1)
        self.assertEqual(_sign("أرباح \"طلعت مصطفى\" تقفز 88% في الربع الثالث"), +1)
        # The old lexicon nets "أرباح" (+) against "تتراجع" (unknown) to +1.
        self.assertGreater(sentiment._score_text("أرباح الشركة تتراجع 41%", "ar")[0], 0)

    def test_losses_shrinking_is_good_and_growing_is_bad(self) -> None:
        self.assertEqual(_sign("خسائر \"جولدن\" تتراجع 60% في النصف الأول"), +1)
        self.assertEqual(_sign("شركة الصلب تقلص خسائرها إلى 12 مليون جنيه"), +1)
        self.assertEqual(_sign("خسائر \"الحديد\" ترتفع 150% خلال 2026"), -1)

    def test_turnarounds(self) -> None:
        self.assertEqual(_sign("\"النصر للملابس\" تتحول للربحية خلال الربع الأول"), +1)
        self.assertEqual(_sign("\"الشمس للفنادق\" تتحول للخسارة في 2026"), -1)
        self.assertEqual(_sign("شركة الأسمدة تتحول إلى الخسائر بنهاية العام"), -1)

    def test_despite_clause_does_not_flip_the_headline(self) -> None:
        self.assertEqual(_sign("رغم نمو الإيرادات.. أرباح \"الكابلات\" تتراجع 30%"), -1)
        self.assertEqual(_sign("أرباح البنك تنمو 20% رغم تراجع العائد على القروض"), +1)

    def test_payouts_ownership_and_governance(self) -> None:
        self.assertEqual(_sign("عمومية \"الشرقية\" تقرر عدم توزيع أرباح عن 2025"), -1)
        self.assertEqual(_sign("عمومية \"الشرقية\" تقر توزيع كوبون نقدي 2 جنيه"), +1)
        self.assertEqual(_sign("مساهم يبيع 12 مليون سهم في \"أوراسكوم\""), -1)
        self.assertEqual(_sign("\"السويدي\" تشتري 1.5 مليون سهم خزينة"), +1)
        self.assertEqual(_sign("استقالة العضو المنتدب لشركة الاتصالات"), -1)
        self.assertEqual(_sign("الرقابة المالية ترفض نشر نشرة الاكتتاب"), -1)

    def test_record_date_notice_and_routine_halt_are_neutral(self) -> None:
        self.assertEqual(_sign("البورصة تعلن نهاية حق وموعد توزيع كوبون \"أبوقير\""), 0)
        self.assertEqual(_sign("تم ايقاف الورقة المالية - كيما لمدة 10 دقائق لتجاوزها نسبة 10 %"), 0)

    def test_normalization_handles_marks_and_alef_forms(self) -> None:
        self.assertEqual(rules.normalize("​إيقاف  أرباحٌ"), "ايقاف  ارباح")
        self.assertEqual(_sign("​أرباحُ الشركة تتراجعُ 10%"), -1)


class BackendTests(unittest.TestCase):
    def test_rules_backend_is_opt_in(self) -> None:
        self.assertEqual(sentiment._resolve_backend("rules", "ar"), "rules")
        self.assertEqual(sentiment._resolve_backend("rules", "en"), "rules")
        self.assertEqual(sentiment._DEFAULT_BACKEND, "lexicon")

    def test_analyze_sentiment_uses_rules_when_asked(self) -> None:
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        feed = {"articles": [{"title": "أرباح الشركة تتراجع 41% في الربع الأول",
                              "date": today, "source": "t", "url": "u"}]}
        with patch.object(sentiment.news, "fetch", return_value=feed), \
             patch.object(sentiment, "resolve_ticker", return_value=("XYZ", None, None)):
            lex = sentiment.analyze_sentiment("XYZ", lang="ar", backend="lexicon")
            rul = sentiment.analyze_sentiment("XYZ", lang="ar", backend="rules")
        self.assertGreater(lex["aggregate_score"], 0)      # the old misread
        self.assertLess(rul["aggregate_score"], 0)
        self.assertEqual(rul["backend"], {"ar": "rules"})


if __name__ == "__main__":
    unittest.main()


class MixedFeedTests(unittest.TestCase):
    def test_arabic_title_in_english_feed_is_scored_as_arabic(self) -> None:
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        feed = {"articles": [{"title": "تراجع أرباح الشركة 30% خلال النصف الأول",
                              "date": today, "source": "Mubasher", "url": "u"}]}
        with patch.object(sentiment.news, "fetch", return_value=feed):
            out = sentiment.analyze_sentiment(None, lang="en", backend="rules")
        h = out["headlines"][0]
        self.assertEqual(h["lang"], "ar")
        self.assertLess(h["score"], 0)       # the English lexicon would give 0


class EnglishRuleTests(unittest.TestCase):
    """Fresh English headlines, not taken from the labeled eval set."""

    def _sign(self, text: str) -> int:
        from egx_mcp.data import egx_news_rules_en as en
        score, _ = en.score_text(text)
        return (score > 0.1) - (score < -0.1)

    def test_direction_attaches_to_the_subject(self) -> None:
        self.assertEqual(self._sign("Juhayna net profit falls 38% in 2Q"), -1)
        self.assertEqual(self._sign("Abu Qir Fertilizers profit jumps 72% on higher urea prices"), +1)
        self.assertEqual(self._sign("Ghabbour Auto loss narrows to EGP 40 mn"), +1)
        self.assertEqual(self._sign("Heliopolis Housing net loss widens in 1H"), -1)
        self.assertEqual(self._sign("Sidi Kerir swings to profit in Q2"), +1)
        # "Group" must not read as "up", nor "downgrade" as "down".
        self.assertEqual(self._sign("Orascom Development Group profit declines 12%"), -1)

    def test_estimates_guidance_and_payouts(self) -> None:
        self.assertEqual(self._sign("Fawry beats consensus estimates on digital payments"), +1)
        self.assertEqual(self._sign("Eastern Co misses analyst estimates"), -1)
        self.assertEqual(self._sign("Telecom Egypt raises full-year guidance"), +1)
        self.assertEqual(self._sign("Kima suspends dividend payments"), -1)
        self.assertEqual(self._sign("Edita announces share buyback program"), +1)

    def test_legal_and_macro(self) -> None:
        self.assertEqual(self._sign("Regulator refers broker to public prosecutor"), -1)
        self.assertEqual(self._sign("CBE tightens lending requirements for consumer loans"), -1)
        self.assertEqual(self._sign("Headline inflation eases to 11.2% in August"), +1)
        self.assertEqual(self._sign("Profit rises 10% despite weaker demand"), +1)

    def test_policy_announcements_stay_neutral(self) -> None:
        self.assertEqual(self._sign("Finance minister reviews draft real estate tax law"), 0)
