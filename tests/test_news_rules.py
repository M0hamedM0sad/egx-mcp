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
    def test_rules_backend_is_arabic_only_and_opt_in(self) -> None:
        self.assertEqual(sentiment._resolve_backend("rules", "ar"), "rules")
        self.assertEqual(sentiment._resolve_backend("rules", "en"), "lexicon")
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
