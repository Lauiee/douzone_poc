"""파이프라인 테스트"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rule_based import (
    apply_rule_based,
    normalize_numbers,
    _parse_sino_korean_number,
)
from src.jamo_corrector import JamoCorrector, to_jamo


class TestSinoKoreanParsing(unittest.TestCase):
    def test_single_digit(self):
        self.assertEqual(_parse_sino_korean_number("오"), 5)

    def test_tens(self):
        self.assertEqual(_parse_sino_korean_number("이십"), 20)

    def test_hundreds(self):
        self.assertEqual(_parse_sino_korean_number("백육십"), 160)


class TestRuleBased(unittest.TestCase):
    def test_number_with_unit(self):
        result, changes = normalize_numbers("오 분 동안 쉬세요")
        self.assertIn("5분", result)

    def test_blood_pressure(self):
        result, _ = normalize_numbers("백육십에 백")
        self.assertEqual(result, "160/100")

    def test_full_rule_based(self):
        result, _ = apply_rule_based("오 분, 십 분 잠깐 쉬면")
        self.assertIn("5분", result)
        self.assertIn("10분", result)

    def test_han_sip_minutes(self):
        result, _ = normalize_numbers("네 한 십분 이십분이면")
        self.assertIn("10분", result)
        self.assertIn("20분", result)

    def test_no_false_positive_oseyo(self):
        self.assertEqual(normalize_numbers("들어오세요")[0], "들어오세요")

    def test_no_false_positive_do(self):
        result, _ = apply_rule_based("걷기도 힘들어요")
        self.assertNotIn("°C", result)

    def test_no_false_positive_jeongdo(self):
        result, _ = normalize_numbers("일주일 정도 된 것")
        self.assertNotIn("1정", result)


class TestJamoCorrector(unittest.TestCase):
    def test_jamo_decomposition(self):
        self.assertNotEqual(to_jamo("한글"), "한글")

    def test_known_word_no_change(self):
        c = JamoCorrector()
        _, changes = c.correct("고혈압 환자입니다")
        self.assertFalse(any(ch["original"] == "고혈압" for ch in changes))

    def test_similar_word_correction(self):
        c = JamoCorrector()
        result, changes = c.correct("극성충수염이 의심됩니다")
        if any("급성충수염" in ch.get("corrected", "") for ch in changes):
            self.assertIn("급성충수염", result)

    def test_no_false_positive_common_words(self):
        c = JamoCorrector()
        _, changes = c.correct("어제 다시 왔어요")
        o = [c_["original"] for c_ in changes]
        self.assertNotIn("어제", o)

    def test_short_word_skip(self):
        c = JamoCorrector(min_term_length=3)
        _, changes = c.correct("열이 나요")
        self.assertFalse(any(len(c_["original"]) < 3 for c_ in changes))

    def test_administrative_word_not_overcorrected(self):
        c = JamoCorrector()
        text = "그럼 바로 이번 수속하면 되나요?"
        result, changes = c.correct(text)
        self.assertIn("수속하면", result)
        self.assertFalse(any(ch["original"] == "수속하면" for ch in changes))


class TestSample1Integration(unittest.TestCase):
    SAMPLE = (
        "오늘 어디가 불편하셔서 오셨어요? 일주일 정도 된 것 같은데 계단 오르내리면 "
        "가슴이 두근거리면서 쪼이는 통증이 있어요. 왼쪽 어깨랑 팔까지 아플 때도 있어요. "
        "오 분, 십 분 잠깐 쉬면 또 괜찮더라고요. 고혈압약은 오 년 전부터 먹고 있는데 "
        "먹다 안 먹다 해요. 혈압이 백육십에 백인데 안정 한번 하시고 다시 한번 재보고 "
        "심장 쪽에 문제 있는지 혈액 검사랑 심전도 한 번 해보시죠."
    )

    def test_rule_based_on_sample1(self):
        result, _ = apply_rule_based(self.SAMPLE)
        self.assertIn("5분", result)
        self.assertIn("160/100", result)

    def test_jamo_on_sample1(self):
        rule_result, _ = apply_rule_based(self.SAMPLE)
        c = JamoCorrector()
        result, changes = c.correct(rule_result)
        self.assertNotIn("다시", [x["original"] for x in changes])
        self.assertIn("심전도", result)


class TestSample2Integration(unittest.TestCase):
    SAMPLE = (
        "박준호님 들어오세요. 어제 저녁부터 체한 것 같더니 지금은 오른쪽 발리뼈가 너무 아파서 "
        "걷기도 힘들어요. 침대에 누워보세요. 개를 좀 눌러보겠습니다. 손을 뗄 때 꾀안이 울리면서 "
        "훨씬 더 아파요. 반동성 약풍이 뚜렷하네요. 맹장염 즉 극성충수염입니다."
    )

    def test_rule_then_jamo(self):
        ruled, _ = apply_rule_based(self.SAMPLE)
        self.assertIn("발리뼈", ruled)
        self.assertIn("개를", ruled)
        self.assertIn("약풍", ruled)
        self.assertIn("극성충수염", ruled)
        c = JamoCorrector()
        final, _ = c.correct(ruled)
        self.assertIn("급성충수염", final)


if __name__ == "__main__":
    unittest.main(verbosity=2)
