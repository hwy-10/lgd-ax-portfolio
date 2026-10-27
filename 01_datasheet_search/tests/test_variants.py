"""품번 추출과 변종 게이트."""

import unittest

from src.variants import (
    chunk_applies_to,
    extract_parts,
    mentions_exactly,
    parse_part,
    query_part,
)


class ParsePartTests(unittest.TestCase):
    def test_splits_family_and_suffix(self):
        part = parse_part("CH334U")
        self.assertEqual(part.family, "CH334")
        self.assertEqual(part.suffix, "U")
        self.assertTrue(part.is_specific)

    def test_family_without_suffix_is_not_specific(self):
        part = parse_part("CH334")
        self.assertEqual(part.family, "CH334")
        self.assertFalse(part.is_specific)

    def test_rejects_tokens_with_one_digit(self):
        # USB2 · INT8 같은 일반 토큰을 품번으로 오인하면 안 된다
        self.assertIsNone(parse_part("USB2"))
        self.assertIsNone(parse_part("INT8"))

    def test_accepts_other_vendors(self):
        self.assertEqual(parse_part("TUSB2046B").family, "TUSB2046")
        self.assertEqual(parse_part("TPS2552").suffix, "")


class ExtractTests(unittest.TestCase):
    def test_extracts_in_order_without_duplicates(self):
        raws = [p.raw for p in extract_parts("CH334U and CH334R and CH334U again")]
        self.assertEqual(raws, ["CH334U", "CH334R"])

    def test_query_part_prefers_specific_variant(self):
        self.assertEqual(query_part("CH334 계열 중 CH334U 전압").raw, "CH334U")

    def test_query_part_returns_none_without_part(self):
        self.assertIsNone(query_part("동작 전압이 얼마인가요?"))


class GateTests(unittest.TestCase):
    """게이트 4가지 경우 — 이 테스트가 이 프로젝트의 핵심이다."""

    def setUp(self):
        self.target = parse_part("CH334U")

    def test_includes_chunk_naming_the_target(self):
        self.assertTrue(chunk_applies_to("CH334U operating voltage 3.3V", self.target))

    def test_includes_family_only_chunk(self):
        # 변종을 특정하지 않은 계열 공통 설명
        self.assertTrue(chunk_applies_to("CH334 series supports USB 2.0", self.target))

    def test_includes_chunk_without_any_part_number(self):
        self.assertTrue(chunk_applies_to("각 다운스트림 포트는 개별 전류 제한", self.target))

    def test_excludes_chunk_specific_to_another_variant(self):
        # CH334U 질의에 CH334R 전용 표가 오면 안 된다
        self.assertFalse(chunk_applies_to("CH334R pin 12 is NC", self.target))

    def test_includes_comparison_table_listing_both(self):
        # 비교표는 target 을 포함하므로 통과해야 한다
        self.assertTrue(
            chunk_applies_to("Package: CH334U QSOP-28, CH334R SSOP-28", self.target)
        )


class EvidenceTests(unittest.TestCase):
    def test_mentions_exactly_is_strict(self):
        target = parse_part("CH334U")
        self.assertTrue(mentions_exactly("see CH334U table", target))
        self.assertFalse(mentions_exactly("see CH334R table", target))
        self.assertFalse(mentions_exactly("see CH334 family", target))


if __name__ == "__main__":
    unittest.main()
