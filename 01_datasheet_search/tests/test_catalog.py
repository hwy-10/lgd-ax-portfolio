"""여러 문서로 넓힌 검색 — 문서 선택, 문서별 변종 규칙, BOM 읽기."""

import unittest

from src.catalog import Catalog, ManifestRules, _expand_refs
from src.variants import PartNumber


class RefsTests(unittest.TestCase):
    def test_range_and_list(self):
        self.assertEqual(_expand_refs("`C1`~`C4`"), ("C1", "C2", "C3", "C4"))
        self.assertEqual(_expand_refs("`R7`,`R9`"), ("R7", "R9"))


class ManifestRulesTests(unittest.TestCase):
    def setUp(self):
        catalog = Catalog()
        self.rules = {d.id: catalog.rules[d.id] for d in catalog.docs}

    def test_ti_variant_and_package_suffix(self):
        tps = self.rules["tps2552"]
        self.assertEqual(tps.variants_in("TPS2552DBVR"), {"TPS2552"})
        self.assertEqual(tps.variants_in("TPS2552-1DBVR"), {"TPS2552-1"})
        self.assertEqual(tps.variants_in("Add –1: TPS2553–1"), {"TPS2553-1"})
        self.assertEqual(tps.variants_in("TPS2552, TPS2553"), {"TPS2552", "TPS2553"})

    def test_letter_variant(self):
        lmr = self.rules["lmr33630"]
        self.assertEqual(lmr.variants_in("LMR33630ADDAR"), {"LMR33630A"})
        self.assertEqual(lmr.variants_in("LMR33630 SIMPLE SWITCHER"), set())
        self.assertIsNotNone(lmr.primary("LMR33630 입력 전압"))
        self.assertFalse(lmr.primary("LMR33630 입력 전압").is_specific)

    def test_gate_uses_registered_variants(self):
        tps = self.rules["tps2552"]
        target = tps.primary("TPS2552 EN")
        self.assertTrue(target.is_specific)
        self.assertFalse(tps.applies("EN = Active High for the TPS2553", target))
        self.assertTrue(tps.applies("Thermal shutdown protection", target))
        # TPS2550 은 이 문서의 변종이 아니다 (CH334 정규식이 CH213K 를 변종으로 잡던 문제가 없다)
        self.assertTrue(tps.applies("Backwards compatible with TPS2550", target))


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    def test_bom_maps_ics_to_documents(self):
        mapped = {ref: doc.id for entry, doc, _ in self.catalog.coverage() if doc for ref in entry.refs}
        self.assertEqual(mapped, {"U3": "ch334", "U1": "tps2552", "U2": "tps2552",
                                  "U5": "lmr33630", "U7": "tps7a20", "U6": "lm66100"})

    def test_refdes_is_rewritten_to_part(self):
        route = self.catalog.route("U5의 스위칭 주파수는?")
        self.assertEqual((route.kind, route.doc.id), ("doc", "lmr33630"))
        self.assertIn("LMR33630ADDAR", route.question)
        self.assertNotIn("U5", route.question)

    def test_blocked_routes(self):
        cases = {
            "U4의 정전기 보호 전압은?": "unregistered",
            "R35 값은?": "unregistered",
            "TPS2560의 전류 제한은?": "unknown",
            "동작 온도 범위는?": "none",
            "TPS2552와 LMR33630의 입력 전압 범위 차이는?": "multi",
        }
        for question, kind in cases.items():
            with self.subTest(question):
                self.assertEqual(self.catalog.route(question).kind, kind)

    def test_single_enabled_document_takes_questions_without_part(self):
        # 문서 하나만 켜면 품번 없이 물어도 그 문서, 변종은 BOM 에 쓴 품번 기준
        route = self.catalog.route("EN 핀 극성은?", {"tps2552"})
        self.assertEqual((route.kind, route.doc.id), ("doc", "tps2552"))
        self.assertIn("TPS2552DBVR", route.question)

    def test_disabled_document_is_not_searched(self):
        route = self.catalog.route("TPS2552의 EN 핀 극성은?", {"lmr33630"})
        self.assertEqual(route.kind, "off")

    def test_two_enabled_documents_need_a_part(self):
        self.assertEqual(self.catalog.route("동작 온도 범위는?", {"lmr33630", "tps2552"}).kind, "none")
        self.assertEqual(self.catalog.route("동작 온도 범위는?", set()).kind, "none")

    def test_pin_name_is_not_a_second_part(self):
        self.assertEqual(self.catalog.route("CH334U의 동작 전압(VDD33)은?").kind, "doc")


if __name__ == "__main__":
    unittest.main()
