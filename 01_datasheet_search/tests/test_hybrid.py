"""RRF 융합 · 변종 게이트 · 검증 게이트."""

import unittest

from src.hybrid import Document, HybridIndex, rrf_fuse


class RRFTests(unittest.TestCase):
    def test_fuses_by_rank_not_score(self):
        # 두 순위에서 모두 상위인 문서가 1위가 되어야 한다
        fused = rrf_fuse([[5, 1, 2], [1, 5, 9]])
        self.assertEqual(fused[0][0], 1)

    def test_document_in_one_ranking_still_appears(self):
        fused = [index for index, _ in rrf_fuse([[1], [2]])]
        self.assertCountEqual(fused, [1, 2])

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            rrf_fuse([[1]], k=0)


class HybridIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = HybridIndex(
            [
                Document("CH334U는 QSOP-28 패키지이며 동작 전압은 3.3V이다", {"page": 2}),
                Document("CH334R는 SSOP-28 패키지이며 동작 전압은 3.3V이다", {"page": 3}),
                Document("CH334 계열은 USB 2.0 허브 컨트롤러이다", {"page": 1}),
                Document("각 다운스트림 포트는 개별 과전류 보호를 지원한다", {"page": 9}),
            ]
        )

    def test_gate_excludes_other_variant(self):
        from src.variants import parse_part

        allowed = self.index.gate(parse_part("CH334U"))
        self.assertIn(0, allowed)
        self.assertNotIn(1, allowed, "CH334R 전용 청크는 제외되어야 한다")
        self.assertIn(2, allowed, "계열 공통 청크는 포함되어야 한다")
        self.assertIn(3, allowed, "품번 없는 청크는 포함되어야 한다")

    def test_search_never_returns_sibling_variant(self):
        """이 프로젝트의 핵심 요구사항."""
        for index in self.index.search("CH334U 패키지가 뭐야?", top_k=4):
            self.assertNotIn(
                "CH334R",
                self.index.documents[index].text,
                "CH334U 질의 결과에 CH334R 문서가 섞이면 안 된다",
            )

    def test_target_chunk_ranks_first(self):
        results = self.index.search("CH334U 동작 전압", top_k=3)
        self.assertEqual(results[0], 0)

    def test_evidence_gate_returns_empty_when_part_absent(self):
        # 문서에 없는 품번을 물으면 답을 만들지 않는다
        results = self.index.search("CH999Z 동작 전압", top_k=3)
        self.assertEqual(results, [])

    def test_query_without_part_number_still_searches(self):
        results = self.index.search("과전류 보호", top_k=2)
        self.assertIn(3, results)

    def test_dense_ranking_is_merged(self):
        # dense 가 3번을 1위로 올리면 융합 결과에 반영되어야 한다
        results = self.index.search(
            "동작 전압", dense_ranking=[3, 2, 0], top_k=4
        )
        self.assertIn(3, results)


if __name__ == "__main__":
    unittest.main()
