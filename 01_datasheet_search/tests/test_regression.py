"""검토에서 찾은 결함의 회귀 테스트. 실제 데이터시트로 돈다.

기준값을 정한 평가셋(evaluation_questions.json)과 섞지 않는다. 같은 셋에 넣고 다시
100% 를 내는 것은 독립 검증이 아니기 때문이다.
"""

import json
import unittest
from pathlib import Path

from config import PDF_PATH

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(PDF_PATH.exists(), "데이터시트 PDF 없음")
class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.pipeline import DatasheetSearch

        cls.search = DatasheetSearch()
        cls.items = json.loads((ROOT / "tests" / "regression_questions.json").read_text(encoding="utf-8"))

    def test_expected_outcome(self):
        for item in self.items:
            with self.subTest(item["id"]):
                trace = self.search.retrieve(item["question"])
                expected_blocked = item["expect"] == "blocked"
                self.assertEqual(trace.blocked, expected_blocked, f"{item['question']} ({trace.sufficiency.reason})")


if __name__ == "__main__":
    unittest.main()
