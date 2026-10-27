import unittest

from src.prompt import NOT_FOUND_ANSWER, build_context
from src.retriever import SearchResult


class PromptTests(unittest.TestCase):
    def test_context_contains_source_fields(self) -> None:
        context = build_context(
            [
                SearchResult(
                    text="Routing description",
                    source="CH334_CH335_V2.5.pdf",
                    page=8,
                    section="DSPORT Routing Logic",
                    similarity=0.8,
                    metadata={},
                )
            ]
        )

        self.assertIn("Source: CH334_CH335_V2.5.pdf", context)
        self.assertIn("Page: 8", context)
        self.assertIn("Section: DSPORT Routing Logic", context)

    def test_not_found_answer_is_fixed_sentence(self) -> None:
        self.assertTrue(NOT_FOUND_ANSWER.endswith("없습니다."))


if __name__ == "__main__":
    unittest.main()
