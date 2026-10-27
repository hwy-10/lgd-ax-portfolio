import unittest

from src.chunker import PageText, create_chunks

NL = chr(10)


class ChunkerTests(unittest.TestCase):
    def test_preserves_page_and_detected_section(self) -> None:
        chunks = create_chunks(
            [PageText(page=12, text="""DSPORT ROUTING LOGIC
Routes downstream ports.""")],
            {"source": "datasheet.pdf", "part_number": "CH334U"},
            chunk_size=100,
            overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["page"], 12)
        self.assertEqual(chunks[0].metadata["section"], "DSPORT ROUTING LOGIC")
        self.assertIn("Routes downstream ports.", chunks[0].text)

    def test_chunks_are_page_bounded_and_overlap(self) -> None:
        text = " ".join(f"word{index}" for index in range(80))
        chunks = create_chunks(
            [PageText(page=3, text=text)],
            {"source": "datasheet.pdf"},
            chunk_size=120,
            overlap=20,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["page"] == 3 for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 120 for chunk in chunks))

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            create_chunks([], {}, chunk_size=100, overlap=100)


class HeadingDetectionTests(unittest.TestCase):
    """핀 표의 낱줄이 제목으로 잡히면 표가 쪼개진다."""

    def sections_of(self, text: str) -> list[str]:
        chunks = create_chunks(
            [PageText(page=1, text=text)], {}, chunk_size=1000, overlap=0
        )
        return [str(chunk.metadata["section"]) for chunk in chunks]

    def test_single_uppercase_token_is_not_a_heading(self) -> None:
        text = NL.join(["OVERVIEW SECTION TITLE", "DP3", "7", "GND", "12"])
        sections = self.sections_of(text)
        self.assertNotIn("DP3", sections)
        self.assertNotIn("GND", sections)

    def test_multiword_uppercase_line_is_still_a_heading(self) -> None:
        text = NL.join(["DSPORT ROUTING LOGIC", "Routes downstream ports."])
        self.assertIn("DSPORT ROUTING LOGIC", self.sections_of(text))

    def test_numbered_heading_survives_single_word_rule(self) -> None:
        text = NL.join(["1.1 Overview", "body text here"])
        self.assertIn("1.1 Overview", self.sections_of(text))


class MergeTests(unittest.TestCase):
    """표의 인접 행을 한 청크로 묶는다."""

    def pin_table(self) -> str:
        """제목이 여러 번 잡혀 작은 구간이 많이 생기는 표를 흉내낸다."""
        blocks: list[str] = []
        for index in range(1, 20):
            blocks.append(f"TABLE BLOCK {index}")
            blocks.append(f"DP{index}")
            blocks.append(str(index))
        return NL.join(blocks)

    def test_merge_reduces_chunk_count(self) -> None:
        pages = [PageText(page=1, text=self.pin_table())]
        without = create_chunks(pages, {}, chunk_size=1000, overlap=0, min_chunk_size=0)
        merged = create_chunks(
            pages, {}, chunk_size=1000, overlap=0, min_chunk_size=200
        )
        self.assertLess(len(merged), len(without))
        self.assertGreaterEqual(max(len(c.text) for c in merged), 200)

    def test_merge_never_exceeds_chunk_size(self) -> None:
        pages = [PageText(page=1, text=self.pin_table())]
        chunks = create_chunks(pages, {}, chunk_size=120, overlap=0, min_chunk_size=500)
        self.assertTrue(all(len(chunk.text) <= 120 for chunk in chunks))

    def test_merge_does_not_cross_pages(self) -> None:
        pages = [
            PageText(page=1, text=NL.join(["AA", "BB"])),
            PageText(page=2, text=NL.join(["CC", "DD"])),
        ]
        chunks = create_chunks(pages, {}, chunk_size=1000, overlap=0, min_chunk_size=500)
        self.assertEqual(sorted({chunk.metadata["page"] for chunk in chunks}), [1, 2])

    def test_min_chunk_size_zero_is_a_no_op(self) -> None:
        pages = [PageText(page=1, text=self.pin_table())]
        explicit = create_chunks(pages, {}, chunk_size=1000, overlap=0, min_chunk_size=0)
        default = create_chunks(pages, {}, chunk_size=1000, overlap=0)
        self.assertEqual(len(explicit), len(default))


if __name__ == "__main__":
    unittest.main()
