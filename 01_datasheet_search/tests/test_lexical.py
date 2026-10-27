"""문자 n-gram BM25 — 한 글자 차이를 가르는지 확인한다."""

import unittest

from src.lexical import BM25Index, char_ngrams, tokenize


class TokenizeTests(unittest.TestCase):
    def test_plain_word_has_no_ngrams(self):
        tokens = tokenize("voltage")
        self.assertEqual(tokens, ["voltage"])

    def test_part_number_gets_char_ngrams(self):
        tokens = tokenize("CH334U")
        self.assertIn("ch334u", tokens)
        self.assertTrue(any(token.startswith("#") for token in tokens))

    def test_korean_particle_does_not_hide_part_number(self):
        tokens = tokenize("CH334U는")
        self.assertIn("ch334u", tokens)
        self.assertIn("는", tokens)
        self.assertTrue(any(token.startswith("#") for token in tokens))

    def test_ngrams_differ_by_last_character(self):
        u = set(char_ngrams("ch334u"))
        r = set(char_ngrams("ch334r"))
        self.assertNotEqual(u, r)
        self.assertTrue(u - r, "마지막 글자가 다르면 고유 n-gram 이 생겨야 한다")


class WordTokenizeTests(unittest.TestCase):
    def test_word_tokenizer_has_no_ngrams(self):
        from src.lexical import word_tokenize

        self.assertEqual(word_tokenize("CH334U voltage"), ["ch334u", "voltage"])

    def test_tokenizer_is_injectable(self):
        from src.lexical import word_tokenize

        index = BM25Index(["CH334U voltage"], tokenizer=word_tokenize)
        self.assertNotIn("#^ch", index.idf)


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "CH334U operating voltage range is 3.3V typical",   # 0
            "CH334R operating voltage range is 3.3V typical",   # 1
            "CH334 series overview and general description",    # 2
            "각 다운스트림 포트는 개별 과전류 보호를 지원한다",        # 3
        ]
        self.index = BM25Index(self.docs)

    def test_exact_variant_outranks_sibling(self):
        ranked = self.index.rank("CH334U operating voltage")
        self.assertEqual(ranked[0][0], 0, "CH334U 문서가 1위여야 한다")
        top_indices = [index for index, _ in ranked[:2]]
        self.assertIn(0, top_indices)
        self.assertGreater(
            self.index.score("CH334U operating voltage", 0),
            self.index.score("CH334U operating voltage", 1),
            "같은 본문이라도 품번이 다르면 점수가 달라야 한다",
        )

    def test_korean_query_matches_korean_document(self):
        ranked = self.index.rank("과전류 보호")
        self.assertEqual(ranked[0][0], 3)

    def test_zero_score_documents_are_dropped(self):
        ranked = self.index.rank("zzzznonexistent")
        self.assertEqual(ranked, [])

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            BM25Index(self.docs, k1=0)
        with self.assertRaises(ValueError):
            BM25Index(self.docs, b=1.5)


if __name__ == "__main__":
    unittest.main()
