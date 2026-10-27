"""문자 n-gram 을 섞은 순수 파이썬 BM25.

임베딩 검색은 CH334U 와 CH334R 을 거의 같은 벡터로 본다.
토크나이저가 두 문자열을 [CH][334][U] / [CH][334][R] 로 쪼개, 3개 토큰 중
1개만 다른 상태가 되기 때문이다.

품번처럼 한 글자가 부품 자체를 바꾸는 토큰은 문자 n-gram 으로 따로 색인해
어휘 단위로 정확히 맞춘다. 외부 의존성을 쓰지 않는다.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# 라틴·숫자와 한글을 다른 토큰으로 끊는다. 하나로 묶으면 "CH334U는" 이
# ch334u는 한 토큰이 되어 품번 n-gram 이 만들어지지 않는다.
WORD_PATTERN = re.compile(r"[0-9a-z]+|[가-힣]+")
# 영문과 숫자가 섞인 토큰 = 품번일 가능성이 높다
ALNUM_MIXED = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]+$")

K1 = 1.5
B = 0.75
NGRAM_SIZES = (3, 4)


def char_ngrams(token: str, sizes: tuple[int, ...] = NGRAM_SIZES) -> list[str]:
    padded = f"^{token}$"
    grams: list[str] = []
    for size in sizes:
        if len(padded) < size:
            continue
        grams.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
    return grams


def word_tokenize(text: str) -> list[str]:
    """단어 토큰만. 임베딩 토크나이저가 CH334U 를 쪼개는 상황의 비교군이다."""
    return WORD_PATTERN.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """단어 토큰 + 품번형 토큰의 문자 n-gram.

    일반 단어는 그대로 색인해 자연어 질문을 받고,
    품번형 토큰만 문자 n-gram 을 덧붙여 CH334U 와 CH334R 을 갈라낸다.
    """
    tokens: list[str] = []
    for word in WORD_PATTERN.findall(text.lower()):
        tokens.append(word)
        if ALNUM_MIXED.match(word):
            tokens.extend(f"#{gram}" for gram in char_ngrams(word))
    return tokens


class BM25Index:
    """메모리 상주 BM25. 데이터시트 한 권 규모(수십~수백 청크)를 대상으로 한다."""

    def __init__(
        self,
        documents: list[str],
        k1: float = K1,
        b: float = B,
        tokenizer=None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or tokenize
        self.doc_count = len(documents)
        self.term_freqs: list[Counter[str]] = [
            Counter(self.tokenizer(d)) for d in documents
        ]
        self.doc_lengths: list[int] = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        document_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            document_freq.update(tf.keys())
        self.idf = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in document_freq.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        tf = self.term_freqs[doc_index]
        length = self.doc_lengths[doc_index] or 1
        norm = self.k1 * (1.0 - self.b + self.b * length / (self.avg_doc_length or 1.0))
        total = 0.0
        for term in self.tokenizer(query):
            frequency = tf.get(term)
            if not frequency:
                continue
            total += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / (
                frequency + norm
            )
        return total

    def rank(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """(문서 인덱스, 점수) 를 점수 내림차순으로. 0점 문서는 제외한다."""
        scored = [
            (index, self.score(query, index))
            for index in range(self.doc_count)
        ]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k] if top_k else scored
