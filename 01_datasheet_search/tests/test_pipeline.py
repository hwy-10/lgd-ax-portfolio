"""질문부터 답변까지 — 용어 확장, 충분성 판정, 답변 대조, 모델 호출 차단."""

import unittest

from src.answer_check import check_answer, cited_pages, numbers_with_units
from src.evidence import judge, term_coverage
from src.glossary import Concept, expand
from src.hybrid import Document, HybridIndex
from src.llm import Completion, parse_spec
from src.pipeline import DatasheetSearch, content_query
from src.prompt import NOT_FOUND_ANSWER


class GlossaryTests(unittest.TestCase):
    def test_longest_korean_term_wins(self):
        expansion = expand("CH334U 동작 전압", {"CH334U"})
        labels = [c.label for c in expansion.concepts]
        self.assertIn("동작 전압", labels)
        self.assertNotIn("전압", labels, "긴 표현을 맞추면 짧은 표현은 다시 세지 않는다")
        self.assertIn("operating", expansion.lexical_query)

    def test_part_number_is_not_a_judging_term(self):
        expansion = expand("CH334U HDMI", {"CH334U"})
        self.assertEqual([c.label for c in expansion.concepts], ["HDMI"])

    def test_stopwords_are_dropped(self):
        expansion = expand("What is the DSPORT routing")
        labels = [c.label.lower() for c in expansion.concepts]
        self.assertNotIn("what", labels)
        self.assertIn("dsport", labels)


class EvidenceTests(unittest.TestCase):
    def test_missing_term_lowers_coverage(self):
        concepts = [Concept("HDMI", ("hdmi",)), Concept("전압", ("voltage",))]
        coverage, found, missing = term_coverage(concepts, ["operating voltage 3.3V"])
        self.assertEqual(coverage, 0.5)
        self.assertEqual(missing, ["HDMI"])

    def test_no_terms_is_zero_coverage(self):
        # 판정할 용어가 없으면 '판단 불가' 로 보고 막는다 (예전에는 1.0 으로 통과시켰다)
        coverage, _, _ = term_coverage([], ["anything"])
        self.assertEqual(coverage, 0.0)

    def test_judge_blocks_on_either_criterion(self):
        concepts = [Concept("HDMI", ("hdmi",))]
        self.assertFalse(judge(concepts, ["usb hub"], 30.0, 2.5, 0.67).passed)
        ok = [Concept("voltage", ("voltage",))]
        self.assertFalse(judge(ok, ["voltage"], 1.0, 2.5, 0.67).passed)
        self.assertTrue(judge(ok, ["voltage"], 9.0, 2.5, 0.67).passed)

    def test_judge_blocks_without_evidence(self):
        self.assertFalse(judge([], [], 0.0, 0.0, 0.0).passed)

    def test_no_judging_term_blocks(self):
        # "CH334U의 색상은?" 처럼 용어표에 없는 질문은 판정할 수 없으므로 답하지 않는다 (GPT 검토 1.1)
        self.assertFalse(judge([], ["CH334U pin description"], 30.0, 2.5, 0.67).passed)

    def test_compound_term_needs_every_word(self):
        # "입력 전압" 은 input 과 voltage 가 모두 있어야 한다 (GPT 검토 1.2)
        concepts = expand("CH334U의 입력 전압과 출력 전압은?", {"CH334U"}).concepts
        coverage, found, missing = term_coverage(concepts, ["supply voltage 3.3 V"])
        self.assertEqual(coverage, 0.0)
        coverage, found, _ = term_coverage(concepts, ["LDO input supply voltage 4.5 V"])
        self.assertEqual(found, ["입력 전압"])


class AnswerCheckTests(unittest.TestCase):
    def test_citations_and_numbers(self):
        answer = "동작 전압은 3.3V 입니다 [p.21]. 입력은 4.5~5.25V 입니다 [p.21]."
        self.assertEqual(cited_pages(answer), [21])
        self.assertEqual(numbers_with_units(answer), ["3.3", "4.5", "5.25"])

    def test_number_not_in_context_is_flagged(self):
        check = check_answer("동작 전압은 1.8V 입니다 [p.21].", [21], ["VDD33 3.2 3.3 3.5 V"])
        self.assertFalse(check.passed)
        self.assertEqual(check.unsupported_numbers, ("1.8",))

    def test_sign_is_kept(self):
        # 근거는 -40 인데 답이 +40 이면 실패 (GPT 검토 1.6)
        self.assertFalse(check_answer("+40°C 입니다 [p.1]", [1], ["range -40°C"]).passed)
        self.assertFalse(check_answer("40°C 입니다 [p.1]", [1], ["range -40~85℃"]).passed)
        self.assertTrue(check_answer("-40~85°C 입니다 [p.1]", [1], ["range: -40~85℃"]).passed)
        self.assertTrue(check_answer("–40°C 입니다 [p.1]", [1], ["TJ –40°C"]).passed)

    def test_range_dash_is_not_a_sign(self):
        self.assertTrue(check_answer("3.2-3.5V 입니다 [p.21]", [21], ["3.2 3.3 3.5 V"]).passed)

    def test_percent_is_checked(self):
        self.assertFalse(check_answer("정확도는 ±9% 입니다 [p.1]", [1], ["±6% accuracy"]).passed)

    def test_citation_outside_evidence_is_flagged(self):
        check = check_answer("3.3V 입니다 [p.30].", [21], ["3.3"])
        self.assertEqual(check.invalid_citations, (30,))
        self.assertFalse(check.passed)

    def test_answer_without_citation_fails(self):
        self.assertFalse(check_answer("3.3V 입니다.", [21], ["3.3"]).passed)

    def test_refusal_passes(self):
        self.assertTrue(check_answer(NOT_FOUND_ANSWER, [], []).refused)
        self.assertTrue(check_answer(NOT_FOUND_ANSWER + " [p.19]", [19], []).refused)

    def test_answer_with_appended_refusal_is_flagged(self):
        # 답을 하고 거절 문장을 덧붙이면 거절이 아니라 형식 위반이다
        check = check_answer(f"입력 전압은 4.5~5.25V [p.21]. {NOT_FOUND_ANSWER}", [21], ["4.5 5.0 5.25"])
        self.assertFalse(check.refused)
        self.assertTrue(check.mixed_refusal)
        self.assertFalse(check.passed)


class LlmSpecTests(unittest.TestCase):
    def test_parse_spec_with_effort(self):
        spec = parse_spec("anthropic:claude-fable-5-1@low")
        self.assertEqual((spec.provider, spec.model, spec.effort), ("anthropic", "claude-fable-5-1", "low"))

    def test_parse_ollama_model_with_tag(self):
        spec = parse_spec("ollama:nemotron-3-nano:4b@nothink")
        self.assertEqual((spec.provider, spec.model, spec.effort), ("ollama", "nemotron-3-nano:4b", "nothink"))

    def test_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            parse_spec("acme:model")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = HybridIndex(
            [
                Document("CH334U package QSOP28 pin description", {"page": 4, "section": "1.3", "source": "ds.pdf"}),
                Document("CH334R package QSOP16 pin description", {"page": 4, "section": "1.3", "source": "ds.pdf"}),
                Document("VDD33 operating supply voltage 3.2 3.3 3.5 V", {"page": 21, "section": "4.2", "source": "ds.pdf"}),
                Document("Industrial grade temperature range -40~85", {"page": 1, "section": "Features", "source": "ds.pdf"}),
            ]
        )
        self.search = DatasheetSearch(self.index, min_score=0.5, min_coverage=0.67, top_k=3)

    def test_content_query_removes_longest_first(self):
        # 계열명과 품번이 함께 잡혀도 품번 꼬리가 남지 않아야 한다 (실행마다 결과가 바뀌던 버그)
        for _ in range(20):
            self.assertEqual(content_query("TPS7A2033의 정지 전류는? quiescent", {"TPS7A20", "TPS7A2033"}).split(),
                             ["정지", "전류는?", "quiescent"])

    def test_content_query_drops_part_number(self):
        self.assertEqual(content_query("CH334U voltage", {"CH334U"}).split(), ["voltage"])
        self.assertEqual(content_query("CH334U 알려줘", {"CH334U"}), "CH334U 알려줘")

    def test_common_spec_reaches_variant_question(self):
        trace = self.search.retrieve("CH334U 동작 전압")
        pages = [e.page for e in trace.evidence]
        self.assertIn(21, pages, "품번 미기재 전기적 특성 표가 근거에 들어와야 한다")
        self.assertTrue(any(e.names_target for e in trace.evidence), "품번 명시 근거 하나는 남아야 한다")
        self.assertNotIn(1, [e.index for e in trace.evidence], "CH334R 전용 청크는 빠져야 한다")

    def test_unknown_variant_never_calls_model(self):
        def fail(*_):
            raise AssertionError("모델을 부르면 안 된다")

        trace = self.search.answer("CH999Z 동작 전압", "openai:x", generate=fail)
        self.assertTrue(trace.verification_failed)
        self.assertEqual(trace.answer, NOT_FOUND_ANSWER)
        self.assertFalse(trace.llm_called)

    def test_comparison_question_never_calls_model(self):
        def fail(*_):
            raise AssertionError("모델을 부르면 안 된다")

        trace = self.search.answer("CH334U와 CH334R의 패키지 차이는?", "openai:x", generate=fail)
        self.assertTrue(trace.blocked)
        self.assertIn("비교 질문", trace.sufficiency.reason)

    def test_unknown_korean_term_never_calls_model(self):
        def fail(*_):
            raise AssertionError("모델을 부르면 안 된다")

        trace = self.search.answer("CH334U의 색상은?", "openai:x", generate=fail)
        self.assertTrue(trace.blocked)

    def test_absent_term_never_calls_model(self):
        def fail(*_):
            raise AssertionError("모델을 부르면 안 된다")

        trace = self.search.answer("CH334U HDMI 지원", "openai:x", generate=fail)
        self.assertTrue(trace.blocked)
        self.assertIn("HDMI", trace.sufficiency.missing)

    def test_answer_is_checked_against_evidence(self):
        def fake(spec, instructions, prompt):
            self.assertIn("Page: 21", prompt)
            return Completion("동작 전압은 3.3V 입니다 [p.21].", "fake", 10, 5, 0.1)

        trace = self.search.answer("CH334U 동작 전압", "openai:x", generate=fake)
        self.assertTrue(trace.llm_called)
        self.assertTrue(trace.check.passed)


if __name__ == "__main__":
    unittest.main()
