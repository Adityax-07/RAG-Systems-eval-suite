"""Tests for the evaluator modules.

These tests run WITHOUT making real LLM API calls.
We mock the LLM client to isolate the evaluator logic.

Teaching note: This is the right way to test LLM applications.
You mock the external dependency (the LLM API) so your tests:
1. Run instantly (no API latency)
2. Are deterministic (no randomness)
3. Don't cost money
4. Can test edge cases (what if the judge returns invalid JSON?)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_framework.types import (
    QAPair, SystemOutput, EvaluationResult, EvaluationMetric
)
from eval_framework.evaluators import (
    FaithfulnessEvaluator,
    RelevanceEvaluator,
    CompletenessEvaluator,
    HallucinationRateEvaluator,
    LatencyEvaluator,
    CostEvaluator,
    ConcisenessEvaluator,
    CoherenceEvaluator,
    ToxicityEvaluator,
    ContextPrecisionEvaluator,
)
from eval_framework.utils.llm_client import LLMClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_qa_pair():
    return QAPair(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        context="France is a country in Western Europe. Its capital and largest city is Paris.",
    )


@pytest.fixture
def faithful_output():
    """A system output that is faithful to the context."""
    return SystemOutput(
        answer="Paris is the capital of France.",
        latency_ms=450.0,
        cost_usd=0.002,
        model="gpt-4",
    )


@pytest.fixture
def hallucinated_output():
    """A system output that contains hallucinated claims."""
    return SystemOutput(
        answer="Paris is the capital of France and has 50 million residents.",
        latency_ms=800.0,
        cost_usd=0.003,
        model="gpt-4",
    )


def make_mock_client(json_response: str) -> LLMClient:
    """Create a mock LLM client that returns a fixed response."""
    mock = MagicMock(spec=LLMClient)
    mock.generate = AsyncMock(return_value=json_response)
    return mock


# ─── FaithfulnessEvaluator tests ─────────────────────────────────────────────

class TestFaithfulnessEvaluator:

    @pytest.mark.asyncio
    async def test_perfect_faithfulness(self, sample_qa_pair, faithful_output):
        """High faithfulness score for a faithful answer."""
        mock_response = '{"score": 0.95, "reasoning": "All claims are supported.", "unsupported_claims": [], "contradictions": [], "supported_claims": ["Paris is capital"]}'
        evaluator = FaithfulnessEvaluator(make_mock_client(mock_response), "mock-model")

        result = await evaluator.evaluate(sample_qa_pair, faithful_output)

        assert isinstance(result, EvaluationResult)
        assert result.metric == EvaluationMetric.FAITHFULNESS
        assert result.score == pytest.approx(0.95)
        assert result.reasoning == "All claims are supported."
        assert result.judge_model == "mock-model"

    @pytest.mark.asyncio
    async def test_hallucinated_answer(self, sample_qa_pair, hallucinated_output):
        """Low faithfulness score for a hallucinated answer."""
        mock_response = '{"score": 0.3, "reasoning": "Claim about 50 million residents is not in context.", "unsupported_claims": ["50 million residents"], "contradictions": [], "supported_claims": ["Paris is capital"]}'
        evaluator = FaithfulnessEvaluator(make_mock_client(mock_response), "mock-model")

        result = await evaluator.evaluate(sample_qa_pair, hallucinated_output)

        assert result.score == pytest.approx(0.3)
        assert "50 million" in result.reasoning or result.score < 0.5

    @pytest.mark.asyncio
    async def test_score_clamped_to_01(self, sample_qa_pair, faithful_output):
        """Score outside [0,1] should be clamped, not raise an error."""
        mock_response = '{"score": 1.5, "reasoning": "Perfect"}'  # Invalid score
        evaluator = FaithfulnessEvaluator(make_mock_client(mock_response), "mock-model")

        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert 0.0 <= result.score <= 1.0  # Clamped to valid range

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self, sample_qa_pair, faithful_output):
        """Should not raise even if judge returns non-JSON response."""
        mock_response = "The score is 0.7, the answer looks mostly faithful."
        evaluator = FaithfulnessEvaluator(make_mock_client(mock_response), "mock-model")

        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_no_context_still_works(self):
        """Evaluator should work even without a context field."""
        qa_no_context = QAPair(
            question="What is 2+2?",
            answer="4",
        )
        system_out = SystemOutput(answer="4", latency_ms=100)
        mock_response = '{"score": 0.9, "reasoning": "Correct math"}'
        evaluator = FaithfulnessEvaluator(make_mock_client(mock_response), "mock")

        result = await evaluator.evaluate(qa_no_context, system_out)
        assert result.score == pytest.approx(0.9)


# ─── RelevanceEvaluator tests ─────────────────────────────────────────────────

class TestRelevanceEvaluator:

    @pytest.mark.asyncio
    async def test_relevant_answer(self, sample_qa_pair, faithful_output):
        mock_response = '{"score": 0.92, "reasoning": "Directly answers the question.", "addressed_aspects": ["capital"], "unaddressed_aspects": [], "off_topic_content": []}'
        evaluator = RelevanceEvaluator(make_mock_client(mock_response), "mock")

        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.metric == EvaluationMetric.RELEVANCE
        assert result.score == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_irrelevant_answer(self, sample_qa_pair):
        off_topic_output = SystemOutput(
            answer="France is a beautiful country known for wine and cheese.",
            latency_ms=300,
        )
        mock_response = '{"score": 0.1, "reasoning": "Does not answer the capital question.", "addressed_aspects": [], "unaddressed_aspects": ["capital city"], "off_topic_content": ["wine", "cheese"]}'
        evaluator = RelevanceEvaluator(make_mock_client(mock_response), "mock")

        result = await evaluator.evaluate(sample_qa_pair, off_topic_output)
        assert result.score < 0.3


# ─── LatencyEvaluator tests ───────────────────────────────────────────────────

class TestLatencyEvaluator:
    """Latency evaluator requires NO LLM mock — it's purely deterministic."""

    def make_evaluator(self):
        # LatencyEvaluator doesn't use judge_client, but needs one for base class
        return LatencyEvaluator(judge_client=None, model_name="none")

    @pytest.mark.asyncio
    async def test_fast_response(self, sample_qa_pair):
        fast_output = SystemOutput(answer="Paris", latency_ms=200.0)
        evaluator = self.make_evaluator()
        result = await evaluator.evaluate(sample_qa_pair, fast_output)
        assert result.score == 1.0
        assert result.metric == EvaluationMetric.LATENCY
        assert result.confidence == 1.0  # Deterministic = perfect confidence

    @pytest.mark.asyncio
    async def test_slow_response(self, sample_qa_pair):
        slow_output = SystemOutput(answer="Paris", latency_ms=12000.0)
        evaluator = self.make_evaluator()
        result = await evaluator.evaluate(sample_qa_pair, slow_output)
        assert result.score == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("latency_ms,expected_score", [
        (200, 1.0),
        (750, 0.8),
        (1500, 0.6),
        (3000, 0.4),
        (7000, 0.2),
        (15000, 0.0),
    ])
    async def test_latency_buckets(self, sample_qa_pair, latency_ms, expected_score):
        output = SystemOutput(answer="Paris", latency_ms=float(latency_ms))
        evaluator = self.make_evaluator()
        result = await evaluator.evaluate(sample_qa_pair, output)
        assert result.score == expected_score


# ─── CostEvaluator tests ──────────────────────────────────────────────────────

class TestCostEvaluator:

    def make_evaluator(self):
        return CostEvaluator(judge_client=None, model_name="none")

    @pytest.mark.asyncio
    async def test_cheap_query(self, sample_qa_pair):
        cheap_output = SystemOutput(answer="Paris", latency_ms=100, cost_usd=0.0005)
        evaluator = self.make_evaluator()
        result = await evaluator.evaluate(sample_qa_pair, cheap_output)
        assert result.score == 1.0
        assert result.raw_score == pytest.approx(0.0005)

    @pytest.mark.asyncio
    async def test_expensive_query(self, sample_qa_pair):
        expensive_output = SystemOutput(answer="Paris", latency_ms=100, cost_usd=0.15)
        evaluator = self.make_evaluator()
        result = await evaluator.evaluate(sample_qa_pair, expensive_output)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_cost_provided(self, sample_qa_pair):
        """When cost is None, treat as $0 (free)."""
        free_output = SystemOutput(answer="Paris", latency_ms=100, cost_usd=None)
        evaluator = self.make_evaluator()
        result = await evaluator.evaluate(sample_qa_pair, free_output)
        assert result.score == 1.0


# ─── HallucinationRateEvaluator tests ─────────────────────────────────────────

class TestHallucinationRateEvaluator:

    @pytest.mark.asyncio
    async def test_no_hallucinations(self, sample_qa_pair, faithful_output):
        """hallucination_rate=0 → score should be 1.0 (inverted)."""
        mock_response = '{"hallucination_rate": 0.0, "total_claims": 2, "hallucinated_claims": [], "grounded_claims": ["Paris is capital", "in France"], "reasoning": "All grounded"}'
        evaluator = HallucinationRateEvaluator(make_mock_client(mock_response), "mock")

        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.metric == EvaluationMetric.HALLUCINATION_RATE
        assert result.score == pytest.approx(1.0)  # 1 - 0.0 = 1.0

    @pytest.mark.asyncio
    async def test_all_hallucinations(self, sample_qa_pair):
        bad_output = SystemOutput(answer="Paris has 50 unicorn parks and 3 moons.", latency_ms=500)
        mock_response = '{"hallucination_rate": 1.0, "total_claims": 2, "hallucinated_claims": ["50 unicorn parks", "3 moons"], "grounded_claims": [], "reasoning": "All hallucinated"}'
        evaluator = HallucinationRateEvaluator(make_mock_client(mock_response), "mock")

        result = await evaluator.evaluate(sample_qa_pair, bad_output)
        assert result.score == pytest.approx(0.0)  # 1 - 1.0 = 0.0


# ─── ConcisenessEvaluator tests ───────────────────────────────────────────────

class TestConcisenessEvaluator:

    @pytest.mark.asyncio
    async def test_concise_answer(self, sample_qa_pair, faithful_output):
        mock_response = '{"score": 0.95, "verbose_phrases": [], "word_count": 6, "reasoning": "Direct and brief."}'
        evaluator = ConcisenessEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.metric == EvaluationMetric.CONCISENESS
        assert result.score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_verbose_answer(self, sample_qa_pair):
        verbose_output = SystemOutput(
            answer="Well, as we know, Paris, which is located in France, and is indeed the capital, is Paris. Paris is definitely the capital. To reiterate, Paris is the capital of France.",
            latency_ms=300,
        )
        mock_response = '{"score": 0.2, "verbose_phrases": ["Well, as we know", "To reiterate"], "word_count": 38, "reasoning": "Heavy repetition."}'
        evaluator = ConcisenessEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, verbose_output)
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self, sample_qa_pair, faithful_output):
        evaluator = ConcisenessEvaluator(make_mock_client("This answer is quite concise, score 0.8"), "mock")
        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.score <= 1.0


# ─── CoherenceEvaluator tests ─────────────────────────────────────────────────

class TestCoherenceEvaluator:

    @pytest.mark.asyncio
    async def test_coherent_answer(self, sample_qa_pair, faithful_output):
        mock_response = '{"score": 0.9, "internal_contradictions": [], "structural_issues": [], "reasoning": "Clear and logical."}'
        evaluator = CoherenceEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.metric == EvaluationMetric.COHERENCE
        assert result.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_incoherent_answer(self, sample_qa_pair):
        incoherent_output = SystemOutput(answer="Paris yes but no. Capital is not. France maybe.", latency_ms=300)
        mock_response = '{"score": 0.1, "internal_contradictions": ["Paris yes but no"], "structural_issues": ["fragmented sentences"], "reasoning": "Very hard to follow."}'
        evaluator = CoherenceEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, incoherent_output)
        assert result.score < 0.3


# ─── ToxicityEvaluator tests ──────────────────────────────────────────────────

class TestToxicityEvaluator:

    @pytest.mark.asyncio
    async def test_safe_answer(self, sample_qa_pair, faithful_output):
        mock_response = '{"score": 1.0, "toxicity_categories": [], "flagged_phrases": [], "reasoning": "Completely safe."}'
        evaluator = ToxicityEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.metric == EvaluationMetric.TOXICITY
        assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_toxic_answer(self, sample_qa_pair):
        toxic_output = SystemOutput(answer="I hate everyone from France, they are all terrible.", latency_ms=200)
        mock_response = '{"score": 0.1, "toxicity_categories": ["hate_speech"], "flagged_phrases": ["I hate everyone from France"], "reasoning": "Clear hate speech detected."}'
        evaluator = ToxicityEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, toxic_output)
        assert result.score < 0.3


# ─── ContextPrecisionEvaluator tests ──────────────────────────────────────────

class TestContextPrecisionEvaluator:

    @pytest.mark.asyncio
    async def test_no_context_defaults_to_one(self):
        qa_no_ctx = QAPair(question="What is 2+2?", answer="4")
        out = SystemOutput(answer="4", latency_ms=50)
        evaluator = ContextPrecisionEvaluator(make_mock_client("{}"), "mock")
        result = await evaluator.evaluate(qa_no_ctx, out)
        assert result.score == pytest.approx(1.0)
        assert result.confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_precise_context(self, sample_qa_pair, faithful_output):
        mock_response = '{"score": 0.95, "useful_sentences": ["France capital is Paris"], "noise_sentences": [], "reasoning": "All context is relevant."}'
        evaluator = ContextPrecisionEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.metric == EvaluationMetric.CONTEXT_PRECISION
        assert result.score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_noisy_context(self, sample_qa_pair, faithful_output):
        mock_response = '{"score": 0.2, "useful_sentences": ["Paris is capital"], "noise_sentences": ["France has wine", "France has cheese", "French cuisine history"], "reasoning": "Mostly noise."}'
        evaluator = ContextPrecisionEvaluator(make_mock_client(mock_response), "mock")
        result = await evaluator.evaluate(sample_qa_pair, faithful_output)
        assert result.score < 0.4
