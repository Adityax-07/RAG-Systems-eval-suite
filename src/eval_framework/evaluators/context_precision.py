"""Context Precision evaluator - How much retrieved context was actually useful?

Context Precision is a RAG-specific metric that measures signal-to-noise ratio
in the retrieved documents.

High precision means:
- Most of the provided context was relevant to the question
- The system did NOT retrieve lots of unrelated chunks

Low precision means:
- The context was mostly irrelevant noise
- The retrieval step fetched documents that did not help answer the question

This is the complement of recall-oriented metrics (like Completeness).
A system could answer fully (high completeness) but waste tokens on irrelevant
context (low precision).

Inspired by RAGAS's context_precision metric.
"""

import re
import json

from ..types import QAPair, SystemOutput, EvaluationMetric
from ..utils.llm_client import LLMClient
from .base import BaseEvaluator


class ContextPrecisionEvaluator(BaseEvaluator):
    """Evaluates how relevant the retrieved context is to the question asked.

    Only meaningful when qa_pair.context is provided.  If no context is given,
    the evaluator returns 1.0 (no retrieval step to penalise).
    """

    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.CONTEXT_PRECISION

    @property
    def system_prompt(self) -> str:
        return """You are an expert evaluator assessing retrieval quality in RAG systems.

Your task: given a QUESTION and a CONTEXT block (retrieved documents), determine
what fraction of the context was actually USEFUL for answering the question.

Useful context = sentences or passages that directly contribute information
needed to answer the question.

Noise = passages unrelated to the question that were retrieved unnecessarily.

Score 1.0: All retrieved context is directly relevant to the question
Score 0.8: Mostly relevant with 1-2 off-topic sentences
Score 0.5: About half the context is useful, half is noise
Score 0.2: Most context is irrelevant; very little signal
Score 0.0: Completely irrelevant context — nothing helps answer the question

Respond with a JSON object:
{
  "score": <float 0-1>,
  "useful_sentences": [<list of context sentences that were relevant>],
  "noise_sentences": [<list of context sentences that were irrelevant>],
  "reasoning": "<brief explanation>"
}"""

    def format_prompt(self, qa_pair: QAPair, system_output: SystemOutput) -> str:
        if not qa_pair.context:
            return "No context provided — context precision cannot be measured."

        return f"""QUESTION:
{qa_pair.question}

RETRIEVED CONTEXT:
{qa_pair.context}

Assess what fraction of the retrieved context above is actually useful for
answering the question. Ignore the system's answer — only judge the context."""

    async def evaluate(self, qa_pair, system_output):
        from datetime import datetime
        from ..types import EvaluationResult

        # If there is no context, there is nothing to evaluate.
        if not qa_pair.context:
            return EvaluationResult(
                metric=self.metric,
                score=1.0,
                raw_score=1.0,
                reasoning="No context provided — context precision defaults to 1.0.",
                judge_model=self.model_name,
                confidence=1.0,
                timestamp=datetime.utcnow(),
            )
        return await super().evaluate(qa_pair, system_output)

    async def parse_judge_response(self, response: str) -> tuple[float, str]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            return score, reasoning
        except json.JSONDecodeError:
            score_match = re.search(r'score["\s:]*(\d+\.?\d*)', response.lower())
            if score_match:
                raw = float(score_match.group(1))
                return max(0.0, min(1.0, raw / 100 if raw > 1 else raw)), response[:200]
            return 0.5, response[:200]
