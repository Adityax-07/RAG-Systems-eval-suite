"""Coherence evaluator - Is the answer logically structured and fluent?

Coherence measures whether the answer:
- Flows logically from one point to the next
- Uses clear sentence structure
- Has no contradictions within the answer itself
- Is grammatically correct and easy to read

This is different from faithfulness (which compares the answer to an external
source).  Coherence only looks at the internal quality of the answer as prose.
"""

import re
import json

from ..types import QAPair, SystemOutput, EvaluationMetric
from ..utils.llm_client import LLMClient
from .base import BaseEvaluator


class CoherenceEvaluator(BaseEvaluator):
    """Evaluates the logical structure and fluency of the system output."""

    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.COHERENCE

    @property
    def system_prompt(self) -> str:
        return """You are an expert evaluator assessing the coherence of LLM responses.

Coherence means:
1. Sentences connect logically — there is a clear flow of ideas
2. No internal contradictions within the answer itself
3. Grammatically correct and easy to read
4. Appropriate use of transitions and structure
5. A clear opening and a conclusion where expected

Score 1.0: Exceptionally clear, structured, and fluent
Score 0.8: Clear and mostly well-structured with minor issues
Score 0.5: Understandable but with noticeable logical gaps or awkward phrasing
Score 0.2: Confusing structure; hard to follow
Score 0.0: Incoherent — contradictions, fragmented sentences, or unreadable

Respond with a JSON object:
{
  "score": <float 0-1>,
  "internal_contradictions": [<list of contradicting statements, if any>],
  "structural_issues": [<list of logical flow problems, if any>],
  "reasoning": "<brief explanation>"
}"""

    def format_prompt(self, qa_pair: QAPair, system_output: SystemOutput) -> str:
        return f"""QUESTION:
{qa_pair.question}

SYSTEM OUTPUT:
{system_output.answer}

Evaluate the coherence (logical structure and fluency) of the system output above.
Do NOT penalise for factual accuracy — only assess how well the answer is written."""

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
