"""Conciseness evaluator - Is the answer appropriately brief?

Conciseness measures whether the answer communicates the necessary information
without unnecessary padding, repetition, or verbosity.

A concise answer:
- Directly addresses the question
- Does not repeat itself
- Does not add irrelevant filler
- Does not omit critical information to be artificially short (that would be
  penalised by CompletenessEvaluator, not here)
"""

import re
import json

from ..types import QAPair, SystemOutput, EvaluationMetric
from ..utils.llm_client import LLMClient
from .base import BaseEvaluator


class ConcisenessEvaluator(BaseEvaluator):
    """Evaluates whether the system output is appropriately concise."""

    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.CONCISENESS

    @property
    def system_prompt(self) -> str:
        return """You are an expert evaluator assessing the conciseness of LLM responses.

Conciseness means:
1. The answer is as short as it can be while still being complete
2. No unnecessary padding, filler phrases, or repetition
3. No restating the question before answering it
4. No over-explaining obvious points

Score 1.0: Perfectly concise — every sentence adds value
Score 0.8: Mostly concise with minor redundancies
Score 0.5: Moderately verbose — noticeable padding but core is clear
Score 0.2: Very verbose — heavy repetition or rambling
Score 0.0: Excessively padded — the answer buries the key information

Respond with a JSON object:
{
  "score": <float 0-1>,
  "verbose_phrases": [<list of specific unnecessary phrases>],
  "word_count": <integer>,
  "reasoning": "<brief explanation>"
}"""

    def format_prompt(self, qa_pair: QAPair, system_output: SystemOutput) -> str:
        return f"""QUESTION:
{qa_pair.question}

REFERENCE ANSWER:
{qa_pair.answer}

SYSTEM OUTPUT:
{system_output.answer}

Evaluate the conciseness of the system output above."""

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
