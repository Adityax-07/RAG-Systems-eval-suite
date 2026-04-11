"""Completeness evaluator - Does the answer cover all aspects?"""

import re
import json

from ..types import (
    QAPair,
    SystemOutput,
    EvaluationMetric,
)
from .base import BaseEvaluator


class CompletenessEvaluator(BaseEvaluator):
    """Evaluates whether the answer addresses all aspects of the question.
    
    Completeness measures if the answer:
    - Covers all parts of multi-part questions
    - Provides sufficient detail
    - Doesn't omit important information
    - Is thorough without being excessive
    
    Example:
        Question: "What are the benefits and drawbacks of Python?"
        Incomplete: "Python is easy to learn" (only mentions benefits)
        Complete: "Benefits: readable, large ecosystem. Drawbacks: slower than C++"
    """
    
    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.COMPLETENESS
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert evaluator assessing answer completeness.

Completeness means the answer addresses all aspects of the question:
- All sub-questions are answered
- Sufficient detail for each aspect  
- Key information is not omitted
- Balanced coverage (not over-focused on one aspect)

You will be given:
- Question asked
- Reference answer (example of complete answer)
- System's answer

Your task: Rate how complete the system answer is.

Score 1.0: All question aspects thoroughly addressed
Score 0.8: All aspects covered but could use more detail
Score 0.5: Some aspects missed or under-explored
Score 0.2: Many important aspects missing
Score 0.0: Severely incomplete, major gaps

For multi-part questions, check:
- Part 1 addressed? 
- Part 2 addressed?
- Part 3 addressed?
- Sufficient depth for each?

Respond with JSON:
{
  "score": <float 0-1>,
  "question_aspects": [<list of aspects in question>],
  "covered_aspects": [<aspects fully addressed>],
  "partially_covered": [<aspects mentioned but lacking detail>],
  "missing_aspects": [<aspects not addressed>],
  "reasoning": "<explanation>"
}"""
    
    def format_prompt(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> str:
        return f"""QUESTION:
{qa_pair.question}

REFERENCE ANSWER (example of complete answer):
{qa_pair.answer}

SYSTEM ANSWER (to evaluate):
{system_output.answer}

How complete is the system answer? Does it address all aspects?"""
    
    async def parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse JSON response from judge."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            return max(0, min(1, score)), reasoning
        except json.JSONDecodeError:
            score_match = re.search(r'score["\s:]*(\d+\.?\d*)', response.lower())
            if score_match:
                score = float(score_match.group(1))
                score = score / 100 if score > 1 else score
                return max(0, min(1, score)), response[:200]
            return 0.5, response[:200]
