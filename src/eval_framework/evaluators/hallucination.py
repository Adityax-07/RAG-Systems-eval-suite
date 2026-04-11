"""Hallucination Rate evaluator - What % of claims are unsupported?"""

import re
import json

from ..types import (
    QAPair,
    SystemOutput,
    EvaluationMetric,
)
from .base import BaseEvaluator


class HallucinationRateEvaluator(BaseEvaluator):
    """Evaluates the hallucination rate - % of unsupported claims.
    
    Similar to Faithfulness but focuses on quantifying hallucination:
    - Identifies individual claims in the answer
    - Checks each claim against context
    - Calculates % of hallucinated vs supported claims
    
    Example:
        Context: "Paris has 2.2 million people"
        Answer: "Paris has 2.2 million people and is surrounded by 5 rivers"
        Hallucination rate: 50% (1/2 claims is unsupported - the rivers)
    """
    
    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.HALLUCINATION_RATE
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert at detecting hallucinations in LLM outputs.

A hallucination is a claim in the answer that is:
- Not found in the provided context
- Not common knowledge
- Presented as fact when unsupported

Your task:
1. Break the answer into individual factual claims
2. Check each claim against the context
3. Identify which claims are hallucinated
4. Calculate hallucination rate as: (hallucinated claims) / (total claims)

Scoring:
- 1.0 = 0% hallucination (all claims grounded)
- 0.8 = 20% hallucination rate
- 0.5 = 50% hallucination rate  
- 0.2 = 80% hallucination rate
- 0.0 = 100% hallucination (all or nearly all unsupported)

Be strict: if something isn't clearly in the context, mark as hallucinated.

Respond with JSON:
{
  "hallucination_rate": <float 0-1, where 1.0 = 0% hallucination>,
  "total_claims": <number>,
  "hallucinated_claims": [<list of unsupported claims>],
  "grounded_claims": [<list of supported claims>],
  "reasoning": "<explanation>"
}"""
    
    def format_prompt(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> str:
        context_section = ""
        if qa_pair.context:
            context_section = f"""CONTEXT (only source of truth):
{qa_pair.context}

"""
        
        return f"""{context_section}QUESTION:
{qa_pair.question}

SYSTEM ANSWER (check each claim):
{system_output.answer}

How many unsupported claims (hallucinations) are in this answer?"""
    
    async def parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse JSON response from judge.
        
        Note: For hallucination rate, the score is 1.0 - hallucination_rate
        (higher is better)
        """
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # The response gives hallucination_rate (0-1 where 1=all hallucinated)
            # We invert it for scoring (1.0 = no hallucinations, 0.0 = all hallucinated)
            hallucination_rate = float(data.get("hallucination_rate", 0.5))
            score = 1.0 - hallucination_rate  # Invert: lower hallucination = higher score
            
            reasoning = data.get("reasoning", "No reasoning provided")
            return max(0, min(1, score)), reasoning
        except json.JSONDecodeError:
            # Try to extract hallucination rate from text
            hal_match = re.search(
                r'hallucination[^0-9]*(\d+\.?\d*)\s*%?',
                response.lower()
            )
            if hal_match:
                hal_rate = float(hal_match.group(1))
                hal_rate = hal_rate / 100 if hal_rate > 1 else hal_rate
                score = 1.0 - hal_rate
                return max(0, min(1, score)), response[:200]
            return 0.5, response[:200]
