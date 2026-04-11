"""Relevance evaluator - Does the answer actually answer the question?"""

import re
import json

from ..types import (
    QAPair,
    SystemOutput,
    EvaluationMetric,
)
from .base import BaseEvaluator


class RelevanceEvaluator(BaseEvaluator):
    """Evaluates whether the system answer is relevant to the question.
    
    Relevance measures if the answer:
    - Addresses the actual question asked
    - Doesn't go off-topic
    - Focuses on what was asked, not tangential info
    
    Example:
        Question: "What is the capital of France?"
        Relevant: "Paris is the capital of France"
        Irrelevant: "France is a beautiful country in Europe"
        Off-topic: "Let me tell you about Napoleon..."
    """
    
    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.RELEVANCE
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert evaluator assessing answer relevance.

Relevance means the answer directly addresses the question asked, without:
- Going off-topic
- Answering a different question
- Providing irrelevant tangential information

You will be given:
- Question asked
- Reference answer (what a good answer looks like)
- System's answer

Your task: Rate how relevant the system answer is to the question.

Score 1.0: Directly and completely addresses the question
Score 0.8: Addresses question but includes some off-topic info
Score 0.5: Partially addresses question, mixed with irrelevant content
Score 0.2: Mostly off-topic, barely addresses the question
Score 0.0: Completely irrelevant or answers a different question

Consider:
- Does it answer WHAT was asked? (not a different question)
- Is it focused? (not rambling to unrelated topics)
- Is it specific to the question? (not generic filler)

Respond with JSON:
{
  "score": <float 0-1>,
  "addressed_aspects": [<list of question aspects answered>],
  "unaddressed_aspects": [<list of aspects ignored>],
  "off_topic_content": [<list of irrelevant parts>],
  "reasoning": "<explanation>"
}"""
    
    def format_prompt(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> str:
        return f"""QUESTION:
{qa_pair.question}

REFERENCE ANSWER (good example):
{qa_pair.answer}

SYSTEM ANSWER (to evaluate):
{system_output.answer}

Is the system answer relevant to the question?"""
    
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
