"""Base evaluator class for all metrics."""

from abc import ABC, abstractmethod
from typing import Optional
import logging
from datetime import datetime

from ..types import (
    QAPair,
    SystemOutput,
    EvaluationResult,
    EvaluationMetric,
)
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators.
    
    Each evaluator measures a specific dimension of LLM output quality.
    They all follow the same pattern:
    1. Receive question, reference answer, and system output
    2. Use an LLM judge to score the output
    3. Return EvaluationResult with score and reasoning
    """
    
    def __init__(self, judge_client: LLMClient, model_name: Optional[str] = None):
        """Initialize evaluator with an LLM client.
        
        Args:
            judge_client: LLMClient instance (OpenAI or Anthropic)
            model_name: Name of the judge model (for logging/display)
        """
        self.judge_client = judge_client
        self.model_name = model_name or "unknown-model"
    
    @property
    @abstractmethod
    def metric(self) -> EvaluationMetric:
        """Return the metric this evaluator measures."""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Judge system prompt - defines the role and task."""
        pass
    
    @abstractmethod
    def format_prompt(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> str:
        """Format the evaluation prompt for the judge.
        
        Args:
            qa_pair: The question and reference answer
            system_output: The system's output to evaluate
            
        Returns:
            Formatted prompt for the judge
        """
        pass
    
    @abstractmethod
    async def parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse judge's response into score and reasoning.
        
        Args:
            response: Raw response from the judge LLM
            
        Returns:
            Tuple of (score_0_to_1, reasoning)
        """
        pass
    
    async def evaluate(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> EvaluationResult:
        """Evaluate system output for this metric.
        
        This is the main entry point. It orchestrates:
        1. Formatting the evaluation prompt
        2. Calling the judge
        3. Parsing the response
        4. Returning a structured result
        
        Args:
            qa_pair: Question and reference answer
            system_output: Output from system under test
            
        Returns:
            EvaluationResult with score, reasoning, confidence
        """
        try:
            # Format prompt for the judge
            user_prompt = self.format_prompt(qa_pair, system_output)
            
            # Get judge's evaluation
            logger.debug(f"Calling judge for {self.metric.value}")
            judge_response = await self.judge_client.generate(
                system_prompt=self.system_prompt,
                user_message=user_prompt,
                temperature=0.1,  # Low temp for consistency
                max_tokens=500,
            )
            
            # Parse the response
            score, reasoning = await self.parse_judge_response(judge_response)
            
            # Validate score is 0-1
            if not (0 <= score <= 1):
                logger.warning(f"Score {score} out of range, clamping to [0, 1]")
                score = max(0, min(1, score))
            
            # Return structured result
            result = EvaluationResult(
                metric=self.metric,
                score=score,
                raw_score=score,  # Could store original if different
                reasoning=reasoning,
                judge_model=self.model_name,
                confidence=0.8,  # Could estimate from judge's confidence
                timestamp=datetime.utcnow(),
            )
            
            logger.info(f"Evaluation complete: {self.metric.value}={score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for {self.metric.value}: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(metric={self.metric.value}, judge={self.model_name})"
