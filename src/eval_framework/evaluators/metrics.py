"""Simple metric evaluators - Latency and Cost"""

from typing import Optional

from ..types import (
    QAPair,
    SystemOutput,
    EvaluationMetric,
    EvaluationResult,
)
from ..utils.llm_client import LLMClient
from .base import BaseEvaluator
from datetime import datetime


class LatencyEvaluator(BaseEvaluator):
    """Evaluates response latency without using an LLM judge.
    
    This is a non-LLM evaluator that directly measures:
    - Response time in milliseconds
    - Compares against acceptable threshold
    - Scores based on latency bucket
    
    Scoring:
    - 1.0 = <500ms (very fast)
    - 0.8 = 500-1000ms (fast)
    - 0.6 = 1-2 seconds (acceptable)
    - 0.4 = 2-5 seconds (slow)
    - 0.2 = 5-10 seconds (very slow)
    - 0.0 = >10 seconds (unacceptable)
    """
    
    def __init__(
        self,
        judge_client: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
        latency_threshold_ms: float = 2000.0,
    ):
        """Initialize latency evaluator.
        
        Args:
            judge_client: Not used (included for interface compatibility)
            model_name: Name for logging
            latency_threshold_ms: Target latency in milliseconds
        """
        # Latency evaluator doesn't need LLM judge
        super().__init__(judge_client, model_name)
        self.latency_threshold_ms = latency_threshold_ms
    
    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.LATENCY
    
    @property
    def system_prompt(self) -> str:
        return "Not used - latency is directly measured"
    
    def format_prompt(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> str:
        return f"Latency evaluation for output with {system_output.latency_ms}ms response time"
    
    async def parse_judge_response(self, response: str) -> tuple[float, str]:
        """Not used - we compute score directly."""
        return 0.5, "Not used"
    
    async def evaluate(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> EvaluationResult:
        """Evaluate latency based on response time.
        
        Uses thresholds:
        - <500ms (1.0), <1s (0.8), <2s (0.6), <5s (0.4), <10s (0.2), else (0.0)
        """
        latency_ms = system_output.latency_ms
        
        # Score based on latency buckets
        if latency_ms < 500:
            score = 1.0
            reasoning = f"Excellent response time: {latency_ms:.0f}ms"
        elif latency_ms < 1000:
            score = 0.8
            reasoning = f"Good response time: {latency_ms:.0f}ms"
        elif latency_ms < 2000:
            score = 0.6
            reasoning = f"Acceptable response time: {latency_ms:.0f}ms"
        elif latency_ms < 5000:
            score = 0.4
            reasoning = f"Slow response time: {latency_ms:.0f}ms"
        elif latency_ms < 10000:
            score = 0.2
            reasoning = f"Very slow response time: {latency_ms:.0f}ms"
        else:
            score = 0.0
            reasoning = f"Unacceptable response time: {latency_ms:.0f}ms"
        
        result = EvaluationResult(
            metric=self.metric,
            score=score,
            raw_score=latency_ms,  # Store raw latency
            reasoning=reasoning,
            judge_model=self.model_name,
            confidence=1.0,  # Perfect confidence (not LLM-based)
            timestamp=datetime.utcnow(),
        )
        
        return result


class CostEvaluator(BaseEvaluator):
    """Evaluates API cost per query without using an LLM judge.
    
    This is a non-LLM evaluator that directly measures:
    - Cost in USD per API call
    - Compares against budget threshold
    - Scores based on cost efficiency
    
    Scoring based on cost efficiency:
    - 1.0 = <$0.001 per query (very cheap)
    - 0.8 = $0.001-0.005 (cheap)
    - 0.6 = $0.005-0.01 (acceptable)
    - 0.4 = $0.01-0.05 (expensive)
    - 0.2 = $0.05-0.10 (very expensive)
    - 0.0 = >$0.10 (prohibitively expensive)
    """
    
    def __init__(
        self,
        judge_client: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
        cost_threshold_usd: float = 0.01,
    ):
        """Initialize cost evaluator.
        
        Args:
            judge_client: Not used (included for interface compatibility)
            model_name: Name for logging
            cost_threshold_usd: Target cost per query in USD
        """
        super().__init__(judge_client, model_name)
        self.cost_threshold_usd = cost_threshold_usd
    
    @property
    def metric(self) -> EvaluationMetric:
        return EvaluationMetric.COST
    
    @property
    def system_prompt(self) -> str:
        return "Not used - cost is directly measured"
    
    def format_prompt(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> str:
        cost = system_output.cost_usd or 0
        return f"Cost evaluation for API call costing ${cost:.6f}"
    
    async def parse_judge_response(self, response: str) -> tuple[float, str]:
        """Not used - we compute score directly."""
        return 0.5, "Not used"
    
    async def evaluate(
        self,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> EvaluationResult:
        """Evaluate cost efficiency based on API cost.
        
        Uses thresholds for cost per query in USD.
        """
        cost_usd = system_output.cost_usd or 0.0
        
        # Score based on cost buckets
        if cost_usd < 0.001:
            score = 1.0
            reasoning = f"Very cost-effective: ${cost_usd:.6f} per query"
        elif cost_usd < 0.005:
            score = 0.8
            reasoning = f"Cost-effective: ${cost_usd:.6f} per query"
        elif cost_usd < 0.01:
            score = 0.6
            reasoning = f"Reasonable cost: ${cost_usd:.6f} per query"
        elif cost_usd < 0.05:
            score = 0.4
            reasoning = f"Expensive: ${cost_usd:.6f} per query"
        elif cost_usd < 0.10:
            score = 0.2
            reasoning = f"Very expensive: ${cost_usd:.6f} per query"
        else:
            score = 0.0
            reasoning = f"Prohibitively expensive: ${cost_usd:.6f} per query"
        
        result = EvaluationResult(
            metric=self.metric,
            score=score,
            raw_score=cost_usd,  # Store raw cost
            reasoning=reasoning,
            judge_model=self.model_name,
            confidence=1.0,  # Perfect confidence
            timestamp=datetime.utcnow(),
        )
        
        return result
