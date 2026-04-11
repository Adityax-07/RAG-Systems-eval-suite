"""Core data types and schemas for the evaluation framework."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Literal
from datetime import datetime
from enum import Enum


class EvaluationMetric(str, Enum):
    """Supported evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    HALLUCINATION_RATE = "hallucination_rate"
    LATENCY = "latency"
    COST = "cost"
    # --- new metrics ---
    CONCISENESS = "conciseness"          # Is the answer appropriately brief?
    COHERENCE = "coherence"              # Is the answer logically structured and fluent?
    TOXICITY = "toxicity"                # Does the answer contain harmful content?
    CONTEXT_PRECISION = "context_precision"  # How much retrieved context was actually useful?


class EvaluationDimension(BaseModel):
    """Definition of an evaluation dimension."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str = Field(..., description="Name of the evaluation dimension")
    metric: EvaluationMetric = Field(..., description="Associated metric")
    description: str = Field(..., description="Human-readable description")
    scale: Literal["0-1", "0-100", "continuous"] = Field(
        default="0-1", description="Scale of the metric"
    )
    lower_is_better: bool = Field(
        default=False, description="Whether lower scores are better"
    )


class QAPair(BaseModel):
    """Question-Answer pair for evaluation."""
    model_config = ConfigDict(use_enum_values=True)
    
    question: str = Field(..., description="Evaluation question")
    answer: str = Field(..., description="Expected or reference answer")
    context: Optional[str] = Field(default=None, description="Source context/documents")
    id: Optional[str] = Field(default=None, description="Unique identifier")


class SystemOutput(BaseModel):
    """Output from a system under test."""
    model_config = ConfigDict(use_enum_values=True)
    
    answer: str = Field(..., description="Generated answer")
    latency_ms: float = Field(..., description="Response time in milliseconds")
    cost_usd: Optional[float] = Field(default=None, description="API cost in USD")
    model: Optional[str] = Field(default=None, description="Model used")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class EvaluationResult(BaseModel):
    """Result of a single evaluation."""
    model_config = ConfigDict(use_enum_values=True)
    
    metric: EvaluationMetric = Field(..., description="Metric evaluated")
    score: float = Field(..., description="Metric score (0-1 normalized)")
    raw_score: Optional[Any] = Field(default=None, description="Raw score before normalization")
    reasoning: str = Field(..., description="Judge's reasoning for the score")
    judge_model: str = Field(..., description="LLM used as judge")
    confidence: float = Field(default=0.5, description="Judge's confidence (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DatasetExample(BaseModel):
    """A single example in an evaluation dataset."""
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="Unique example ID")
    qa_pair: QAPair = Field(..., description="Question-answer pair")
    gold_answer: Optional[str] = Field(default=None, description="Gold standard answer")
    source_document: Optional[str] = Field(default=None, description="Source document")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Estimated difficulty"
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")


class EvaluationReport(BaseModel):
    """Complete evaluation report for a system."""
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="Unique report ID")
    system_name: str = Field(..., description="Name of system under test")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_examples_evaluated: int = Field(..., description="Number of examples evaluated")
    
    # Results aggregated by metric
    results: dict[EvaluationMetric, list[EvaluationResult]] = Field(
        ..., description="Results grouped by metric"
    )
    
    # Summary statistics
    summary_scores: dict[EvaluationMetric, float] = Field(
        ..., description="Average score per metric"
    )
    
    # Metadata
    judge_models_used: list[str] = Field(..., description="LLM judges used")
    total_cost: float = Field(default=0.0, description="Total evaluation cost in USD")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class JudgePromptTemplate(BaseModel):
    """Template for LLM judge prompts."""
    model_config = ConfigDict(use_enum_values=True)
    
    metric: EvaluationMetric = Field(..., description="Metric this prompt evaluates")
    system_prompt: str = Field(..., description="System/role prompt")
    user_prompt_template: str = Field(..., description="User prompt with {placeholders}")
    output_format: str = Field(..., description="Expected judge output format (JSON schema)")
    version: str = Field(default="1.0", description="Prompt template version")
