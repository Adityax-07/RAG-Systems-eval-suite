"""Judge calibration - measuring how well your LLM judge agrees with humans.

This is what separates a toy eval framework from a production one.

The problem: An LLM judge might be biased, inconsistent, or miscalibrated.
You need to PROVE it agrees with humans before trusting it.

How calibration works:
1. Collect ~100 examples where humans have scored the output (0-1)
2. Run your LLM judge on the same examples
3. Measure statistical agreement between judge and human scores

Key metrics we compute:
- Pearson r: Linear correlation (-1 to 1). You want > 0.7.
- Mean Absolute Error (MAE): Average size of disagreement. Want < 0.15.
- Accuracy within threshold: % of time judge is within ±0.2 of human. Want > 80%.
- Cohen's kappa: Agreement adjusted for chance (discretized scores).

Teaching note: This is a real technique used at AI labs to validate eval systems.
Companies like Anthropic and OpenAI run similar calibration pipelines before
trusting automated evals. Showing this on your resume is rare and impressive.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..types import QAPair, SystemOutput, EvaluationMetric
from ..evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class HumanLabeledExample:
    """A single example with a human-assigned ground truth score.

    Collect these by having a human (or yourself) score outputs manually.
    Even 50-100 examples is enough to see if your judge is calibrated.
    """
    qa_pair: QAPair
    system_output: SystemOutput
    human_score: float          # Human-assigned score, 0.0 to 1.0
    human_reasoning: str = ""   # Optional: why the human gave this score
    annotator_id: str = "human" # Track who annotated (for inter-annotator agreement)


@dataclass
class CalibrationResult:
    """Statistical summary of judge-vs-human agreement for one metric."""
    metric: EvaluationMetric
    n_examples: int

    # Core agreement metrics
    pearson_r: float            # Linear correlation. >0.7 is good.
    mean_absolute_error: float  # Average |judge - human|. <0.15 is good.
    accuracy_within_02: float   # % of time |judge - human| < 0.2. Want >0.8.

    # Discretized agreement (treating 0-0.33=low, 0.33-0.66=mid, 0.67-1=high)
    cohens_kappa: float         # Chance-adjusted agreement. >0.6 is good.

    # Per-example details for debugging
    judge_scores: list[float] = field(default_factory=list)
    human_scores: list[float] = field(default_factory=list)

    # Diagnosis
    is_calibrated: bool = False     # True if all metrics pass thresholds
    bias: float = 0.0               # mean(judge) - mean(human). Positive = judge over-scores.

    def summary(self) -> str:
        """Human-readable calibration summary."""
        status = "PASS" if self.is_calibrated else "FAIL"
        return (
            f"[{status}] {self.metric.value} calibration ({self.n_examples} examples):\n"
            f"  Pearson r:     {self.pearson_r:.3f} (want >0.70)\n"
            f"  MAE:           {self.mean_absolute_error:.3f} (want <0.15)\n"
            f"  Acc ±0.2:      {self.accuracy_within_02:.1%} (want >80%)\n"
            f"  Cohen's kappa: {self.cohens_kappa:.3f} (want >0.60)\n"
            f"  Bias:          {self.bias:+.3f} (want ≈0.00)\n"
        )


class JudgeCalibrator:
    """Calibrates LLM judges against human-labeled examples.

    Usage:
        calibrator = JudgeCalibrator(faithfulness_evaluator)

        # After you've collected human labels (even 50 is fine to start):
        examples = [
            HumanLabeledExample(qa_pair=..., system_output=..., human_score=0.9),
            HumanLabeledExample(qa_pair=..., system_output=..., human_score=0.3),
            ...
        ]

        result = await calibrator.calibrate(examples)
        print(result.summary())
    """

    def __init__(self, evaluator: BaseEvaluator, concurrency: int = 5):
        self.evaluator = evaluator
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(self, example: HumanLabeledExample) -> float:
        """Run the judge on one example and return its score."""
        async with self._semaphore:
            result = await self.evaluator.evaluate(
                example.qa_pair,
                example.system_output,
            )
            return result.score

    async def calibrate(
        self,
        examples: list[HumanLabeledExample],
    ) -> CalibrationResult:
        """Run calibration: compare judge scores vs human scores.

        Args:
            examples: Human-labeled examples. Aim for 50-100 minimum.

        Returns:
            CalibrationResult with all agreement statistics.
        """
        if len(examples) < 10:
            logger.warning(
                f"Only {len(examples)} examples for calibration. "
                "Use 50+ for reliable statistics."
            )

        logger.info(
            f"Calibrating {self.evaluator.metric.value} judge on {len(examples)} examples..."
        )

        # Run all judge evaluations concurrently
        judge_scores = await asyncio.gather(
            *[self._judge_one(ex) for ex in examples],
            return_exceptions=True,
        )

        # Separate successes from failures
        valid_judge: list[float] = []
        valid_human: list[float] = []
        for ex, score in zip(examples, judge_scores):
            if isinstance(score, Exception):
                logger.warning(f"Judge failed on example: {score}")
                continue
            valid_judge.append(float(score))
            valid_human.append(ex.human_score)

        if len(valid_judge) < 5:
            raise RuntimeError(
                f"Too few successful evaluations ({len(valid_judge)}) to calibrate. "
                "Check your API key and evaluator configuration."
            )

        return _compute_calibration_stats(
            metric=self.evaluator.metric,
            judge_scores=valid_judge,
            human_scores=valid_human,
        )


def _compute_calibration_stats(
    metric: EvaluationMetric,
    judge_scores: list[float],
    human_scores: list[float],
) -> CalibrationResult:
    """Compute all calibration statistics from score lists.

    This is pure statistics — no LLM calls needed.
    Kept separate so you can also call it with scores from a CSV file.
    """
    import math

    n = len(judge_scores)
    assert n == len(human_scores), "Score lists must be same length"

    # --- Pearson correlation ---
    # Measures linear agreement between judge and human scores
    j_mean = sum(judge_scores) / n
    h_mean = sum(human_scores) / n
    numerator = sum((j - j_mean) * (h - h_mean) for j, h in zip(judge_scores, human_scores))
    j_std = math.sqrt(sum((j - j_mean) ** 2 for j in judge_scores))
    h_std = math.sqrt(sum((h - h_mean) ** 2 for h in human_scores))
    pearson_r = numerator / (j_std * h_std) if (j_std * h_std) > 0 else 0.0

    # --- Mean Absolute Error ---
    mae = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / n

    # --- Accuracy within ±0.2 threshold ---
    within_02 = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= 0.2)
    accuracy_within_02 = within_02 / n

    # --- Cohen's kappa (discretized into 3 buckets) ---
    # We bin scores: low (0-0.33), mid (0.33-0.67), high (0.67-1.0)
    def discretize(score: float) -> int:
        if score < 0.33:
            return 0  # low
        elif score < 0.67:
            return 1  # mid
        else:
            return 2  # high

    j_discrete = [discretize(s) for s in judge_scores]
    h_discrete = [discretize(s) for s in human_scores]

    cohens_kappa = _cohens_kappa(j_discrete, h_discrete, n_categories=3)

    # --- Bias ---
    bias = j_mean - h_mean  # Positive = judge tends to over-score

    # --- Pass/fail thresholds ---
    is_calibrated = (
        pearson_r > 0.70
        and mae < 0.15
        and accuracy_within_02 > 0.80
        and cohens_kappa > 0.60
    )

    return CalibrationResult(
        metric=metric,
        n_examples=n,
        pearson_r=pearson_r,
        mean_absolute_error=mae,
        accuracy_within_02=accuracy_within_02,
        cohens_kappa=cohens_kappa,
        judge_scores=judge_scores,
        human_scores=human_scores,
        is_calibrated=is_calibrated,
        bias=bias,
    )


def _cohens_kappa(
    rater1: list[int],
    rater2: list[int],
    n_categories: int,
) -> float:
    """Compute Cohen's kappa for two lists of categorical labels.

    Kappa = (Po - Pe) / (1 - Pe)
    - Po = observed agreement (fraction of examples where both raters agree)
    - Pe = expected agreement by chance (if raters assigned randomly)

    This is better than raw accuracy because it accounts for chance agreement.
    Two raters who always guess "high" would have high accuracy but kappa ≈ 0.
    """
    n = len(rater1)
    if n == 0:
        return 0.0

    # Po: proportion of times they agree
    observed_agree = sum(1 for a, b in zip(rater1, rater2) if a == b)
    po = observed_agree / n

    # Pe: expected chance agreement
    # For each category, compute: (count_in_rater1 / n) * (count_in_rater2 / n)
    pe = 0.0
    for cat in range(n_categories):
        p1 = sum(1 for r in rater1 if r == cat) / n
        p2 = sum(1 for r in rater2 if r == cat) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0  # Perfect agreement edge case

    return (po - pe) / (1.0 - pe)


def load_calibration_examples_from_csv(filepath: str) -> list[HumanLabeledExample]:
    """Load human-labeled calibration examples from a CSV file.

    Expected CSV columns:
        question, answer, context, system_answer, latency_ms, human_score, human_reasoning

    This lets you collect labels in a spreadsheet and import them here.
    """
    import csv
    from ..types import QAPair, SystemOutput

    examples = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qa_pair = QAPair(
                question=row["question"],
                answer=row["answer"],
                context=row.get("context") or None,
            )
            system_output = SystemOutput(
                answer=row["system_answer"],
                latency_ms=float(row.get("latency_ms", 1000)),
                cost_usd=float(row["cost_usd"]) if row.get("cost_usd") else None,
            )
            examples.append(
                HumanLabeledExample(
                    qa_pair=qa_pair,
                    system_output=system_output,
                    human_score=float(row["human_score"]),
                    human_reasoning=row.get("human_reasoning", ""),
                )
            )
    return examples
