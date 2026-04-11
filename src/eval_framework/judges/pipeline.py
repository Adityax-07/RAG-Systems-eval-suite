"""Evaluation pipeline - orchestrates running all judges on a system under test.

This is the heart of the framework. The pipeline:
1. Takes a "system under test" (a function that answers questions)
2. Runs it on every QA pair in a dataset
3. Evaluates each answer with all configured evaluators concurrently
4. Aggregates everything into a structured EvaluationReport

Teaching note: Notice how asyncio.gather lets us run multiple LLM judge
calls at the same time. Without this, evaluating 100 examples with 4 metrics
would be 400 sequential API calls. With gather, they run in batches.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime
from typing import Callable, Awaitable

from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log

from ..types import (
    QAPair,
    SystemOutput,
    EvaluationResult,
    EvaluationReport,
    EvaluationMetric,
)
from ..evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

# Type alias: a "system under test" is any async function that takes a QAPair
# and returns a SystemOutput. This is the interface your RAG/chatbot must implement.
SystemUnderTest = Callable[[QAPair], Awaitable[SystemOutput]]


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "rate_limit" in str(exc).lower() or "rate limit" in str(exc).lower()


def _retry_after_secs(exc: Exception) -> float:
    match = re.search(r"try again in ([\d\.]+)s", str(exc))
    return min(float(match.group(1)) + 1.0, 60.0) if match else 5.0


class EvaluationPipeline:
    """Orchestrates running a full eval suite against a system under test.

    Usage:
        pipeline = EvaluationPipeline(
            evaluators=[faithfulness_eval, relevance_eval, completeness_eval],
            concurrency=5,  # max 5 judge calls at once (rate limit protection)
        )

        report = await pipeline.run(
            system=my_rag_system,
            dataset=[qa_pair_1, qa_pair_2, ...],
            system_name="my-rag-v2",
        )

        print(report.summary_scores)
        # {"faithfulness": 0.87, "relevance": 0.91, "completeness": 0.73}
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        concurrency: int = 5,
    ):
        """
        Args:
            evaluators: List of evaluator instances to run on each output.
            concurrency: Max simultaneous LLM judge calls. Keep low to avoid
                         hitting rate limits (5-10 is safe for most providers).
        """
        self.evaluators = evaluators
        self.concurrency = concurrency
        # Semaphore limits concurrent LLM API calls — critical for rate limiting
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _call_system(
        self,
        system: SystemUnderTest,
        qa_pair: QAPair,
    ) -> SystemOutput:
        """Call the system under test with automatic retry on rate-limit errors."""
        attempts = 0
        while True:
            try:
                return await system(qa_pair)
            except Exception as e:
                if _is_rate_limit(e) and attempts < 8:
                    wait = _retry_after_secs(e)
                    attempts += 1
                    logger.warning(
                        f"System 429 (attempt {attempts}/8) — waiting {wait:.1f}s: "
                        f"{qa_pair.question[:40]}..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"System call failed for question: {qa_pair.question[:50]}... Error: {e}"
                    )
                    raise

    async def _evaluate_one(
        self,
        evaluator: BaseEvaluator,
        qa_pair: QAPair,
        system_output: SystemOutput,
    ) -> EvaluationResult:
        """Run one evaluator on one output, respecting the concurrency limit."""
        async with self._semaphore:  # Blocks if too many calls are already in flight
            return await evaluator.evaluate(qa_pair, system_output)

    async def _evaluate_example(
        self,
        system: SystemUnderTest,
        qa_pair: QAPair,
        example_index: int,
    ) -> tuple[SystemOutput, list[EvaluationResult]]:
        """Call system + run all evaluators on one QA pair concurrently."""
        logger.info(f"[{example_index + 1}] Evaluating: {qa_pair.question[:60]}...")

        # 1. Call the system under test to get its answer
        system_output = await self._call_system(system, qa_pair)

        # 2. Run ALL evaluators on this output concurrently
        # asyncio.gather fires all coroutines at the same time and waits for all
        eval_tasks = [
            self._evaluate_one(evaluator, qa_pair, system_output)
            for evaluator in self.evaluators
        ]
        results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        # 3. Filter out failures (return_exceptions=True prevents one failure from killing the batch)
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Evaluator failed: {result}")
            else:
                successful_results.append(result)

        return system_output, successful_results

    async def run(
        self,
        system: SystemUnderTest,
        dataset: list[QAPair],
        system_name: str = "system-under-test",
    ) -> EvaluationReport:
        """Run the full evaluation suite.

        Args:
            system: Async function (QAPair) -> SystemOutput representing your RAG/chatbot.
            dataset: List of QA pairs to evaluate on.
            system_name: Human-readable name for the report (e.g. "rag-v2-gpt4").

        Returns:
            EvaluationReport with per-metric scores and summary statistics.
        """
        logger.info(
            f"Starting evaluation: {len(dataset)} examples × {len(self.evaluators)} metrics"
        )
        start_time = datetime.utcnow()

        # Run all examples concurrently (the semaphore controls actual API concurrency)
        tasks = [
            self._evaluate_example(system, qa_pair, i)
            for i, qa_pair in enumerate(dataset)
        ]
        all_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results by metric
        results_by_metric: dict[EvaluationMetric, list[EvaluationResult]] = {
            evaluator.metric: [] for evaluator in self.evaluators
        }
        total_cost = 0.0

        for output in all_outputs:
            if isinstance(output, Exception):
                logger.error(f"Example failed entirely: {output}")
                continue
            system_output, eval_results = output
            if system_output.cost_usd:
                total_cost += system_output.cost_usd
            for result in eval_results:
                results_by_metric[result.metric].append(result)

        # Calculate summary scores (mean per metric)
        summary_scores: dict[EvaluationMetric, float] = {}
        for metric, metric_results in results_by_metric.items():
            if metric_results:
                summary_scores[metric] = sum(r.score for r in metric_results) / len(metric_results)
            else:
                summary_scores[metric] = 0.0

        report = EvaluationReport(
            id=str(uuid.uuid4()),
            system_name=system_name,
            timestamp=start_time,
            total_examples_evaluated=len(dataset),
            results=results_by_metric,
            summary_scores=summary_scores,
            judge_models_used=list({e.model_name for e in self.evaluators}),
            total_cost=total_cost,
            metadata={
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "concurrency": self.concurrency,
                "evaluator_count": len(self.evaluators),
            },
        )

        logger.info(
            f"Evaluation complete. Scores: "
            + ", ".join(f"{k.value}={v:.2f}" for k, v in summary_scores.items())
        )
        return report
