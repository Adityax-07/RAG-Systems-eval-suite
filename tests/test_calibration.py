"""Tests for judge calibration statistics.

These are pure math tests — no LLM calls needed.
Tests the statistical functions that measure judge-human agreement.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_framework.types import EvaluationMetric
from eval_framework.judges.calibration import _compute_calibration_stats, _cohens_kappa


class TestCalibrationStats:

    def test_perfect_agreement(self):
        """When judge = human exactly, all metrics should be perfect."""
        scores = [0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.9, 0.5, 0.3, 0.7]
        result = _compute_calibration_stats(
            metric=EvaluationMetric.FAITHFULNESS,
            judge_scores=scores,
            human_scores=scores,  # Identical
        )
        assert result.pearson_r == pytest.approx(1.0, abs=1e-6)
        assert result.mean_absolute_error == pytest.approx(0.0, abs=1e-6)
        assert result.accuracy_within_02 == pytest.approx(1.0, abs=1e-6)
        assert result.bias == pytest.approx(0.0, abs=1e-6)

    def test_perfectly_wrong(self):
        """Judge that always predicts the opposite of human."""
        human = [1.0, 1.0, 1.0, 0.0, 0.0]
        judge = [0.0, 0.0, 0.0, 1.0, 1.0]
        result = _compute_calibration_stats(
            metric=EvaluationMetric.FAITHFULNESS,
            judge_scores=judge,
            human_scores=human,
        )
        assert result.pearson_r == pytest.approx(-1.0, abs=1e-6)
        assert result.mean_absolute_error == pytest.approx(1.0, abs=1e-6)
        assert result.is_calibrated is False

    def test_is_calibrated_threshold(self):
        """Calibrated means: r>0.7, MAE<0.15, acc>0.8, kappa>0.6."""
        # Create scores that just pass all thresholds
        # High correlation, low error
        human = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.95]
        # Slightly perturbed but very close
        judge = [0.12, 0.28, 0.52, 0.68, 0.91, 0.22, 0.78, 0.42, 0.61, 0.93]
        result = _compute_calibration_stats(
            metric=EvaluationMetric.FAITHFULNESS,
            judge_scores=judge,
            human_scores=human,
        )
        # These should be well-calibrated
        assert result.pearson_r > 0.99
        assert result.mean_absolute_error < 0.05
        assert result.accuracy_within_02 == 1.0

    def test_positive_bias(self):
        """Judge that over-scores should have positive bias."""
        human = [0.3, 0.5, 0.7]
        judge = [0.5, 0.7, 0.9]  # Always 0.2 higher
        result = _compute_calibration_stats(
            metric=EvaluationMetric.FAITHFULNESS,
            judge_scores=judge,
            human_scores=human,
        )
        assert result.bias == pytest.approx(0.2, abs=1e-6)

    def test_n_examples_recorded(self):
        human = [0.5, 0.6, 0.7]
        judge = [0.5, 0.6, 0.7]
        result = _compute_calibration_stats(
            metric=EvaluationMetric.FAITHFULNESS,
            judge_scores=judge,
            human_scores=human,
        )
        assert result.n_examples == 3


class TestCohensKappa:

    def test_perfect_agreement(self):
        """Same labels = kappa of 1.0."""
        labels = [0, 0, 1, 1, 2, 2]
        kappa = _cohens_kappa(labels, labels, n_categories=3)
        assert kappa == pytest.approx(1.0, abs=1e-6)

    def test_all_same_category(self):
        """If both raters always use same category, kappa is edge case."""
        labels1 = [2, 2, 2]
        labels2 = [2, 2, 2]
        kappa = _cohens_kappa(labels1, labels2, n_categories=3)
        assert kappa == pytest.approx(1.0, abs=1e-6)

    def test_empty_list(self):
        """Empty input should return 0.0, not crash."""
        kappa = _cohens_kappa([], [], n_categories=3)
        assert kappa == 0.0

    def test_random_agreement(self):
        """Complete disagreement should be near 0 (adjusted for chance)."""
        rater1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        rater2 = [1, 2, 0, 2, 0, 1, 0, 1, 2]  # Mostly different
        kappa = _cohens_kappa(rater1, rater2, n_categories=3)
        # Near-random agreement → kappa near 0
        assert kappa < 0.2
