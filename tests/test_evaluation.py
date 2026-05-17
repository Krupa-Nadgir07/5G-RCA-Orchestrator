"""Tests for evaluation metrics."""

import pytest
from services.evaluation.metrics import (
    CalibrationMetric,
    ClassificationAccuracyMetric,
    CostMetric,
    HallucinationMetric,
    LatencyMetric,
    ReasoningAccuracyMetric,
)


class TestClassificationMetric:
    def test_perfect_predictions(self):
        metric = ClassificationAccuracyMetric()
        preds = ["interference", "congestion", "handover"]
        truth = ["interference", "congestion", "handover"]
        result = metric.compute(preds, truth)
        assert result.value == 1.0

    def test_all_wrong(self):
        metric = ClassificationAccuracyMetric()
        preds = ["interference", "interference", "interference"]
        truth = ["congestion", "handover", "hardware"]
        result = metric.compute(preds, truth)
        assert result.value == 0.0


class TestCalibrationMetric:
    def test_perfect_calibration(self):
        metric = CalibrationMetric()
        # All predictions are 90% confident and 90% are correct
        confs = [0.9] * 10
        correct = [True] * 9 + [False]
        result = metric.compute(confs, correct)
        assert result.value < 0.1

    def test_overconfident(self):
        metric = CalibrationMetric()
        confs = [0.95] * 10
        correct = [True, False, False, False, False, True, False, False, False, False]
        result = metric.compute(confs, correct)
        # ECE should be high (overconfident)
        assert result.value > 0.5


class TestLatencyMetric:
    def test_p95(self):
        metric = LatencyMetric(percentile=95)
        latencies = list(range(1, 101))  # 1-100ms
        result = metric.compute(latencies)
        assert 94 <= result.value <= 96

    def test_empty(self):
        metric = LatencyMetric()
        result = metric.compute([])
        assert result.value == 0.0


class TestHallucinationMetric:
    def test_no_hallucinations(self):
        metric = HallucinationMetric()
        responses = [
            "The root cause is co-channel interference from cell-102",
            "PRB congestion due to high UE count",
        ]
        result = metric.compute(responses)
        assert result.value == 0.0

    def test_detects_hallucination(self):
        metric = HallucinationMetric()
        responses = [
            "This 6g network has zero latency guaranteed",
        ]
        result = metric.compute(responses)
        assert result.value > 0.0


class TestCostMetric:
    def test_average_cost(self):
        metric = CostMetric()
        costs = [0.001, 0.001, 0.015, 0.001]
        result = metric.compute(costs)
        assert 0.004 < result.value < 0.005
