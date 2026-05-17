"""
Evaluation Metrics - Comprehensive metrics for model benchmarking.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: float
    details: dict = field(default_factory=dict)


class ClassificationAccuracyMetric:
    """
    Weighted F1 score for root cause classification.
    Handles multi-class classification with imbalanced classes.
    """
    name = "classification_accuracy"

    def compute(self, predictions: list[str], ground_truth: list[str]) -> MetricResult:
        """Compute weighted F1 score."""
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        accuracy = accuracy_score(ground_truth, predictions)
        f1_weighted = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
        precision = precision_score(ground_truth, predictions, average="weighted", zero_division=0)
        recall = recall_score(ground_truth, predictions, average="weighted", zero_division=0)
        
        return MetricResult(
            name=self.name,
            value=f1_weighted,
            details={
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "precision_weighted": precision,
                "recall_weighted": recall,
            }
        )


class ReasoningAccuracyMetric:
    """
    Measures the correctness of Chain-of-Thought reasoning steps.
    Compares generated reasoning steps against reference steps.
    """
    name = "reasoning_accuracy"

    def compute(
        self, 
        predicted_steps: list[list[str]], 
        reference_steps: list[list[str]]
    ) -> MetricResult:
        """
        Compute reasoning accuracy as average step correctness.
        Uses fuzzy matching for step comparison.
        """
        if not predicted_steps or not reference_steps:
            return MetricResult(name=self.name, value=0.0)

        total_score = 0.0
        total_samples = len(predicted_steps)

        for pred, ref in zip(predicted_steps, reference_steps):
            if not pred or not ref:
                continue
            # Compute step-level similarity
            step_scores = []
            for pred_step in pred:
                max_sim = max(
                    self._similarity(pred_step, ref_step) 
                    for ref_step in ref
                )
                step_scores.append(max_sim)
            
            if step_scores:
                total_score += np.mean(step_scores)

        avg_score = total_score / total_samples if total_samples > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            value=avg_score,
            details={"num_samples": total_samples}
        )

    def _similarity(self, text_a: str, text_b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return len(intersection) / len(union) if union else 0.0


class LatencyMetric:
    """Measures inference latency at a given percentile."""
    
    def __init__(self, percentile: int = 95):
        self.percentile = percentile
        self.name = f"latency_p{percentile}"

    def compute(self, latencies_ms: list[float]) -> MetricResult:
        """Compute percentile latency."""
        if not latencies_ms:
            return MetricResult(name=self.name, value=0.0)
        
        p_value = np.percentile(latencies_ms, self.percentile)
        
        return MetricResult(
            name=self.name,
            value=p_value,
            details={
                "p50": np.percentile(latencies_ms, 50),
                "p95": np.percentile(latencies_ms, 95),
                "p99": np.percentile(latencies_ms, 99),
                "mean": np.mean(latencies_ms),
                "std": np.std(latencies_ms),
                "num_samples": len(latencies_ms),
            }
        )


class CalibrationMetric:
    """
    Expected Calibration Error (ECE) - measures confidence calibration.
    Lower is better; perfect calibration = 0.0.
    """
    name = "confidence_calibration"

    def compute(
        self, 
        confidences: list[float], 
        correctness: list[bool], 
        n_bins: int = 10
    ) -> MetricResult:
        """Compute Expected Calibration Error."""
        if not confidences or not correctness:
            return MetricResult(name=self.name, value=1.0)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidences)
        
        bin_details = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = [
                (c, g) for c, g in zip(confidences, correctness)
                if bin_lower <= c < bin_upper
            ]
            
            if not in_bin:
                continue
            
            bin_confidence = np.mean([c for c, _ in in_bin])
            bin_accuracy = np.mean([int(g) for _, g in in_bin])
            bin_size = len(in_bin)
            
            ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)
            
            bin_details.append({
                "bin": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                "count": bin_size,
                "avg_confidence": round(bin_confidence, 3),
                "accuracy": round(bin_accuracy, 3),
            })

        return MetricResult(
            name=self.name,
            value=round(ece, 4),
            details={"bins": bin_details, "n_bins": n_bins}
        )


class HallucinationMetric:
    """
    Measures the rate of hallucinated (factually incorrect) claims.
    Checks if generated text contains known incorrect assertions.
    """
    name = "hallucination_rate"

    # Known facts that should not be contradicted
    TELECOM_FACTS = {
        "5g_max_bandwidth": 400,  # MHz
        "nr_max_prb": 273,  # for 100MHz
        "max_mcs_index": 28,
        "max_cqi": 15,
    }

    def compute(self, responses: list[str], reference_facts: list[dict] = None) -> MetricResult:
        """Compute hallucination rate."""
        if not responses:
            return MetricResult(name=self.name, value=0.0)
        
        hallucinations = 0
        total = len(responses)
        
        for response in responses:
            if self._contains_hallucination(response):
                hallucinations += 1
        
        rate = hallucinations / total if total > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            value=round(rate, 4),
            details={"hallucinations": hallucinations, "total": total}
        )

    def _contains_hallucination(self, text: str) -> bool:
        """Check if text contains obvious hallucinations."""
        text_lower = text.lower()
        
        # Check for impossible claims
        impossible_claims = [
            "6g",  # If claiming 6G when analyzing 5G
            "10 gbps guaranteed",  # Impossible guarantee
            "zero latency",  # Physically impossible
        ]
        
        for claim in impossible_claims:
            if claim in text_lower:
                return True
        
        return False


class CostMetric:
    """Measures cost per query."""
    name = "cost_per_query"

    def compute(self, costs: list[float]) -> MetricResult:
        """Compute average cost per query."""
        if not costs:
            return MetricResult(name=self.name, value=0.0)
        
        return MetricResult(
            name=self.name,
            value=round(np.mean(costs), 6),
            details={
                "total_cost": round(sum(costs), 4),
                "min_cost": round(min(costs), 6),
                "max_cost": round(max(costs), 6),
                "num_queries": len(costs),
            }
        )
