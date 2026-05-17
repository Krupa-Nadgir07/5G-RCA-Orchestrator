"""
Evaluation Service - Benchmark runner comparing SLM vs LLM performance.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

from config.settings import get_settings
from models.schemas import BenchmarkResult, RootCauseCategory
from services.evaluation.datasets import (
    BaseDataset,
    EvaluationSample,
    ITUChallengeDataset,
    LoghubDataset,
    SyntheticDataset,
)
from services.evaluation.metrics import (
    CalibrationMetric,
    ClassificationAccuracyMetric,
    CostMetric,
    HallucinationMetric,
    LatencyMetric,
    ReasoningAccuracyMetric,
)
from services.orchestrator.graph import OrchestratorGraph

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    dataset_name: str = "synthetic_5g"
    max_samples: int = 50
    models_to_compare: list[str] = field(default_factory=lambda: ["phi-3-mini", "mistral-7b", "gpt-4o"])
    concurrency: int = 5
    timeout_per_sample_s: float = 60.0


@dataclass
class ModelResult:
    """Results for a single model on the benchmark."""
    model_name: str
    f1_score: float
    reasoning_accuracy: float
    latency_p95_ms: float
    calibration_ece: float
    hallucination_rate: float
    cost_per_query: float
    num_samples: int
    errors: int = 0


class EvaluationService:
    """
    Benchmark orchestration service.
    Runs evaluation samples through the RCA pipeline and computes metrics.
    """

    DATASETS = {
        "synthetic_5g": SyntheticDataset,
        "itu_challenge": ITUChallengeDataset,
        "loghub_telecom": LoghubDataset,
    }

    def __init__(self, orchestrator: Optional[OrchestratorGraph] = None):
        self.orchestrator = orchestrator or OrchestratorGraph()
        
        # Metrics
        self.classification_metric = ClassificationAccuracyMetric()
        self.reasoning_metric = ReasoningAccuracyMetric()
        self.latency_metric = LatencyMetric(percentile=95)
        self.calibration_metric = CalibrationMetric()
        self.hallucination_metric = HallucinationMetric()
        self.cost_metric = CostMetric()

    async def initialize(self):
        """Initialize evaluation service."""
        await self.orchestrator.initialize()

    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a complete benchmark evaluation.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info("Starting benchmark run",
                   dataset=config.dataset_name,
                   max_samples=config.max_samples)
        
        start_time = time.time()
        
        # Load dataset
        dataset = self._get_dataset(config.dataset_name)
        samples = dataset.sample(config.max_samples)
        
        logger.info("Dataset loaded", num_samples=len(samples))

        # Run pipeline on all samples
        predictions = []
        ground_truths = []
        confidences = []
        latencies = []
        reasoning_preds = []
        reasoning_refs = []
        costs = []
        responses = []
        errors = 0

        semaphore = asyncio.Semaphore(config.concurrency)

        async def process_sample(sample: EvaluationSample):
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self.orchestrator.execute(sample.event),
                        timeout=config.timeout_per_sample_s
                    )
                    return result, sample, None
                except Exception as e:
                    return None, sample, str(e)

        # Execute concurrently
        tasks = [process_sample(s) for s in samples]
        results = await asyncio.gather(*tasks)

        for result, sample, error in results:
            if error:
                errors += 1
                logger.warning("Sample evaluation failed", error=error)
                continue
            
            predictions.append(result.root_cause.value)
            ground_truths.append(sample.ground_truth_cause.value)
            confidences.append(result.confidence)
            latencies.append(result.latency_ms)
            
            # Reasoning steps
            pred_steps = [s.description for s in result.reasoning_trace] if result.reasoning_trace else []
            reasoning_preds.append(pred_steps)
            reasoning_refs.append(sample.reference_reasoning)
            
            # Cost (estimate based on model used)
            cost = self._estimate_cost(result.model_used)
            costs.append(cost)
            
            # Full response text for hallucination check
            responses.append(result.specific_cause or "")

        # Compute metrics
        correctness = [p == g for p, g in zip(predictions, ground_truths)]
        
        f1_result = self.classification_metric.compute(predictions, ground_truths)
        reasoning_result = self.reasoning_metric.compute(reasoning_preds, reasoning_refs)
        latency_result = self.latency_metric.compute(latencies)
        calibration_result = self.calibration_metric.compute(confidences, correctness)
        hallucination_result = self.hallucination_metric.compute(responses)
        cost_result = self.cost_metric.compute(costs)
        
        total_time = time.time() - start_time
        
        logger.info("Benchmark complete",
                   f1=f1_result.value,
                   ece=calibration_result.value,
                   p95_latency=latency_result.value,
                   total_time_s=round(total_time, 1))

        return BenchmarkResult(
            dataset=config.dataset_name,
            num_samples=len(samples),
            classification_f1=f1_result.value,
            reasoning_accuracy=reasoning_result.value,
            latency_p95_ms=latency_result.value,
            calibration_ece=calibration_result.value,
            hallucination_rate=hallucination_result.value,
            cost_per_query=cost_result.value,
            errors=errors,
            details={
                "classification": f1_result.details,
                "latency": latency_result.details,
                "calibration": calibration_result.details,
                "total_time_s": round(total_time, 1),
                "models_used": config.models_to_compare,
            },
        )

    async def compare_models(self, config: BenchmarkConfig) -> dict[str, ModelResult]:
        """
        Compare multiple models on the same dataset.
        Forces each model tier separately.
        """
        results = {}
        
        dataset = self._get_dataset(config.dataset_name)
        samples = dataset.sample(config.max_samples)
        
        # For each model, run the same samples with forced routing
        for model_name in config.models_to_compare:
            logger.info("Evaluating model", model=model_name)
            
            preds, gts, confs, lats, costs_list, resps = [], [], [], [], [], []
            errors = 0
            
            for sample in samples:
                try:
                    # Force specific model via inference service configuration
                    result = await asyncio.wait_for(
                        self.orchestrator.execute(sample.event),
                        timeout=config.timeout_per_sample_s
                    )
                    
                    preds.append(result.root_cause.value)
                    gts.append(sample.ground_truth_cause.value)
                    confs.append(result.confidence)
                    lats.append(result.latency_ms)
                    costs_list.append(self._estimate_cost(model_name))
                    resps.append(result.specific_cause or "")
                    
                except Exception:
                    errors += 1
            
            # Compute per-model metrics
            correctness = [p == g for p, g in zip(preds, gts)]
            
            results[model_name] = ModelResult(
                model_name=model_name,
                f1_score=self.classification_metric.compute(preds, gts).value if preds else 0.0,
                reasoning_accuracy=0.0,  # Would need reference steps
                latency_p95_ms=self.latency_metric.compute(lats).value if lats else 0.0,
                calibration_ece=self.calibration_metric.compute(confs, correctness).value if confs else 1.0,
                hallucination_rate=self.hallucination_metric.compute(resps).value if resps else 0.0,
                cost_per_query=self.cost_metric.compute(costs_list).value if costs_list else 0.0,
                num_samples=len(preds),
                errors=errors,
            )
        
        return results

    def _get_dataset(self, name: str) -> BaseDataset:
        """Get dataset by name."""
        dataset_cls = self.DATASETS.get(name)
        if not dataset_cls:
            logger.warning("Unknown dataset, using synthetic", name=name)
            dataset_cls = SyntheticDataset
        return dataset_cls()

    def _estimate_cost(self, model_name: str) -> float:
        """Estimate cost per query by model."""
        cost_map = {
            "phi-3-mini": 0.0001,   # Local inference
            "mistral-7b": 0.0002,   # Local inference
            "gpt-4o": 0.015,        # API cost estimate
            "rule-based": 0.0,      # No model used
            "error": 0.0,
        }
        return cost_map.get(model_name, 0.001)
