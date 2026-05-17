"""
Confidence Calibrator - Temperature scaling for reliable confidence scores.
Ensures model confidence is well-calibrated (low ECE).
"""

import math
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ConfidenceCalibrator:
    """
    Calibrates raw model confidence scores using temperature scaling.
    
    Temperature scaling applies a learned temperature T to logits:
    calibrated_confidence = softmax(logits / T)
    
    For post-hoc calibration on model outputs, we use a simpler
    Platt scaling approach: calibrated = sigmoid(a * raw + b)
    """

    def __init__(self):
        # Default Platt scaling parameters (learned from validation data)
        self._scale_a: float = 1.5  # Slope
        self._scale_b: float = -0.3  # Intercept
        self._is_fitted = False
        self._calibration_data: list[tuple[float, bool]] = []

    def calibrate(self, raw_confidence: float, features: Optional[dict] = None) -> float:
        """
        Calibrate a raw confidence score.
        
        Args:
            raw_confidence: Raw confidence from model (0-1)
            features: Optional features for feature-aware calibration
            
        Returns:
            Calibrated confidence score (0-1)
        """
        if not self._is_fitted:
            # Use default parameters
            return self._platt_scale(raw_confidence)
        
        # Feature-aware calibration
        if features:
            adjustment = self._feature_adjustment(features)
            raw_confidence = max(0.0, min(1.0, raw_confidence + adjustment))
        
        return self._platt_scale(raw_confidence)

    def _platt_scale(self, confidence: float) -> float:
        """Apply Platt scaling (sigmoid transformation)."""
        logit = self._scale_a * confidence + self._scale_b
        calibrated = 1.0 / (1.0 + math.exp(-logit))
        return round(calibrated, 4)

    def _feature_adjustment(self, features: dict) -> float:
        """
        Adjust confidence based on input features.
        E.g., queries with more context tend to be more reliable.
        """
        adjustment = 0.0
        
        # More reasoning steps → higher confidence
        num_steps = features.get("num_reasoning_steps", 0)
        if num_steps >= 4:
            adjustment += 0.05
        elif num_steps <= 1:
            adjustment -= 0.1
        
        # Higher RAG relevance → higher confidence
        rag_score = features.get("avg_rag_score", 0)
        if rag_score >= 0.8:
            adjustment += 0.05
        elif rag_score < 0.5:
            adjustment -= 0.1
        
        # Multiple supporting evidence → higher confidence
        evidence_count = features.get("evidence_count", 0)
        if evidence_count >= 3:
            adjustment += 0.05
        
        return adjustment

    def fit(self, predictions: list[float], ground_truth: list[bool]):
        """
        Fit calibration parameters from validation data.
        
        Args:
            predictions: Raw model confidence scores
            ground_truth: Whether the prediction was actually correct
        """
        if len(predictions) < 50:
            logger.warning("Insufficient data for calibration", n=len(predictions))
            return

        # Simple grid search for Platt scaling parameters
        best_ece = float("inf")
        best_a, best_b = self._scale_a, self._scale_b
        
        for a in np.arange(0.5, 3.0, 0.1):
            for b in np.arange(-1.0, 1.0, 0.1):
                self._scale_a = a
                self._scale_b = b
                calibrated = [self._platt_scale(p) for p in predictions]
                ece = self._compute_ece(calibrated, ground_truth)
                
                if ece < best_ece:
                    best_ece = ece
                    best_a, best_b = a, b

        self._scale_a = best_a
        self._scale_b = best_b
        self._is_fitted = True
        
        logger.info("Calibrator fitted", 
                   scale_a=best_a, scale_b=best_b, ece=round(best_ece, 4))

    def _compute_ece(self, calibrated: list[float], ground_truth: list[bool], n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(calibrated)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = [
                (c, g) for c, g in zip(calibrated, ground_truth)
                if bin_lower <= c < bin_upper
            ]
            
            if not in_bin:
                continue
            
            bin_confidence = np.mean([c for c, _ in in_bin])
            bin_accuracy = np.mean([int(g) for _, g in in_bin])
            bin_size = len(in_bin)
            
            ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)
        
        return ece

    def add_observation(self, predicted_confidence: float, was_correct: bool):
        """Add a calibration observation for online learning."""
        self._calibration_data.append((predicted_confidence, was_correct))
        
        # Refit periodically
        if len(self._calibration_data) % 100 == 0 and len(self._calibration_data) >= 100:
            predictions = [p for p, _ in self._calibration_data]
            truths = [t for _, t in self._calibration_data]
            self.fit(predictions, truths)
