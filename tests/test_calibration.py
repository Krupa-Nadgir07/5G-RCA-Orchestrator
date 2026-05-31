"""
Unit tests for the Confidence Calibrator and feedback loop.
"""

import pytest
from services.inference.confidence_calibrator import ConfidenceCalibrator
from services.inference.service import InferenceService
from config.settings import get_settings

settings = get_settings()


def test_platt_scaling_math():
    """Verify that Platt scaling correctly translates raw confidence using a sigmoid function."""
    calibrator = ConfidenceCalibrator()
    # default parameters are scale_a=1.5, scale_b=-0.3
    # logit for 0.5: 1.5 * 0.5 - 0.3 = 0.45
    # sigmoid(0.45): 1 / (1 + exp(-0.45)) = 0.6106
    val = calibrator.calibrate(0.5)
    assert 0.60 <= val <= 0.62

    # logit for 0.0: 1.5 * 0.0 - 0.3 = -0.3
    # sigmoid(-0.3): 1 / (1 + exp(0.3)) = 0.4255
    val_zero = calibrator.calibrate(0.0)
    assert 0.41 <= val_zero <= 0.44


def test_feature_aware_adjustment():
    """Verify feature-aware adjustments shift confidence values appropriately."""
    calibrator = ConfidenceCalibrator()
    raw = 0.6

    # Test baseline calibrate without features
    baseline = calibrator.calibrate(raw)

    # Test positive adjustment (e.g. high reasoning steps + high RAG score + multiple evidence)
    adjusted_up = calibrator.calibrate(raw, features={
        "num_reasoning_steps": 5,  # +0.05
        "avg_rag_score": 0.85,      # +0.05
        "evidence_count": 4        # +0.05
    })
    assert adjusted_up > baseline

    # Test negative adjustment (e.g. low reasoning steps + low RAG score)
    adjusted_down = calibrator.calibrate(raw, features={
        "num_reasoning_steps": 1,  # -0.1
        "avg_rag_score": 0.4       # -0.1
    })
    assert adjusted_down < baseline


def test_calibration_fit():
    """Verify fitting calibrator parameters yields lower expected calibration error (ECE)."""
    calibrator = ConfidenceCalibrator()
    
    # Generate mock validation data: raw confidence and whether prediction was correct
    # A set of predictions that are over-confident (e.g. raw=0.9 but only correct 50% of the time)
    predictions = [0.9] * 30 + [0.3] * 30
    ground_truth = [True] * 15 + [False] * 15 + [True] * 5 + [False] * 25
    
    # We override app_debug to allow small dataset fitting
    original_debug = settings.app_debug
    settings.app_debug = True
    try:
        calibrator.fit(predictions, ground_truth)
        assert calibrator._is_fitted is True
        
        # Verify fitted parameters result in a calibration update
        calibrated_high = calibrator.calibrate(0.9)
        # Raw 0.9 should have been scaled down because truth rate was only 50%
        assert calibrated_high < 0.9
    finally:
        settings.app_debug = original_debug


def test_online_calibration_updates():
    """Verify that submitting observations triggers refitting periodically."""
    calibrator = ConfidenceCalibrator()
    assert calibrator._is_fitted is False

    original_debug = settings.app_debug
    settings.app_debug = True
    try:
        # Feed observations in debug mode (interval is 5)
        for i in range(4):
            calibrator.add_observation(0.8, True)
        assert calibrator._is_fitted is False
        
        # 5th observation should trigger fit
        calibrator.add_observation(0.8, True)
        assert calibrator._is_fitted is True
    finally:
        settings.app_debug = original_debug


@pytest.mark.asyncio
async def test_feedback_loop_wiring(sample_event):
    """Verify that feedback route successfully integrates with the calibrator."""
    from services.orchestrator.graph import OrchestratorGraph
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services.api.routes import router
    
    app = FastAPI()
    app.state.orchestrator = OrchestratorGraph()
    app.include_router(router, prefix="/api/v1")
    
    # Store a prediction in InferenceService's cache
    rca_id = "11111111-2222-3333-4444-555555555555"
    app.state.orchestrator.inference_service._recent_predictions[rca_id] = 0.85
    
    client = TestClient(app)
    response = client.post("/api/v1/feedback", json={
        "rca_id": rca_id,
        "correct": True
    })
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "accepted"
    assert "updated successfully" in res_data["message"]
    assert res_data["prediction"] == 0.85
    assert res_data["correction"] is True
    
    # Verify calibrator registered the observation
    assert len(app.state.orchestrator.inference_service.calibrator._calibration_data) == 1
