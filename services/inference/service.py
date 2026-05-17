"""
Inference Service - Unified interface for model inference.
"""

from config.settings import get_settings
from models.schemas import InferenceRequest, InferenceResponse
from services.inference.confidence_calibrator import ConfidenceCalibrator
from services.inference.router import InferenceRouter

settings = get_settings()


class InferenceService:
    """
    Unified inference service combining routing, calibration, and metrics.
    """

    def __init__(self):
        self.router = InferenceRouter()
        self.calibrator = ConfidenceCalibrator()

    async def generate(
        self, 
        prompt: str, 
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        force_llm: bool = False,
    ) -> InferenceResponse:
        """Generate text with automatic model routing and confidence calibration."""
        request = InferenceRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens or settings.slm_max_tokens,
            temperature=temperature or settings.slm_temperature,
        )

        response = await self.router.route(request, force_llm=force_llm)
        
        # Calibrate confidence
        calibrated = self.calibrator.calibrate(
            response.confidence,
            features={"num_reasoning_steps": response.text.count("Step")}
        )
        response.confidence = calibrated
        
        return response

    async def health_check(self) -> dict:
        """Check inference service health."""
        return await self.router.health_check()

    def get_metrics(self) -> dict:
        """Return inference service metrics."""
        return self.router.get_metrics()

    async def close(self):
        """Cleanup resources."""
        await self.router.close()
