"""
Inference Router - Simplified single-tier routing using Ollama.
All inference goes through local Ollama + Mistral.
"""

import structlog

from config.settings import get_settings
from models.schemas import InferenceRequest, InferenceResponse
from services.inference.slm_client import SLMClient

logger = structlog.get_logger(__name__)
settings = get_settings()


class InferenceRouter:
    """
    Simplified inference router using Ollama for all requests.
    Routes all inference through the local SLM (Mistral via Ollama).
    """

    def __init__(self):
        self.slm = SLMClient(
            model_name=settings.slm_model,
            base_url=settings.ollama_base_url,
        )
        self._routing_stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
        }

    async def route(self, request: InferenceRequest, force_llm: bool = False) -> InferenceResponse:
        """
        Route request to Ollama SLM.
        force_llm parameter is kept for API compatibility but uses same model.
        """
        self._routing_stats["total_requests"] += 1

        response = await self.slm.generate(request)

        if response.finish_reason in ("stop", "length") and response.text:
            self._routing_stats["successful"] += 1
        else:
            self._routing_stats["failed"] += 1

        return response

    async def health_check(self) -> dict:
        """Check health of inference backend."""
        slm_healthy = await self.slm.health_check()
        return {
            "ollama": slm_healthy,
            "model": settings.slm_model,
        }

    def get_metrics(self) -> dict:
        """Return routing metrics."""
        total = self._routing_stats["total_requests"] or 1
        return {
            **self._routing_stats,
            "success_rate_pct": round(self._routing_stats["successful"] / total * 100, 1),
            "slm": self.slm.get_metrics(),
        }

    async def close(self):
        """Cleanup (no persistent connections to close with httpx)."""
        pass

