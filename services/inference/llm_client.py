"""
LLM Client - Uses Ollama with a larger/more capable model as fallback.
In the simplified architecture, this also uses Ollama (same as SLM).
"""

import time
from typing import Optional

import httpx
import structlog

from config.settings import get_settings
from models.schemas import InferenceRequest, InferenceResponse

logger = structlog.get_logger(__name__)
settings = get_settings()


class LLMClient:
    """
    Fallback LLM client using Ollama.
    In the minimal research prototype, this uses the same Ollama backend
    but can be pointed at a different/larger model if available.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.slm_model
        self.base_url = settings.ollama_base_url
        self._request_count = 0
        self._total_latency_ms = 0.0

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using Ollama (fallback path)."""
        start_time = time.time()
        model = request.model or self.model_name

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": request.max_tokens or settings.slm_max_tokens,
                            "temperature": request.temperature or settings.slm_temperature,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000
            self._request_count += 1
            self._total_latency_ms += latency_ms

            text = data.get("message", {}).get("content", "")
            tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

            return InferenceResponse(
                text=text,
                model=model,
                tokens_used=tokens_used,
                latency_ms=round(latency_ms, 2),
                confidence=0.85,
                finish_reason="stop" if data.get("done", False) else "length",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("LLM inference failed", model=model, error=str(e))
            return InferenceResponse(
                text="",
                model=model,
                tokens_used=0,
                latency_ms=round(latency_ms, 2),
                confidence=0.0,
                finish_reason="error",
            )

    async def health_check(self) -> bool:
        """Check if the LLM API is accessible."""
        try:
            client = self._get_client()
            # Simple models list call to verify API connectivity
            await client.models.list()
            return True
        except Exception:
            return False

    def get_metrics(self) -> dict:
        """Return client metrics."""
        avg_latency = (
            self._total_latency_ms / self._request_count 
            if self._request_count > 0 else 0
        )
        return {
            "model": self.model_name,
            "request_count": self._request_count,
            "avg_latency_ms": round(avg_latency, 2),
            "total_cost_usd": round(self._total_cost, 4),
        }
