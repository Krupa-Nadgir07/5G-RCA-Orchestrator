"""
SLM Client - Cloud-hosted Small Language Model inference via Groq.
Uses Groq OpenAI-compatible API for inference.
"""

import os
import time
from typing import Optional

import httpx
import structlog

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config.settings import get_settings
from models.schemas import InferenceRequest, InferenceResponse

load_dotenv()

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------
# Groq OpenAI-Compatible Client
# ---------------------------------------------------------

groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


class SLMClient:
    """
    Client for cloud-hosted Small Language Models via Groq.

    Supports:
    - Llama 3
    - DeepSeek Distill
    - Gemma
    - Any Groq-supported OpenAI-compatible model
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):

        # Default model from settings
        self.model_name = model_name or settings.slm_model

        # Retained for compatibility
        self.base_url = base_url or "https://api.groq.com/openai/v1"

        # Metrics
        self._request_count = 0
        self._total_latency_ms = 0.0

    async def generate(
        self,
        request: InferenceRequest
    ) -> InferenceResponse:
        """
        Generate text completion using Groq cloud inference.
        """

        start_time = time.time()

        model = request.model or self.model_name

        try:

            response = await groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": request.system_prompt or ""
                    },
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            text = response.choices[0].message.content or ""

            latency_ms = (time.time() - start_time) * 1000

            # Metrics tracking
            self._request_count += 1
            self._total_latency_ms += latency_ms

            confidence = self._estimate_confidence(text)

            return InferenceResponse(
                text=text,
                model=model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=round(latency_ms, 2),
                confidence=confidence,
                finish_reason=response.choices[0].finish_reason or "stop",
            )

        except httpx.TimeoutException:

            latency_ms = (time.time() - start_time) * 1000

            logger.error(
                "Groq inference timeout",
                model=model,
                latency_ms=latency_ms
            )

            return InferenceResponse(
                text="",
                model=model,
                tokens_used=0,
                latency_ms=round(latency_ms, 2),
                confidence=0.0,
                finish_reason="timeout",
            )

        except Exception as e:

            latency_ms = (time.time() - start_time) * 1000

            logger.error(
                "Groq inference failed",
                model=model,
                error=str(e),
            )

            return InferenceResponse(
                text="",
                model=model,
                tokens_used=0,
                latency_ms=round(latency_ms, 2),
                confidence=0.0,
                finish_reason="error",
            )

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence based on response quality signals.
        """

        if not text:
            return 0.0

        confidence = 0.6

        # Structured reasoning indicators
        if any(
            marker in text.lower()
            for marker in [
                "step",
                "because",
                "therefore",
                "evidence",
                "analysis",
                "root cause",
            ]
        ):
            confidence += 0.1

        # Longer responses tend to be better
        if len(text) > 200:
            confidence += 0.1

        # JSON structure indicator
        if "{" in text and "}" in text:
            confidence += 0.1

        return min(confidence, 0.95)

    def get_metrics(self) -> dict:
        """
        Return client metrics.
        """

        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count > 0
            else 0
        )

        return {
            "model": self.model_name,
            "requests": self._request_count,
            "avg_latency_ms": round(avg_latency, 2),
        }

    async def health_check(self) -> bool:
        """
        Check if Groq API is available.
        """

        try:
            await groq_client.models.list()
            return True

        except Exception as e:

            logger.error(
                "Groq health check failed",
                error=str(e)
            )

            return False