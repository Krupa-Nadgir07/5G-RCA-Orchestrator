"""
Base Agent - Abstract base class for all RCA agents.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import structlog

from models.schemas import AgentMessage, AgentType, MessageType, ProcessedEvent
from services.orchestrator.working_memory import WorkingMemory

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the RCA system.
    Each agent has:
    - A specific role in the reasoning pipeline
    - Access to shared working memory
    - Structured input/output via AgentMessage
    """

    agent_type: AgentType

    def __init__(self):
        self._invocation_count = 0
        self._total_latency_ms = 0.0

    @abstractmethod
    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Execute the agent's analysis. Must be implemented by subclasses."""
        pass

    def _create_message(
        self,
        message_type: MessageType,
        content: dict,
        confidence: float,
        reasoning_steps: list[str],
        metadata: dict = None,
    ) -> AgentMessage:
        """Helper to create a properly structured AgentMessage."""
        return AgentMessage(
            agent_id=self.agent_type,
            timestamp=datetime.utcnow(),
            message_type=message_type,
            content=content,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            metadata=metadata or {},
        )

    def get_metrics(self) -> dict:
        """Return agent metrics."""
        avg_latency = (
            self._total_latency_ms / self._invocation_count
            if self._invocation_count > 0 else 0
        )
        return {
            "agent": self.agent_type.value,
            "invocations": self._invocation_count,
            "avg_latency_ms": round(avg_latency, 2),
        }
