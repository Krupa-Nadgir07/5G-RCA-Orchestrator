"""
Agent Orchestrator Graph - LangGraph-style directed graph for agent coordination.
Defines the execution flow between agents with conditional routing.
"""

import time
from typing import Optional
from uuid import uuid4

import structlog

from config.settings import get_settings
from models.schemas import AgentMessage, ProcessedEvent, RCAResult, RootCauseCategory
from services.inference.service import InferenceService
from services.orchestrator.agents.decision_agent import DecisionAgent
from services.orchestrator.agents.hypothesis_agent import HypothesisAgent
from services.orchestrator.agents.knowledge_agent import KnowledgeRetrievalAgent
from services.orchestrator.agents.signal_agent import SignalAnalysisAgent
from services.orchestrator.agents.validation_agent import ValidationAgent
from services.orchestrator.working_memory import WorkingMemory
from services.rag.service import RAGService

logger = structlog.get_logger(__name__)
settings = get_settings()


class OrchestratorGraph:
    """
    Directed graph orchestrator for multi-agent RCA.
    
    Execution Flow:
    1. Signal Agent (analyze KPIs)
    2. Knowledge Agent + Context Agent (parallel)
    3. Hypothesis Agent (generate hypotheses using CoT)
    4. Validation Agent (validate against evidence)
    5. Decision Agent (final output, may escalate to LLM)
    
    The graph supports conditional routing:
    - If Signal Agent finds no anomalies → skip to rule-based result
    - If Validation confidence < threshold → escalate in Decision Agent
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        inference_service: Optional[InferenceService] = None,
    ):
        self.rag_service = rag_service or RAGService()
        self.inference_service = inference_service or InferenceService()
        
        # Initialize agents
        self.signal_agent = SignalAnalysisAgent()
        self.knowledge_agent = KnowledgeRetrievalAgent(self.rag_service)
        self.hypothesis_agent = HypothesisAgent(self.inference_service)
        self.validation_agent = ValidationAgent()
        self.decision_agent = DecisionAgent(self.inference_service)
        
        self._execution_count = 0
        self._total_latency_ms = 0.0

    async def initialize(self):
        """Initialize all services."""
        await self.rag_service.initialize()
        logger.info("Orchestrator graph initialized")

    async def execute(self, event: ProcessedEvent, force_llm: bool = False) -> RCAResult:
        """
        Execute the full RCA pipeline for a processed event.
        
        Args:
            event: The processed log event with KPIs
            force_llm: Force LLM usage (skip SLM routing)
            
        Returns:
            RCAResult with root cause, confidence, and reasoning trace
        """
        start_time = time.time()
        session_id = str(uuid4())
        self._execution_count += 1
        
        logger.info("Starting RCA pipeline",
                   session_id=session_id,
                   cell_id=event.cell_id,
                   event_type=event.event_type)

        # Initialize working memory for this session
        memory = WorkingMemory(session_id)
        await memory.initialize()

        try:
            # Store event context in working memory
            await memory.store("event", event.model_dump(mode="json"))

            # === Stage 1: Signal Analysis ===
            signal_result = await self.signal_agent.execute(event, memory)
            logger.debug("Signal analysis complete", 
                        confidence=signal_result.confidence,
                        anomalies=len(signal_result.content.get("anomalies", [])))

            # Early exit if no anomalies detected
            if not signal_result.content.get("anomalies"):
                logger.info("No anomalies detected, returning clean result")
                return self._create_clean_result(event, start_time)

            # === Stage 2: Knowledge Retrieval (parallel-capable) ===
            knowledge_result = await self.knowledge_agent.execute(event, memory)
            logger.debug("Knowledge retrieval complete",
                        confidence=knowledge_result.confidence,
                        sources=knowledge_result.content.get("num_sources", 0))

            # === Stage 3: Hypothesis Generation ===
            hypothesis_result = await self.hypothesis_agent.execute(event, memory)
            logger.debug("Hypothesis generation complete",
                        confidence=hypothesis_result.confidence,
                        num_hypotheses=hypothesis_result.metadata.get("num_hypotheses", 0))

            # === Stage 4: Validation ===
            validation_result = await self.validation_agent.execute(event, memory)
            logger.debug("Validation complete",
                        confidence=validation_result.confidence)

            # === Stage 5: Decision ===
            decision_result = await self.decision_agent.execute(event, memory)
            
            # Extract RCA result from decision
            rca_data = decision_result.content.get("rca_result", {})
            rca_result = RCAResult(**rca_data)
            
            # Update latency with full pipeline time
            total_latency_ms = int((time.time() - start_time) * 1000)
            rca_result.latency_ms = total_latency_ms
            self._total_latency_ms += total_latency_ms

            logger.info("RCA pipeline complete",
                       session_id=session_id,
                       root_cause=rca_result.root_cause.value,
                       confidence=rca_result.confidence,
                       latency_ms=total_latency_ms,
                       escalated=rca_result.escalated)

            return rca_result

        except Exception as e:
            logger.error("RCA pipeline failed", 
                        session_id=session_id, error=str(e))
            return self._create_error_result(event, str(e), start_time)
        
        finally:
            await memory.clear()
            await memory.close()

    def _create_clean_result(self, event: ProcessedEvent, start_time: float) -> RCAResult:
        """Create result when no issues detected."""
        return RCAResult(
            root_cause=RootCauseCategory.UNKNOWN,
            specific_cause="No anomalies detected - system operating within normal parameters",
            confidence=0.95,
            reasoning_trace=[],
            supporting_evidence=["All KPIs within normal thresholds"],
            recommended_actions=["Continue monitoring"],
            model_used="rule-based",
            escalated=False,
            latency_ms=int((time.time() - start_time) * 1000),
            cell_id=event.cell_id,
            gnb_id=event.gnb_id,
            event_ids=[event.event_id],
        )

    def _create_error_result(self, event: ProcessedEvent, error: str, start_time: float) -> RCAResult:
        """Create result when pipeline fails."""
        return RCAResult(
            root_cause=RootCauseCategory.UNKNOWN,
            specific_cause=f"Analysis failed: {error[:200]}",
            confidence=0.0,
            reasoning_trace=[],
            supporting_evidence=[],
            recommended_actions=["Manual investigation required", "Check system logs"],
            model_used="error",
            escalated=False,
            latency_ms=int((time.time() - start_time) * 1000),
            cell_id=event.cell_id,
            gnb_id=event.gnb_id,
            event_ids=[event.event_id],
        )

    def get_metrics(self) -> dict:
        """Return orchestrator metrics."""
        avg_latency = (
            self._total_latency_ms / self._execution_count
            if self._execution_count > 0 else 0
        )
        return {
            "total_executions": self._execution_count,
            "avg_latency_ms": round(avg_latency, 2),
            "inference": self.inference_service.get_metrics(),
            "agents": {
                "signal": self.signal_agent.get_metrics(),
                "knowledge": self.knowledge_agent.get_metrics(),
                "hypothesis": self.hypothesis_agent.get_metrics(),
                "validation": self.validation_agent.get_metrics(),
                "decision": self.decision_agent.get_metrics(),
            },
        }
