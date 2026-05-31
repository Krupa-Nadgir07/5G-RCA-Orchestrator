"""
Decision Agent - Final decision maker, produces RCA output.
Handles escalation to LLM when confidence is insufficient.
"""

import time
from typing import Optional

import structlog

from config.settings import get_settings
from models.schemas import (
    AgentMessage,
    AgentType,
    MessageType,
    ProcessedEvent,
    RCAResult,
    ReasoningStep,
    RootCauseCategory,
)
from services.inference.service import InferenceService
from services.orchestrator.agents.base import BaseAgent
from services.orchestrator.working_memory import WorkingMemory

logger = structlog.get_logger(__name__)
settings = get_settings()


class DecisionAgent(BaseAgent):
    """
    Final decision maker in the RCA pipeline.
    - Produces the final RCA output with confidence and reasoning trace
    - Decides whether to escalate to LLM based on confidence threshold
    - Compiles the full reasoning trace from all agents
    - Generates recommended actions
    """

    agent_type = AgentType.DECISION

    # Recommended actions for each root cause category
    RECOMMENDED_ACTIONS = {
        RootCauseCategory.INTERFERENCE: [
            "Analyze neighboring cell PCI and frequency reuse pattern",
            "Check for external interference sources (radar, other systems)",
            "Consider ICIC power control adjustments",
            "Evaluate beam management configuration",
            "Check antenna tilt and azimuth settings",
        ],
        RootCauseCategory.HANDOVER_FAILURE: [
            "Review A3 event threshold and hysteresis parameters",
            "Check T310/N310 timer configurations",
            "Verify neighbor cell relations (NRT) completeness",
            "Analyze coverage overlap between source and target cells",
            "Consider adding missing neighbor relations",
        ],
        RootCauseCategory.RESOURCE_CONGESTION: [
            "Evaluate cell split or capacity expansion options",
            "Review QoS priority configurations",
            "Consider load balancing (MLB) parameter adjustments",
            "Analyze traffic patterns for capacity planning",
            "Check for abnormal UE behavior consuming excessive resources",
        ],
        RootCauseCategory.HARDWARE_FAULT: [
            "Inspect physical hardware components (antenna, cables, RRU)",
            "Check environmental conditions (temperature, power supply)",
            "Review hardware alarm history",
            "Schedule maintenance window for component replacement",
            "Verify redundancy and failover mechanisms",
        ],
        RootCauseCategory.SOFTWARE_FAULT: [
            "Check recent software updates or configuration changes",
            "Review system logs for crash dumps or memory issues",
            "Verify software version compatibility",
            "Consider rollback to previous stable version",
            "Engage vendor support with diagnostic logs",
        ],
        RootCauseCategory.CONFIGURATION_ERROR: [
            "Audit recent configuration changes",
            "Compare configuration with baseline/golden template",
            "Verify parameter consistency across related cells",
            "Check for configuration drift from planned values",
            "Validate inter-cell parameter alignment",
        ],
    }

    def __init__(self, inference_service: InferenceService):
        super().__init__()
        self.inference_service = inference_service
        self.confidence_threshold = settings.agent_confidence_threshold
        self.escalation_threshold = settings.agent_escalation_threshold

    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Make final RCA decision."""
        start_time = time.time()
        self._invocation_count += 1

        # Retrieve validated hypotheses
        validated = await memory.retrieve("validated_hypotheses") or []
        
        if not validated:
            return self._create_insufficient_evidence_result(event, memory, start_time)

        top_hypothesis = validated[0]
        validation_score = top_hypothesis.get("validation_score", 0.0)

        # Check if escalation to LLM is needed
        escalated = False
        if validation_score < self.escalation_threshold:
            logger.info("Escalating to LLM", 
                       validation_score=validation_score,
                       threshold=self.escalation_threshold)
            top_hypothesis, escalated = await self._escalate_to_llm(event, memory)

        # Calculate total tokens used across hypothesis & decision phases
        hypothesis_data = await memory.retrieve("hypotheses") or {}
        hyp_tokens = hypothesis_data.get("tokens_used", 0)
        esc_tokens = top_hypothesis.get("tokens_used", 0) if escalated else 0
        total_tokens = hyp_tokens + esc_tokens

        # Extract features for probabilistic confidence calibration
        reasoning_trace = self._compile_reasoning_trace(top_hypothesis)
        num_steps = len(reasoning_trace)

        knowledge_context = await memory.retrieve("knowledge_context") or {}
        contexts = knowledge_context.get("contexts", [])
        scores = []
        for ctx in contexts:
            for chunk in ctx.get("chunks", []):
                if "score" in chunk:
                    scores.append(chunk["score"])
        avg_rag_score = sum(scores) / len(scores) if scores else 0.0

        evidence_count = len(top_hypothesis.get("supporting_evidence", []))

        # Calibrate final confidence
        raw_final_confidence = top_hypothesis.get("validation_score", top_hypothesis.get("confidence", 0.5))
        features = {
            "num_reasoning_steps": num_steps,
            "avg_rag_score": avg_rag_score,
            "evidence_count": evidence_count,
        }
        calibrated_confidence = self.inference_service.calibrator.calibrate(
            raw_final_confidence,
            features=features
        )
        top_hypothesis["calibrated_confidence"] = calibrated_confidence

        # Retrieve signal analysis details
        signal_analysis = await memory.retrieve("signal_analysis") or {}
        anomalies = signal_analysis.get("anomalies", [])
        correlations = signal_analysis.get("correlations", [])

        # Build the RCA result
        rca_result = self._build_rca_result(
            event,
            top_hypothesis,
            memory,
            escalated,
            start_time,
            tokens_used=total_tokens,
            anomalies=anomalies,
            correlations=correlations,
            hypotheses=validated,
        )
        
        # Store final result in working memory
        await memory.store("rca_result", rca_result.model_dump(mode="json"))

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        return self._create_message(
            message_type=MessageType.DECISION,
            content={"rca_result": rca_result.model_dump(mode="json")},
            confidence=rca_result.confidence,
            reasoning_steps=[
                f"Final decision: {rca_result.root_cause.value}",
                f"Confidence: {rca_result.confidence:.2f}",
                f"Escalated to LLM: {escalated}",
            ],
            metadata={"latency_ms": round(latency_ms, 2)},
        )

    async def _escalate_to_llm(self, event: ProcessedEvent, memory: WorkingMemory) -> tuple[dict, bool]:
        """Escalate to LLM for more accurate analysis."""
        full_context = await memory.get_full_context()
        
        escalation_prompt = self._build_escalation_prompt(event, full_context)
        
        response = await self.inference_service.generate(
            prompt=escalation_prompt,
            system_prompt="You are a senior 5G network engineer. Analyze the evidence and provide a root cause determination. Respond with a JSON object containing: root_cause (category), specific_cause (description), confidence (0-1).",
            force_llm=True,
            max_tokens=1024,
        )
        
        if response.text and response.confidence > 0:
            # Parse LLM response
            import json
            try:
                # Try to parse JSON from response
                data = json.loads(response.text)
                return {
                    "root_cause": data.get("root_cause", "unknown"),
                    "specific_cause": data.get("specific_cause", response.text[:200]),
                    "confidence": data.get("confidence", response.confidence),
                    "validation_score": data.get("confidence", response.confidence),
                    "supporting_evidence": data.get("supporting_evidence", []),
                    "model_used": response.model,
                    "tokens_used": response.tokens_used,
                }, True
            except (json.JSONDecodeError, TypeError):
                return {
                    "root_cause": "unknown",
                    "specific_cause": response.text[:200],
                    "confidence": response.confidence,
                    "validation_score": response.confidence,
                    "supporting_evidence": [],
                    "model_used": response.model,
                    "tokens_used": response.tokens_used,
                }, True
        
        # If LLM also fails, return best available hypothesis
        validated = await memory.retrieve("validated_hypotheses") or []
        fallback_res = dict(validated[0]) if validated else {
            "root_cause": "unknown",
            "specific_cause": "Unable to determine root cause",
            "confidence": 0.2,
            "validation_score": 0.2,
        }
        fallback_res["tokens_used"] = response.tokens_used
        return fallback_res, True

    def _build_escalation_prompt(self, event: ProcessedEvent, context: dict) -> str:
        """Build prompt for LLM escalation."""
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        
        parts = [
            f"## 5G gNB RCA Escalation",
            f"Cell: {event.cell_id}, Time: {event.timestamp}",
            f"Event Type: {event.event_type}",
            f"\n## KPIs: {kpi_dict}",
        ]
        
        if "signal_analysis" in context:
            parts.append(f"\n## Signal Analysis: {context['signal_analysis']}")
        if "knowledge_context" in context:
            kc = context["knowledge_context"]
            if isinstance(kc, dict):
                parts.append(f"\n## 3GPP Context: {kc.get('formatted_context', '')[:1000]}")
        if "hypotheses" in context:
            parts.append(f"\n## Prior Hypotheses: {context['hypotheses']}")
        
        return "\n".join(parts)

    def _build_rca_result(
        self, 
        event: ProcessedEvent, 
        top_hypothesis: dict, 
        memory: WorkingMemory, 
        escalated: bool,
        start_time: float,
        tokens_used: int = 0,
        anomalies: list = None,
        correlations: list = None,
        hypotheses: list = None,
    ) -> RCAResult:
        """Build the final RCA result."""
        # Determine root cause category
        root_cause_str = top_hypothesis.get("root_cause", "unknown")
        try:
            root_cause = RootCauseCategory(root_cause_str)
        except ValueError:
            root_cause = RootCauseCategory.UNKNOWN

        # Get recommended actions
        actions = self.RECOMMENDED_ACTIONS.get(root_cause, [
            "Collect additional diagnostic data",
            "Review recent changes in the network",
            "Engage domain expert for further analysis",
        ])

        # Build reasoning trace from hypothesis steps
        reasoning_trace = self._compile_reasoning_trace(top_hypothesis)

        latency_ms = int((time.time() - start_time) * 1000)

        return RCAResult(
            root_cause=root_cause,
            specific_cause=top_hypothesis.get("specific_cause", "Unknown"),
            confidence=top_hypothesis.get(
                "calibrated_confidence",
                min(1.0, max(0.0, top_hypothesis.get("validation_score", top_hypothesis.get("confidence", 0.5))))
            ),
            reasoning_trace=reasoning_trace,
            supporting_evidence=top_hypothesis.get("supporting_evidence", []),
            recommended_actions=actions[:3],
            model_used=top_hypothesis.get("model_used", settings.slm_model),
            escalated=escalated,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            kpis=event.kpis,
            anomalies=anomalies or [],
            correlations=correlations or [],
            hypotheses=hypotheses or [],
            cell_id=event.cell_id,
            gnb_id=event.gnb_id,
            event_ids=[event.event_id],
        )

    def _compile_reasoning_trace(self, hypothesis: dict) -> list[ReasoningStep]:
        """Compile full reasoning trace."""
        steps = []
        
        # Signal analysis step
        steps.append(ReasoningStep(
            step_number=1,
            agent=AgentType.SIGNAL,
            action="Analyze KPI metrics",
            observation="Detected anomalies in network metrics",
            conclusion="KPI anomalies indicate potential issue",
            confidence=0.9,
        ))
        
        # Knowledge retrieval step
        steps.append(ReasoningStep(
            step_number=2,
            agent=AgentType.KNOWLEDGE,
            action="Retrieve 3GPP context",
            observation="Found relevant specifications and patterns",
            conclusion="3GPP standards provide context for diagnosis",
            confidence=0.85,
        ))
        
        # Hypothesis step
        steps.append(ReasoningStep(
            step_number=3,
            agent=AgentType.HYPOTHESIS,
            action="Generate root cause hypotheses",
            observation=f"Primary hypothesis: {hypothesis.get('root_cause', 'unknown')}",
            conclusion=hypothesis.get("specific_cause", "Unknown"),
            confidence=hypothesis.get("confidence", 0.5),
        ))
        
        # Validation step
        validation_details = hypothesis.get("validation_details", {})
        steps.append(ReasoningStep(
            step_number=4,
            agent=AgentType.VALIDATION,
            action="Validate hypothesis against evidence",
            observation=f"Support: {validation_details.get('support_score', 'N/A')}, "
                       f"Contradictions: {validation_details.get('contradiction_score', 'N/A')}",
            conclusion=f"Validation score: {hypothesis.get('validation_score', 'N/A')}",
            confidence=hypothesis.get("validation_score", 0.5),
        ))
        
        # Decision step
        steps.append(ReasoningStep(
            step_number=5,
            agent=AgentType.DECISION,
            action="Final root cause determination",
            observation=f"Root cause: {hypothesis.get('root_cause', 'unknown')}",
            conclusion=hypothesis.get("specific_cause", "Determination made"),
            confidence=hypothesis.get("validation_score", 0.5),
        ))
        
        return steps

    def _create_insufficient_evidence_result(
        self, event: ProcessedEvent, memory: WorkingMemory, start_time: float
    ) -> AgentMessage:
        """Create result when evidence is insufficient."""
        return self._create_message(
            message_type=MessageType.DECISION,
            content={
                "rca_result": RCAResult(
                    root_cause=RootCauseCategory.UNKNOWN,
                    specific_cause="Insufficient evidence for root cause determination",
                    confidence=0.1,
                    reasoning_trace=[],
                    recommended_actions=["Collect additional diagnostic data"],
                    model_used="rule-based",
                    escalated=False,
                    latency_ms=int((time.time() - start_time) * 1000),
                    cell_id=event.cell_id,
                    gnb_id=event.gnb_id,
                ).model_dump(mode="json")
            },
            confidence=0.1,
            reasoning_steps=["Insufficient validated hypotheses for decision"],
        )
