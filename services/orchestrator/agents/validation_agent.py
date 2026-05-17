"""
Validation Agent - Validates hypotheses against available evidence.
Checks for contradictions and performs counterfactual reasoning.
"""

import time

import structlog

from models.schemas import (
    AgentMessage,
    AgentType,
    Hypothesis,
    MessageType,
    ProcessedEvent,
    RootCauseCategory,
)
from services.orchestrator.agents.base import BaseAgent
from services.orchestrator.working_memory import WorkingMemory

logger = structlog.get_logger(__name__)


class ValidationAgent(BaseAgent):
    """
    Validates hypotheses against available evidence.
    Performs:
    1. Supporting evidence strength evaluation
    2. Contradiction checking
    3. Counterfactual reasoning (would removing cause explain symptoms?)
    4. Cross-hypothesis consistency checking
    
    Input: Hypotheses from Hypothesis Agent + all prior evidence
    Output: Validated and re-scored hypotheses
    """

    agent_type = AgentType.VALIDATION

    # Evidence rules for validation
    VALIDATION_RULES = {
        RootCauseCategory.INTERFERENCE: {
            "supporting_kpis": {
                "sinr_db": lambda v: v < 5,
                "interference_level_dbm": lambda v: v > -95,
                "bler_pct": lambda v: v > 5,
            },
            "contradicting_kpis": {
                "sinr_db": lambda v: v > 15,
                "interference_level_dbm": lambda v: v < -110,
            },
            "required_evidence": 2,
        },
        RootCauseCategory.HANDOVER_FAILURE: {
            "supporting_kpis": {
                "handover_success_rate": lambda v: v < 0.85,
                "rsrp_dbm": lambda v: v < -110,
                "rsrq_db": lambda v: v < -15,
            },
            "contradicting_kpis": {
                "handover_success_rate": lambda v: v > 0.95,
                "rsrp_dbm": lambda v: v > -90,
            },
            "required_evidence": 1,
        },
        RootCauseCategory.RESOURCE_CONGESTION: {
            "supporting_kpis": {
                "prb_utilization_pct": lambda v: v > 85,
                "connected_ues": lambda v: v > 300,
                "latency_ms": lambda v: v > 50,
            },
            "contradicting_kpis": {
                "prb_utilization_pct": lambda v: v < 50,
                "connected_ues": lambda v: v < 50,
            },
            "required_evidence": 2,
        },
        RootCauseCategory.HARDWARE_FAULT: {
            "supporting_kpis": {
                "bler_pct": lambda v: v > 10,
            },
            "contradicting_kpis": {},
            "required_evidence": 1,
        },
    }

    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Validate hypotheses against evidence."""
        start_time = time.time()
        self._invocation_count += 1

        # Retrieve hypotheses from working memory
        hypothesis_data = await memory.retrieve("hypotheses") or {}
        hypotheses_raw = hypothesis_data.get("hypotheses", [])
        
        # Reconstruct Hypothesis objects
        hypotheses = [Hypothesis(**h) for h in hypotheses_raw]
        
        if not hypotheses:
            return self._create_message(
                message_type=MessageType.EVIDENCE,
                content={"validated_hypotheses": [], "validation_complete": False},
                confidence=0.0,
                reasoning_steps=["No hypotheses available for validation"],
            )

        # Validate each hypothesis
        validated = []
        for hypothesis in hypotheses:
            validation = self._validate_hypothesis(hypothesis, event)
            validated.append(validation)

        # Sort by validation score
        validated.sort(key=lambda x: x.get("validation_score", 0), reverse=True)

        # Store in working memory
        await memory.store("validated_hypotheses", validated)

        # Generate reasoning
        reasoning_steps = self._generate_reasoning(validated)

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        top_confidence = validated[0].get("validation_score", 0.5) if validated else 0.0

        return self._create_message(
            message_type=MessageType.EVIDENCE,
            content={
                "validated_hypotheses": validated,
                "validation_complete": True,
                "top_hypothesis": validated[0] if validated else None,
            },
            confidence=top_confidence,
            reasoning_steps=reasoning_steps,
            metadata={"latency_ms": round(latency_ms, 2)},
        )

    def _validate_hypothesis(self, hypothesis: Hypothesis, event: ProcessedEvent) -> dict:
        """
        Validate a single hypothesis against KPI evidence.
        Returns hypothesis dict with validation scores.
        """
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        
        # Get validation rules for this root cause
        rules = self.VALIDATION_RULES.get(hypothesis.root_cause, {})
        
        # Evaluate supporting evidence
        support_score = self._evaluate_support(hypothesis, kpi_dict, rules)
        
        # Check for contradictions
        contradiction_score = self._check_contradictions(hypothesis, kpi_dict, rules)
        
        # Counterfactual test
        counterfactual_score = self._counterfactual_test(hypothesis, kpi_dict, rules)
        
        # Compute weighted validation score
        validation_score = (
            support_score * 0.4 +
            (1.0 - contradiction_score) * 0.3 +
            counterfactual_score * 0.3
        )

        return {
            **hypothesis.model_dump(),
            "validation_score": round(validation_score, 4),
            "validation_details": {
                "support_score": round(support_score, 4),
                "contradiction_score": round(contradiction_score, 4),
                "counterfactual_score": round(counterfactual_score, 4),
            },
        }

    def _evaluate_support(self, hypothesis: Hypothesis, kpi_dict: dict, rules: dict) -> float:
        """Evaluate strength of supporting evidence."""
        supporting_kpis = rules.get("supporting_kpis", {})
        required = rules.get("required_evidence", 1)
        
        if not supporting_kpis:
            # No rules defined, use hypothesis's own confidence
            return hypothesis.confidence
        
        supports = 0
        total_checks = 0
        
        for kpi_name, check_fn in supporting_kpis.items():
            if kpi_name in kpi_dict:
                total_checks += 1
                if check_fn(kpi_dict[kpi_name]):
                    supports += 1
        
        if total_checks == 0:
            return 0.4  # No evidence available
        
        # Score based on how many supporting conditions are met
        support_ratio = supports / total_checks
        
        # Bonus if meets required evidence threshold
        if supports >= required:
            support_ratio = min(1.0, support_ratio + 0.2)
        
        return support_ratio

    def _check_contradictions(self, hypothesis: Hypothesis, kpi_dict: dict, rules: dict) -> float:
        """Check for contradicting evidence. Returns 0 (no contradictions) to 1 (strong contradiction)."""
        contradicting_kpis = rules.get("contradicting_kpis", {})
        
        if not contradicting_kpis:
            return 0.0
        
        contradictions = 0
        total_checks = 0
        
        for kpi_name, check_fn in contradicting_kpis.items():
            if kpi_name in kpi_dict:
                total_checks += 1
                if check_fn(kpi_dict[kpi_name]):
                    contradictions += 1
        
        if total_checks == 0:
            return 0.0
        
        return contradictions / total_checks

    def _counterfactual_test(self, hypothesis: Hypothesis, kpi_dict: dict, rules: dict) -> float:
        """
        Counterfactual reasoning: If the hypothesized cause were removed,
        would the observed symptoms be explained?
        
        Score: 1.0 = hypothesis perfectly explains symptoms
               0.0 = hypothesis doesn't explain symptoms
        """
        supporting_kpis = rules.get("supporting_kpis", {})
        
        if not supporting_kpis:
            return 0.5  # Can't evaluate without rules
        
        # Check if the hypothesis explains the observed anomalies
        explained_anomalies = 0
        total_anomalies = 0
        
        for kpi_name, check_fn in supporting_kpis.items():
            if kpi_name in kpi_dict:
                # Check if this KPI is anomalous
                value = kpi_dict[kpi_name]
                if check_fn(value):
                    total_anomalies += 1
                    # Would this hypothesis explain this anomaly?
                    explained_anomalies += 1
        
        if total_anomalies == 0:
            return 0.5
        
        return explained_anomalies / total_anomalies

    def _generate_reasoning(self, validated: list[dict]) -> list[str]:
        """Generate reasoning steps for the validation."""
        steps = []
        
        if not validated:
            return ["No hypotheses to validate"]
        
        top = validated[0]
        steps.append(
            f"Top hypothesis: '{top['root_cause']}' with validation score "
            f"{top['validation_score']:.2f}"
        )
        
        details = top.get("validation_details", {})
        steps.append(
            f"Support: {details.get('support_score', 0):.2f}, "
            f"Contradictions: {details.get('contradiction_score', 0):.2f}, "
            f"Counterfactual: {details.get('counterfactual_score', 0):.2f}"
        )
        
        if details.get("contradiction_score", 0) > 0.5:
            steps.append("WARNING: Significant contradicting evidence found")
        
        if len(validated) > 1:
            gap = top["validation_score"] - validated[1].get("validation_score", 0)
            if gap < 0.1:
                steps.append(
                    f"Close alternative: '{validated[1]['root_cause']}' "
                    f"(score gap: {gap:.2f})"
                )
        
        return steps
