"""
Hypothesis Generation Agent - Uses CoT reasoning to generate root cause hypotheses.
"""

import json
import time
from typing import Optional

import structlog

from models.schemas import (
    AgentMessage,
    AgentType,
    Hypothesis,
    MessageType,
    ProcessedEvent,
    RootCauseCategory,
)
from services.inference.service import InferenceService
from services.orchestrator.agents.base import BaseAgent
from services.orchestrator.working_memory import WorkingMemory

logger = structlog.get_logger(__name__)


class HypothesisAgent(BaseAgent):
    """
    Generates ranked hypotheses for the root cause using CoT reasoning.
    Leverages SLM with structured prompting to produce explainable hypotheses.
    
    Input: Observations from Signal Agent + Evidence from Knowledge Agent
    Output: Ranked list of hypotheses with supporting evidence
    """

    agent_type = AgentType.HYPOTHESIS

    COT_SYSTEM_PROMPT = """You are a 5G network expert performing root cause analysis on gNB (next-generation Node B) systems.
Your task is to analyze network anomalies and generate hypotheses for the root cause.

You must think step by step and provide structured reasoning.
Always base your analysis on 3GPP standards and telecom domain knowledge.

Output your response as valid JSON with this exact structure:
{
  "reasoning_steps": [
    "Step 1: <analysis>",
    "Step 2: <analysis>",
    "Step 3: <analysis>"
  ],
  "hypotheses": [
    {
      "root_cause": "<category>",
      "specific_cause": "<detailed description>",
      "confidence": <0.0-1.0>,
      "supporting_evidence": ["<evidence1>", "<evidence2>"],
      "contradicting_evidence": ["<evidence1>"]
    }
  ]
}

Root cause categories: interference, handover_failure, resource_congestion, hardware_fault, software_fault, configuration_error, unknown
"""

    COT_USER_PROMPT = """Analyze the following 5G gNB event and generate root cause hypotheses.

## Event Information
- Cell ID: {cell_id}
- Timestamp: {timestamp}
- Event Type: {event_type}

## KPI Observations
{observations}

## Detected Anomalies
{anomalies}

## Correlation Patterns
{correlations}

## 3GPP Knowledge Context
{knowledge_context}

## Known Failure Patterns
{known_patterns}

Think step by step:
Step 1: Identify the primary symptoms from the KPI anomalies
Step 2: Map symptoms to potential root causes using 3GPP knowledge
Step 3: Consider temporal and topological correlations
Step 4: Rank hypotheses by likelihood based on available evidence

Generate your hypotheses now:"""

    def __init__(self, inference_service: InferenceService):
        super().__init__()
        self.inference_service = inference_service

    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Generate root cause hypotheses using CoT reasoning."""
        start_time = time.time()
        self._invocation_count += 1

        # Retrieve context from working memory
        signal_analysis = await memory.retrieve("signal_analysis") or {}
        knowledge_context = await memory.retrieve("knowledge_context") or {}

        # Construct the prompt
        prompt = self._build_prompt(event, signal_analysis, knowledge_context)

        # Generate hypotheses using SLM
        response = await self.inference_service.generate(
            prompt=prompt,
            system_prompt=self.COT_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.3,
        )

        print("RAW LLM RESPONSE:")
        print(response)

        # Parse the response
        hypotheses, reasoning_steps = self._parse_response(response.text)

        # If parsing fails, create a rule-based hypothesis
        if not hypotheses:
            hypotheses = self._fallback_hypothesis(event, signal_analysis)
            reasoning_steps = ["Fallback: Generated rule-based hypothesis due to parsing failure"]

        # Store in working memory
        hypothesis_result = {
            "hypotheses": [h.model_dump() for h in hypotheses],
            "model_used": response.model,
            "model_confidence": response.confidence,
            "tokens_used": response.tokens_used,
        }
        await memory.store("hypotheses", hypothesis_result)

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        # Confidence is the max hypothesis confidence
        max_confidence = max((h.confidence for h in hypotheses), default=0.5)

        return self._create_message(
            message_type=MessageType.HYPOTHESIS,
            content={"hypotheses": [h.model_dump() for h in hypotheses]},
            confidence=max_confidence,
            reasoning_steps=reasoning_steps,
            metadata={
                "latency_ms": round(latency_ms, 2),
                "model_used": response.model,
                "num_hypotheses": len(hypotheses),
            },
        )

    def _build_prompt(self, event: ProcessedEvent, signal_analysis: dict, knowledge_context: dict) -> str:
        """Build the CoT prompt with all available context."""
        # Format observations
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        observations = "\n".join(f"- {k}: {v}" for k, v in kpi_dict.items()) or "No KPIs available"

        # Append rich scenario flags from parsed_template if present
        # (injected by the benchmark's scenario_to_event for structured evaluation)
        if event.parsed_template:
            observations += "\n\n## Scenario-Specific Degradation Flags\n" + event.parsed_template

        # Format anomalies
        anomalies_list = signal_analysis.get("anomalies", [])
        anomalies = "\n".join(
            f"- {a.get('description', a.get('kpi', 'unknown'))}" for a in anomalies_list
        ) or "No anomalies detected"

        # Format correlations
        correlations_list = signal_analysis.get("correlations", [])
        correlations = "\n".join(
            f"- {c.get('description', c.get('pattern', 'unknown'))}" for c in correlations_list
        ) or "No correlation patterns detected"

        # Format knowledge context
        ctx = knowledge_context.get("formatted_context", "No relevant 3GPP context available")
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."

        # Format known patterns
        patterns = knowledge_context.get("known_patterns", [])
        known_patterns = "\n".join(f"- {p}" for p in patterns) or "No known pattern matches"

        return self.COT_USER_PROMPT.format(
            cell_id=event.cell_id,
            timestamp=str(event.timestamp),
            event_type=event.event_type or "unknown",
            observations=observations,
            anomalies=anomalies,
            correlations=correlations,
            knowledge_context=ctx,
            known_patterns=known_patterns,
        )

    def _parse_response(self, response_text: str) -> tuple[list[Hypothesis], list[str]]:
        """Parse the LLM response into structured hypotheses."""
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response_text)
            if json_str:
                data = json.loads(json_str)
            else:
                data = json.loads(response_text)

            reasoning_steps = data.get("reasoning_steps", [])
            hypotheses = []

            for h in data.get("hypotheses", []):
                root_cause = h.get("root_cause", "unknown")
                # Map to enum
                try:
                    category = RootCauseCategory(root_cause)
                except ValueError:
                    category = RootCauseCategory.UNKNOWN

                hypotheses.append(Hypothesis(
                    root_cause=category,
                    specific_cause=h.get("specific_cause", "Unknown cause"),
                    confidence=min(1.0, max(0.0, float(h.get("confidence", 0.5)))),
                    supporting_evidence=h.get("supporting_evidence", []),
                    contradicting_evidence=h.get("contradicting_evidence", []),
                ))

            return hypotheses, reasoning_steps

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse hypothesis response", error=str(e))
            return [], []

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text that might have surrounding content."""
        # Try to find JSON block
        start_markers = ["{", "```json\n{", "```\n{"]
        
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # Find the actual { start
                json_start = text.find("{", idx)
                if json_start == -1:
                    continue
                
                # Find matching closing brace
                depth = 0
                for i in range(json_start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return text[json_start:i + 1]
        
        return None

    def _fallback_hypothesis(self, event: ProcessedEvent, signal_analysis: dict) -> list[Hypothesis]:
        """Generate rule-based hypothesis when SLM parsing fails."""
        hypotheses = []
        
        anomalies = signal_analysis.get("anomalies", [])
        correlations = signal_analysis.get("correlations", [])
        
        # Rule-based hypothesis generation
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        
        if kpi_dict.get("sinr_db", 100) < 5 and kpi_dict.get("interference_level_dbm", -200) > -95:
            hypotheses.append(Hypothesis(
                root_cause=RootCauseCategory.INTERFERENCE,
                specific_cause="High interference causing low SINR",
                confidence=0.7,
                supporting_evidence=[
                    f"SINR={kpi_dict.get('sinr_db')}dB below threshold",
                    f"Interference level={kpi_dict.get('interference_level_dbm')}dBm above normal",
                ],
            ))
        
        if kpi_dict.get("handover_success_rate", 1.0) < 0.85:
            hypotheses.append(Hypothesis(
                root_cause=RootCauseCategory.HANDOVER_FAILURE,
                specific_cause="Handover success rate below acceptable threshold",
                confidence=0.65,
                supporting_evidence=[
                    f"Handover success rate={kpi_dict.get('handover_success_rate')}",
                ],
            ))
        
        if kpi_dict.get("prb_utilization_pct", 0) > 90:
            hypotheses.append(Hypothesis(
                root_cause=RootCauseCategory.RESOURCE_CONGESTION,
                specific_cause="Cell resource exhaustion due to high PRB utilization",
                confidence=0.7,
                supporting_evidence=[
                    f"PRB utilization={kpi_dict.get('prb_utilization_pct')}% above critical threshold",
                ],
            ))
        
        # Default hypothesis if nothing specific detected
        if not hypotheses:
            hypotheses.append(Hypothesis(
                root_cause=RootCauseCategory.UNKNOWN,
                specific_cause="Insufficient evidence for definitive root cause determination",
                confidence=0.3,
                supporting_evidence=["Event type: " + (event.event_type or "unknown")],
            ))
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses
