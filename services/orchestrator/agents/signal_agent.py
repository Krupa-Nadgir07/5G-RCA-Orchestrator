"""
Signal Analysis Agent - Interprets KPIs and detects anomalies.
First agent in the RCA pipeline.
"""

import time

import structlog

from models.schemas import AgentMessage, AgentType, MessageType, ProcessedEvent, Severity
from services.orchestrator.agents.base import BaseAgent
from services.orchestrator.working_memory import WorkingMemory

logger = structlog.get_logger(__name__)


class SignalAnalysisAgent(BaseAgent):
    """
    Analyzes raw KPIs, detects anomalies, identifies trends,
    and computes cross-KPI correlations.
    
    Input: ProcessedEvent with KPIs
    Output: Anomaly report with severity scores and initial observations
    """

    agent_type = AgentType.SIGNAL

    # KPI thresholds based on 3GPP recommendations
    THRESHOLDS = {
        "sinr_db": {"critical": 0, "warning": 5, "normal": 10},
        "rsrp_dbm": {"critical": -130, "warning": -120, "normal": -100},
        "rsrq_db": {"critical": -20, "warning": -15, "normal": -10},
        "prb_utilization_pct": {"critical": 95, "warning": 85, "normal": 70},
        "bler_pct": {"critical": 10, "warning": 5, "normal": 2},
        "handover_success_rate": {"critical": 0.7, "warning": 0.85, "normal": 0.95},
        "latency_ms": {"critical": 100, "warning": 50, "normal": 20},
        "interference_level_dbm": {"critical": -85, "warning": -95, "normal": -105},
        "connected_ues": {"critical": 500, "warning": 300, "normal": 200},
    }

    # Known correlation patterns between KPIs
    CORRELATION_PATTERNS = [
        {
            "pattern": "interference_congestion",
            "conditions": {
                "sinr_db": lambda v: v is not None and v < 5,
                "prb_utilization_pct": lambda v: v is not None and v > 85,
            },
            "description": "Low SINR combined with high PRB utilization suggests interference-induced congestion",
        },
        {
            "pattern": "handover_signal_quality",
            "conditions": {
                "handover_success_rate": lambda v: v is not None and v < 0.85,
                "rsrp_dbm": lambda v: v is not None and v < -110,
            },
            "description": "Low handover success rate with weak signal suggests coverage gap at cell edge",
        },
        {
            "pattern": "overload_degradation",
            "conditions": {
                "prb_utilization_pct": lambda v: v is not None and v > 90,
                "connected_ues": lambda v: v is not None and v > 300,
                "latency_ms": lambda v: v is not None and v > 50,
            },
            "description": "High PRB usage + many UEs + high latency indicates cell overload",
        },
        {
            "pattern": "interference_only",
            "conditions": {
                "sinr_db": lambda v: v is not None and v < 3,
                "interference_level_dbm": lambda v: v is not None and v > -90,
            },
            "description": "Low SINR with high interference level indicates external interference source",
        },
    ]

    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Analyze signal metrics and detect anomalies."""
        start_time = time.time()
        self._invocation_count += 1

        # Detect anomalies
        anomalies = self._detect_anomalies(event)
        
        # Detect correlation patterns
        correlations = self._detect_correlations(event)
        
        # Compute overall severity
        severity = self._compute_severity(anomalies)
        
        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning(event, anomalies, correlations)

        # Store in working memory
        analysis_result = {
            "anomalies": anomalies,
            "correlations": correlations,
            "severity": severity,
            "kpi_summary": self._summarize_kpis(event),
        }
        await memory.store("signal_analysis", analysis_result)

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        return self._create_message(
            message_type=MessageType.OBSERVATION,
            content=analysis_result,
            confidence=0.9 if anomalies else 0.5,
            reasoning_steps=reasoning_steps,
            metadata={"latency_ms": round(latency_ms, 2)},
        )

    def _detect_anomalies(self, event: ProcessedEvent) -> list[dict]:
        """Detect KPI anomalies based on thresholds."""
        anomalies = []
        kpi_dict = event.kpis.model_dump(exclude_none=True)

        for kpi_name, value in kpi_dict.items():
            if kpi_name not in self.THRESHOLDS:
                continue
            
            thresholds = self.THRESHOLDS[kpi_name]
            level = self._classify_level(kpi_name, value, thresholds)
            
            if level in ("critical", "warning"):
                anomalies.append({
                    "kpi": kpi_name,
                    "value": value,
                    "level": level,
                    "threshold": thresholds[level],
                    "description": f"{kpi_name}={value} is at {level} level (threshold: {thresholds[level]})",
                })

        return anomalies

    def _classify_level(self, kpi_name: str, value: float, thresholds: dict) -> str:
        """Classify KPI value into severity level."""
        # KPIs where lower is worse
        lower_is_worse = {"sinr_db", "rsrp_dbm", "rsrq_db", "handover_success_rate", 
                         "rrc_connection_setup_success_rate"}
        
        if kpi_name in lower_is_worse:
            if value <= thresholds["critical"]:
                return "critical"
            elif value <= thresholds["warning"]:
                return "warning"
            return "normal"
        elif kpi_name == "interference_level_dbm":
            # Higher (less negative) is worse
            if value >= thresholds["critical"]:
                return "critical"
            elif value >= thresholds["warning"]:
                return "warning"
            return "normal"
        else:
            # Higher is worse (prb_util, bler, latency, ues)
            if value >= thresholds["critical"]:
                return "critical"
            elif value >= thresholds["warning"]:
                return "warning"
            return "normal"

    def _detect_correlations(self, event: ProcessedEvent) -> list[dict]:
        """Detect known correlation patterns between KPIs."""
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        detected = []

        for pattern in self.CORRELATION_PATTERNS:
            all_met = True
            for kpi_name, condition_fn in pattern["conditions"].items():
                value = kpi_dict.get(kpi_name)
                if not condition_fn(value):
                    all_met = False
                    break
            
            if all_met:
                detected.append({
                    "pattern": pattern["pattern"],
                    "description": pattern["description"],
                })

        return detected

    def _compute_severity(self, anomalies: list[dict]) -> str:
        """Compute overall severity from detected anomalies."""
        if any(a["level"] == "critical" for a in anomalies):
            return "critical"
        elif any(a["level"] == "warning" for a in anomalies):
            return "warning"
        return "normal"

    def _summarize_kpis(self, event: ProcessedEvent) -> dict:
        """Create a summary of KPI values."""
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        return {
            "cell_id": event.cell_id,
            "timestamp": str(event.timestamp),
            "kpis": kpi_dict,
            "num_anomalies": len(event.anomaly_scores),
        }

    def _generate_reasoning(self, event: ProcessedEvent, anomalies: list, correlations: list) -> list[str]:
        """Generate human-readable reasoning steps."""
        steps = []
        
        kpi_dict = event.kpis.model_dump(exclude_none=True)
        if kpi_dict:
            kpi_summary = ", ".join(f"{k}={v}" for k, v in list(kpi_dict.items())[:5])
            steps.append(f"Analyzed KPI values: {kpi_summary}")
        
        if anomalies:
            anomaly_desc = "; ".join(a["description"] for a in anomalies[:3])
            steps.append(f"Detected {len(anomalies)} anomalies: {anomaly_desc}")
        else:
            steps.append("No KPI anomalies detected above threshold levels")
        
        if correlations:
            for corr in correlations[:2]:
                steps.append(f"Correlation pattern detected: {corr['description']}")
        
        return steps
