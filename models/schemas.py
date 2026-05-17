"""
Core data models shared across all services.
Defines the canonical data contracts for the RCA system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RootCauseCategory(str, Enum):
    INTERFERENCE = "interference"
    HANDOVER_FAILURE = "handover_failure"
    RESOURCE_CONGESTION = "resource_congestion"
    HARDWARE_FAULT = "hardware_fault"
    SOFTWARE_FAULT = "software_fault"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class AgentType(str, Enum):
    SIGNAL = "signal_agent"
    KNOWLEDGE = "knowledge_agent"
    HYPOTHESIS = "hypothesis_agent"
    VALIDATION = "validation_agent"
    DECISION = "decision_agent"


class MessageType(str, Enum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    DECISION = "decision"


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    NORMAL = "normal"
    INFO = "info"


# =============================================================================
# Log Models
# =============================================================================

class RawLogEvent(BaseModel):
    """Raw log event as received from sources."""
    event_id: UUID = Field(default_factory=uuid4)
    source_id: str
    timestamp: datetime
    log_level: LogLevel = LogLevel.INFO
    raw_message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogMetadata(BaseModel):
    """Metadata associated with a log event."""
    cell_id: Optional[str] = None
    gnb_id: Optional[str] = None
    region: Optional[str] = None
    frequency_band: Optional[str] = None
    bandwidth_mhz: Optional[float] = None


# =============================================================================
# KPI Models
# =============================================================================

class KPIMetrics(BaseModel):
    """Extracted KPI metrics from a log event."""
    sinr_db: Optional[float] = None
    rsrp_dbm: Optional[float] = None
    rsrq_db: Optional[float] = None
    prb_utilization_pct: Optional[float] = None
    bler_pct: Optional[float] = None
    handover_success_rate: Optional[float] = None
    rrc_connection_setup_success_rate: Optional[float] = None
    throughput_dl_mbps: Optional[float] = None
    throughput_ul_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    connected_ues: Optional[int] = None
    cqi_avg: Optional[float] = None
    mcs_dl_avg: Optional[float] = None
    ta_advance_us: Optional[float] = None
    interference_level_dbm: Optional[float] = None


class CellContext(BaseModel):
    """Contextual information about the cell."""
    neighboring_cells: list[str] = Field(default_factory=list)
    frequency_band: Optional[str] = None
    bandwidth_mhz: Optional[float] = None
    antenna_config: Optional[str] = None
    max_ues: Optional[int] = None


# =============================================================================
# Processed Event
# =============================================================================

class ProcessedEvent(BaseModel):
    """Fully processed and enriched log event."""
    event_id: UUID = Field(default_factory=uuid4)
    cell_id: str
    gnb_id: str
    timestamp: datetime
    event_type: Optional[str] = None
    kpis: KPIMetrics
    parsed_template: Optional[str] = None
    severity: Severity = Severity.INFO
    context: CellContext = Field(default_factory=CellContext)
    raw_message: Optional[str] = None
    anomaly_scores: dict[str, float] = Field(default_factory=dict)


# =============================================================================
# Agent Communication Models
# =============================================================================

class AgentMessage(BaseModel):
    """Structured message passed between agents."""
    agent_id: AgentType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: MessageType
    content: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """A single hypothesis about the root cause."""
    root_cause: RootCauseCategory
    specific_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    validation_score: Optional[float] = None
    validation_details: Optional[dict[str, float]] = None


# =============================================================================
# RCA Result Models
# =============================================================================

class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""
    step_number: int
    agent: AgentType
    action: str
    observation: str
    conclusion: str
    confidence: float = Field(ge=0.0, le=1.0)


class RCAResult(BaseModel):
    """Final Root Cause Analysis result."""
    model_config = {"protected_namespaces": ()}
    
    rca_id: UUID = Field(default_factory=uuid4)
    root_cause: RootCauseCategory
    specific_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    model_used: str = ""
    escalated: bool = False
    latency_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cell_id: Optional[str] = None
    gnb_id: Optional[str] = None
    event_ids: list[UUID] = Field(default_factory=list)


# =============================================================================
# RAG Models
# =============================================================================

class RAGQuery(BaseModel):
    """Query for the RAG knowledge service."""
    query: str
    top_k: int = 5
    max_hops: int = 2
    filters: dict[str, Any] = Field(default_factory=dict)


class RAGChunk(BaseModel):
    """A retrieved knowledge chunk."""
    chunk_id: str
    content: str
    source_document: str
    section: Optional[str] = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    """Result from RAG retrieval."""
    query: str
    chunks: list[RAGChunk] = Field(default_factory=list)
    total_tokens: int = 0
    retrieval_time_ms: float = 0.0


# =============================================================================
# API Request/Response Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request to analyze a single event for RCA."""
    cell_id: str
    gnb_id: str = "gnb_unknown"
    timestamp: Optional[datetime] = None
    event_type: Optional[str] = None
    raw_log: Optional[str] = None
    kpis: Optional[KPIMetrics] = None
    severity: Optional[str] = "warning"
    force_llm: bool = False


class AnalyzeResponse(BaseModel):
    """Response for analysis."""
    request_id: str
    result: RCAResult
    status: str = "completed"


class FeedbackRequest(BaseModel):
    """Human feedback on an RCA result."""
    rca_id: UUID
    correct: bool
    correct_root_cause: Optional[RootCauseCategory] = None
    comments: Optional[str] = None
    analyst_id: Optional[str] = None


class BenchmarkRequest(BaseModel):
    """Request to run a benchmark."""
    models: list[str]
    datasets: list[str]
    num_samples: int = 500
    metrics: list[str] = Field(default_factory=lambda: [
        "classification_accuracy",
        "reasoning_accuracy",
        "latency_p95"
    ])


class BenchmarkResult(BaseModel):
    """Result of a benchmark run."""
    benchmark_id: UUID = Field(default_factory=uuid4)
    results: dict[str, dict[str, float]] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Inference Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request for model inference."""
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.3
    model: Optional[str] = None
    system_prompt: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response from model inference."""
    model_config = {"protected_namespaces": ()}
    
    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    confidence: float = 0.0
    finish_reason: str = "stop"
