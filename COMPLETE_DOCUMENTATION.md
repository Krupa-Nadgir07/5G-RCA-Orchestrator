# COMPLETE SYSTEM DOCUMENTATION
# Multi-Agent Orchestrator for Automated Root Cause Analysis in 5G gNB

> **Analysis Date:** May 7, 2026  
> **Codebase:** `c:\Users\pnadgir\Desktop\Major_Project\Paper_project\Agent`  
> **Language:** Python 3.11+  
> **Framework:** FastAPI + LangGraph-style multi-agent orchestration

---

## TABLE OF CONTENTS

1. [System Overview & Problem Statement](#1-system-overview--problem-statement)
2. [Software Requirements Specification (SRS)](#2-software-requirements-specification-srs)
3. [High-Level Architecture (HLD)](#3-high-level-architecture-hld)
4. [Low-Level Design (LLD)](#4-low-level-design-lld)
5. [Data Flow Analysis](#5-data-flow-analysis)
6. [Module-by-Module Documentation](#6-module-by-module-documentation)
7. [Core Algorithms & AI/ML Insights](#7-core-algorithms--aiml-insights)
8. [Diagrams](#8-diagrams)
9. [End-to-End Execution Walkthrough](#9-end-to-end-execution-walkthrough)
10. [Configuration & Environment](#10-configuration--environment)
11. [Design Decisions & Justifications](#11-design-decisions--justifications)
12. [Testing Strategy](#12-testing-strategy)
13. [Improvement Suggestions](#13-improvement-suggestions)

---

## 1. SYSTEM OVERVIEW & PROBLEM STATEMENT

### 1.1 What Problem Does This Solve?

Modern 5G networks are composed of thousands of gNodeBs (gNBs — the base stations in 5G New Radio). Each gNB continuously emits logs, KPIs (Key Performance Indicators), alarms, and measurement reports. When a performance degradation occurs — e.g., a sudden spike in Block Error Rate (BLER), a drop in Signal-to-Interference-plus-Noise Ratio (SINR), or a wave of failed handovers — a **network engineer** must manually analyze logs across multiple systems, cross-reference 3GPP standards, and determine the root cause.

This is:
- **Slow** (manual analysis takes hours)
- **Expensive** (requires highly-trained telecom engineers)
- **Error-prone** (humans miss subtle multi-KPI correlations)
- **Not scalable** (one analyst cannot monitor thousands of cells simultaneously)

**This system automates that process.** It ingests raw 5G gNB log events, extracts KPI metrics, coordinates five specialized AI agents, retrieves relevant 3GPP knowledge, generates structured chain-of-thought reasoning, and produces a human-readable root cause analysis in under 2 seconds for the majority of cases.

### 1.2 Primary Use Cases

| Use Case | Description | User |
|----------|-------------|------|
| UC-1 | Automated RCA on streaming log events | NOC Operations |
| UC-2 | Batch RCA on historical log archives | Network Planning |
| UC-3 | Benchmark SLM vs LLM accuracy for 5G RCA tasks | AI/ML Researchers |
| UC-4 | Human feedback for model improvement | Network Engineers |
| UC-5 | 3GPP knowledge base management | System Admin |

### 1.3 System Identity

| Property | Value |
|----------|-------|
| System Name | Multi-Agent RCA System |
| Domain | 5G Telecom — Network Fault Management |
| Architecture | Event-driven microservices + multi-agent AI |
| Primary Language | Python 3.11 |
| Entry Point | `main.py` → `services/api/app.py` |
| API Style | REST (FastAPI), async |

---

## 2. SOFTWARE REQUIREMENTS SPECIFICATION (SRS)

### 2.1 Functional Requirements

#### FR-01: Log Ingestion
- Accept raw log events from REST API (single and batch)
- Accept log events from Apache Kafka streaming topic
- Rate limit ingestion: 100 requests/min per IP (REST), 10K events/sec per source (Kafka)
- Schema validation of all incoming events

#### FR-02: Log Preprocessing
- Parse semi-structured 5G gNB logs using Drain3 (ML-based template mining) and regex patterns
- Extract at least 15 KPI metrics per event (SINR, RSRP, RSRQ, PRB utilization, BLER, HO success rate, throughput, latency, connected UEs, CQI, MCS, TA advance, interference level)
- Detect event types: RRC failure, handover, interference, PRB congestion, radio link failure, beam failure, SINR alarm, hardware alarm, KPI report
- Compute anomaly scores (0.0–1.0) for each KPI
- Determine overall event severity (INFO, NORMAL, WARNING, CRITICAL)

#### FR-03: RAG Knowledge Retrieval
- Maintain knowledge base of 3GPP specifications and telecom documents
- Perform hybrid retrieval (dense vector via Qdrant + sparse BM25 via Elasticsearch)
- Apply Reciprocal Rank Fusion (RRF) to merge dense and sparse results
- Apply cross-encoder re-ranking (BAAI/bge-reranker-v2-m3) for final ranking
- Support multi-hop retrieval (up to 2 hops) for complex queries
- Cache retrieval results for 1 hour (in-memory dict)

#### FR-04: Multi-Agent RCA Pipeline
- Signal Analysis Agent: Detect KPI anomalies and cross-KPI correlation patterns
- Knowledge Retrieval Agent: Fetch relevant 3GPP context for detected anomalies
- Hypothesis Agent: Generate ranked root-cause hypotheses using CoT + SLM
- Validation Agent: Validate hypotheses via supporting evidence, contradiction checking, counterfactual reasoning
- Decision Agent: Produce final RCA with confidence, reasoning trace, recommended actions
- Tiered inference: Phi-3-mini → Mistral-7B → GPT-4o (with automatic escalation)

#### FR-05: RCA Output
- Output: root cause category, specific cause description, confidence (0–1), full CoT reasoning trace, supporting evidence list, recommended actions
- Root cause categories: interference, handover_failure, resource_congestion, hardware_fault, software_fault, configuration_error, unknown
- Recommended actions: 3–5 domain-specific remediation steps per category

#### FR-06: Evaluation Framework
- Support 3 benchmark datasets: synthetic_5g, itu_challenge, loghub_telecom
- Compute: weighted F1, reasoning accuracy, P95 latency, ECE (calibration), hallucination rate, cost per query
- Support concurrent evaluation (configurable semaphore, default 5 parallel samples)

#### FR-07: REST API
- `POST /api/v1/analyze` — Single event RCA
- `POST /api/v1/analyze/batch` — Batch RCA (max 50 events)
- `GET /api/v1/health` — Health check
- `GET /api/v1/metrics` — Orchestrator metrics
- `POST /api/v1/feedback` — Submit operator feedback
- `POST /api/v1/benchmark` — Run evaluation benchmark

### 2.2 Non-Functional Requirements

| NFR | Target | Implementation Mechanism |
|-----|--------|--------------------------|
| P95 Latency (SLM path) | < 2,000 ms | Phi-3 via vLLM, Redis working memory |
| P95 Latency (LLM path) | < 8,000 ms | GPT-4o API with retry |
| API Rate Limit | 100 req/min per IP | `RateLimitMiddleware` (sliding window) |
| Ingestion Rate Limit | 10K events/sec | Token bucket rate limiter |
| Scalability | Horizontal + vertical | Docker/K8s, 4 uvicorn workers |
| Redis Working Memory TTL | 3600 seconds | Per-session, auto-expire |
| Embedding Cache | 10,000 entries (in-memory LRU) | `EmbeddingService._cache` dict |
| Confidence Calibration | ECE ≤ 0.10 | Platt scaling with grid search |
| Availability | Graceful degradation | Redis fallback to local dict; services fail-open |

### 2.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 16 GB | 32 GB |
| GPU (for vLLM) | NVIDIA T4 (16GB) | NVIDIA A10G (24GB) |
| Disk | 50 GB SSD | 200 GB NVMe |
| Network | 1 Gbps | 10 Gbps |

For containerized deployment (Docker Compose):
- Redis: 256 MB RAM
- MongoDB: 512 MB RAM
- Qdrant: 2 GB RAM
- Elasticsearch: 1 GB RAM (JVM heap: 512m–512m)
- Kafka + Zookeeper: 1 GB RAM
- RCA API: 4–8 GB RAM (embedding model + SLM if local)

---

## 3. HIGH-LEVEL ARCHITECTURE (HLD)

### 3.1 Architecture Style

The system follows an **event-driven, service-oriented monolith** with an internal multi-agent AI pipeline:

- **Monolith deployment**: All Python services run in the same process/container
- **Event-driven communication**: Kafka for asynchronous log streaming
- **Synchronous REST API**: For external callers
- **Agent orchestration**: Synchronous directed-graph pipeline (LangGraph-style)
- **External services**: Qdrant, Elasticsearch, Redis, MongoDB, Kafka (via Docker)

### 3.2 Major Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
│        REST Clients / NOC Dashboard / Kafka Producers               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP / Kafka
┌──────────────────────────────▼──────────────────────────────────────┐
│                         API GATEWAY LAYER                            │
│  FastAPI App  │  RateLimitMiddleware  │  CORS  │  REST Routes        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
       ┌───────────────────────┼────────────────────────┐
       │                       │                        │
┌──────▼──────┐      ┌─────────▼──────┐      ┌────────▼────────┐
│  Ingestion  │      │  Preprocessing │      │  Evaluation     │
│  Service    │      │  Service       │      │  Service        │
└──────┬──────┘      └───────┬────────┘      └────────┬────────┘
       │                     │                        │
       │          ┌──────────▼──────────┐             │
       │          │  ProcessedEvent     │             │
       │          └──────────┬──────────┘             │
       │                     │                        │
       │          ┌──────────▼──────────┐             │
       └─────────▶│  ORCHESTRATOR GRAPH │◀────────────┘
                  │  (Multi-Agent RCA)  │
                  └──────────┬──────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
┌───────▼──────┐   ┌─────────▼─────┐   ┌──────────▼──────┐
│  RAG Service │   │  Inference    │   │  Working Memory  │
│  (Qdrant+ES) │   │  Service      │   │  (Redis)         │
└──────────────┘   └───────────────┘   └──────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
   ┌──────────▼──┐  ┌──────▼──────┐  ┌▼──────────────┐
   │ SLM Client  │  │ SLM Client  │  │  LLM Client   │
   │ (Phi-3)     │  │ (Mistral-7B)│  │  (GPT-4o)     │
   └─────────────┘  └─────────────┘  └───────────────┘
```

### 3.3 Data Store Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA STORES                           │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │    Redis     │   │   MongoDB    │   │   Kafka    │  │
│  │  (working    │   │  (RCA result │   │ (log event │  │
│  │   memory)    │   │   storage)   │   │  streaming)│  │
│  └──────────────┘   └──────────────┘   └────────────┘  │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐                   │
│  │   Qdrant     │   │Elasticsearch │                   │
│  │  (vector     │   │ (BM25 index  │                   │
│  │   search)    │   │  for 3GPP)   │                   │
│  └──────────────┘   └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. LOW-LEVEL DESIGN (LLD)

### 4.1 Class Hierarchy

```
BaseAgent (ABC)
├── SignalAnalysisAgent
├── KnowledgeRetrievalAgent
├── HypothesisAgent
├── ValidationAgent
└── DecisionAgent

OrchestratorGraph
├── SignalAnalysisAgent
├── KnowledgeRetrievalAgent
├── HypothesisAgent
├── ValidationAgent
├── DecisionAgent
├── RAGService
│   ├── EmbeddingService
│   ├── MultiHopRetriever
│   │   ├── EmbeddingService
│   │   ├── QdrantClient
│   │   └── AsyncElasticsearch
│   └── KnowledgeBaseManager
│       └── EmbeddingService
└── InferenceService
    ├── InferenceRouter
    │   ├── SLMClient (Phi-3)
    │   ├── SLMClient (Mistral-7B)
    │   └── LLMClient (GPT-4o)
    └── ConfidenceCalibrator

WorkingMemory
PreprocessingService
├── HybridLogParser
│   └── TemplateMiner (Drain3)
└── KPIExtractor

IngestionService
├── SchemaValidator
└── TokenBucketRateLimiter

EvaluationService
├── OrchestratorGraph
├── ClassificationAccuracyMetric
├── ReasoningAccuracyMetric
├── LatencyMetric
├── CalibrationMetric
├── HallucinationMetric
└── CostMetric
```

### 4.2 Key Interfaces and Contracts

#### BaseAgent Contract
```python
class BaseAgent(ABC):
    agent_type: AgentType          # Identity of the agent
    _invocation_count: int         # Metrics tracking
    _total_latency_ms: float       # Metrics tracking

    async def execute(
        self,
        event: ProcessedEvent,     # The 5G event being analyzed
        memory: WorkingMemory,     # Shared session state
        **kwargs
    ) -> AgentMessage:             # Structured output
        ...

    def get_metrics(self) -> dict  # Returns execution metrics
```

#### AgentMessage Schema
```python
class AgentMessage(BaseModel):
    agent_id: AgentType            # Which agent produced this
    timestamp: datetime            # When produced
    message_type: MessageType      # OBSERVATION, HYPOTHESIS, EVIDENCE, DECISION
    content: dict[str, Any]        # Agent-specific payload
    confidence: float              # 0.0 to 1.0
    reasoning_steps: list[str]     # Human-readable reasoning steps
    metadata: dict[str, Any]       # Latency, model used, etc.
```

#### ProcessedEvent Schema
```python
class ProcessedEvent(BaseModel):
    event_id: UUID
    cell_id: str                   # Which cell (e.g., "cell-101")
    gnb_id: str                    # Which gNB (e.g., "gnb-001")
    timestamp: datetime
    event_type: Optional[str]      # "interference_detected", "handover_event", etc.
    kpis: KPIMetrics               # All extracted KPI values
    parsed_template: Optional[str]
    severity: Severity             # INFO, NORMAL, WARNING, CRITICAL
    context: CellContext           # Neighboring cells, band, bandwidth
    raw_message: Optional[str]     # Truncated to 512 chars
    anomaly_scores: dict[str, float]  # Per-KPI anomaly scores [0.0, 1.0]
```

#### RCAResult Schema
```python
class RCAResult(BaseModel):
    rca_id: UUID
    root_cause: RootCauseCategory  # Enum: interference, handover_failure, etc.
    specific_cause: str            # Detailed natural language description
    confidence: float              # Calibrated confidence [0, 1]
    reasoning_trace: list[ReasoningStep]  # Full 5-step CoT trace
    supporting_evidence: list[str]
    recommended_actions: list[str] # 3-5 remediation steps
    model_used: str                # Which model produced the result
    escalated: bool                # Whether LLM fallback was used
    latency_ms: int
    timestamp: datetime
    cell_id: Optional[str]
    gnb_id: Optional[str]
    event_ids: list[UUID]
```

---

## 5. DATA FLOW ANALYSIS

### 5.1 End-to-End Data Flow

```
INPUT                     TRANSFORMATION                    OUTPUT
─────                     ──────────────                    ──────

RawLogEvent               1. Rate Limiting                 (reject if over limit)
{                         2. Schema Validation             (reject if invalid)
  source_id               3. Kafka publish                 (async queue)
  timestamp           ──▶                              ──▶
  raw_message             4. HybridLogParser               ParsedLog
  metadata                   a. Regex match (10 patterns)   {template, params,
}                             b. Drain3 template mining       event_type}
                              c. Fallback (raw)
                                    │
                                    ▼
                         5. KPIExtractor                   KPIMetrics
                            a. Regex patterns               {sinr_db, rsrp_dbm,
                            b. Param map extraction          bler_pct, ...}
                                    │
                                    ▼
                         6. Anomaly Score Computation       dict[str, float]
                            (threshold distance,             {sinr_db: 0.8,
                             linear interpolation)           prb_util: 0.6, ...}
                                    │
                                    ▼
                         7. Severity Classification        Severity enum
                            (max anomaly score → tier)      CRITICAL/WARNING/...
                                    │
                                    ▼
ProcessedEvent                                ◀── assembled here
{cell_id, gnb_id, kpis,
 event_type, anomaly_scores,
 severity, context}
                                    │
                                    ▼
                    ┌──────── ORCHESTRATOR GRAPH ────────┐

                    Stage 1: Signal Analysis Agent
                    Input:  ProcessedEvent
                    Output: {anomalies[], correlations[], severity}
                    Store → working_memory["signal_analysis"]
                                    │
                                    ▼
                    Stage 2: Knowledge Retrieval Agent
                    Input:  working_memory["signal_analysis"]
                    Output: {contexts[], formatted_context,
                              thresholds{}, known_patterns[]}
                    RAG calls: 1–5 hybrid queries
                    Store → working_memory["knowledge_context"]
                                    │
                                    ▼
                    Stage 3: Hypothesis Agent
                    Input:  working_memory["signal_analysis"]
                            working_memory["knowledge_context"]
                    Action: Build CoT prompt → SLM inference
                    Output: {hypotheses[{root_cause, confidence,
                              supporting_evidence}]}
                    Store → working_memory["hypotheses"]
                                    │
                                    ▼
                    Stage 4: Validation Agent
                    Input:  working_memory["hypotheses"]
                    Action: Score each hypothesis:
                            support_score × 0.4 +
                            (1 - contradiction) × 0.3 +
                            counterfactual × 0.3
                    Output: sorted validated_hypotheses[]
                    Store → working_memory["validated_hypotheses"]
                                    │
                                    ▼
                    Stage 5: Decision Agent
                    Input:  working_memory["validated_hypotheses"]
                    Check:  top validation_score vs threshold
                    If low: escalate to LLM
                    Output: RCAResult with full reasoning trace

                    └────────────────────────────────────┘
                                    │
                                    ▼
OUTPUT
RCAResult
{
  root_cause: "interference",
  specific_cause: "Co-channel interference...",
  confidence: 0.84,
  reasoning_trace: [Step1(Signal), Step2(Knowledge),
                    Step3(Hypothesis), Step4(Validation),
                    Step5(Decision)],
  recommended_actions: ["Analyze neighboring cell PCI...", ...],
  model_used: "microsoft/Phi-3-mini-4k-instruct",
  escalated: false,
  latency_ms: 847
}
```

### 5.2 Working Memory State Transitions

```
Session Start
     │
     ▼
wm["event"] = event.model_dump()
     │
     ▼ (Signal Agent)
wm["signal_analysis"] = {anomalies, correlations, severity, kpi_summary}
     │
     ▼ (Knowledge Agent)
wm["knowledge_context"] = {contexts, formatted_context, thresholds, known_patterns}
     │
     ▼ (Hypothesis Agent)
wm["hypotheses"] = {hypotheses[], model_used, model_confidence}
     │
     ▼ (Validation Agent)
wm["validated_hypotheses"] = [{...hypothesis, validation_score, validation_details}]
     │
     ▼ (Decision Agent)
wm["rca_result"] = RCAResult.model_dump()
     │
     ▼ (Finally block)
memory.clear() → Redis DEL wm:{session_id}
memory.close() → Redis connection close
```

---

## 6. MODULE-BY-MODULE DOCUMENTATION

---

### 6.1 `main.py` — Entry Point

**Purpose:** Boot the FastAPI application via uvicorn.

**Key Functions:**
- Reads `Settings.debug` to toggle hot-reload
- Starts 4 uvicorn worker processes
- Points at `services.api.app:app` as the ASGI application

**Notes:** The 4-worker configuration provides basic process-level concurrency. Workers do not share in-memory state (each has its own embedding cache, router stats, etc.). This is appropriate for stateless request handling; stateful data lives in Redis/MongoDB.

---

### 6.2 `config/settings.py` — Configuration Management

**Purpose:** Centralized, validated settings using Pydantic Settings with `.env` file support.

**Key Class: `Settings(BaseSettings)`**

Uses `@lru_cache()` on `get_settings()` to return a singleton instance, meaning the `.env` file is parsed only once per process.

**Configuration Sections:**

| Section | Key Settings |
|---------|-------------|
| Application | `app_name`, `app_env`, `app_port`, `app_debug` |
| Kafka | `kafka_bootstrap_servers`, topic names, consumer group |
| Redis | `redis_url`, `redis_max_connections`, `redis_working_memory_ttl` |
| MongoDB | `mongodb_url`, `mongodb_database` |
| Qdrant | `qdrant_host`, `qdrant_port`, `qdrant_collection` |
| Elasticsearch | `elasticsearch_url`, `elasticsearch_index` |
| SLM | `vllm_base_url`, model names, `slm_max_tokens`, `slm_temperature` |
| LLM | `openai_api_key`, `llm_fallback_model`, `llm_max_tokens` |
| Embedding | `embedding_model` (BAAI/bge-base-en-v1.5), `embedding_dimension` (768) |
| RAG | `rag_top_k` (5), `rag_max_hops` (2), `rag_similarity_threshold` (0.7) |
| Agent | `agent_confidence_threshold` (0.75), `agent_escalation_threshold` (0.5) |
| Auth | JWT secret, algorithm, expiration |
| Rate Limiting | `rate_limit_requests_per_minute` (100), `rate_limit_burst` (20) |

**Security Note:** `jwt_secret_key` defaults to `"change-me-in-production"` — must be overridden in production via environment variable.

---

### 6.3 `models/schemas.py` — Data Models

**Purpose:** Canonical Pydantic v2 data models shared across all services. Acts as the single source of truth for all data contracts.

**Enumerations:**

| Enum | Values | Purpose |
|------|--------|---------|
| `LogLevel` | debug, info, warning, error, critical | Raw log severity |
| `RootCauseCategory` | interference, handover_failure, resource_congestion, hardware_fault, software_fault, configuration_error, unknown | The 7 possible RCA categories |
| `Priority` | critical, high, normal, low | Request priority for rate limiting |
| `AgentType` | signal_agent, knowledge_agent, hypothesis_agent, validation_agent, decision_agent | Agent identities |
| `MessageType` | observation, hypothesis, evidence, decision | Inter-agent message types |
| `Severity` | critical, warning, normal, info | Event severity |

**KPI Metrics (`KPIMetrics`):**
All fields are `Optional[float]` or `Optional[int]`. The 15 supported KPIs:
- `sinr_db` — Signal-to-Interference-plus-Noise Ratio
- `rsrp_dbm` — Reference Signal Received Power
- `rsrq_db` — Reference Signal Received Quality
- `prb_utilization_pct` — Physical Resource Block utilization (%)
- `bler_pct` — Block Error Rate (%)
- `handover_success_rate` — Handover success fraction (0–1)
- `rrc_connection_setup_success_rate` — RRC setup success fraction
- `throughput_dl_mbps` / `throughput_ul_mbps` — Throughput in Mbps
- `latency_ms` — Round-trip latency in milliseconds
- `connected_ues` — Number of connected User Equipment
- `cqi_avg` — Average Channel Quality Indicator
- `mcs_dl_avg` — Average DL Modulation and Coding Scheme
- `ta_advance_us` — Timing Advance in microseconds
- `interference_level_dbm` — Interference power in dBm

**Additional Models:**
- `InferenceRequest` / `InferenceResponse` — Contract between services and inference layer
- `BenchmarkRequest` / `BenchmarkResult` — Evaluation API models

---

### 6.4 `services/api/app.py` — FastAPI Application Factory

**Purpose:** Creates and configures the FastAPI application with lifespan management.

**Lifespan Management (`@asynccontextmanager async def lifespan`):**

On startup:
1. Instantiates `OrchestratorGraph()` (with sub-services: RAG, Inference)
2. Instantiates `EvaluationService(orchestrator)`
3. Instantiates `IngestionService()`
4. Instantiates `PreprocessingService()`
5. Calls `orchestrator.initialize()` — which initializes the RAG service (connects to Qdrant and Elasticsearch)

All service instances are stored on `app.state` for access in route handlers.

**Middleware Stack (inner to outer):**
1. `RateLimitMiddleware` — 100 requests/minute per IP, sliding window
2. `CORSMiddleware` — Allow all origins by default (configurable)

**Design Pattern:** The factory function `create_app()` returns an `app` instance, which is imported by uvicorn. This clean separation enables testing via `TestClient(create_app())`.

---

### 6.5 `services/api/middleware.py` — Rate Limiting

**Purpose:** Sliding-window, per-IP rate limiting at the HTTP middleware level.

**Algorithm:** Sliding Window (timestamp-based)
- Maintains a dict of `{ip: [timestamp1, timestamp2, ...]}` 
- On each request: prune timestamps older than `window_seconds`, count remaining
- If count ≥ `max_requests`: return HTTP 429 with `Retry-After` header
- Returns HTTP 429 with JSON `{"detail": "Rate limit exceeded. Try again later."}`

**Limitation:** In-memory state means the rate limiter is per-worker, not shared across uvicorn workers. For true rate limiting across workers, Redis would be required.

---

### 6.6 `services/api/routes.py` — REST Endpoints

**Purpose:** Defines all REST routes using FastAPI's `APIRouter`.

**Routes:**

| Route | Method | Description | Key Logic |
|-------|--------|-------------|-----------|
| `/health` | GET | Health check | Returns static status dict |
| `/metrics` | GET | Orchestrator metrics | `orchestrator.get_metrics()` |
| `/analyze` | POST | Single event RCA | Builds `ProcessedEvent` → `orchestrator.execute()` |
| `/analyze/batch` | POST | Batch RCA (≤50) | Sequential loop (not parallel) |
| `/rca/{id}` | GET | Get stored result | Stub: returns 404 (MongoDB not wired) |
| `/rca` | GET | List results | Stub: returns empty list |
| `/feedback` | POST | Submit feedback | Logs and returns 200 (no persistence wired) |
| `/benchmark` | POST | Run benchmark | `evaluation.run_benchmark(config)` |

**Notable Pattern:** Route handlers access services via `request.app.state.orchestrator`, `request.app.state.preprocessing`, etc. — enabling clean dependency injection without FastAPI's `Depends`.

---

### 6.7 `services/ingestion/service.py` — Log Ingestion

**Purpose:** Accept log events from multiple sources, validate, rate-limit, and publish to Kafka.

**Key Methods:**

| Method | Complexity | Description |
|--------|-----------|-------------|
| `ingest_single()` | O(1) | Validate + rate-limit + Kafka publish |
| `ingest_batch()` | O(n) | Sequential single ingestion for each event |
| `consume_stream()` | Continuous | AsyncIterator over Kafka consumer group |

**Kafka Topology:**
- Consumer topic: `gnb-logs` (raw log events)
- Producer topic: `processed-events` (validated raw events)
- Consumer group: `rca-orchestrator`
- Auto-commit interval: 5 seconds

**Schema Validation:** Delegated to `SchemaValidator.validate()` (not analyzed in detail but confirms required fields are present).

---

### 6.8 `services/ingestion/rate_limiter.py` — Token Bucket

**Purpose:** Per-source token bucket rate limiter for ingestion.

**Algorithm: Token Bucket**

The token bucket algorithm works as follows:
- Each source has a bucket with `burst` capacity
- Tokens replenish at `rate` tokens/second
- Each request consumes 1 token
- If bucket is empty, request is denied

**Implementation Detail:**
```
tokens = min(burst, tokens + elapsed × rate)
if tokens >= 1.0: tokens -= 1.0; return True
else: return False
```

**Two-Level Rate Limiting:**
1. Global bucket: `rate=10000/sec`, `burst=20000`
2. Per-source bucket: `rate=1000/sec`, `burst=2000` (10% of global)

This prevents any single source from monopolizing capacity while still enforcing a global ceiling.

**Thread Safety:** Protected by `threading.Lock` for the `_buckets` dict (handles concurrent REST requests).

---

### 6.9 `services/preprocessing/service.py` — Preprocessing Pipeline

**Purpose:** Multi-stage pipeline converting raw log events into structured `ProcessedEvent` objects with KPIs, anomaly scores, and severity.

**Pipeline Stages:**

**Stage 1: Parse** — `HybridLogParser.parse(raw_message)`
  - Try regex patterns → Try Drain3 → Fallback to raw

**Stage 2: Extract KPIs** — `KPIExtractor.extract(raw_message, params)`
  - Regex extraction + parameter map extraction

**Stage 3: Identify Cell/gNB** — `_extract_cell_id()` / `_extract_gnb_id()`
  - Check `event.metadata` → parsed params → fallback to `source_id`

**Stage 4: Anomaly Scoring** — `_compute_anomaly_scores(kpis)`
  - For each KPI with thresholds, compute a score [0.0, 1.0]:
    - Score 1.0 at critical threshold
    - Score 0.5–1.0 between warning and critical (linear interpolation)
    - Score 0.0–0.5 between normal and warning

**Stage 5: Severity** — `_determine_severity(kpis, anomaly_scores)`
  - max_score ≥ 0.8 → CRITICAL
  - max_score ≥ 0.5 → WARNING
  - max_score ≥ 0.2 → NORMAL
  - else → INFO

**Stage 6: Context** — `_build_context(event, params)`
  - Extracts `neighboring_cells`, `frequency_band`, `bandwidth_mhz` from metadata

**Metrics:** Maintains running average of processing time using Welford's online algorithm:
```
avg = avg × (n-1)/n + new_time/n
```

---

### 6.10 `services/preprocessing/log_parser.py` — Hybrid Log Parser

**Purpose:** Parse semi-structured 5G gNB logs using multiple strategies.

**Strategy Cascade:**

**1. Regex Patterns (fastest, ~0.1ms)**

10 predefined patterns for known 5G log formats:

| Pattern Name | Matches | Extracts |
|-------------|---------|---------|
| `rrc_failure` | RRC connection failure/release/setup | ue_id, cause, target_cell |
| `handover` | Handover initiated/completed/failed | source_cell, target_cell, ue_id |
| `throughput_degradation` | Throughput degradation detected | cell_id, current throughput, unit |
| `interference` | High interference detected | cell_id, interference level |
| `prb_congestion` | PRB utilization high/critical | cell_id, utilization % |
| `rlf` | Radio Link Failure detected | ue_id, cell_id, cause |
| `beam_failure` | Beam failure detected/recovery | cell_id, beam_id |
| `sinr_alarm` | SINR below threshold | cell_id, SINR value |
| `kpi_report` | KPI Report | cell_id, sinr, prb, bler |
| `hardware_alarm` | Hardware alarm/fault | component, severity |

**2. Drain3 Template Mining (~1ms)**

Drain3 is an online log parsing algorithm that builds a prefix tree of log templates:
- `sim_th=0.4` — Minimum similarity to assign to existing cluster
- `depth=4` — Prefix tree depth
- `max_children=100` — Max branches per node

When a new log message is processed:
1. Tokenize by whitespace
2. Traverse prefix tree by first `depth` tokens
3. Find or create matching cluster
4. Template tokens that vary across messages become `<*>` placeholders

**3. Fallback**
Returns raw message as template, uses `_infer_event_type()` keyword matching.

---

### 6.11 `services/preprocessing/kpi_extractor.py` — KPI Extraction

**Purpose:** Extract numerical KPI values from log text using regex patterns.

**Two Extraction Paths:**

**Path A: Regex on raw message**
Each KPI has 2 regex patterns (primary + alternative). The extractor tries each pattern and returns the first match as `float`.

Example for `sinr_db`:
```
Pattern 1: SINR[:\s=]*(-?[\d.]+)\s*(?:dB)?
Pattern 2: sinr[_\s]*(?:value)?[:\s=]*(-?[\d.]+)
```

**Path B: Parameter map from parser**
Maps parsed parameter names to KPI names:
```
"sinr" → "sinr_db"
"rsrp" → "rsrp_dbm"
"prb"  → "prb_utilization_pct"
"level" → "interference_level_dbm"
```

**Derived KPI Computation** (not used in main pipeline but available):
- `spectral_efficiency = throughput_dl / (prb_utilization / 100)`
- `per_user_throughput = throughput_dl / connected_ues`
- `handover_failure_rate = 1.0 - handover_success_rate`

---

### 6.12 `services/orchestrator/graph.py` — Orchestrator Graph

**Purpose:** The master controller that sequences all 5 agents and produces the final `RCAResult`.

**Key Logic in `execute()`:**

```
1. Create unique session_id (UUID)
2. Initialize WorkingMemory for this session
3. Store event in wm["event"]
4. Signal Agent → if no anomalies AND low confidence → return clean result (EARLY EXIT)
5. Knowledge Agent → parallel-capable (currently sequential)
6. Hypothesis Agent → SLM inference
7. Validation Agent → evidence scoring
8. Decision Agent → final decision, may LLM-escalate
9. Extract RCAResult from decision
10. Clear working memory
11. Return RCAResult
```

**Early Exit Optimization:**
```python
if not signal_result.content.get("anomalies") and signal_result.confidence < 0.3:
    return self._create_clean_result(event, start_time)
```
This avoids running all 5 agents (especially the expensive SLM call) when there's clearly no issue.

**Error Isolation:** The entire pipeline is wrapped in `try/except`. On any failure, `_create_error_result()` returns a graceful RCA with `confidence=0.0` and instructions for manual investigation.

**Metrics:** Tracks total executions and total latency for average computation.

---

### 6.13 `services/orchestrator/agents/base.py` — BaseAgent

**Purpose:** Abstract base class enforcing the agent contract.

**Key Design:**
- `agent_type` is a class attribute, not instance attribute — ensures all agents declare their identity
- `_create_message()` is a helper that constructs a well-formed `AgentMessage` — reduces boilerplate and ensures consistent output format
- `get_metrics()` returns per-agent invocation count and average latency — useful for profiling the pipeline

---

### 6.14 `services/orchestrator/agents/signal_agent.py` — Signal Analysis Agent

**Purpose:** First agent in the pipeline. Analyzes KPI values against 3GPP-derived thresholds to detect anomalies and correlation patterns.

**KPI Thresholds (3GPP-derived):**

| KPI | Critical | Warning | Normal |
|-----|---------|---------|--------|
| sinr_db | 0 dB | 5 dB | 10 dB |
| rsrp_dbm | -130 dBm | -120 dBm | -100 dBm |
| rsrq_db | -20 dB | -15 dB | -10 dB |
| prb_utilization_pct | 95% | 85% | 70% |
| bler_pct | 10% | 5% | 2% |
| handover_success_rate | 0.70 | 0.85 | 0.95 |
| latency_ms | 100 ms | 50 ms | 20 ms |
| interference_level_dbm | -85 dBm | -95 dBm | -105 dBm |
| connected_ues | 500 | 300 | 200 |

**Threshold Direction:** The agent handles both "lower is worse" (sinr, rsrp, rsrq, handover_success_rate) and "higher is worse" (prb_util, bler, latency) KPIs. `interference_level_dbm` is special: higher (less negative) means stronger interference.

**Correlation Patterns (4 patterns):**

| Pattern | Conditions | Interpretation |
|---------|-----------|----------------|
| `interference_congestion` | sinr < 5 AND prb_util > 85 | Interference-induced congestion |
| `handover_signal_quality` | ho_rate < 0.85 AND rsrp < -110 | Coverage gap at cell edge |
| `overload_degradation` | prb > 90 AND ues > 300 AND latency > 50 | Cell overload |
| `interference_only` | sinr < 3 AND interference > -90 | External interference source |

**Confidence Output:**
- 0.9 if anomalies detected
- 0.5 if no anomalies

---

### 6.15 `services/orchestrator/agents/knowledge_agent.py` — Knowledge Retrieval Agent

**Purpose:** Construct domain-specific queries based on detected anomalies and retrieve relevant 3GPP context.

**Query Construction Logic:**

Queries are built from three sources (in priority order):
1. **Event type** → Pre-defined query templates (e.g., `handover_event` → 2 queries about handover failure conditions and RRC re-establishment)
2. **Anomaly KPI names** → Dynamic queries (e.g., `sinr` anomaly → "interference mitigation 5G NR cell")
3. **Correlation patterns** → Spec-specific queries (e.g., `interference_congestion` → "inter-cell interference 3GPP TS 38.213")

Maximum 5 queries (deduplicated) are sent to RAG.

**Threshold Extraction:**
Scans retrieved chunks for known 3GPP parameter names (T310, N310, T311, A3 offset, etc.) and extracts the surrounding sentence for context.

**Pattern Matching:**
6 known failure pattern keywords are searched in retrieved content: "ping-pong handover", "pilot pollution", "coverage hole", "overloaded cell", "hardware failure", "configuration mismatch".

**Confidence Formula:**
```
confidence = min(0.9, 0.5 + 0.1 × num_retrieved_contexts)
```

---

### 6.16 `services/orchestrator/agents/hypothesis_agent.py` — Hypothesis Agent

**Purpose:** The most complex agent. Uses an SLM with structured CoT prompting to generate ranked root-cause hypotheses.

**System Prompt Design:**
The system prompt establishes the agent as a "5G network expert" and mandates JSON output with exact schema (reasoning_steps + hypotheses array). This structured prompting approach reduces hallucination by constraining the output format.

**User Prompt Template:**
Fills in 6 slots: cell_id, timestamp, event_type, observations (KPI values), anomalies, correlations, 3GPP context (truncated to 1500 chars), known patterns.

**Response Parsing:**
1. Try to extract JSON block from response text (handles prefix text before JSON)
2. Parse `reasoning_steps` and `hypotheses` arrays
3. Map `root_cause` string to `RootCauseCategory` enum (with UNKNOWN fallback)
4. Clamp `confidence` to [0.0, 1.0]

**Fallback Rule-Based Hypotheses:**
If SLM parsing fails, the agent generates hypotheses from hard-coded KPI rules:
- sinr < 5 AND interference > -95 → INTERFERENCE (confidence 0.7)
- handover_success_rate < 0.85 → HANDOVER_FAILURE (confidence 0.65)
- prb_util > 90 → RESOURCE_CONGESTION (confidence 0.7)
- Default → UNKNOWN (confidence 0.3)

Sorted by confidence descending before returning.

---

### 6.17 `services/orchestrator/agents/validation_agent.py` — Validation Agent

**Purpose:** Score each hypothesis against KPI evidence using three validation dimensions.

**Validation Formula:**
```
validation_score = support_score × 0.4
                 + (1 - contradiction_score) × 0.3
                 + counterfactual_score × 0.3
```

**Validation Rules (per root cause):**

| Root Cause | Supporting KPIs | Contradicting KPIs | Required Evidence |
|-----------|----------------|-------------------|------------------|
| INTERFERENCE | sinr < 5, interference > -95, bler > 5 | sinr > 15, interference < -110 | 2 |
| HANDOVER_FAILURE | ho_rate < 0.85, rsrp < -110, rsrq < -15 | ho_rate > 0.95, rsrp > -90 | 1 |
| RESOURCE_CONGESTION | prb > 85, ues > 300, latency > 50 | prb < 50, ues < 50 | 2 |
| HARDWARE_FAULT | bler > 10 | (none) | 1 |

**Support Score Calculation:**
```
support_ratio = (KPIs meeting supporting conditions) / (total KPIs with rules)
if actual_count >= required_evidence: support_ratio += 0.2 (bonus)
clamped to [0.0, 1.0]
```

**Contradiction Score:**
```
contradiction_ratio = (KPIs meeting contradicting conditions) / (total contradicting KPIs)
```

**Counterfactual Score:**
Asks: "If this root cause were removed, would the observed anomalies be explained?" Checks how many of the supporting KPIs are actually anomalous. Returns `explained / total_anomalies`.

---

### 6.18 `services/orchestrator/agents/decision_agent.py` — Decision Agent

**Purpose:** Makes the final RCA determination. May escalate to LLM.

**Escalation Logic:**
```
if validation_score < escalation_threshold (0.5):
    escalate to LLM
    parse LLM JSON response
    return (llm_hypothesis, escalated=True)
else:
    use top validated hypothesis
    return (hypothesis, escalated=False)
```

**LLM Escalation Context:**
Builds a comprehensive prompt with:
- Event metadata (cell_id, timestamp, event_type)
- KPIs dict
- Signal analysis summary
- 3GPP context (first 1000 chars)
- Prior hypotheses

Sends to `InferenceService.generate(force_llm=True)`.

**Recommended Actions (per root cause):** Each of the 6 root cause categories has 5 recommended remediation actions.

**Reasoning Trace Compilation:**
Assembles 5 `ReasoningStep` objects (one per agent), each with step_number, agent type, action taken, observation, conclusion, and confidence.

---

### 6.19 `services/orchestrator/working_memory.py` — Working Memory

**Purpose:** Redis-backed shared state store for inter-agent communication within a single RCA session.

**Redis Key Pattern:** `wm:{session_id}` (Redis hash)

**Graceful Degradation:** If Redis is unavailable (exception on `ping()`), silently falls back to an in-memory dict (`_local_cache`). This means the system continues to function without Redis, at the cost of memory growth under high load.

**TTL:** 3600 seconds (1 hour). Each `store()` call refreshes the TTL via `EXPIRE`.

**Serialization:** JSON with `default=str` (handles datetime, UUID, etc.).

**Session Cleanup:** `clear()` is called in the `finally` block of `OrchestratorGraph.execute()` regardless of success or failure.

---

### 6.20 `services/rag/service.py` — RAG Service

**Purpose:** Unified entry point for all RAG operations.

**Initialization Flow:**
1. Creates `EmbeddingService` (lazy-loads model on first use)
2. Creates `MultiHopRetriever` (connects to Qdrant + Elasticsearch)
3. Creates `KnowledgeBaseManager` (for document ingestion)
4. `initialize()` calls `retriever.initialize()` → connects to backends

**Format Context:** Token-budget-aware formatter. Truncates chunks to fit within `max_tokens` (approximate word count). Includes source attribution: `[Source: TS 38.331, Section: 5.3.5, Relevance: 0.87]`.

---

### 6.21 `services/rag/embeddings.py` — Embedding Service

**Purpose:** Text-to-vector encoding using sentence-transformers.

**Model:** `BAAI/bge-base-en-v1.5` (768 dimensions)

**Key Optimizations:**

**Lazy Loading:** The `SentenceTransformer` model is not loaded at import time — only on the first `encode()` call. This avoids slow startup if embeddings aren't needed.

**Embedding Cache:** In-memory dict with max 10,000 entries. No eviction policy beyond the size cap — entries are never removed until the process restarts. This is fine for the 3GPP query patterns which are highly repetitive.

**Normalization:** All embeddings are L2-normalized (`normalize_embeddings=True`). This means `cosine_similarity = dot_product`, simplifying downstream operations.

**Batch Encoding:** Only encodes uncached texts in a single batch call, then assembles the full result. Batch size: 32.

---

### 6.22 `services/rag/retriever.py` — Multi-Hop Retriever

**Purpose:** Implements the full hybrid retrieval pipeline.

**Retrieval Pipeline:**

```
Query
  │
  ▼
Dense Retrieve (Qdrant)    Sparse Retrieve (ES/BM25)
  │                              │
  └──────── RRF Merge ───────────┘
                │
                ▼
          Cross-Encoder Rerank
                │
                ▼
         Sufficient? (score ≥ 0.7)
                │
         No ────┘
                │
                ▼
         Refine Query
                │
                ▼
         Hop 2 Hybrid Retrieve
                │
                ▼
         Merge + Final Rerank
                │
                ▼
           RAGResult
```

**Reciprocal Rank Fusion (RRF) Algorithm:**
```
RRF(d) = Σ  1 / (k + rank(d))
       results

where k=60 (standard value from the RRF paper by Cormack et al., 2009)
```

RRF is used because:
1. It is rank-based, not score-based — combines two incomparable scoring systems (cosine similarity vs BM25 score)
2. It is robust to outliers (the `k` constant dampens the effect of very high-ranked documents)
3. Theoretical properties: it approximates optimal score fusion without requiring calibration

**Query Refinement for Hop 2:**
```python
def _refine_query(self, original_query, hop1_results):
    # Extract key terms from top-k results
    all_content = " ".join([c.content[:200] for c in hop1_results[:3]])
    top_terms = self._extract_key_terms(all_content)
    return f"{original_query} {' '.join(top_terms[:5])}"
```

**Cross-Encoder Re-ranking:**
Uses `BAAI/bge-reranker-v2-m3` (cross-encoder) which jointly encodes query and document for higher precision than bi-encoder retrieval. Input: (query, chunk) pairs. Output: relevance scores.

**Caching:** In-memory dict keyed by `"{query}:{top_k}"`. TTL of 3600s. No eviction — can grow unboundedly in long-running servers.

---

### 6.23 `services/rag/knowledge_base.py` — Knowledge Base Manager

**Purpose:** Document ingestion pipeline for the 3GPP knowledge base.

**Ingestion Pipeline:**
1. Recursively scan directory for `.txt` and `.md` files
2. For each file: extract spec number from filename, split into sections, chunk each section
3. For each chunk: generate embedding, index in Qdrant, index in Elasticsearch

**Semantic Chunking:**
- Chunk size: 512 tokens
- Overlap: 64 tokens
- Preserves section boundaries by splitting on header patterns before word-splitting

**Chunk ID:** SHA-256 hash of `{source_document}:{section}:{chunk_index}` — ensures deterministic, content-addressable IDs.

---

### 6.24 `services/inference/service.py` — Inference Service

**Purpose:** Unified interface. Delegates to router and applies calibration.

**Calibration Applied:**
After routing returns a response, `ConfidenceCalibrator.calibrate()` is called with `features={"num_reasoning_steps": count_of_"Step"_in_response}`. This allows feature-aware confidence adjustment.

---

### 6.25 `services/inference/router.py` — Inference Router

**Purpose:** The core model routing logic — decides which model tier to use.

**3-Tier Strategy:**

| Tier | Model | When Used | Config |
|------|-------|-----------|--------|
| Tier 1 | Phi-3-mini-4k | Primary attempt | `slm_primary_model` |
| Tier 2 | Mistral-7B | If Tier 1 confidence < 0.75 and > 0.5 | `slm_secondary_model` |
| Tier 3 | GPT-4o | If all SLM confidence < 0.5, or `force_llm=True` | `llm_fallback_model` |

**Routing Logic Detail:**
```
Route(request):
  if force_llm: return tier3(request)
  
  r1 = tier1(request)
  if r1.confidence >= 0.75 and r1.finish_reason == "stop":
    return r1  # Happy path
  
  if r1.confidence >= 0.5:  # Worth trying tier 2
    r2 = tier2(request)
    if r2.confidence >= 0.75 and r2.finish_reason == "stop":
      return r2
    # Use best SLM response
    if r2.confidence > r1.confidence: r1 = r2
  
  if r1.confidence < 0.5:  # Must escalate
    r3 = tier3(request)
    if r3.confidence > 0: return r3
  
  return r1  # Best available
```

**Routing Statistics:** Tracks tier1/2/3 served counts and total requests — accessible via `get_metrics()`.

---

### 6.26 `services/inference/slm_client.py` — SLM Client

**Purpose:** HTTP client for the vLLM-served SLM via OpenAI-compatible API.

**Endpoint:** `POST {vllm_base_url}/chat/completions` (OpenAI chat completions format)

**Request Format:**
```json
{
  "model": "microsoft/Phi-3-mini-4k-instruct",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "max_tokens": 1024,
  "temperature": 0.3,
  "top_p": 0.95,
  "stream": false
}
```

**Timeouts:** 60s total, 10s connect timeout.

**Confidence Estimation Heuristic:**
```
if finish_reason != "stop": return 0.3

confidence = 0.5
if len(text) > 100: confidence += 0.1
if "therefore" or "conclusion" in text: confidence += 0.1
if "uncertain" or "unclear" in text: confidence -= 0.2

return clamp(confidence, 0.0, 1.0)
```

This is a weak heuristic. The `ConfidenceCalibrator` is designed to improve upon this.

---

### 6.27 `services/inference/llm_client.py` — LLM Client

**Purpose:** Async OpenAI client for GPT-4o fallback.

**Cost Tracking:** Estimates cost per request based on approximate GPT-4o pricing:
- Prompt: $0.005 / 1K tokens
- Completion: $0.015 / 1K tokens

**Fixed Confidence:** LLM responses are assigned `confidence=0.85` regardless of content. This reflects the assumption that cloud LLM outputs are more reliable than local SLM outputs.

**Health Check:** Calls `client.models.list()` to verify API connectivity.

---

### 6.28 `services/inference/confidence_calibrator.py` — Confidence Calibrator

**Purpose:** Improves reliability of confidence scores using Platt scaling.

**Problem:** Raw model confidence scores are often not well-calibrated. A model might output "80% confident" but only be correct 60% of the time. The Expected Calibration Error (ECE) measures this gap.

**Platt Scaling:**
```
calibrated = sigmoid(a × raw_confidence + b)
           = 1 / (1 + exp(-(a × raw_confidence + b)))
```

Default parameters: `a=1.5`, `b=-0.3` (without fitting on real data these are reasonable priors that slightly compress overconfident scores).

**Fitting Process:**
- Requires ≥ 50 observations
- Grid search over `a ∈ [0.5, 3.0]` (step 0.1) and `b ∈ [-1.0, 1.0]` (step 0.1)
- Minimizes ECE (Expected Calibration Error) over 10 bins

**ECE Computation:**
```
ECE = Σ_bins |bin_size/total| × |bin_accuracy - bin_confidence|
```

**Feature Adjustments:**
- `num_reasoning_steps >= 4` → +0.05 to confidence
- `num_reasoning_steps <= 1` → -0.10 to confidence
- `avg_rag_score >= 0.8` → +0.05
- `avg_rag_score < 0.5` → -0.10
- `evidence_count >= 3` → +0.05

**Online Learning:** Accumulates observations via `add_observation()`. Refits every 100 observations.

---

### 6.29 `services/evaluation/service.py` — Evaluation Service

**Purpose:** Benchmark runner that evaluates RCA accuracy across datasets.

**Concurrency Model:**
- Uses `asyncio.Semaphore(config.concurrency)` (default 5) to limit parallel executions
- All samples are processed with `asyncio.gather()` — I/O-bound operations run concurrently within the semaphore
- Each sample has a per-sample timeout (default 60s) via `asyncio.wait_for()`

**Cost Estimation:**
```python
def _estimate_cost(model_used):
    if "phi" in model_used: return 0.001
    elif "mistral" in model_used: return 0.002
    else: return 0.03  # GPT-4o
```

**Dataset Registry:**
```python
DATASETS = {
    "synthetic_5g": SyntheticDataset,
    "itu_challenge": ITUChallengeDataset,
    "loghub_telecom": LoghubDataset,
}
```

---

### 6.30 `services/evaluation/metrics.py` — Evaluation Metrics

**Purpose:** Six specialized metrics for evaluating RCA quality.

**ClassificationAccuracyMetric:**
Uses `sklearn.metrics.f1_score(average="weighted")` for multi-class classification. Also computes accuracy, precision, recall.

**ReasoningAccuracyMetric:**
Measures quality of CoT reasoning steps. For each predicted step, finds the maximum Jaccard similarity (word overlap) against all reference steps. Average over all steps and samples.

```
similarity(A, B) = |words_A ∩ words_B| / |words_A ∪ words_B|
```

**LatencyMetric:**
Uses `numpy.percentile()`. Reports P50, P95, P99, mean, std.

**CalibrationMetric:**
Implements ECE with 10 equal-width bins across [0.0, 1.0] confidence range. Each bin contributes: `(bin_size/total) × |avg_confidence - accuracy|`.

**HallucinationMetric:**
Checks if generated specific_cause text contains impossible claims (e.g., asserting more than 400 MHz bandwidth for 5G, or MCS > 28, or negative throughput).

**CostMetric:**
Simple mean over cost estimates.

---

### 6.31 `services/evaluation/datasets.py` — Evaluation Datasets

**Purpose:** Provides labeled evaluation samples.

**SyntheticDataset (10 scenarios):**
Carefully crafted ground-truth scenarios covering all 6 root cause categories at 3 difficulty levels.

| Root Cause | Scenarios | Difficulty |
|-----------|-----------|-----------|
| INTERFERENCE | 2 (co-channel, adjacent channel) | easy, medium |
| HANDOVER_FAILURE | 2 (weak signal, ping-pong) | easy, hard |
| RESOURCE_CONGESTION | 2 (UE overload, traffic burst) | easy, medium |
| HARDWARE_FAULT | 1 (RRU degradation) | hard |
| SOFTWARE_FAULT | 1 (scheduler bug) | hard |
| CONFIGURATION_ERROR | 2 (antenna tilt, missing NRT) | medium, hard |

Each scenario includes:
- KPI signature (dict of KPI → value)
- Ground truth root cause category
- Ground truth specific cause description
- Reference reasoning steps (4 steps explaining the diagnosis)

---

## 7. CORE ALGORITHMS & AI/ML INSIGHTS

### 7.1 Chain-of-Thought (CoT) Prompting

**What:** Instead of asking the model "what is the root cause?", the system instructs the model to reason step by step before concluding. This is implemented via the `COT_SYSTEM_PROMPT` and `COT_USER_PROMPT` in `HypothesisAgent`.

**Why it works:** CoT has been empirically shown (Wei et al., 2022) to improve performance on reasoning tasks, especially for models ≥7B parameters. By making the reasoning explicit, the model "thinks" before committing to an answer, which reduces superficial pattern-matching.

**Structure:**
```
Step 1: Identify primary symptoms (symptom extraction)
Step 2: Map symptoms to causes (domain knowledge application)
Step 3: Consider temporal/topological correlations
Step 4: Rank hypotheses by likelihood
```

**Structured Output:** The model is required to output JSON with exactly defined keys. This is enforced by the system prompt ("Output your response as valid JSON with this exact structure"). This reduces post-processing complexity and enables programmatic parsing.

**Temperature 0.3:** Low temperature reduces output variance (makes the model more deterministic), which is desirable for a diagnostic task where consistency matters more than creativity.

### 7.2 Reciprocal Rank Fusion (RRF)

**What:** Combines ranked lists from dense (cosine similarity) and sparse (BM25) retrievers.

**Formula:** $RRF(d) = \sum_{r \in Results} \frac{1}{k + rank_r(d)}$

where $k = 60$ (standard constant).

**Why 60?** The RRF paper found that $k=60$ empirically provides the best balance between high-ranked and low-ranked documents across a variety of test collections. It prevents a single highly-ranked document from dominating.

**Why hybrid?** Dense retrieval captures semantic similarity ("interference management" matches "ICIC") while sparse retrieval captures exact terminology ("T310 timer" matches "T310 timer"). Neither alone is sufficient for the 3GPP domain, which mixes formal specification language with domain jargon.

### 7.3 Platt Scaling / Confidence Calibration

**What:** Transforms raw model confidence $p$ via a sigmoid: $calibrated = \sigma(a \cdot p + b)$

**Why:** Neural network outputs are typically overconfident. A model trained to predict class probabilities often outputs probabilities that don't match empirical accuracy (e.g., 90% confident but only 70% accurate). Platt scaling is a lightweight post-hoc correction that maps raw outputs to better-calibrated probabilities.

**ECE (Expected Calibration Error):** Measures calibration quality.
$ECE = \sum_{b=1}^{B} \frac{|B_b|}{n} |acc(B_b) - conf(B_b)|$

Perfect calibration = ECE 0. The system targets ECE ≤ 0.10.

### 7.4 Validation Scoring

**What:** A 3-component weighted score for each hypothesis.

**Components:**
- **Support Score (weight 0.4):** How strongly does the KPI evidence support this hypothesis?
- **Contradiction Score (weight 0.3):** Is there contradicting KPI evidence? (1 = strong contradiction; penalized as `1 - contradiction_score`)
- **Counterfactual Score (weight 0.3):** Would this hypothesis explain all observed anomalies?

**Why this design?** Support alone can be high even when contradicting evidence exists. The contradiction check adds a falsification dimension (inspired by Popper's philosophy of science). The counterfactual ensures the hypothesis is complete, not just consistent.

### 7.5 Anomaly Score Computation

**What:** Each KPI gets a continuous anomaly score in [0, 1] based on threshold distance.

**For "lower is worse" KPIs (e.g., sinr_db):**
- If value ≤ critical threshold: score = 1.0
- If critical < value ≤ warning: score = 0.5 + 0.5 × (warning - value) / (warning - critical)  [linear interpolation]
- If value > warning: score = max(0, 0.5 × (warning × 1.5 - value) / (warning × 0.5))

**For "higher is worse" KPIs (e.g., prb_utilization):**
- Symmetrical logic

**Why continuous scores?** A binary threshold (above/below warning) loses information. A value of 84% PRB utilization and 96% PRB utilization are both "above warning" but represent very different severity levels. The continuous score captures this gradient.

### 7.6 Token Bucket Rate Limiting

**What:** Provides smooth rate limiting with burst tolerance.

**Why Token Bucket over Fixed Window?**
- Fixed window: 100 requests allowed in first second, 0 in next 59 seconds — bursty and unfair
- Token bucket: 100 requests per minute, with ability to burst up to `burst` capacity — smoother

**Mathematical model:**
$tokens(t) = \min(burst, tokens(t_{prev}) + rate \times (t - t_{prev}))$

### 7.7 Drain3 Log Parsing

**What:** An online log parsing algorithm that clusters similar log messages and mines templates.

**How:** Builds a prefix tree (trie) where:
- Root children are log tokens at depth 0
- Each path represents a prefix of a log message
- Leaf nodes are clusters with templates like `"Connection failure for UE <*> cause: <*>"`

**Why Drain3 over pure regex?** Regex patterns must be manually written for each log format. Drain3 automatically discovers patterns from streaming data, handling unknown or vendor-specific log formats that don't match pre-defined regex.

---

## 8. DIAGRAMS

### 8.1 System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         5G gNB RCA System                              │
│                                                                        │
│  ┌──────────┐  REST/POST     ┌─────────────────────────────────────┐  │
│  │  NOC     │ ─────────────▶ │         FastAPI Server              │  │
│  │Operators │               │  +RateLimitMiddleware  +CORS         │  │
│  └──────────┘               └──────────────┬──────────────────────┘  │
│                                             │                         │
│  ┌──────────┐  Kafka Topics  │             │                         │
│  │  gNB     │ ─────────────▶ │  Ingestion  │  Preprocessing         │
│  │  Logs    │                │  Service    │  Service                │
│  └──────────┘               └──────────────┘                         │
│                                             │                         │
│                                    ProcessedEvent                     │
│                                             │                         │
│                              ┌──────────────▼──────────────────────┐ │
│                              │      OrchestratorGraph              │ │
│                              │  ┌──────────────────────────────┐  │ │
│                              │  │     WorkingMemory (Redis)    │  │ │
│                              │  └──────────────────────────────┘  │ │
│                              │                                     │ │
│                              │  1.SignalAgent ──▶ 2.KnowledgeAgent │ │
│                              │         └──────────────────────┐    │ │
│                              │              3.HypothesisAgent  │    │ │
│                              │              (SLM CoT)          ▼    │ │
│                              │  4.ValidationAgent ◀────────────┘    │ │
│                              │  5.DecisionAgent ──▶ (LLM fallback) │ │
│                              └──────────────┬───────────────────────┘ │
│                                             │                         │
│  ┌─────────────────────────────────────────▼─────────────────────┐   │
│  │                        AI/ML Layer                             │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │   │
│  │  │  vLLM Server │   │  RAG Engine  │   │   GPT-4o (API)   │  │   │
│  │  │  Phi-3 mini  │   │  Qdrant +    │   │   (fallback)     │  │   │
│  │  │  Mistral-7B  │   │  Elasticsearch   └──────────────────┘  │   │
│  │  └──────────────┘   │  + BGE embed │                         │   │
│  │                     └──────────────┘                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Module Dependency Graph

```
main.py
  └── config.settings
  └── services.api.app
        ├── services.api.middleware
        ├── services.api.routes
        │     ├── models.schemas
        │     ├── services.evaluation.service
        │     └── services.orchestrator.graph
        ├── services.orchestrator.graph
        │     ├── services.orchestrator.working_memory
        │     ├── services.orchestrator.agents.signal_agent
        │     │     └── models.schemas
        │     ├── services.orchestrator.agents.knowledge_agent
        │     │     └── services.rag.service
        │     ├── services.orchestrator.agents.hypothesis_agent
        │     │     └── services.inference.service
        │     ├── services.orchestrator.agents.validation_agent
        │     └── services.orchestrator.agents.decision_agent
        │           └── services.inference.service
        ├── services.rag.service
        │     ├── services.rag.embeddings          (SentenceTransformer)
        │     ├── services.rag.retriever            (Qdrant + ES + CrossEncoder)
        │     └── services.rag.knowledge_base       (doc ingestion)
        ├── services.inference.service
        │     ├── services.inference.router
        │     │     ├── services.inference.slm_client   (vLLM API)
        │     │     └── services.inference.llm_client   (OpenAI API)
        │     └── services.inference.confidence_calibrator
        ├── services.ingestion.service
        │     ├── services.ingestion.schema_validator
        │     └── services.ingestion.rate_limiter
        └── services.preprocessing.service
              ├── services.preprocessing.log_parser  (Drain3 + regex)
              └── services.preprocessing.kpi_extractor
```

### 8.3 Data Flow Diagram (Level 0 — Context DFD)

```
                   Log Events
    ┌──────┐    (REST/Kafka)   ┌──────────────────────────────┐
    │ 5G   │ ─────────────────▶│                              │
    │ gNB  │                  │   RCA Orchestrator System    │
    └──────┘                  │                              │─────── RCA Results ──────▶ NOC Ops
                              │                              │
    ┌──────┐  Operator Input  │                              │─── Reasoning Traces ─────▶ Dashboard
    │ NOC  │ ─────────────────▶│                              │
    │ Ops  │  Feedback        │                              │─── Metrics/Health ───────▶ Monitoring
    └──────┘                  └──────────────────────────────┘
                                        │         ▲
                              3GPP Knowledge  Model API
                                        ▼         │
                              ┌─────────────────────┐
                              │   External Services  │
                              │  (Qdrant, ES, Redis, │
                              │   vLLM, OpenAI API)  │
                              └─────────────────────┘
```

### 8.4 Agent Pipeline Sequence Diagram

```
Client          Orchestrator    Signal    Knowledge    Hypothesis    Validation    Decision
  │                  │           │            │             │              │           │
  │ POST /analyze    │           │            │             │              │           │
  │─────────────────▶│           │            │             │              │           │
  │                  │           │            │             │              │           │
  │                  │─execute───▶           │             │              │           │
  │                  │           │─analyze   │             │              │           │
  │                  │           │  KPIs     │             │              │           │
  │                  │           │  detect   │             │              │           │
  │                  │           │  anomalies│             │              │           │
  │                  │◀──anomaly ─│           │             │              │           │
  │                  │  report   │           │             │              │           │
  │                  │           │           │             │              │           │
  │                  │─store wm["signal"] ──────────────────────────────────────────▶│
  │                  │           │           │             │              │           │
  │                  │─retrieve──────────────▶            │              │           │
  │                  │           │           │─construct  │              │           │
  │                  │           │           │ queries    │              │           │
  │                  │           │           │─────RAG──▶ Qdrant+ES      │           │
  │                  │           │           │◀───chunks──               │           │
  │                  │◀──3GPP context ────────           │              │           │
  │                  │           │           │             │              │           │
  │                  │─store wm["knowledge"] ───────────────────────────────────────▶│
  │                  │           │           │             │              │           │
  │                  │─generate──────────────────────────▶│              │           │
  │                  │           │           │             │─build CoT    │           │
  │                  │           │           │             │ prompt       │           │
  │                  │           │           │             │──SLM─────────────────────────▶ vLLM
  │                  │           │           │             │◀─response──────────────────────
  │                  │           │           │             │─parse JSON   │           │
  │                  │◀──hypotheses ──────────────────────│              │           │
  │                  │           │           │             │              │           │
  │                  │─validate──────────────────────────────────────────▶           │
  │                  │           │           │             │              │─score each│
  │                  │           │           │             │              │ hypothesis│
  │                  │◀──validated hyps ────────────────────────────────│           │
  │                  │           │           │             │              │           │
  │                  │─decide────────────────────────────────────────────────────────▶
  │                  │           │           │             │              │           │─check conf
  │                  │           │           │             │              │           │─compile trace
  │                  │◀──RCAResult ──────────────────────────────────────────────────│
  │                  │           │           │             │              │           │
  │◀─── HTTP 200 ───│           │           │             │              │           │
  │    RCAResult    │           │           │             │              │           │
```

### 8.5 Inference Routing Flowchart

```
                    ┌─────────────────┐
                    │  InferenceRouter│
                    │  .route(request)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  force_llm=True?│
                    └────────┬────────┘
                        Yes  │  No
                             │  ────▶ Tier 3 (GPT-4o)
                             │
                    ┌────────▼────────┐
                    │  Tier 1: Phi-3  │
                    │  generate()     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────────┐
                    │  confidence ≥ 0.75      │
                    │  AND finish_reason=stop?│
                    └────────┬────────────────┘
                        Yes  │  No
                    Return   │
                    Tier 1   │
                             │
                    ┌────────▼────────────────┐
                    │  confidence ≥ 0.5?       │
                    └────────┬────────────────┘
                        Yes  │  No ────────────────────────▶ skip tier 2
                             │
                    ┌────────▼────────┐
                    │  Tier 2:        │
                    │  Mistral-7B     │
                    │  generate()     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────────┐
                    │  confidence ≥ 0.75      │
                    │  AND finish_reason=stop?│
                    └────────┬────────────────┘
                        Yes  │  No
                    Return   │
                    Tier 2   │
                             │
                    ┌────────▼────────────────┐
                    │  Best SLM conf < 0.5?   │
                    └────────┬────────────────┘
                        Yes  │  No
                             │  ────▶ Return best SLM
                    ┌────────▼────────┐
                    │  Tier 3: GPT-4o │
                    │  generate()     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────────┐
                    │  LLM confidence > 0?    │
                    └────────┬────────────────┘
                        Yes  │  No ────────────────▶ Return best SLM
                    Return   │
                    Tier 3   │
```

### 8.6 Class Diagram (Core Classes)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         models.schemas                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ KPIMetrics  │  │ProcessedEvt │  │ AgentMessage │  │RCAResult │ │
│  │─────────────│  │─────────────│  │──────────────│  │──────────│ │
│  │sinr_db      │  │event_id     │  │agent_id      │  │rca_id    │ │
│  │rsrp_dbm     │  │cell_id      │  │timestamp     │  │root_cause│ │
│  │prb_util_pct │  │kpis◀───────▶│  │message_type  │  │confidence│ │
│  │bler_pct     │  │event_type   │  │content       │  │reasoning │ │
│  │...          │  │severity     │  │confidence    │  │trace     │ │
│  └─────────────┘  │anomaly_scrs │  │reasoning_    │  │actions   │ │
│                   └─────────────┘  │steps         │  │escalated │ │
│                                    └──────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        Orchestrator Layer                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    <<abstract>> BaseAgent                    │   │
│  │  agent_type: AgentType                                       │   │
│  │  + execute(event, memory) -> AgentMessage  [abstract]        │   │
│  │  + get_metrics() -> dict                                     │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│         ┌─────────────┬──────┴────────┬────────────┬──────────┐    │
│         ▼             ▼               ▼            ▼          ▼    │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────┐  │
│  │Signal     │  │Knowledge │  │Hypothesis│  │Validation│  │Dec.│  │
│  │Agent      │  │Agent     │  │Agent     │  │Agent     │  │Agt.│  │
│  │-THRESHOLDS│  │-TEMPLATES│  │-PROMPTS  │  │-RULES    │  │-ACT│  │
│  │-PATTERNS  │  │-rag_svc  │  │-infer_svc│  │          │  │-inf│  │
│  └───────────┘  └──────────┘  └──────────┘  └──────────┘  └────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              OrchestratorGraph                               │   │
│  │  + execute(event, force_llm) -> RCAResult                    │   │
│  │  + initialize()                                              │   │
│  │  + get_metrics() -> dict                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Layer                                    │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  RAGService    │    │ MultiHopRetriever│    │EmbeddingService│  │
│  │  + initialize()│───▶│ + retrieve()     │───▶│ + encode()     │  │
│  │  + retrieve()  │    │ + _hybrid_()     │    │ + encode_batch()│  │
│  │  + format_ctx()│    │ + _rerank()      │    │ _cache: dict   │  │
│  └────────────────┘    │ + _rrf_merge()   │    └────────────────┘  │
│                        └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       Inference Layer                                │
│  ┌─────────────────┐    ┌──────────────────┐                       │
│  │ InferenceService│───▶│  InferenceRouter │                       │
│  │ + generate()    │    │  + route()       │                       │
│  │ + health_check()│    └───────┬──────────┘                       │
│  └─────────────────┘          /│\                                  │
│                               / │ \                                 │
│  ┌─────────────────┐  ┌──────┘  │  └──────┐  ┌─────────────────┐  │
│  │ConfidenceCalib. │  │SLMClient│  SLMClient  │   LLMClient     │  │
│  │ + calibrate()   │  │(Phi-3)  │  (Mistral) │   (GPT-4o)      │  │
│  │ + fit()         │  └─────────┘  └─────────┘  └─────────────────┘│
│  │ + _ece()        │                                               │
│  └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.7 State Machine — RCA Pipeline

```
           ┌─────────────────┐
           │    INIT         │
           │  (session_id,   │
           │   WorkingMemory)│
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐           ┌──────────────────────┐
           │ SIGNAL_ANALYSIS │──no anoms──▶  CLEAN_RESULT (exit)│
           │ (KPI anomalies, │           └──────────────────────┘
           │  correlations)  │
           └────────┬────────┘
                    │ anomalies found
                    ▼
           ┌─────────────────┐
           │ KNOWLEDGE_FETCH │
           │ (RAG retrieval, │
           │  3GPP context)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐           ┌──────────────────────┐
           │ HYPOTHESIS_GEN  │──fail─────▶  RULE_BASED_FALLBACK │
           │ (SLM CoT        │           └──────────┬───────────┘
           │  inference)     │                      │
           └────────┬────────┘                      │
                    │                               │
                    ▼◀──────────────────────────────┘
           ┌─────────────────┐
           │ VALIDATION      │
           │ (score each     │
           │  hypothesis)    │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────────┐
           │  confidence check   │
           │  val_score vs 0.5   │
           └──────────┬──────────┘
                      │
          ┌───────────┴───────────┐
          │ ≥ 0.5                 │ < 0.5
          ▼                       ▼
 ┌─────────────────┐    ┌─────────────────────┐
 │  DECISION       │    │  LLM_ESCALATION     │
 │  (final RCA,    │    │  (full context to   │
 │   compile trace)│    │   GPT-4o, parse)    │
 └────────┬────────┘    └──────────┬──────────┘
          │                        │
          └────────────┬───────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  RCA_COMPLETE   │
              │  (RCAResult     │
              │   returned)     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  CLEANUP        │
              │  (wm.clear()    │
              │   wm.close())   │
              └─────────────────┘
```

### 8.8 Structure Chart

```
main()
├── uvicorn.run()
│   └── create_app()
│       ├── lifespan()
│       │   ├── OrchestratorGraph()
│       │   │   ├── SignalAnalysisAgent()
│       │   │   ├── KnowledgeRetrievalAgent(rag_service)
│       │   │   ├── HypothesisAgent(inference_service)
│       │   │   ├── ValidationAgent()
│       │   │   ├── DecisionAgent(inference_service)
│       │   │   ├── RAGService()
│       │   │   │   ├── EmbeddingService()
│       │   │   │   ├── MultiHopRetriever()
│       │   │   │   └── KnowledgeBaseManager()
│       │   │   └── InferenceService()
│       │   │       ├── InferenceRouter()
│       │   │       │   ├── SLMClient(phi-3)
│       │   │       │   ├── SLMClient(mistral)
│       │   │       │   └── LLMClient(gpt-4o)
│       │   │       └── ConfidenceCalibrator()
│       │   ├── EvaluationService(orchestrator)
│       │   ├── IngestionService()
│       │   └── PreprocessingService()
│       ├── add_middleware(RateLimitMiddleware)
│       ├── add_middleware(CORSMiddleware)
│       └── include_router(router)
│
└── POST /api/v1/analyze
    ├── build ProcessedEvent
    ├── OrchestratorGraph.execute(event)
    │   ├── WorkingMemory.initialize()
    │   ├── SignalAnalysisAgent.execute(event, memory)
    │   │   ├── _detect_anomalies(event.kpis)
    │   │   ├── _detect_correlations(event.kpis)
    │   │   ├── _compute_severity(anomalies)
    │   │   └── memory.store("signal_analysis", ...)
    │   ├── KnowledgeRetrievalAgent.execute(event, memory)
    │   │   ├── memory.retrieve("signal_analysis")
    │   │   ├── _construct_queries(event, signal_analysis)
    │   │   ├── RAGService.retrieve(query) × N queries
    │   │   │   └── MultiHopRetriever.retrieve(rag_query)
    │   │   │       ├── _hybrid_retrieve() [Qdrant + ES + RRF]
    │   │   │       └── _rerank() [CrossEncoder]
    │   │   └── memory.store("knowledge_context", ...)
    │   ├── HypothesisAgent.execute(event, memory)
    │   │   ├── _build_prompt(event, signal, knowledge)
    │   │   ├── InferenceService.generate(prompt)
    │   │   │   ├── InferenceRouter.route(request)
    │   │   │   │   ├── SLMClient(phi3).generate(request) → vLLM API
    │   │   │   │   └── [escalate if needed]
    │   │   │   └── ConfidenceCalibrator.calibrate(response)
    │   │   ├── _parse_response(text) → list[Hypothesis]
    │   │   └── memory.store("hypotheses", ...)
    │   ├── ValidationAgent.execute(event, memory)
    │   │   ├── memory.retrieve("hypotheses")
    │   │   ├── _validate_hypothesis(hyp, event) × N
    │   │   │   ├── _evaluate_support()
    │   │   │   ├── _check_contradictions()
    │   │   │   └── _counterfactual_test()
    │   │   └── memory.store("validated_hypotheses", ...)
    │   ├── DecisionAgent.execute(event, memory)
    │   │   ├── memory.retrieve("validated_hypotheses")
    │   │   ├── [if val_score < 0.5] _escalate_to_llm()
    │   │   ├── _build_rca_result()
    │   │   └── memory.store("rca_result", ...)
    │   └── memory.clear(); memory.close()
    └── return AnalyzeResponse(result=rca_result)
```

---

## 9. END-TO-END EXECUTION WALKTHROUGH

### 9.1 Scenario: Interference-Induced Throughput Degradation

**Input:**
```json
POST /api/v1/analyze
{
  "cell_id": "cell-101",
  "gnb_id": "gnb-001",
  "event_type": "interference_detected",
  "kpis": {
    "sinr_db": 2.5,
    "interference_level_dbm": -88,
    "bler_pct": 8.5,
    "prb_utilization_pct": 75,
    "throughput_dl_mbps": 12
  }
}
```

---

**Step 1: API Receives Request** (services/api/routes.py)
- `RateLimitMiddleware` checks IP: 1 request in last 60s → allow
- Route handler `/api/v1/analyze` builds `ProcessedEvent`:
  - `event_id = UUID4()`
  - `cell_id = "cell-101"`, `gnb_id = "gnb-001"`
  - `kpis = KPIMetrics(sinr_db=2.5, interference_level_dbm=-88, ...)`
  - `severity = "warning"` (from request)

---

**Step 2: Orchestrator Begins** (services/orchestrator/graph.py)
- New `session_id = "a1b2c3d4-..."`
- `WorkingMemory.initialize()` → connects to Redis (or falls back to local dict)
- `await memory.store("event", event.model_dump())`

---

**Step 3: Signal Analysis Agent** (services/orchestrator/agents/signal_agent.py)
- `_detect_anomalies(event.kpis)`:
  - `sinr_db = 2.5` → threshold critical=0, warning=5 → **WARNING** level anomaly
  - `interference_level_dbm = -88` → threshold critical=-85, warning=-95 → **WARNING** level (value > warning)
  - `bler_pct = 8.5` → threshold critical=10, warning=5 → **WARNING** level

- `_detect_correlations(event.kpis)`:
  - Pattern `interference_only`: `sinr < 3` ✓ AND `interference > -90` ✓ → **DETECTED**
  - Pattern `interference_congestion`: `sinr < 5` ✓ AND `prb > 85` ✗ (75 < 85) → not detected

- `_compute_severity(anomalies)` → 3 warnings → **"warning"** severity

- Reasoning steps generated:
  ```
  "Analyzed KPI values: sinr_db=2.5, interference_level_dbm=-88, bler_pct=8.5"
  "Detected 3 anomalies: sinr_db=2.5 is at warning level (threshold: 5); ..."
  "Correlation pattern detected: Low SINR with high interference level indicates external interference"
  ```

- Confidence: 0.9 (anomalies found)
- `memory.store("signal_analysis", {anomalies: [...], correlations: [...], severity: "warning"})`

---

**Step 4: Knowledge Retrieval Agent** (services/orchestrator/agents/knowledge_agent.py)
- `_construct_queries(event, signal_analysis)`:
  - Event type `"interference_detected"` → queries:
    1. `"5G NR interference management procedures 3GPP"`
    2. `"Inter-cell interference coordination ICIC mechanisms"`
  - Anomaly `sinr` detected → query: `"interference mitigation 5G NR cell"`
  - Correlation `interference_only` → query: `"inter-cell interference 3GPP TS 38.213"`
  - Total: 4 unique queries

- For each query, calls `RAGService.retrieve(query, top_k=3)`:
  - `MultiHopRetriever.retrieve()`:
    - Hop 1: Dense retrieve from Qdrant (top 6) + Sparse from ES (top 6)
    - RRF merge → top 6
    - CrossEncoder rerank → top 3
    - Check if sufficient (score ≥ 0.7) → assume yes
  - Returns `RAGResult` with chunks

- `_extract_thresholds()`: Finds "T310" in retrieved content → "T310 timer expires when...{sentence}"
- `_match_patterns()`: Finds "pilot pollution" in retrieved chunk → adds to known_patterns

- Confidence: `min(0.9, 0.5 + 0.1 × 4) = 0.9`
- `memory.store("knowledge_context", {...})`

---

**Step 5: Hypothesis Agent** (services/orchestrator/agents/hypothesis_agent.py)
- `_build_prompt()` assembles:
  ```
  ## Event Information
  Cell ID: cell-101, Timestamp: 2026-05-07T...
  
  ## KPI Observations
  - sinr_db: 2.5
  - interference_level_dbm: -88
  - bler_pct: 8.5
  - prb_utilization_pct: 75
  - throughput_dl_mbps: 12
  
  ## Detected Anomalies
  - sinr_db=2.5 is at warning level (threshold: 5)
  - interference_level_dbm=-88 is at warning level
  - bler_pct=8.5 is at warning level
  
  ## Correlation Patterns
  - Low SINR with high interference indicates external interference source
  
  ## 3GPP Knowledge Context
  [Source: TS 38.213]: ICIC procedures in NR allow...
  
  ## Known Failure Patterns
  - Excessive interference from multiple strong cells (pilot pollution)
  ```

- `InferenceService.generate(prompt)`:
  - Router tries `SLMClient(Phi-3)` → returns JSON response with confidence 0.7
  - 0.7 < 0.75 threshold → tries `SLMClient(Mistral-7B)` → confidence 0.82
  - 0.82 ≥ 0.75 → return Mistral response

- SLM Response (parsed):
  ```json
  {
    "reasoning_steps": [
      "Step 1: Primary symptoms are low SINR (2.5 dB) and elevated BLER (8.5%), indicating poor signal quality",
      "Step 2: Interference level at -88 dBm is above normal baseline, consistent with co-channel interference",
      "Step 3: PRB utilization is normal (75%), ruling out resource congestion as primary cause",
      "Step 4: Pattern matches external interference source based on SINR/interference correlation"
    ],
    "hypotheses": [
      {
        "root_cause": "interference",
        "specific_cause": "Co-channel interference from neighboring cell causing SINR degradation",
        "confidence": 0.85,
        "supporting_evidence": ["SINR=2.5dB below threshold", "interference=-88dBm above normal"]
      }
    ]
  }
  ```

- `ConfidenceCalibrator.calibrate(0.85, features={"num_reasoning_steps": 4})`:
  - `a=1.5, b=-0.3`: `logit = 1.5 × 0.85 + (-0.3) = 0.975`
  - `calibrated = sigmoid(0.975) = 0.726`
  - Feature: 4 steps → +0.05 → `0.726 + 0.05 = 0.776`

- `memory.store("hypotheses", {hypotheses: [...]})`

---

**Step 6: Validation Agent** (services/orchestrator/agents/validation_agent.py)
- Retrieves hypothesis: INTERFERENCE, confidence 0.85

- Validation rules for INTERFERENCE:
  - Supporting: `sinr < 5` → 2.5 < 5 ✓, `interference > -95` → -88 > -95 ✓, `bler > 5` → 8.5 > 5 ✓
  - Contradicting: `sinr > 15` → 2.5 NOT > 15 ✗, `interference < -110` → -88 NOT < -110 ✗
  - Required: 2

- `_evaluate_support()`:
  - `supports = 3` (all 3 supporting KPIs met), `total_checks = 3`
  - `support_ratio = 1.0`
  - `supports(3) >= required(2)` → bonus +0.2 → `min(1.0, 1.2) = 1.0`

- `_check_contradictions()`:
  - `contradictions = 0`, `total_checks = 2`
  - `contradiction_score = 0.0`

- `_counterfactual_test()`:
  - Supporting KPIs that ARE anomalous: sinr ✓, interference ✓, bler ✓ → 3/3 = 1.0

- `validation_score = 1.0 × 0.4 + (1-0.0) × 0.3 + 1.0 × 0.3 = 1.0`

- Top hypothesis: validation_score = 1.0
- Reasoning: "Top hypothesis: 'interference' with validation score 1.00"

---

**Step 7: Decision Agent** (services/orchestrator/agents/decision_agent.py)
- `validation_score = 1.0 ≥ escalation_threshold (0.5)` → **NO ESCALATION**

- `_build_rca_result()`:
  - `root_cause = RootCauseCategory.INTERFERENCE`
  - `confidence = min(1.0, max(0.0, 1.0)) = 1.0`
  - Recommended actions (5 for INTERFERENCE, top 3 taken):
    1. "Analyze neighboring cell PCI and frequency reuse pattern"
    2. "Check for external interference sources (radar, other systems)"
    3. "Consider ICIC power control adjustments"
  - `model_used = "mistralai/Mistral-7B-Instruct-v0.3"`
  - `escalated = False`

---

**Step 8: Result & Cleanup**
- `rca_result.latency_ms` set to total pipeline time (e.g., 1247 ms)
- `memory.clear()` → Redis DEL
- `memory.close()` → Redis connection close

**Output:**
```json
{
  "request_id": "b2c3d4e5-...",
  "status": "completed",
  "result": {
    "rca_id": "c3d4e5f6-...",
    "root_cause": "interference",
    "specific_cause": "Co-channel interference from neighboring cell causing SINR degradation",
    "confidence": 1.0,
    "reasoning_trace": [
      {"step_number": 1, "agent": "signal_agent", "action": "Analyze KPI metrics",
       "observation": "Detected anomalies in network metrics", "confidence": 0.9},
      {"step_number": 2, "agent": "knowledge_agent", "action": "Retrieve 3GPP context",
       "observation": "Found relevant specifications and patterns", "confidence": 0.85},
      {"step_number": 3, "agent": "hypothesis_agent", "action": "Generate root cause hypotheses",
       "observation": "Primary hypothesis: interference", "confidence": 0.85},
      {"step_number": 4, "agent": "validation_agent", "action": "Validate hypothesis against evidence",
       "observation": "Support: 1.0, Contradictions: 0.0", "confidence": 1.0},
      {"step_number": 5, "agent": "decision_agent", "action": "Final root cause determination",
       "observation": "Root cause: interference", "confidence": 1.0}
    ],
    "supporting_evidence": [
      "SINR=2.5dB below threshold",
      "interference=-88dBm above normal"
    ],
    "recommended_actions": [
      "Analyze neighboring cell PCI and frequency reuse pattern",
      "Check for external interference sources (radar, other systems)",
      "Consider ICIC power control adjustments"
    ],
    "model_used": "mistralai/Mistral-7B-Instruct-v0.3",
    "escalated": false,
    "latency_ms": 1247
  }
}
```

---

## 10. CONFIGURATION & ENVIRONMENT

### 10.1 Environment Variables (`.env`)

```bash
# Application
APP_ENV=development
APP_DEBUG=false
APP_PORT=8000

# Security (MUST CHANGE IN PRODUCTION)
JWT_SECRET_KEY=your-secret-here-at-least-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=rca-orchestrator
KAFKA_LOG_TOPIC=gnb-logs

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_WORKING_MEMORY_TTL=3600

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=rca_orchestrator

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=3gpp_knowledge

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=3gpp_sparse

# vLLM / SLM
VLLM_BASE_URL=http://localhost:8001/v1
SLM_PRIMARY_MODEL=microsoft/Phi-3-mini-4k-instruct
SLM_SECONDARY_MODEL=mistralai/Mistral-7B-Instruct-v0.3
SLM_MAX_TOKENS=1024
SLM_TEMPERATURE=0.3

# LLM Fallback
OPENAI_API_KEY=sk-...
LLM_FALLBACK_MODEL=gpt-4o
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.2

# Embedding
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# RAG
RAG_TOP_K=5
RAG_MAX_HOPS=2
RAG_SIMILARITY_THRESHOLD=0.7

# Agent
AGENT_CONFIDENCE_THRESHOLD=0.75
AGENT_ESCALATION_THRESHOLD=0.5
AGENT_MAX_REASONING_STEPS=5
AGENT_TIMEOUT_SECONDS=30

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Monitoring
OTLP_ENDPOINT=http://localhost:4317
```

### 10.2 Build & Run Instructions

**Prerequisites:**
```bash
Python 3.11+
Docker & Docker Compose
GPU (optional, for local vLLM serving)
```

**Setup:**
```bash
# 1. Clone and navigate
cd Agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies (Windows: vllm skipped automatically)
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings
```

**Start Infrastructure (Docker Compose):**
```bash
# Core services (Kafka, Redis, MongoDB, Qdrant, Elasticsearch)
docker-compose up -d

# With monitoring (Prometheus + Grafana on port 3000)
docker-compose --profile monitoring up -d

# Check service health
docker-compose ps
```

**Run the Application:**
```bash
# Development (single worker, hot reload)
uvicorn services.api.app:app --host 0.0.0.0 --port 8000 --reload

# Production (4 workers)
python main.py
# OR
uvicorn services.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Run Tests:**
```bash
pytest tests/ -v
pytest tests/test_agents.py -v  # Agent tests only
pytest tests/ -v --tb=short     # Brief error output
```

**Docker Build:**
```bash
docker build -t rca-system:latest .
docker run -p 8000:8000 --env-file .env rca-system:latest
```

### 10.3 Docker Compose Services

| Service | Port | Health Check | Notes |
|---------|------|--------------|-------|
| rca-api | 8000 | httpx GET /api/v1/health | Main application |
| kafka | 9092 | kafka-topics --list | Requires zookeeper |
| zookeeper | 2181 | echo ruok | Kafka dependency |
| redis | 6379 | redis-cli ping | Working memory |
| mongodb | 27017 | mongosh --eval "db.adminCommand('ping')" | Result storage |
| qdrant | 6333 | — | Vector search |
| elasticsearch | 9200 | curl /_cluster/health | BM25 search |
| prometheus | 9090 | — | Profile: monitoring |
| grafana | 3000 | — | Profile: monitoring |

---

## 11. DESIGN DECISIONS & JUSTIFICATIONS

### 11.1 Why Multi-Agent Instead of Single-Chain?

**Challenge:** 5G RCA requires distinct cognitive tasks: signal analysis (numerical), knowledge retrieval (information), hypothesis generation (reasoning), validation (verification), decision (synthesis). A single monolithic prompt handles all poorly.

**Solution:** Specialized agents, each expert in its task, communicating via structured messages through shared working memory.

**Justification:**
- Modularity: Each agent can be improved or replaced independently
- Testability: Each agent can be unit tested in isolation
- Parallelism: Knowledge and context agents could run in parallel in a future optimization
- Debuggability: The working memory and reasoning trace provide full observability

### 11.2 Why SLM-First with LLM Fallback?

**Challenge:** GPT-4o gives excellent accuracy (~90%) but costs $0.03/query and adds 3–5s latency. Running it for every event in a 10K-cell network is prohibitively expensive.

**Solution:** Tier 1 (Phi-3, ~$0.001, ~400ms) handles 70% of cases. Tier 2 (Mistral-7B, ~$0.002, ~800ms) handles 20%. LLM fallback (GPT-4o, ~$0.03, ~3s) handles the remaining 10% that require deep reasoning.

**Justification:** The validation agent's scoring naturally identifies low-confidence cases. Most 5G faults follow well-known patterns (interference, congestion, handover) that a 3.8B-7B model with domain-specific prompting can diagnose accurately.

### 11.3 Why Hybrid RAG (Dense + Sparse + Reranker)?

**Challenge:** 3GPP specifications use precise technical terminology (e.g., "T310 timer", "A3 event", "Qrxlevmin"). Semantic embedding alone may not capture exact parameter names.

**Solution:** 
- Dense retrieval (Qdrant/cosine similarity): Captures semantic similarity
- Sparse retrieval (Elasticsearch/BM25): Captures exact keyword matches
- RRF fusion: Combines both signals without calibration
- Cross-encoder reranker: Higher-precision re-scoring of top candidates

**Justification:** Hybrid retrieval consistently outperforms either approach alone by 5–15% in telecom RAG benchmarks. The cross-encoder is slower than the bi-encoder but runs only on top-10 candidates (not the full corpus).

### 11.4 Why Redis for Working Memory?

**Challenge:** Agents need to share context within an RCA session. Options: function parameters (tight coupling), database (too slow), message queue (async complexity).

**Solution:** Redis hashes keyed by session_id, TTL-expired after 1 hour.

**Justification:**
- Sub-millisecond read/write latency
- Automatic cleanup via TTL
- Enables future horizontal scaling (agents on different machines can share state)
- Graceful degradation to local dict if Redis unavailable

### 11.5 Why Platt Scaling for Calibration?

**Challenge:** SLM confidence scores are poorly calibrated (often overconfident). The heuristic in `_estimate_confidence()` is simplistic.

**Solution:** Platt scaling transforms raw confidence via `sigmoid(a × raw + b)`. Parameters are fitted on labeled data using ECE minimization.

**Justification:** 
- Low computational cost (single sigmoid transform)
- Proven effectiveness on neural network outputs
- Supports online learning via `add_observation()`
- More theoretically principled than ad-hoc adjustments

### 11.6 Why Drain3 for Log Parsing?

**Challenge:** 5G gNB logs are semi-structured — same event type appears with different UE IDs, cell IDs, parameter values. Writing regex for every possible format is maintenance-intensive.

**Solution:** Drain3 mines templates automatically using an online algorithm, requiring no labeled data.

**Justification:**
- Zero-shot: works on unseen log formats immediately
- Online: updates templates as new patterns are discovered
- Fast: O(depth × max_children) per message
- Proven: state-of-the-art in log parsing benchmarks (Loghub)

---

## 12. TESTING STRATEGY

### 12.1 Test Structure

```
tests/
├── conftest.py          # Shared fixtures (sample events, KPIs)
├── test_agents.py       # Unit tests for all 5 agents
├── test_api.py          # Integration tests for REST endpoints
├── test_evaluation.py   # Evaluation service and metrics tests
└── test_preprocessing.py # Preprocessing pipeline tests
```

### 12.2 Key Fixtures (`conftest.py`)

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_kpis` | `KPIMetrics` | Multi-anomaly KPI set (SINR=3, RSRP=-110, BLER=8, HO=0.78) |
| `sample_event` | `ProcessedEvent` | interference_detected event using sample_kpis |
| `sample_raw_log` | `RawLogEvent` | Raw alarm log with SINR anomaly |
| `congestion_event` | `ProcessedEvent` | Critical congestion (PRB=96, UEs=400, latency=80) |
| `handover_event` | `ProcessedEvent` | Critical handover failure (HO=0.60, RSRP=-125) |

### 12.3 Testing Philosophy
- Agents tested with mocked services (no real SLM/RAG calls in unit tests)
- API tests use `TestClient` with `app.state` overridden
- Evaluation tests use `SyntheticDataset` (deterministic, no I/O)
- Preprocessing tests use hardcoded log strings matching known patterns

---

## 13. IMPROVEMENT SUGGESTIONS

### 13.1 Code Quality

| Issue | Location | Improvement |
|-------|---------|-------------|
| `jwt_secret_key` default is weak | `config/settings.py` | Add `@validator` to raise ValueError if production and default key |
| Rate limiter is per-worker | `services/api/middleware.py` | Replace with Redis-backed rate limiter for multi-worker correctness |
| `retrieve()` calls are sequential in KnowledgeAgent | `knowledge_agent.py` | Use `asyncio.gather(*[rag.retrieve(q) for q in queries])` |
| Embedding cache has no eviction | `services/rag/embeddings.py` | Implement LRU eviction using `collections.OrderedDict` |
| RRF cache never expires in-memory | `services/rag/retriever.py` | Add timestamp-based TTL or `cachetools.TTLCache` |
| MongoDB not wired for persistence | `services/api/routes.py` | Implement MongoDB `motor` client for RCA result storage |
| `specific_cause` truncated to 200 chars in error path | `decision_agent.py` | Increase to 500 chars or return full error |
| Missing `@pytest.mark.asyncio` | tests/ | Add `asyncio_mode = "auto"` to `pytest.ini` |

### 13.2 Architectural Improvements

1. **Parallel Agent Execution:** The Knowledge Agent and any context enrichment agents can run in parallel with `asyncio.gather()`. Currently they are sequential. This could save 100–300ms per request.

2. **Persistent RCA Storage:** MongoDB is in docker-compose but the `GET /rca/{id}` endpoint returns 404. Wire in `motor` async MongoDB client to persist and retrieve results.

3. **WebSocket Push:** Add WebSocket endpoint for real-time RCA result streaming to dashboards. FastAPI supports this natively.

4. **Feedback Loop Wiring:** The `/feedback` endpoint logs but doesn't persist or trigger calibration updates. Wire feedback to `ConfidenceCalibrator.add_observation()`.

5. **Model Registry:** No model versioning. Add MLflow or a simple database table to track which model version produced which RCA results (critical for reproducibility).

6. **Circuit Breaker:** No circuit breaker for Qdrant/Elasticsearch/vLLM. If any external service fails, every request fails. Use `tenacity` for retry with exponential backoff, and circuit breaker pattern.

### 13.3 Performance Optimizations

1. **Batch Embedding:** When multiple RAG queries are made for one event, encode all queries in a single batch call to `embedding_service.encode_batch()` rather than sequential `encode()` calls.

2. **vLLM Connection Pooling:** `SLMClient` creates a new `httpx.AsyncClient` per instance. Use a shared connection pool across workers.

3. **Lazy Qdrant/ES Init:** `MultiHopRetriever.initialize()` is called on every server start. If backends are unavailable, it logs warnings and continues — this is correct, but in production, startup should retry with backoff.

4. **KPI Threshold Dict Performance:** `SignalAnalysisAgent.THRESHOLDS` is a class-level dict — this is already optimal (no per-call allocation). ✓

5. **JSON Serialization:** `memory.store()` uses `json.dumps(value, default=str)`. For complex nested objects, `orjson` is 2–5x faster and should be preferred.

### 13.4 Scalability Enhancements

1. **Stateless Workers:** Currently, each uvicorn worker has its own in-memory embedding cache and routing stats. In a Kubernetes deployment with multiple pods, stats and caches are not shared. Move `_routing_stats` to Redis for accurate metrics.

2. **Horizontal Scaling of Inference:** The `InferenceRouter` points at a single `vllm_base_url`. For high throughput, deploy multiple vLLM instances behind a load balancer and configure the URL accordingly.

3. **Topic Partitioning:** The Kafka topics are created with default 1 partition. For throughput > 10K events/sec, increase partitions and add more consumer group members.

4. **Async Batch Processing:** The `/analyze/batch` endpoint processes events sequentially. Replace with `asyncio.gather(*[execute(e) for e in events[:50]])` for true parallelism.

### 13.5 Security Enhancements

1. **Authentication:** JWT authentication is configured but not enforced in routes. Add `Depends(get_current_user)` to sensitive endpoints.

2. **Secret Management:** `JWT_SECRET_KEY` and `OPENAI_API_KEY` should come from a secrets manager (AWS Secrets Manager, HashiCorp Vault) rather than `.env` files in production.

3. **Input Sanitization:** `raw_message` is stored after truncation but is not sanitized. If stored in MongoDB and later rendered in a dashboard, ensure proper escaping to prevent XSS.

4. **Rate Limiter Bypass for Localhost:** Consider allowing health checks from localhost without rate limiting to avoid false positives from monitoring.

5. **CORS:** Current CORS config allows `allow_origins=["*"]`. In production, restrict to specific frontend origins.

---

*This documentation was generated through complete analysis of all 35+ source files in the codebase as of May 7, 2026.*
