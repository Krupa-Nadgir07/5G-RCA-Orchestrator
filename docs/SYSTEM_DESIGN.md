# COMPLETE SYSTEM DOCUMENTATION
# Multi-Agent Orchestrator for Automated Root Cause Analysis in 5G gNB Environments

## Production-Grade System Design & Architecture

---

## 1. Executive Summary

This document presents an optimized, production-grade system design for a **Multi-Agent Orchestrator** that performs **Automated Root Cause Analysis (RCA)** in 5G gNB (next-generation Node B) environments. The system leverages **Small Language Models (SLMs)** enhanced with **Retrieval-Augmented Generation (RAG)** and **Chain-of-Thought (CoT)** reasoning to diagnose network issues including throughput degradation, interference, handover failure, and resource congestion.

**Key Differentiators:**
- Multi-agent collaborative reasoning with domain-specialized agents
- Hybrid SLM/LLM inference with intelligent escalation
- 3GPP standards-grounded RAG pipeline for telecom domain accuracy
- Explainable AI through structured CoT reasoning traces
- Research-grade evaluation framework for SLM vs LLM benchmarking

**Architecture Philosophy:** Event-driven microservices with an agent-orchestrator pattern, optimized for low-latency inference (<2s P95 for SLM path), horizontal scalability, and full observability of reasoning paths.

---

## 2. Extracted Requirements & Assumptions

### 2.1 Functional Requirements (Refined)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-01 | Ingest 5G gNB logs (streaming + batch) | P0 | Support ≥10K events/sec streaming, batch files up to 10GB |
| FR-02 | Parse and normalize semi-structured logs | P0 | ≥95% parse success rate on standard gNB formats |
| FR-03 | Extract KPIs (SINR, PRB util, HO rate, etc.) | P0 | Extract ≥15 standard KPIs per log event |
| FR-04 | RAG pipeline with 3GPP specifications | P0 | Retrieve relevant context with ≥0.85 recall@5 |
| FR-05 | Multi-agent orchestration for RCA | P0 | 5 specialized agents with directed-graph orchestration |
| FR-06 | CoT reasoning for explainable diagnosis | P0 | Generate ≥3 reasoning steps per diagnosis |
| FR-07 | Classify root causes (interference, HO failure, congestion, HW/SW fault) | P0 | ≥85% classification accuracy on benchmark datasets |
| FR-08 | Provide confidence score + reasoning trace | P0 | Calibrated confidence with ≤0.1 ECE |
| FR-09 | SLM vs LLM benchmarking pipeline | P1 | Compare ≥3 models on ≥3 metrics |
| FR-10 | REST API for external integration | P0 | OpenAPI 3.0 compliant |
| FR-11 | Dashboard for visualization | P1 | Real-time RCA results + reasoning traces |
| FR-12 | Hybrid escalation (SLM → LLM) | P0 | Automatic escalation when confidence < threshold |

### 2.2 Non-Functional Requirements (Quantified)

| NFR | Target | Measurement |
|-----|--------|-------------|
| Latency (SLM path) | P95 < 2s, P99 < 5s | End-to-end from log event to RCA |
| Latency (LLM fallback) | P95 < 8s, P99 < 15s | Including network round-trip |
| Throughput | ≥1000 RCA queries/min | Under sustained load |
| Availability | 99.9% uptime | Monthly SLA |
| Scalability | Linear scaling to 100 cells | Horizontal pod autoscaling |
| Observability | 100% reasoning trace coverage | Every RCA has full CoT trace |
| Data freshness | <30s from log event to availability | Streaming pipeline latency |

### 2.3 Constraints (Explicitly Stated)

1. **Context Window Limitation:** SLMs (1B-7B params) have 4K-8K token context windows; RAG retrieval must be concise
2. **Domain Specificity:** 3GPP specifications are highly structured; embeddings must capture technical semantics
3. **Log Noise:** Semi-structured logs with inconsistent formats require robust parsing
4. **Cost Envelope:** Primary inference on SLMs (GPU cost ~$0.001/query vs $0.03/query for LLM)
5. **Regulatory:** Telecom data may be subject to data sovereignty requirements

### 2.4 Resolved Assumptions

| Assumption | Resolution | Impact |
|------------|------------|--------|
| SLM size range | 1B-7B parameters (Phi-3 Mini to Mistral 7B) | Determines GPU memory requirements |
| Log format standardization | Logs follow O-RAN/3GPP formats or can be normalized via configurable parsers | Parser extensibility required |
| 3GPP document availability | Full text of TS 38.xxx series available for chunking | ~500 documents, ~50K chunks |
| Infrastructure | Kubernetes with GPU nodes (T4/A10G) | Mixed CPU/GPU node pools |
| Real-time definition | "Real-time" = streaming with <30s end-to-end | Not sub-second |
| Ground truth availability | ITU Challenge dataset + Loghub provide labeled examples | ~10K labeled samples |

### 2.5 Identified Gaps in Original Design (Resolved)

1. **No feedback loop** → Added human-in-the-loop feedback mechanism for continuous learning
2. **No model versioning** → Added ML model registry and A/B testing framework
3. **No rate limiting/backpressure** → Added admission control and backpressure mechanisms
4. **Unclear agent communication protocol** → Defined structured message format between agents
5. **No graceful degradation strategy** → Defined tiered degradation: full CoT → simplified → rule-based fallback
6. **No data retention/lifecycle policy** → Defined retention tiers (hot/warm/cold)
7. **Missing conflict resolution** → Added consensus mechanism for conflicting agent outputs

---

## 3. Critical Evaluation of Input Design

### 3.1 Strengths
- Sound architectural intuition (microservices + agent pattern)
- Correct identification of RAG as grounding mechanism
- Appropriate hybrid SLM/LLM strategy
- Good coverage of core components

### 3.2 Weaknesses & Improvements

| Issue | Severity | Problem | Improvement |
|-------|----------|---------|-------------|
| Agent communication undefined | High | No protocol for inter-agent messaging | Define structured JSON message schema with typed fields |
| No backpressure mechanism | High | Log flood can overwhelm inference | Add admission control + priority queuing |
| RAG retrieval strategy simplistic | Medium | Single top-k retrieval insufficient for complex RCA | Implement multi-hop retrieval with re-ranking |
| No model serving strategy | High | Raw model loading is not production-grade | Use vLLM/TensorRT-LLM with batched inference |
| Evaluation pipeline is offline-only | Medium | No way to evaluate in production | Add shadow mode + online evaluation |
| Dashboard is afterthought | Low | Unclear data flow to frontend | Define WebSocket push for real-time updates |
| No agent memory/context sharing | High | Agents can't build on prior reasoning | Add shared context store (working memory) |
| Single vector DB query strategy | Medium | Different failure modes need different retrieval | Implement hybrid retrieval (dense + sparse + keyword) |
| No confidence calibration | Medium | Raw model confidence is unreliable | Add temperature scaling + calibration layer |
| Missing data preprocessing pipeline | Medium | Feature extraction is hand-waved | Define concrete feature engineering pipeline |

### 3.3 Architectural Risks

1. **Cold start latency:** First inference after scaling requires model loading (~30s for 7B model)
   - *Mitigation:* Pre-warm pods, use model caching, keep minimum replicas

2. **RAG context pollution:** Irrelevant 3GPP chunks degrade reasoning
   - *Mitigation:* Multi-stage retrieval with cross-encoder re-ranking

3. **Agent deadlock:** Circular dependencies in agent graph
   - *Mitigation:* DAG validation at orchestration compile time, timeout-based circuit breaking

4. **Concept drift:** Network behavior changes over time
   - *Mitigation:* Online monitoring of prediction distribution, periodic retraining triggers

---

## 4. Optimized Architecture

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  React/Next  │  │  REST API    │  │  gRPC Internal Gateway       │  │
│  │  Dashboard   │  │  Gateway     │  │                              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   AGENT ORCHESTRATOR (LangGraph)                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │ │
│  │  │ Signal   │ │Knowledge │ │Hypothesis│ │Validation│ │Decision│ │ │
│  │  │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │ │ Agent  │ │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │ │
│  │         ▲           ▲           ▲            ▲           ▲        │ │
│  │         └───────────┴───────────┴────────────┴───────────┘        │ │
│  │                    SHARED WORKING MEMORY                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  Ingestion   │  │ Preprocessing│  │  Evaluation Engine            │  │
│  │  Service     │  │ Service      │  │                              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI/ML LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  SLM Serving │  │  RAG Engine  │  │  LLM Fallback Gateway        │  │
│  │  (vLLM)     │  │  (Multi-hop) │  │  (API-based)                 │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                                    │
│  │  Confidence  │  │  Model       │                                    │
│  │  Calibrator  │  │  Registry    │                                    │
│  └──────────────┘  └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Kafka       │  │  Vector DB   │  │  Time-Series │  │  Document  │ │
│  │  (Streaming) │  │  (Qdrant)    │  │  (VictoriaM) │  │  Store     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
│  ┌──────────────┐  ┌──────────────┐                                    │
│  │  Redis       │  │  MongoDB     │                                    │
│  │  (Cache/WM)  │  │  (Results)   │                                    │
│  └──────────────┘  └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Kubernetes  │  │  Prometheus  │  │  Jaeger      │  │  ArgoCD    │ │
│  │  (EKS/GKE)  │  │  + Grafana   │  │  (Tracing)   │  │  (GitOps)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Architecture Justification

| Decision | Rationale |
|----------|-----------|
| LangGraph for orchestration | Native support for stateful, directed agent graphs with conditional edges; better than raw LangChain for complex multi-step workflows |
| vLLM for SLM serving | Continuous batching + PagedAttention gives 2-4x throughput vs naive serving; critical for meeting latency targets |
| Qdrant for vector DB | Native support for filtering + payload storage; better than FAISS for production (persistence, horizontal scaling, metadata filtering) |
| Redis for working memory | Sub-ms latency for inter-agent state sharing; TTL-based expiration for session cleanup |
| Kafka for streaming | Battle-tested for telecom-scale event streaming; exactly-once semantics; schema registry integration |
| VictoriaMetrics for time-series | 10x more efficient than Prometheus for long-term storage; compatible with PromQL |
| Event-driven over request-response | Decouples ingestion from inference; enables backpressure; supports replay |

---

## 5. Detailed Component Design

### 5.1 Log Ingestion Service

**Responsibility:** Accept, validate, and route log events from multiple sources.

```
┌─────────────────────────────────────────────────┐
│             LOG INGESTION SERVICE                 │
│                                                   │
│  ┌───────────┐    ┌──────────────┐    ┌────────┐│
│  │ Kafka     │───▶│ Schema       │───▶│ Router ││
│  │ Consumer  │    │ Validator    │    │        ││
│  └───────────┘    └──────────────┘    └────────┘│
│  ┌───────────┐    ┌──────────────┐       │      │
│  │ REST      │───▶│ Rate Limiter │───────┘      │
│  │ Endpoint  │    │ + Admission  │              │
│  └───────────┘    └──────────────┘              │
│  ┌───────────┐                                   │
│  │ S3 Batch  │───▶ (same pipeline)              │
│  │ Trigger   │                                   │
│  └───────────┘                                   │
└─────────────────────────────────────────────────┘
```

**Internal Design:**

```python
# Ingestion Service - Core Interface
class LogIngestionService:
    """
    Accepts logs from streaming (Kafka), REST API, and batch (S3) sources.
    Validates schema, applies rate limiting, and routes to preprocessing.
    """
    
    async def consume_stream(self, topic: str) -> AsyncIterator[RawLogEvent]:
        """Kafka consumer with consumer group management."""
        pass
    
    async def ingest_batch(self, s3_uri: str) -> BatchJob:
        """Triggers batch processing job for S3-stored log files."""
        pass
    
    async def ingest_single(self, event: RawLogEvent) -> IngestionResult:
        """REST endpoint for single log event ingestion."""
        pass
```

**Schema (Raw Log Event):**
```json
{
  "source_id": "gnb-001",
  "timestamp": "2026-01-15T10:30:00.123Z",
  "log_level": "WARNING",
  "raw_message": "RRC connection failure for UE 0x3FA1, cause: handoverFailure, target_cell: 102",
  "metadata": {
    "cell_id": "cell-102",
    "gnb_id": "gnb-001",
    "region": "us-east-1"
  }
}
```

**Rate Limiting Strategy:**
- Token bucket: 10K events/sec per source
- Backpressure: Kafka consumer lag monitoring → pause consumption at 100K lag
- Priority queue: Critical alarms bypass rate limiting

---

### 5.2 Preprocessing & Feature Extraction Service

**Responsibility:** Parse raw logs, extract structured features and KPIs.

**Internal Design:**

```python
class PreprocessingPipeline:
    """
    Multi-stage pipeline: Parse → Normalize → Extract → Enrich → Emit
    """
    
    def __init__(self):
        self.parser = HybridLogParser()       # Drain3 + regex fallback
        self.normalizer = LogNormalizer()       # Timestamp/format normalization
        self.kpi_extractor = KPIExtractor()    # Domain-specific KPI extraction
        self.enricher = ContextEnricher()      # Add cell topology context
    
    async def process(self, raw_event: RawLogEvent) -> ProcessedEvent:
        parsed = self.parser.parse(raw_event)
        normalized = self.normalizer.normalize(parsed)
        kpis = self.kpi_extractor.extract(normalized)
        enriched = self.enricher.enrich(normalized, kpis)
        return enriched


class HybridLogParser:
    """
    Uses Drain3 for template mining + regex patterns for known formats.
    Falls back to LLM-based parsing for unrecognized formats.
    """
    
    def parse(self, event: RawLogEvent) -> ParsedLog:
        # 1. Try regex patterns (fastest, ~0.1ms)
        result = self._try_regex(event)
        if result: return result
        
        # 2. Try Drain3 template matching (~1ms)
        result = self._try_drain(event)
        if result: return result
        
        # 3. Fallback: SLM-based parsing (~100ms)
        return self._slm_parse(event)
```

**Extracted KPI Schema:**
```json
{
  "event_id": "evt-uuid-001",
  "cell_id": "cell-102",
  "gnb_id": "gnb-001",
  "timestamp": "2026-01-15T10:30:00.123Z",
  "event_type": "handover_failure",
  "kpis": {
    "sinr_db": 3.2,
    "rsrp_dbm": -118,
    "rsrq_db": -15,
    "prb_utilization_pct": 92,
    "bler_pct": 12.5,
    "handover_success_rate": 0.67,
    "rrc_connection_setup_success_rate": 0.85,
    "throughput_dl_mbps": 12.3,
    "throughput_ul_mbps": 3.1,
    "latency_ms": 45,
    "connected_ues": 287,
    "cqi_avg": 7,
    "mcs_dl_avg": 14,
    "ta_advance_us": 12.5,
    "interference_level_dbm": -95
  },
  "parsed_template": "RRC connection failure for UE {ue_id}, cause: {cause}, target_cell: {cell}",
  "severity": "warning",
  "context": {
    "neighboring_cells": ["cell-101", "cell-103"],
    "frequency_band": "n78",
    "bandwidth_mhz": 100
  }
}
```

**Feature Engineering Pipeline:**
```
Raw KPIs → Windowed Statistics (5min, 15min, 1hr)
         → Anomaly Scores (Z-score, isolation forest)
         → Trend Detection (slope over window)
         → Correlation Features (cross-KPI correlations)
         → Topology Features (neighbor cell state)
```

---

### 5.3 RAG Knowledge Service

**Responsibility:** Manage 3GPP knowledge base, perform multi-hop retrieval with re-ranking.

**Architecture:**

```
┌────────────────────────────────────────────────────────────┐
│                  RAG KNOWLEDGE SERVICE                       │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Query       │───▶│ Hybrid       │───▶│ Cross-Encoder │  │
│  │ Analyzer    │    │ Retriever    │    │ Re-ranker     │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         │                   │                    │           │
│         ▼                   ▼                    ▼           │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Query       │    │ Dense + BM25 │    │ Context       │  │
│  │ Expansion   │    │ Fusion       │    │ Compressor    │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│                                                              │
│  Data Stores:                                                │
│  ┌─────────────┐    ┌──────────────┐                       │
│  │ Qdrant      │    │ Elasticsearch│                       │
│  │ (Dense)     │    │ (BM25/Sparse)│                       │
│  └─────────────┘    └──────────────┘                       │
└────────────────────────────────────────────────────────────┘
```

**Multi-Hop Retrieval Strategy:**

```python
class MultiHopRAGRetriever:
    """
    Implements iterative retrieval for complex telecom queries.
    Hop 1: Direct retrieval on original query
    Hop 2: Retrieval on refined query using Hop 1 context
    Hop 3: (Optional) Retrieval on specific sub-questions
    """
    
    async def retrieve(self, query: str, max_hops: int = 2) -> RetrievalResult:
        # Hop 1: Initial retrieval
        initial_results = await self._hybrid_retrieve(query, top_k=10)
        
        # Re-rank with cross-encoder
        reranked = await self._rerank(query, initial_results, top_k=5)
        
        if self._is_sufficient(reranked):
            return self._compress_context(reranked)
        
        # Hop 2: Refined retrieval
        refined_query = await self._refine_query(query, reranked)
        hop2_results = await self._hybrid_retrieve(refined_query, top_k=10)
        
        # Merge and re-rank all results
        merged = self._merge_results(reranked, hop2_results)
        final = await self._rerank(query, merged, top_k=5)
        
        return self._compress_context(final)
    
    async def _hybrid_retrieve(self, query: str, top_k: int):
        """Reciprocal Rank Fusion of dense + sparse retrieval."""
        dense_results = await self.qdrant.search(
            self.embedding_model.encode(query), top_k=top_k*2
        )
        sparse_results = await self.elasticsearch.search(
            query, top_k=top_k*2
        )
        return reciprocal_rank_fusion(dense_results, sparse_results, k=60)
```

**Knowledge Base Preparation:**

| Source | Documents | Chunks | Embedding Model |
|--------|-----------|--------|-----------------|
| 3GPP TS 38.300 (NR Overall Description) | 1 | ~200 | BGE-base-en-v1.5 |
| 3GPP TS 38.331 (RRC Protocol) | 1 | ~500 | BGE-base-en-v1.5 |
| 3GPP TS 38.401 (Architecture) | 1 | ~150 | BGE-base-en-v1.5 |
| 3GPP TS 38.213 (Physical Layer) | 1 | ~300 | BGE-base-en-v1.5 |
| 3GPP TS 38.214 (Data Procedures) | 1 | ~250 | BGE-base-en-v1.5 |
| O-RAN Specifications | ~20 | ~2000 | BGE-base-en-v1.5 |
| Internal KPI Thresholds | 1 | ~50 | BGE-base-en-v1.5 |
| Historical RCA Reports | ~1000 | ~5000 | BGE-base-en-v1.5 |
| **Total** | **~1025** | **~8450** | - |

**Chunking Strategy:**
- Semantic chunking with 512-token chunks, 64-token overlap
- Preserve section boundaries (chapter/subsection awareness)
- Metadata: document title, section hierarchy, specification number, version

---

### 5.4 Multi-Agent Orchestrator

**Responsibility:** Coordinate specialized agents through a directed graph to produce RCA output.

**Agent Graph (LangGraph):**

```
                    ┌───────────────┐
                    │   START       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Signal Agent  │  ← Analyzes KPIs, detects anomalies
                    └───────┬───────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
          ┌───────────────┐  ┌───────────────┐
          │ Knowledge     │  │ Context       │  ← Parallel execution
          │ Agent         │  │ Agent         │
          └───────┬───────┘  └───────┬───────┘
                   └────────┬────────┘
                            ▼
                    ┌───────────────┐
                    │ Hypothesis    │  ← Generates ranked hypotheses
                    │ Agent         │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Validation    │  ← Tests hypotheses against evidence
                    │ Agent         │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │ confidence    │
                    │ >= threshold? │
                    └───┬───────┬───┘
                   Yes  │       │  No
                        ▼       ▼
               ┌────────────┐  ┌────────────┐
               │ Decision   │  │ LLM        │  ← Escalation path
               │ Agent      │  │ Escalation │
               └────────────┘  └────────────┘
                        │              │
                        ▼              ▼
                    ┌───────────────┐
                    │   RCA OUTPUT  │
                    └───────────────┘
```

**Agent Definitions:**

```python
# Agent Message Protocol
@dataclass
class AgentMessage:
    """Structured message passed between agents."""
    agent_id: str
    timestamp: datetime
    message_type: Literal["observation", "hypothesis", "evidence", "decision"]
    content: dict
    confidence: float  # 0.0 - 1.0
    reasoning_steps: list[str]
    metadata: dict


# Signal Analysis Agent
class SignalAnalysisAgent:
    """
    Interprets raw KPIs and detects anomalies.
    Input: ProcessedEvent with KPIs
    Output: Anomaly report with severity scores
    """
    
    THRESHOLDS = {
        "sinr_db": {"critical": 0, "warning": 5, "normal": 10},
        "prb_utilization_pct": {"critical": 95, "warning": 85, "normal": 70},
        "handover_success_rate": {"critical": 0.7, "warning": 0.85, "normal": 0.95},
        "bler_pct": {"critical": 10, "warning": 5, "normal": 2},
    }
    
    async def analyze(self, event: ProcessedEvent, memory: WorkingMemory) -> AgentMessage:
        anomalies = self._detect_anomalies(event.kpis)
        trends = await self._analyze_trends(event.cell_id, window="15min")
        correlations = self._cross_kpi_correlation(event.kpis)
        
        return AgentMessage(
            agent_id="signal_agent",
            message_type="observation",
            content={
                "anomalies": anomalies,
                "trends": trends,
                "correlations": correlations,
                "severity": self._compute_severity(anomalies)
            },
            confidence=0.9,
            reasoning_steps=[
                f"SINR={event.kpis['sinr_db']}dB is below warning threshold (5dB)",
                f"PRB utilization at {event.kpis['prb_utilization_pct']}% exceeds critical threshold",
                f"Correlation detected: high PRB + low SINR suggests interference"
            ]
        )


# Knowledge Retrieval Agent
class KnowledgeRetrievalAgent:
    """
    Fetches relevant 3GPP context based on detected anomalies.
    Input: Anomaly observations from Signal Agent
    Output: Relevant 3GPP specifications and historical context
    """
    
    async def retrieve(self, observations: AgentMessage, memory: WorkingMemory) -> AgentMessage:
        # Construct domain-specific queries
        queries = self._construct_queries(observations.content["anomalies"])
        
        # Multi-hop retrieval
        contexts = []
        for query in queries:
            result = await self.rag_service.retrieve(query, max_hops=2)
            contexts.append(result)
        
        return AgentMessage(
            agent_id="knowledge_agent",
            message_type="evidence",
            content={
                "specifications": contexts,
                "relevant_thresholds": self._extract_thresholds(contexts),
                "known_failure_patterns": self._match_patterns(contexts, observations)
            },
            confidence=0.85,
            reasoning_steps=[
                "Retrieved TS 38.331 Section 5.3.5: Handover failure conditions",
                "Found threshold: T310 expiry with N310=1 triggers RLF declaration",
                "Historical pattern match: similar KPI signature in 3 prior incidents"
            ]
        )


# Hypothesis Generation Agent
class HypothesisAgent:
    """
    Generates ranked hypotheses using CoT reasoning.
    Input: Observations + Evidence from prior agents
    Output: Ranked list of hypotheses with supporting evidence
    """
    
    COT_PROMPT = """
    You are a 5G network expert performing root cause analysis.
    
    Given the following observations and evidence, generate hypotheses for the root cause.
    
    ## Observations
    {observations}
    
    ## 3GPP Context
    {context}
    
    ## Historical Patterns
    {patterns}
    
    Think step by step:
    Step 1: Identify the primary symptoms
    Step 2: Map symptoms to potential causes using 3GPP knowledge
    Step 3: Consider temporal correlations and topology
    Step 4: Rank hypotheses by likelihood
    
    Output format:
    {{
      "hypotheses": [
        {{
          "root_cause": "category",
          "specific_cause": "detailed description",
          "confidence": 0.0-1.0,
          "supporting_evidence": ["..."],
          "contradicting_evidence": ["..."]
        }}
      ]
    }}
    """
    
    async def generate(self, observations: AgentMessage, 
                       evidence: AgentMessage, memory: WorkingMemory) -> AgentMessage:
        prompt = self.COT_PROMPT.format(
            observations=observations.content,
            context=evidence.content["specifications"],
            patterns=evidence.content["known_failure_patterns"]
        )
        
        response = await self.slm_client.generate(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.3
        )
        
        hypotheses = self._parse_hypotheses(response)
        return AgentMessage(
            agent_id="hypothesis_agent",
            message_type="hypothesis",
            content={"hypotheses": hypotheses},
            confidence=max(h["confidence"] for h in hypotheses),
            reasoning_steps=self._extract_cot_steps(response)
        )


# Validation Agent
class ValidationAgent:
    """
    Validates hypotheses against available evidence.
    Checks for contradictions, performs counterfactual reasoning.
    """
    
    async def validate(self, hypotheses: AgentMessage, 
                       all_evidence: list[AgentMessage], 
                       memory: WorkingMemory) -> AgentMessage:
        validated = []
        for hypothesis in hypotheses.content["hypotheses"]:
            # Check supporting evidence strength
            support_score = self._evaluate_support(hypothesis, all_evidence)
            # Check for contradictions
            contradiction_score = self._check_contradictions(hypothesis, all_evidence)
            # Counterfactual: would removing this cause explain the symptoms?
            counterfactual_score = self._counterfactual_test(hypothesis, all_evidence)
            
            validated.append({
                **hypothesis,
                "validation_score": (support_score * 0.4 + 
                                    (1 - contradiction_score) * 0.3 + 
                                    counterfactual_score * 0.3),
                "validation_details": {
                    "support": support_score,
                    "contradictions": contradiction_score,
                    "counterfactual": counterfactual_score
                }
            })
        
        # Sort by validation score
        validated.sort(key=lambda x: x["validation_score"], reverse=True)
        
        return AgentMessage(
            agent_id="validation_agent",
            message_type="evidence",
            content={"validated_hypotheses": validated},
            confidence=validated[0]["validation_score"] if validated else 0.0,
            reasoning_steps=[
                f"Top hypothesis '{validated[0]['root_cause']}' has support score {validated[0]['validation_score']:.2f}",
                f"No contradicting evidence found for primary hypothesis",
                f"Counterfactual test passed: removing cause would explain symptom resolution"
            ]
        )


# Decision Agent
class DecisionAgent:
    """
    Final decision maker. Produces RCA output with confidence and full reasoning trace.
    Decides whether to escalate to LLM based on confidence threshold.
    """
    
    CONFIDENCE_THRESHOLD = 0.75
    ESCALATION_THRESHOLD = 0.5
    
    async def decide(self, validated: AgentMessage, 
                     full_trace: list[AgentMessage], 
                     memory: WorkingMemory) -> RCAResult:
        top_hypothesis = validated.content["validated_hypotheses"][0]
        
        if top_hypothesis["validation_score"] < self.ESCALATION_THRESHOLD:
            # Escalate to LLM
            return await self._escalate_to_llm(full_trace)
        
        # Compile reasoning trace
        reasoning_trace = self._compile_trace(full_trace)
        
        # Calibrate confidence
        calibrated_confidence = await self.calibrator.calibrate(
            raw_confidence=top_hypothesis["validation_score"],
            features=self._extract_calibration_features(full_trace)
        )
        
        return RCAResult(
            root_cause=top_hypothesis["root_cause"],
            specific_cause=top_hypothesis["specific_cause"],
            confidence=calibrated_confidence,
            reasoning_trace=reasoning_trace,
            supporting_evidence=top_hypothesis["supporting_evidence"],
            recommended_actions=self._generate_actions(top_hypothesis),
            escalated=False
        )
```

**Shared Working Memory:**

```python
class WorkingMemory:
    """
    Redis-backed shared state for agent collaboration.
    Scoped per RCA session with TTL-based expiration.
    """
    
    def __init__(self, session_id: str, redis_client: Redis):
        self.session_id = session_id
        self.redis = redis_client
        self.ttl = 3600  # 1 hour session TTL
    
    async def store(self, key: str, value: dict):
        await self.redis.hset(f"wm:{self.session_id}", key, json.dumps(value))
        await self.redis.expire(f"wm:{self.session_id}", self.ttl)
    
    async def retrieve(self, key: str) -> dict:
        data = await self.redis.hget(f"wm:{self.session_id}", key)
        return json.loads(data) if data else None
    
    async def get_full_context(self) -> dict:
        """Retrieve all working memory for the session."""
        return await self.redis.hgetall(f"wm:{self.session_id}")
```

---

### 5.5 SLM Inference Service

**Responsibility:** High-throughput, low-latency model serving for SLMs.

**Architecture:**

```
┌────────────────────────────────────────────────────────────┐
│                 SLM INFERENCE SERVICE                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              vLLM Serving Engine                      │   │
│  │  ┌─────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │Continuous│  │ PagedAttention│  │ Speculative   │  │   │
│  │  │ Batching │  │ (Memory Mgmt)│  │ Decoding      │  │   │
│  │  └─────────┘  └──────────────┘  └───────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐     │
│  │ Model       │  │ Request      │  │ Confidence    │     │
│  │ Router      │  │ Queue        │  │ Calibration   │     │
│  │ (A/B test)  │  │ (Priority)   │  │ Layer         │     │
│  └─────────────┘  └──────────────┘  └───────────────┘     │
└────────────────────────────────────────────────────────────┘
```

**Model Configuration:**

| Model | Parameters | Context Window | Use Case | GPU Memory | Latency (P50) |
|-------|-----------|---------------|----------|------------|---------------|
| Phi-3-mini-4k | 3.8B | 4K tokens | Fast initial analysis | 8GB | ~400ms |
| Mistral-7B-v0.3 | 7B | 8K tokens | Complex reasoning | 14GB | ~800ms |
| Qwen2.5-7B | 7B | 32K tokens | Long-context scenarios | 14GB | ~900ms |

**Inference Optimization:**
- Quantization: AWQ 4-bit for 2x memory reduction, <2% accuracy loss
- Continuous batching: Process up to 32 concurrent requests
- KV-cache optimization: PagedAttention for efficient memory usage
- Speculative decoding: Draft model (Phi-3-mini) + verify with Mistral-7B for 1.5x speedup

**Escalation Logic:**
```python
class InferenceRouter:
    """Routes requests to appropriate model based on complexity and confidence."""
    
    async def route(self, request: InferenceRequest) -> InferenceResponse:
        # Stage 1: Try fast SLM (Phi-3-mini)
        response = await self.phi3_client.generate(request)
        
        if response.confidence >= 0.8:
            return response
        
        # Stage 2: Try larger SLM (Mistral-7B)
        response = await self.mistral_client.generate(request)
        
        if response.confidence >= 0.6:
            return response
        
        # Stage 3: Escalate to LLM (API call)
        return await self.llm_fallback.generate(request)
```

---

### 5.6 Evaluation Engine

**Responsibility:** Benchmark SLM vs LLM reasoning accuracy in telecom domain.

**Evaluation Framework:**

```python
class EvaluationEngine:
    """
    Comprehensive benchmarking framework for model comparison.
    """
    
    METRICS = {
        "reasoning_accuracy": ReasoningAccuracyMetric(),    # CoT step correctness
        "classification_accuracy": ClassificationMetric(),   # Root cause F1
        "latency_p50": LatencyMetric(percentile=50),
        "latency_p95": LatencyMetric(percentile=95),
        "confidence_calibration": CalibrationMetric(),       # ECE score
        "reasoning_completeness": CompletenessMetric(),      # Steps coverage
        "hallucination_rate": HallucinationMetric(),         # Factual consistency
        "cost_per_query": CostMetric(),
    }
    
    BENCHMARK_DATASETS = {
        "itu_challenge": ITUChallengeDataset(),
        "loghub_telecom": LoghubTelecomDataset(),
        "synthetic_5g": Synthetic5GDataset(),  # Generated from 3GPP specs
    }
    
    async def run_benchmark(self, models: list[str], 
                           datasets: list[str],
                           num_samples: int = 500) -> BenchmarkReport:
        results = {}
        for model in models:
            for dataset in datasets:
                samples = self.BENCHMARK_DATASETS[dataset].sample(num_samples)
                model_results = await self._evaluate_model(model, samples)
                results[f"{model}/{dataset}"] = model_results
        
        return BenchmarkReport(
            results=results,
            comparison=self._compute_comparison(results),
            statistical_significance=self._compute_significance(results)
        )
```

**Evaluation Metrics Detail:**

| Metric | Definition | Target (SLM) | Target (LLM) |
|--------|-----------|--------------|--------------|
| Classification F1 | Weighted F1 for root cause categories | ≥0.82 | ≥0.90 |
| Reasoning Accuracy | % of CoT steps that are factually correct | ≥0.75 | ≥0.88 |
| Confidence ECE | Expected Calibration Error | ≤0.12 | ≤0.08 |
| Hallucination Rate | % of responses with fabricated claims | ≤8% | ≤3% |
| Latency P95 | 95th percentile end-to-end | ≤2s | ≤10s |
| Cost/Query | Average inference cost | ~$0.002 | ~$0.03 |

---

### 5.7 API Gateway

**Endpoints:**

```yaml
openapi: 3.0.3
info:
  title: 5G RCA Multi-Agent API
  version: 1.0.0

paths:
  /api/v1/analyze:
    post:
      summary: Submit log event for RCA
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AnalyzeRequest'
      responses:
        '202':
          description: Analysis accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalyzeResponse'

  /api/v1/rca/{rca_id}:
    get:
      summary: Get RCA result
      parameters:
        - name: rca_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: RCA result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RCAResult'

  /api/v1/rca/{rca_id}/trace:
    get:
      summary: Get full reasoning trace
      responses:
        '200':
          description: Detailed CoT trace

  /api/v1/benchmark:
    post:
      summary: Run model benchmark
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/BenchmarkRequest'

  /api/v1/feedback:
    post:
      summary: Submit human feedback on RCA result
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/FeedbackRequest'

  /api/v1/health:
    get:
      summary: Health check

  /ws/v1/stream:
    get:
      summary: WebSocket for real-time RCA results

components:
  schemas:
    AnalyzeRequest:
      type: object
      required: [log_events]
      properties:
        log_events:
          type: array
          items:
            $ref: '#/components/schemas/LogEvent'
        priority:
          type: string
          enum: [critical, high, normal, low]
        force_llm:
          type: boolean
          default: false

    RCAResult:
      type: object
      properties:
        rca_id:
          type: string
          format: uuid
        root_cause:
          type: string
          enum: [interference, handover_failure, resource_congestion, hardware_fault, software_fault, configuration_error]
        specific_cause:
          type: string
        confidence:
          type: number
          minimum: 0
          maximum: 1
        reasoning_trace:
          type: array
          items:
            $ref: '#/components/schemas/ReasoningStep'
        supporting_evidence:
          type: array
          items:
            type: string
        recommended_actions:
          type: array
          items:
            type: string
        model_used:
          type: string
        escalated:
          type: boolean
        latency_ms:
          type: integer
        timestamp:
          type: string
          format: date-time

    ReasoningStep:
      type: object
      properties:
        step_number:
          type: integer
        agent:
          type: string
        action:
          type: string
        observation:
          type: string
        conclusion:
          type: string
        confidence:
          type: number
```

---

### 5.8 Dashboard Service

**Technology:** Next.js 14 + React + TailwindCSS + Recharts

**Views:**
1. **Real-time RCA Feed** — Live stream of RCA results via WebSocket
2. **Cell Health Overview** — Heatmap of cell status across topology
3. **Reasoning Explorer** — Interactive CoT trace visualization
4. **Benchmark Dashboard** — Model comparison charts
5. **KPI Trends** — Time-series visualization of cell metrics

---

## 6. Data Flow & Sequence Descriptions

### 6.1 End-to-End RCA Flow (Happy Path)

```
┌────┐     ┌─────────┐     ┌──────────┐     ┌─────┐     ┌────────────┐
│gNB │     │Ingestion│     │Preprocess│     │Kafka│     │Orchestrator│
└──┬─┘     └────┬────┘     └────┬─────┘     └──┬──┘     └─────┬──────┘
   │             │               │              │               │
   │ Log Event   │               │              │               │
   │────────────▶│               │              │               │
   │             │ Validate      │              │               │
   │             │──────────────▶│              │               │
   │             │               │ Extract KPIs │               │
   │             │               │─────────────▶│               │
   │             │               │              │ Processed     │
   │             │               │              │ Event         │
   │             │               │              │──────────────▶│
   │             │               │              │               │
   │             │               │              │     ┌─────────┴──────────┐
   │             │               │              │     │ Agent Orchestration│
   │             │               │              │     │                    │
   │             │               │              │     │ 1. Signal Agent    │
   │             │               │              │     │ 2. Knowledge Agent │
   │             │               │              │     │ 3. Hypothesis Agent│
   │             │               │              │     │ 4. Validation Agent│
   │             │               │              │     │ 5. Decision Agent  │
   │             │               │              │     └─────────┬──────────┘
   │             │               │              │               │
   │             │               │              │               │ Store Result
   │             │               │              │               │────────────▶ MongoDB
   │             │               │              │               │
   │             │               │              │               │ Push via WS
   │             │               │              │               │────────────▶ Dashboard
```

### 6.2 Agent Orchestration Sequence (Detailed)

```
Time ──────────────────────────────────────────────────────────────────▶

Orchestrator    Signal      Knowledge    Hypothesis   Validation   Decision
    │              │            │             │            │           │
    │─ invoke ────▶│            │             │            │           │
    │              │─ analyze   │             │            │           │
    │              │  KPIs      │             │            │           │
    │              │            │             │            │           │
    │◀─ anomalies ─│            │             │            │           │
    │              │            │             │            │           │
    │─── store in working memory ────────────────────────────────────▶│
    │              │            │             │            │           │
    │─ invoke (parallel) ─────▶│             │            │           │
    │─ invoke (parallel) ─────────────────────────────────▶           │
    │              │            │             │            │           │
    │              │            │─ RAG query  │            │           │
    │              │            │  (multi-hop)│            │           │
    │              │            │             │            │           │
    │◀─ 3GPP context ──────────│             │            │           │
    │◀─ cell topology ──────────────────────────────────── │          │
    │              │            │             │            │           │
    │─── store in working memory ────────────────────────────────────▶│
    │              │            │             │            │           │
    │─ invoke ────────────────────────────── ▶│            │           │
    │              │            │             │─ CoT       │           │
    │              │            │             │  reasoning │           │
    │              │            │             │  (SLM)     │           │
    │◀─ hypotheses ──────────────────────────│            │           │
    │              │            │             │            │           │
    │─ invoke ─────────────────────────────────────────── ▶│          │
    │              │            │             │            │─validate  │
    │              │            │             │            │ hypotheses│
    │◀─ validated ─────────────────────────────────────────│          │
    │              │            │             │            │           │
    │─ check confidence ──────────────────────────────────────────── ▶│
    │              │            │             │            │           │
    │  [confidence >= 0.75]    │             │            │           │
    │◀─ RCA Result ────────────────────────────────────────────────── │
    │              │            │             │            │           │
    │  [confidence < 0.5 → ESCALATE TO LLM]  │            │           │
```

### 6.3 Escalation Flow

```
Decision Agent ─── confidence < 0.5 ──▶ Compile full context
                                              │
                                              ▼
                                        LLM API Call
                                        (GPT-4/Claude)
                                              │
                                              ▼
                                        Parse LLM response
                                              │
                                              ▼
                                        Merge with agent traces
                                              │
                                              ▼
                                        Return RCA (escalated=true)
```

### 6.4 Feedback Loop Flow

```
User ─── submits feedback ──▶ Feedback Service
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                   Update RAG    Log for       Trigger
                   (add to       retraining    confidence
                   history DB)   pipeline      recalibration
```

---

## 7. Technology Stack

### 7.1 Complete Stack

| Layer | Component | Technology | Justification |
|-------|-----------|-----------|---------------|
| **Frontend** | Dashboard | Next.js 14 + React 18 | SSR for initial load, WebSocket for real-time |
| **Frontend** | Visualization | Recharts + D3.js | Interactive CoT trace visualization |
| **Frontend** | State Management | Zustand | Lightweight, sufficient for dashboard state |
| **API** | Gateway | FastAPI (Python 3.11+) | Async-native, auto-OpenAPI docs, Pydantic validation |
| **API** | Internal RPC | gRPC + Protobuf | Type-safe, efficient inter-service communication |
| **API** | WebSocket | FastAPI WebSockets | Native async WebSocket support |
| **Orchestration** | Agent Framework | LangGraph 0.2+ | Stateful agent graphs with conditional routing |
| **Orchestration** | Workflow Engine | Temporal (optional) | Long-running workflow reliability |
| **AI/ML** | SLM Serving | vLLM 0.5+ | Continuous batching, PagedAttention, best throughput |
| **AI/ML** | Embedding | BGE-base-en-v1.5 | Strong retrieval performance, 768-dim, fast inference |
| **AI/ML** | Re-ranking | BGE-reranker-v2-m3 | Cross-encoder re-ranking for precision |
| **AI/ML** | SLM Models | Phi-3-mini-4k, Mistral-7B-v0.3 | Best accuracy/size ratio for 2026 |
| **AI/ML** | LLM Fallback | GPT-4o / Claude 3.5 (API) | High-accuracy fallback |
| **AI/ML** | Log Parsing | Drain3 | State-of-the-art online log parsing |
| **Data** | Streaming | Apache Kafka 3.7+ | Exactly-once, high throughput, schema registry |
| **Data** | Vector DB | Qdrant 1.8+ | Filtering, horizontal scaling, payload storage |
| **Data** | Sparse Search | Elasticsearch 8.x | BM25 for keyword-based retrieval |
| **Data** | Time-Series | VictoriaMetrics | 10x more efficient than Prometheus for storage |
| **Data** | Cache/WM | Redis 7+ (Cluster) | Sub-ms working memory, pub/sub for events |
| **Data** | Document Store | MongoDB 7+ | Flexible schema for RCA results |
| **Data** | Object Storage | MinIO / S3 | Batch log storage, model artifacts |
| **Infra** | Container Orchestration | Kubernetes (EKS/GKE) | GPU node pools, HPA, production-grade |
| **Infra** | Service Mesh | Istio (optional) | mTLS, traffic management, observability |
| **Infra** | CI/CD | GitHub Actions + ArgoCD | GitOps deployment, canary releases |
| **Infra** | Container Registry | ECR / GCR | Secure image storage |
| **Observability** | Metrics | Prometheus + Grafana | Standard k8s monitoring |
| **Observability** | Tracing | OpenTelemetry + Jaeger | Distributed tracing across agents |
| **Observability** | Logging | Fluentd + Elasticsearch | Centralized log aggregation |
| **Observability** | AI Tracing | LangSmith / Langfuse | LLM-specific observability (token usage, latency) |
| **Security** | Auth | OAuth2 + JWT (Keycloak) | Standard enterprise auth |
| **Security** | Secrets | HashiCorp Vault | Secure credential management |
| **Security** | API Security | Rate limiting + WAF | Protection against abuse |

### 7.2 Model Selection Rationale

| Model | Why Selected | Trade-off |
|-------|-------------|-----------|
| **Phi-3-mini-4k (3.8B)** | Excellent reasoning for size, fast inference (~400ms), fits on single T4 GPU | Limited context window (4K), may miss complex multi-factor issues |
| **Mistral-7B-v0.3 (7B)** | Best open-source 7B model for reasoning tasks, 8K context | 2x memory/latency vs Phi-3, but significantly better on complex cases |
| **Qwen2.5-7B (7B)** | 32K context window for cases requiring extensive log history | Similar performance to Mistral but with extended context capability |
| **GPT-4o (fallback)** | Highest accuracy, best for ambiguous/complex cases | High cost ($0.03/query), high latency (2-5s), API dependency |

---

## 8. Scalability, Reliability & Security

### 8.1 Scalability Strategy

**Horizontal Scaling:**

| Component | Scaling Trigger | Strategy | Min/Max Replicas |
|-----------|----------------|----------|-----------------|
| Ingestion Service | Kafka consumer lag > 50K | Add consumers (partition-based) | 3 / 20 |
| Preprocessing | CPU > 70% | HPA on CPU/memory | 3 / 15 |
| Agent Orchestrator | Queue depth > 100 | HPA on custom metric | 2 / 10 |
| SLM Inference | GPU utilization > 80% | GPU node autoscaling | 2 / 8 |
| RAG Service | Request latency P95 > 500ms | HPA on latency | 2 / 10 |
| Vector DB (Qdrant) | Memory > 80% | Add shards | 3 / 9 |
| Redis | Memory > 75% | Cluster expansion | 3 / 9 |

**GPU Scaling Strategy:**
```yaml
# Kubernetes GPU Autoscaler Config
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: slm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: slm-inference
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 120
    scaleDown:
      stabilizationWindowSeconds: 300
```

**Caching Strategy:**
```
Layer 1: In-process LRU cache (embeddings, parsed templates) — ~10ms savings
Layer 2: Redis cache (frequent RAG queries, KPI thresholds) — ~50ms savings  
Layer 3: Qdrant result cache (identical vector queries) — ~100ms savings

Cache invalidation: TTL-based (RAG: 1hr, KPI: 5min, embeddings: 24hr)
Hit rate target: >60% for RAG queries in steady state
```

### 8.2 Reliability & Fault Tolerance

**Failure Modes & Mitigations:**

| Failure Mode | Impact | Detection | Mitigation | RTO |
|--------------|--------|-----------|------------|-----|
| SLM inference pod crash | RCA requests fail | Health check + readiness probe | Auto-restart + spare capacity | <30s |
| Kafka broker failure | Log ingestion pauses | Broker health monitoring | Multi-broker cluster (RF=3) | <10s |
| Vector DB unavailable | RAG fails | Connection pool monitoring | Read replicas + degraded mode (rule-based) | <60s |
| LLM API outage | Escalation fails | API health check | Retry with backoff + use best SLM result | N/A |
| Redis failure | Working memory lost | Sentinel monitoring | Redis Cluster with failover | <5s |
| Model loading failure | Cold start fails | Startup probe | Pre-pulled images + init containers | <60s |
| Network partition | Inter-service communication fails | Mesh health | Circuit breaker + local fallback | <30s |

**Graceful Degradation Tiers:**

```
Tier 1 (Full): Multi-agent CoT with RAG + SLM reasoning
   ↓ (SLM unavailable)
Tier 2 (Reduced): Rule-based analysis + cached RAG context
   ↓ (RAG unavailable)  
Tier 3 (Minimal): Pure rule-based threshold analysis
   ↓ (All services degraded)
Tier 4 (Alert only): Forward raw anomaly alerts without RCA
```

**Circuit Breaker Configuration:**
```python
# Circuit breaker for SLM inference
circuit_breaker_config = {
    "failure_threshold": 5,          # Open after 5 failures
    "success_threshold": 3,          # Close after 3 successes
    "timeout": 30,                   # Half-open after 30s
    "expected_exception": TimeoutError,
    "fallback": rule_based_analysis   # Degraded mode function
}
```

### 8.3 Security Design

**Authentication & Authorization:**

```
┌──────────┐      ┌──────────┐      ┌──────────────┐
│  Client  │─────▶│  OAuth2  │─────▶│  API Gateway │
│          │ JWT  │ (Keycloak)│      │  (Validate)  │
└──────────┘      └──────────┘      └──────────────┘
                                            │
                                     ┌──────┴──────┐
                                     │    RBAC     │
                                     │  Policies   │
                                     └─────────────┘
```

**Role-Based Access Control:**

| Role | Permissions |
|------|------------|
| `rca-viewer` | Read RCA results, view dashboard |
| `rca-analyst` | Submit logs, trigger analysis, view traces |
| `rca-admin` | Run benchmarks, manage models, configure agents |
| `system-admin` | Full access including infrastructure |

**Data Security:**

| Control | Implementation |
|---------|---------------|
| Encryption at rest | AES-256 for all stored data |
| Encryption in transit | TLS 1.3 for all inter-service communication |
| Data anonymization | PII stripping from logs (UE IDs, IMSIs) before storage |
| Audit logging | All API calls logged with user identity |
| Network isolation | Kubernetes NetworkPolicies, private subnets |
| Secret management | HashiCorp Vault for API keys, model credentials |
| Input validation | Pydantic schema validation on all API inputs |
| Rate limiting | Token bucket: 100 req/min for viewers, 1000 req/min for analysts |

**Threat Model:**

| Threat | Risk | Mitigation |
|--------|------|-----------|
| Prompt injection via log content | Medium | Input sanitization + output validation |
| Model extraction attacks | Low | No raw model weights exposed via API |
| Data exfiltration | Medium | Network policies + egress controls |
| DoS via expensive queries | High | Rate limiting + query complexity limits |
| Unauthorized model access | Medium | mTLS between services + RBAC |

---

## 9. Trade-offs & Alternatives

### 9.1 Key Design Decisions

| Decision | Chosen | Alternative | Why Chosen |
|----------|--------|-------------|-----------|
| Multi-agent vs single chain | Multi-agent | Single reasoning chain | Modularity, testability, parallel execution; each agent can be improved independently |
| RAG vs fine-tuning | RAG (primary) | Fine-tuned telecom model | Flexibility, updateability (3GPP specs change); fine-tuning has high maintenance cost |
| vLLM vs TensorRT-LLM | vLLM | TensorRT-LLM | Easier deployment, broader model support, continuous batching; TRT-LLM is faster but harder to maintain |
| Qdrant vs FAISS | Qdrant | FAISS, Pinecone | Production features (filtering, scaling, persistence) without vendor lock-in; FAISS is in-memory only |
| LangGraph vs custom orchestration | LangGraph | Custom DAG engine | Proven framework, good debugging tools, active development; custom would give more control but high maintenance |
| Kafka vs Pulsar | Kafka | Apache Pulsar | Larger ecosystem, more operational expertise available; Pulsar has better multi-tenancy but less mature tooling |
| FastAPI vs Node.js | FastAPI | Express/Fastify | Native Python ML ecosystem integration, async support, auto-docs; Node would add language boundary |

### 9.2 Cost Analysis

| Configuration | Monthly Cost (est.) | RCA/month Capacity | Cost/RCA |
|---------------|--------------------|--------------------|----------|
| **Minimal (dev)** | ~$2,000 | ~100K | $0.02 |
| 2x T4 GPU, 3-node k8s | | | |
| **Standard (prod)** | ~$8,000 | ~1M | $0.008 |
| 4x A10G GPU, 6-node k8s | | | |
| **High-scale** | ~$20,000 | ~5M | $0.004 |
| 8x A10G GPU, 12-node k8s | | | |

*Assumes 90% SLM, 10% LLM escalation. LLM costs additional ~$0.03/escalated query.*

### 9.3 Future Considerations

1. **Model evolution:** As SLMs improve (2026-2027), the escalation threshold can be raised, reducing LLM costs
2. **Fine-tuning path:** Once sufficient labeled data is collected (~50K samples), consider fine-tuning a telecom-specific SLM
3. **Edge deployment:** For ultra-low-latency use cases, quantized Phi-3 can run on edge devices (NVIDIA Jetson)
4. **Multi-modal:** Future integration of RF signal visualizations (spectrograms) for visual RCA
5. **Federated learning:** Cross-operator model improvement without sharing sensitive data

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up Kubernetes cluster with GPU nodes
- [ ] Deploy Kafka + schema registry
- [ ] Implement log ingestion service with 3 source adapters
- [ ] Implement preprocessing pipeline (Drain3 + regex)
- [ ] Deploy Qdrant + prepare 3GPP embedding pipeline
- [ ] Basic FastAPI gateway with health endpoints

### Phase 2: AI Core (Weeks 5-8)
- [ ] Deploy vLLM with Phi-3 and Mistral-7B
- [ ] Implement RAG pipeline (hybrid retrieval + re-ranking)
- [ ] Build Signal Analysis Agent
- [ ] Build Knowledge Retrieval Agent
- [ ] Build Hypothesis Agent with CoT prompting
- [ ] Implement LangGraph orchestration (basic linear flow)

### Phase 3: Multi-Agent System (Weeks 9-12)
- [ ] Build Validation Agent
- [ ] Build Decision Agent with confidence calibration
- [ ] Implement shared working memory (Redis)
- [ ] Add conditional routing (escalation logic)
- [ ] Implement LLM fallback gateway
- [ ] End-to-end integration testing

### Phase 4: Evaluation & Benchmarking (Weeks 13-16)
- [ ] Prepare ITU Challenge dataset integration
- [ ] Prepare Loghub dataset integration
- [ ] Implement evaluation engine with all metrics
- [ ] Run initial benchmarks (SLM vs LLM)
- [ ] Confidence calibration training
- [ ] Generate benchmark report

### Phase 5: Production Hardening (Weeks 17-20)
- [ ] Implement circuit breakers + graceful degradation
- [ ] Add observability (OpenTelemetry, Grafana dashboards)
- [ ] Security hardening (OAuth2, rate limiting, data anonymization)
- [ ] Load testing + performance optimization
- [ ] Build Next.js dashboard
- [ ] CI/CD pipeline (GitHub Actions + ArgoCD)

### Phase 6: Optimization (Weeks 21-24)
- [ ] Model quantization (AWQ 4-bit)
- [ ] Caching optimization
- [ ] Speculative decoding implementation
- [ ] Feedback loop integration
- [ ] Documentation + knowledge transfer

---

## 11. Monitoring & Observability

### 11.1 Key Dashboards

**System Health Dashboard:**
- Pod status across all services
- GPU utilization and memory
- Kafka consumer lag
- Request latency percentiles (P50, P95, P99)
- Error rates by service

**AI/ML Dashboard:**
- Model inference latency distribution
- Token usage per request
- Escalation rate (SLM → LLM)
- Confidence score distribution
- Cache hit rates (RAG, embedding)

**RCA Quality Dashboard:**
- Classification accuracy (rolling 7-day)
- Reasoning step completeness
- User feedback scores
- False positive/negative rates
- Concept drift indicators

### 11.2 Alerting Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High inference latency | P95 > 5s for 5min | Warning | Scale up inference pods |
| Low classification accuracy | Accuracy < 0.75 for 1hr | Critical | Investigate, potential model degradation |
| High escalation rate | >30% escalations for 30min | Warning | Check SLM health, potential data drift |
| Kafka lag critical | Consumer lag > 500K | Critical | Scale consumers, check processing bottleneck |
| GPU OOM | Any OOM event | Critical | Check batch size, memory leak |
| RAG retrieval failure | Error rate > 5% for 5min | Warning | Check Qdrant health, fallback to cache |

---

## 12. Summary

This design delivers a **production-grade, research-publishable** system that:

1. **Solves the core problem:** Automated, explainable RCA for 5G gNB environments
2. **Balances cost and accuracy:** SLM-first with intelligent LLM escalation
3. **Ensures domain accuracy:** Multi-hop RAG grounded in 3GPP specifications
4. **Provides explainability:** Full CoT reasoning traces through multi-agent collaboration
5. **Enables research:** Rigorous benchmarking framework for SLM vs LLM comparison
6. **Scales horizontally:** Event-driven architecture with GPU-aware autoscaling
7. **Degrades gracefully:** 4-tier degradation strategy ensures availability
8. **Maintains security:** Defense-in-depth with telecom-grade data protection

The architecture is designed to be incrementally buildable (6-phase roadmap) while maintaining a clear path from prototype to production deployment.
