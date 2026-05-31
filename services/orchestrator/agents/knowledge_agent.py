"""
Knowledge Retrieval Agent - Fetches relevant 3GPP context and scenario
data from the Qdrant-backed RAG pipeline for downstream RCA agents.
"""

import time

import structlog

from models.schemas import AgentMessage, AgentType, MessageType, ProcessedEvent
from services.orchestrator.agents.base import BaseAgent
from services.orchestrator.working_memory import WorkingMemory
from services.rag.service import RAGService

logger = structlog.get_logger(__name__)


class KnowledgeRetrievalAgent(BaseAgent):
    """
    Fetches relevant 3GPP specifications **and** historical 5G drive-test
    scenario data (from the JSONL RAG dataset collection) based on
    anomalies detected by the Signal Agent.

    Input: Signal analysis results from working memory
    Output: Relevant 3GPP context, scenario-level evidence, thresholds,
            and known failure patterns
    """

    agent_type = AgentType.KNOWLEDGE

    QUERY_TEMPLATES = {
        "interference_detected": [
            "5G NR interference management procedures 3GPP",
            "Inter-cell interference coordination ICIC mechanisms",
            "overlapping coverage co-frequency neighbor cell SINR degradation",
        ],
        "handover_event": [
            "5G NR handover failure conditions and procedures",
            "RRC connection re-establishment after handover failure",
            "Measurement report criteria for handover A3 event",
        ],
        "resource_congestion": [
            "5G NR PRB scheduling and resource allocation",
            "QoS flow management under congestion conditions",
            "Load balancing between cells MLB",
        ],
        "radio_link_failure": [
            "Radio Link Failure detection T310 N310 parameters",
            "RLF recovery procedures in NR",
        ],
        "throughput_degradation": [
            "5G NR throughput degradation causes and diagnosis",
            "Adaptive modulation and coding scheme selection MCS",
            "drive test low throughput cell configuration optimization",
        ],
        "signal_quality_alarm": [
            "SINR measurement and reporting in 5G NR",
            "Beam management procedures beam failure recovery",
        ],
        "hardware_fault": [
            "gNB hardware fault detection and recovery",
            "O-RAN fault management procedures",
        ],
    }

    # Chunk types from the JSONL datasets that carry scenario-level evidence
    _SCENARIO_CHUNK_TYPES = {
        "scenario_overview", "drive_test_analysis", "signaling_analysis",
        "traffic_analysis", "mr_analysis", "cell_configuration",
        "diagnostic_options", "full_scenario", "combined_analysis",
        "root_cause_reference", "domain_knowledge", "raw_timeseries",
    }

    def __init__(self, rag_service: RAGService):
        super().__init__()
        self.rag_service = rag_service

    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Retrieve relevant knowledge and scenario data from Qdrant."""
        start_time = time.time()
        self._invocation_count += 1

        signal_analysis = await memory.retrieve("signal_analysis")

        queries = self._construct_queries(event, signal_analysis)
        filters = {"cell_id": event.cell_id} if event.cell_id else None

        all_contexts = []
        scenario_evidence: list[dict] = []

        for query in queries:
            result = await self.rag_service.retrieve(query, top_k=5, filters=filters)
            if not result.chunks:
                continue

            chunk_dicts = []
            for chunk in result.chunks:
                chunk_type = chunk.metadata.get("chunk_type", "")
                entry = {
                    "content": chunk.content,
                    "source": chunk.source_document,
                    "section": chunk.section,
                    "score": chunk.score,
                    "chunk_type": chunk_type,
                    "scenario_id": chunk.metadata.get("scenario_id", ""),
                    "collection": chunk.metadata.get("collection", ""),
                }
                chunk_dicts.append(entry)

                if chunk_type in self._SCENARIO_CHUNK_TYPES:
                    scenario_evidence.append(entry)

            all_contexts.append({"query": query, "chunks": chunk_dicts})

        thresholds = self._extract_thresholds(all_contexts)
        known_patterns = self._match_patterns(all_contexts, signal_analysis)
        formatted_context = self._format_context(all_contexts)

        knowledge_result = {
            "contexts": all_contexts,
            "formatted_context": formatted_context,
            "thresholds": thresholds,
            "known_patterns": known_patterns,
            "scenario_evidence": scenario_evidence[:10],
            "num_sources": len(all_contexts),
            "num_scenario_chunks": len(scenario_evidence),
        }
        await memory.store("knowledge_context", knowledge_result)

        reasoning_steps = self._generate_reasoning(queries, all_contexts, known_patterns, scenario_evidence)

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        confidence = min(0.95, 0.5 + 0.08 * len(all_contexts) + 0.05 * min(len(scenario_evidence), 5))

        return self._create_message(
            message_type=MessageType.EVIDENCE,
            content=knowledge_result,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            metadata={"latency_ms": round(latency_ms, 2), "queries_made": len(queries)},
        )

    def _construct_queries(self, event: ProcessedEvent, signal_analysis: dict = None) -> list[str]:
        """Construct domain-specific queries from event and anomalies."""
        queries = []

        if event.event_type and event.event_type in self.QUERY_TEMPLATES:
            queries.extend(self.QUERY_TEMPLATES[event.event_type][:2])

        if signal_analysis and signal_analysis.get("anomalies"):
            for anomaly in signal_analysis["anomalies"][:3]:
                kpi = anomaly.get("kpi", "")
                if "sinr" in kpi or "interference" in kpi:
                    queries.append("interference mitigation 5G NR cell")
                elif "handover" in kpi:
                    queries.append("handover optimization parameters 5G")
                elif "prb" in kpi:
                    queries.append("resource block utilization optimization 5G")
                elif "bler" in kpi:
                    queries.append("BLER high causes link adaptation NR")

        if signal_analysis and signal_analysis.get("correlations"):
            for corr in signal_analysis["correlations"]:
                pattern = corr.get("pattern", "")
                if "interference" in pattern:
                    queries.append("inter-cell interference 3GPP TS 38.213")
                elif "handover" in pattern:
                    queries.append("mobility management 3GPP TS 38.331")
                elif "overload" in pattern:
                    queries.append("cell overload management procedures")

        # Cell-specific query to pull matching cell configs from the dataset
        if event.cell_id:
            queries.append(f"Cell Configuration {event.cell_id}")

        if not queries:
            queries.append(f"5G gNB {event.event_type or 'fault'} diagnosis root cause")

        return list(dict.fromkeys(queries))[:7]

    def _extract_thresholds(self, contexts: list[dict]) -> dict[str, str]:
        """Extract numeric thresholds mentioned in retrieved context."""
        thresholds = {}
        threshold_keywords = [
            "T310", "T311", "N310", "N311", "A3 offset", "hysteresis",
            "SINR threshold", "RSRP threshold", "Qrxlevmin",
            "A2 Threshold", "A5 Threshold", "downtilt",
        ]

        for ctx in contexts:
            for chunk in ctx.get("chunks", []):
                content = chunk.get("content", "")
                for keyword in threshold_keywords:
                    if keyword.lower() in content.lower():
                        sentences = content.split(".")
                        for sentence in sentences:
                            if keyword.lower() in sentence.lower():
                                thresholds[keyword] = sentence.strip()[:200]
                                break
        return thresholds

    def _match_patterns(self, contexts: list[dict], signal_analysis: dict = None) -> list[str]:
        """Identify known failure patterns from retrieved knowledge."""
        patterns = []
        pattern_keywords = {
            "ping-pong handover": "Frequent handovers between cells (ping-pong effect)",
            "pilot pollution": "Excessive interference from multiple strong cells",
            "coverage hole": "Area with insufficient signal coverage",
            "overloaded cell": "Cell capacity exceeded, causing degradation",
            "hardware failure": "Physical component malfunction detected",
            "configuration mismatch": "Network parameters inconsistently configured",
            "weak coverage": "Weak RSRP at cell edge due to downtilt or power",
            "overshooting": "Coverage extends beyond intended range causing interference",
            "overlapping coverage": "Multiple co-frequency cells causing interference",
            "missing neighbor": "Missing neighbor relationship causing handover failures",
            "downtilt": "Antenna tilt misconfiguration affecting coverage footprint",
        }

        for ctx in contexts:
            for chunk in ctx.get("chunks", []):
                content = chunk.get("content", "").lower()
                for keyword, description in pattern_keywords.items():
                    if keyword in content:
                        patterns.append(description)

        return list(set(patterns))

    def _format_context(self, contexts: list[dict]) -> str:
        """Format all contexts into a structured string for LLM consumption.

        Groups chunks by type so the downstream LLM sees domain knowledge,
        cell configurations, and scenario analyses as distinct sections.
        """
        domain_parts: list[str] = []
        scenario_parts: list[str] = []
        other_parts: list[str] = []

        for ctx in contexts[:6]:
            for chunk in ctx.get("chunks", [])[:3]:
                source = chunk.get("source", "Unknown")
                chunk_type = chunk.get("chunk_type", "")
                content = chunk.get("content", "")[:600]
                score = chunk.get("score", 0.0)
                label = f"[{source} | {chunk_type} | score={score:.2f}]"

                if chunk_type in ("domain_knowledge", "root_cause_reference"):
                    domain_parts.append(f"{label}\n{content}")
                elif chunk_type in self._SCENARIO_CHUNK_TYPES:
                    scenario_parts.append(f"{label}\n{content}")
                else:
                    other_parts.append(f"{label}\n{content}")

        sections = []
        if domain_parts:
            sections.append("=== Domain Knowledge ===\n" + "\n\n".join(domain_parts))
        if scenario_parts:
            sections.append("=== Scenario Evidence ===\n" + "\n\n".join(scenario_parts))
        if other_parts:
            sections.append("=== Additional Context ===\n" + "\n\n".join(other_parts))

        return "\n\n".join(sections)

    def _generate_reasoning(self, queries, contexts, patterns, scenario_evidence) -> list[str]:
        """Generate reasoning steps."""
        steps = [f"Constructed {len(queries)} domain-specific queries based on detected anomalies"]

        if contexts:
            sources = set()
            for ctx in contexts:
                for chunk in ctx.get("chunks", []):
                    if chunk.get("source"):
                        sources.add(chunk["source"])
            if sources:
                steps.append(f"Retrieved relevant context from: {', '.join(list(sources)[:5])}")

        if scenario_evidence:
            types_seen = set(e.get("chunk_type", "") for e in scenario_evidence)
            steps.append(
                f"Found {len(scenario_evidence)} scenario-level evidence chunks "
                f"(types: {', '.join(sorted(types_seen))})"
            )

        if patterns:
            steps.append(f"Matched {len(patterns)} known failure patterns: {'; '.join(patterns[:3])}")
        else:
            steps.append("No exact match with previously known failure patterns")

        return steps
