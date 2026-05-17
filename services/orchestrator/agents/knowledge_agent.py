"""
Knowledge Retrieval Agent - Fetches relevant 3GPP context for the RCA.
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
    Fetches relevant 3GPP specifications and historical context
    based on anomalies detected by the Signal Agent.
    
    Input: Signal analysis results from working memory
    Output: Relevant 3GPP context, thresholds, and known failure patterns
    """

    agent_type = AgentType.KNOWLEDGE

    # Pre-defined query templates for common failure modes
    QUERY_TEMPLATES = {
        "interference_detected": [
            "5G NR interference management procedures 3GPP",
            "Inter-cell interference coordination ICIC mechanisms",
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

    def __init__(self, rag_service: RAGService):
        super().__init__()
        self.rag_service = rag_service

    async def execute(self, event: ProcessedEvent, memory: WorkingMemory, **kwargs) -> AgentMessage:
        """Retrieve relevant 3GPP knowledge based on detected anomalies."""
        start_time = time.time()
        self._invocation_count += 1

        # Get signal analysis from working memory
        signal_analysis = await memory.retrieve("signal_analysis")
        
        # Construct queries based on anomalies and event type
        queries = self._construct_queries(event, signal_analysis)
        
        # Retrieve context for each query
        all_contexts = []
        for query in queries:
            result = await self.rag_service.retrieve(query, top_k=3)
            if result.chunks:
                all_contexts.append({
                    "query": query,
                    "chunks": [
                        {
                            "content": chunk.content,
                            "source": chunk.source_document,
                            "section": chunk.section,
                            "score": chunk.score,
                        }
                        for chunk in result.chunks
                    ],
                })

        # Extract relevant thresholds from context
        thresholds = self._extract_thresholds(all_contexts)
        
        # Match known failure patterns
        known_patterns = self._match_patterns(all_contexts, signal_analysis)

        # Build formatted context for downstream agents
        formatted_context = self._format_context(all_contexts)

        # Store in working memory
        knowledge_result = {
            "contexts": all_contexts,
            "formatted_context": formatted_context,
            "thresholds": thresholds,
            "known_patterns": known_patterns,
            "num_sources": len(all_contexts),
        }
        await memory.store("knowledge_context", knowledge_result)

        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning(queries, all_contexts, known_patterns)

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        confidence = min(0.9, 0.5 + 0.1 * len(all_contexts))

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

        # Query based on event type
        if event.event_type and event.event_type in self.QUERY_TEMPLATES:
            queries.extend(self.QUERY_TEMPLATES[event.event_type][:2])

        # Query based on detected anomalies
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

        # Query based on correlation patterns
        if signal_analysis and signal_analysis.get("correlations"):
            for corr in signal_analysis["correlations"]:
                pattern = corr.get("pattern", "")
                if "interference" in pattern:
                    queries.append("inter-cell interference 3GPP TS 38.213")
                elif "handover" in pattern:
                    queries.append("mobility management 3GPP TS 38.331")
                elif "overload" in pattern:
                    queries.append("cell overload management procedures")

        # Ensure at least one generic query
        if not queries:
            queries.append(f"5G gNB {event.event_type or 'fault'} diagnosis root cause")

        # Deduplicate
        return list(dict.fromkeys(queries))[:5]

    def _extract_thresholds(self, contexts: list[dict]) -> dict[str, str]:
        """Extract numeric thresholds mentioned in retrieved context."""
        thresholds = {}
        
        threshold_keywords = [
            "T310", "T311", "N310", "N311", "A3 offset", "hysteresis",
            "SINR threshold", "RSRP threshold", "Qrxlevmin"
        ]
        
        for ctx in contexts:
            for chunk in ctx.get("chunks", []):
                content = chunk.get("content", "")
                for keyword in threshold_keywords:
                    if keyword.lower() in content.lower():
                        # Extract a relevant sentence
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
        }
        
        for ctx in contexts:
            for chunk in ctx.get("chunks", []):
                content = chunk.get("content", "").lower()
                for keyword, description in pattern_keywords.items():
                    if keyword in content:
                        patterns.append(description)

        return list(set(patterns))

    def _format_context(self, contexts: list[dict]) -> str:
        """Format all contexts into a single string for LLM consumption."""
        parts = []
        for ctx in contexts[:5]:
            for chunk in ctx.get("chunks", [])[:2]:
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")[:500]
                parts.append(f"[{source}]: {content}")
        
        return "\n\n".join(parts)

    def _generate_reasoning(self, queries: list, contexts: list, patterns: list) -> list[str]:
        """Generate reasoning steps."""
        steps = []
        
        steps.append(f"Constructed {len(queries)} domain-specific queries based on detected anomalies")
        
        if contexts:
            sources = set()
            for ctx in contexts:
                for chunk in ctx.get("chunks", []):
                    if chunk.get("source"):
                        sources.add(chunk["source"])
            if sources:
                steps.append(f"Retrieved relevant context from: {', '.join(list(sources)[:3])}")
        
        if patterns:
            steps.append(f"Matched {len(patterns)} known failure patterns: {'; '.join(patterns[:2])}")
        else:
            steps.append("No exact match with previously known failure patterns")
        
        return steps
