"""
RAG Service - Main entry point for the RAG knowledge pipeline.
Combines embedding, retrieval, and re-ranking into a unified service.
Supports both plain-text knowledge documents and JSONL RAG datasets.
"""

import structlog

from config.settings import get_settings
from models.schemas import RAGQuery, RAGResult
from services.rag.embeddings import EmbeddingService
from services.rag.knowledge_base import KnowledgeBaseManager
from services.rag.retriever import MultiHopRetriever

logger = structlog.get_logger(__name__)
settings = get_settings()


class RAGService:
    """
    Unified RAG service providing knowledge retrieval capabilities.
    Manages the full pipeline from query to relevant context, searching
    across both the 3GPP knowledge base and the JSONL RAG dataset
    collection in Qdrant.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.retriever = MultiHopRetriever(self.embedding_service)
        self.knowledge_base = KnowledgeBaseManager(self.embedding_service)

    async def initialize(self):
        """Initialize all RAG components."""
        self.embedding_service.initialize()
        await self.retriever.initialize()
        logger.info("RAG service initialized")

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        max_hops: int = None,
        filters: dict[str, any] = None
    ) -> RAGResult:
        """
        Retrieve relevant knowledge for a query.

        Searches both the 3GPP knowledge collection and the JSONL RAG
        dataset collection, merges by score, and returns the top-k.

        Args:
            query: Natural language query about 5G/telecom domain
            top_k: Number of results to return
            max_hops: Maximum retrieval hops
            filters: Dictionary of metadata filters to apply

        Returns:
            RAGResult with ranked relevant chunks
        """
        rag_query = RAGQuery(
            query=query,
            top_k=top_k or settings.rag_top_k,
            max_hops=max_hops or settings.rag_max_hops,
            filters=filters or {},
        )

        result = await self.retriever.retrieve(rag_query)

        logger.debug("RAG retrieval complete",
                     query=query[:50],
                     num_chunks=len(result.chunks),
                     time_ms=result.retrieval_time_ms)

        return result

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------

    async def ingest_knowledge_base(self, documents_dir: str) -> dict:
        """Ingest plain-text documents into the 3GPP knowledge base."""
        return await self.knowledge_base.ingest_documents(documents_dir)

    async def ingest_jsonl_datasets(self, data_dir: str = "data", recreate: bool = False) -> dict:
        """
        Ingest all JSONL files under *data_dir* into the RAG dataset
        Qdrant collection.  After ingestion the retriever's collection
        list is refreshed so new data is immediately searchable.

        Returns ingestion statistics.
        """
        stats = await self.knowledge_base.ingest_jsonl_directory(data_dir, recreate=recreate)
        self.retriever.refresh_collections()
        logger.info("JSONL dataset ingestion complete", **stats)
        return stats

    async def ingest_single_jsonl(self, jsonl_path: str, recreate: bool = False) -> dict:
        """Ingest a single JSONL file into the RAG dataset collection."""
        stats = await self.knowledge_base.ingest_jsonl(jsonl_path, recreate=recreate)
        self.retriever.refresh_collections()
        return stats

    async def ingest_json_datasets(self, data_dir: str = "data/3PGPP", recreate: bool = False) -> dict:
        """
        Ingest all structured JSON knowledge files under *data_dir* into the
        3GPP knowledge Qdrant collection.  Handles both flat-array concept
        files (``3gpp_rca_knowledge.json``) and nested-object spec extracts
        (``handover_rca_knowledge.json``).

        Returns ingestion statistics.
        """
        stats = await self.knowledge_base.ingest_json_directory(data_dir, recreate=recreate)
        self.retriever.refresh_collections()
        logger.info("JSON dataset ingestion complete", **stats)
        return stats

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context(self, result: RAGResult, max_tokens: int = 2048) -> str:
        """
        Format RAG results into a context string for LLM prompting.
        Respects token budget to fit within SLM context window.

        Chunks from the JSONL RAG dataset include their chunk_type and
        scenario_id so the downstream LLM can distinguish between
        domain knowledge, cell configurations, and scenario analyses.
        """
        context_parts = []
        total_tokens = 0

        for chunk in result.chunks:
            chunk_tokens = len(chunk.content.split())
            if total_tokens + chunk_tokens > max_tokens:
                break

            source_info = f"[Source: {chunk.source_document}"
            if chunk.section:
                source_info += f", Section: {chunk.section}"
            chunk_type = chunk.metadata.get("chunk_type", "")
            scenario_id = chunk.metadata.get("scenario_id", "")
            if chunk_type:
                source_info += f", Type: {chunk_type}"
            if scenario_id and scenario_id != "global":
                source_info += f", Scenario: {scenario_id[:8]}"
            source_info += f", Relevance: {chunk.score:.2f}]"

            context_parts.append(f"{source_info}\n{chunk.content}")
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)
