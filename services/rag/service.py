"""
RAG Service - Main entry point for the RAG knowledge pipeline.
Combines embedding, retrieval, and re-ranking into a unified service.
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
    Manages the full pipeline from query to relevant context.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.retriever = MultiHopRetriever(self.embedding_service)
        self.knowledge_base = KnowledgeBaseManager(self.embedding_service)

    async def initialize(self):
        """Initialize all RAG components."""
        await self.retriever.initialize()
        logger.info("RAG service initialized")

    async def retrieve(self, query: str, top_k: int = None, max_hops: int = None) -> RAGResult:
        """
        Retrieve relevant knowledge for a query.
        
        Args:
            query: Natural language query about 5G/telecom domain
            top_k: Number of results to return
            max_hops: Maximum retrieval hops
            
        Returns:
            RAGResult with ranked relevant chunks
        """
        rag_query = RAGQuery(
            query=query,
            top_k=top_k or settings.rag_top_k,
            max_hops=max_hops or settings.rag_max_hops,
        )
        
        result = await self.retriever.retrieve(rag_query)
        
        logger.debug("RAG retrieval complete",
                    query=query[:50],
                    num_chunks=len(result.chunks),
                    time_ms=result.retrieval_time_ms)
        
        return result

    async def ingest_knowledge_base(self, documents_dir: str) -> dict:
        """Ingest documents into the knowledge base."""
        return await self.knowledge_base.ingest_documents(documents_dir)

    def format_context(self, result: RAGResult, max_tokens: int = 2048) -> str:
        """
        Format RAG results into a context string for LLM prompting.
        Respects token budget to fit within SLM context window.
        """
        context_parts = []
        total_tokens = 0
        
        for chunk in result.chunks:
            chunk_tokens = len(chunk.content.split())  # Approximate
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            source_info = f"[Source: {chunk.source_document}"
            if chunk.section:
                source_info += f", Section: {chunk.section}"
            source_info += f", Relevance: {chunk.score:.2f}]"
            
            context_parts.append(f"{source_info}\n{chunk.content}")
            total_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
