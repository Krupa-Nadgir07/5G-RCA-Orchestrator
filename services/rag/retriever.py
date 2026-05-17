"""
RAG Retriever - Vector retrieval using Qdrant.
Simplified for the research prototype (no Elasticsearch, no cross-encoder).
"""

import time
from typing import Optional

import structlog

from config.settings import get_settings
from models.schemas import RAGChunk, RAGQuery, RAGResult
from services.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)
settings = get_settings()


class MultiHopRetriever:
    """
    Vector retrieval using Qdrant for the RAG pipeline.
    Simplified single-hop dense retrieval for the research prototype.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self._qdrant_client = None
        self._retrieval_cache: dict[str, RAGResult] = {}

    async def initialize(self):
        """Initialize connection to Qdrant."""
        try:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key or None,
            )
            # Verify connection
            self._qdrant_client.get_collections()
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.warning("Qdrant connection failed, RAG will return empty results", error=str(e))
            self._qdrant_client = None

    async def retrieve(self, query: RAGQuery) -> RAGResult:
        """Perform vector retrieval from Qdrant."""
        start_time = time.time()

        # Check cache
        cache_key = f"{query.query}:{query.top_k}"
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        # Dense retrieval from Qdrant
        chunks = await self._dense_retrieve(query.query, top_k=query.top_k)

        # Build result
        retrieval_time_ms = (time.time() - start_time) * 1000
        result = RAGResult(
            query=query.query,
            chunks=chunks,
            total_tokens=sum(len(c.content.split()) for c in chunks),
            retrieval_time_ms=round(retrieval_time_ms, 2),
        )

        self._retrieval_cache[cache_key] = result
        return result

    async def _dense_retrieve(self, query: str, top_k: int) -> list[RAGChunk]:
        """Dense retrieval using Qdrant vector search."""
        if not self._qdrant_client:
            return []

        try:
            query_embedding = self.embedding_service.encode(query)

            results = self._qdrant_client.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_embedding.tolist(),
                limit=top_k,
            )

            chunks = []
            for result in results:
                payload = result.payload or {}
                chunks.append(RAGChunk(
                    chunk_id=str(result.id),
                    content=payload.get("content", ""),
                    source_document=payload.get("source_document", ""),
                    section=payload.get("section"),
                    score=result.score,
                    metadata=payload.get("metadata", {}),
                ))
            return chunks

        except Exception as e:
            logger.error("Dense retrieval failed", error=str(e))
            return []

    def clear_cache(self):
        """Clear retrieval cache."""
        self._retrieval_cache.clear()
