"""
RAG Retriever - Multi-collection vector retrieval using Qdrant.

Searches both the 3GPP knowledge base collection and the JSONL RAG
dataset collection, merges the results by score, applies the
similarity threshold, and returns the top-k chunks.
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
    Multi-collection dense retrieval using Qdrant.

    At query time the retriever searches **both** the 3GPP knowledge
    collection and the JSONL RAG-dataset collection, merges the hits
    by cosine score, filters by ``rag_similarity_threshold``, and
    returns the top-k results.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self._qdrant_client = None
        self._retrieval_cache: dict[str, RAGResult] = {}
        self._available_collections: list[str] = []

    async def initialize(self):
        """Initialize connection to Qdrant and discover available collections."""
        try:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key or None,
            )
            collections = self._qdrant_client.get_collections().collections
            self._available_collections = [c.name for c in collections]
            logger.info("Qdrant client initialized",
                        collections=self._available_collections)
        except Exception as e:
            logger.warning("Qdrant connection failed, RAG will return empty results",
                           error=str(e))
            self._qdrant_client = None

    async def retrieve(self, query: RAGQuery) -> RAGResult:
        """Perform multi-collection vector retrieval from Qdrant."""
        start_time = time.time()

        # Build unique cache key including filters if present
        filter_str = ",".join(f"{k}={v}" for k, v in sorted(query.filters.items())) if query.filters else ""
        cache_key = f"{query.query}:{query.top_k}:{filter_str}"
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        query_embedding = self.embedding_service.encode(query.query)
        vector = query_embedding.tolist()

        all_chunks: list[RAGChunk] = []

        # Search the 3GPP knowledge collection (global, no cell-specific filters)
        all_chunks.extend(
            await self._search_collection(
                settings.qdrant_collection, vector, top_k=query.top_k
            )
        )

        # Compile Qdrant Filter conditions for the scenario collection
        qdrant_filter = None
        if query.filters and self._qdrant_client:
            from qdrant_client.http import models as qmodels
            conditions = []
            for k, v in query.filters.items():
                if not v:
                    continue
                if k == "cell_id":
                    conditions.append(
                        qmodels.FieldCondition(
                            key="metadata.cell_ids",
                            match=qmodels.MatchValue(value=v)
                        )
                    )
                else:
                    conditions.append(
                        qmodels.FieldCondition(
                            key=k,
                            match=qmodels.MatchValue(value=v)
                        )
                    )
            if conditions:
                qdrant_filter = qmodels.Filter(must=conditions)

        # Search the JSONL RAG dataset collection (filtered conditionally)
        all_chunks.extend(
            await self._search_collection(
                settings.qdrant_rag_collection, vector, top_k=query.top_k, query_filter=qdrant_filter
            )
        )

        # Merge: sort by score descending, apply similarity threshold, keep top-k
        threshold = settings.rag_similarity_threshold
        filtered = [c for c in all_chunks if c.score >= threshold]
        if not filtered:
            filtered = all_chunks  # fallback: skip threshold if nothing passes

        filtered.sort(key=lambda c: c.score, reverse=True)
        top_chunks = filtered[: query.top_k]

        retrieval_time_ms = (time.time() - start_time) * 1000
        result = RAGResult(
            query=query.query,
            chunks=top_chunks,
            total_tokens=sum(len(c.content.split()) for c in top_chunks),
            retrieval_time_ms=round(retrieval_time_ms, 2),
        )

        self._retrieval_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search_collection(
        self, collection_name: str, vector: list[float], top_k: int, query_filter = None
    ) -> list[RAGChunk]:
        """Search a single Qdrant collection, returning RAGChunk objects."""
        if not self._qdrant_client:
            return []
        if collection_name not in self._available_collections:
            return []

        try:
            results = self._qdrant_client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
            )

            chunks = []
            for hit in results:
                payload = hit.payload or {}
                chunks.append(RAGChunk(
                    chunk_id=str(hit.id),
                    content=payload.get("content", ""),
                    source_document=payload.get("source_document", ""),
                    section=payload.get("section"),
                    score=hit.score,
                    metadata={
                        **payload.get("metadata", {}),
                        "collection": collection_name,
                        "chunk_type": payload.get("chunk_type", ""),
                        "scenario_id": payload.get("scenario_id", ""),
                    },
                ))
            return chunks

        except Exception as e:
            logger.error("Collection search failed",
                         collection=collection_name, error=str(e))
            return []

    def clear_cache(self):
        """Clear retrieval cache."""
        self._retrieval_cache.clear()

    def refresh_collections(self):
        """Re-discover available Qdrant collections (call after ingestion)."""
        if self._qdrant_client:
            try:
                collections = self._qdrant_client.get_collections().collections
                self._available_collections = [c.name for c in collections]
            except Exception:
                pass
