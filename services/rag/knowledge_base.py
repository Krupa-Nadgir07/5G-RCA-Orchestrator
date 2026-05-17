"""
Knowledge Base Manager - Handles 3GPP document ingestion, chunking, and indexing.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from config.settings import get_settings
from services.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class DocumentChunk:
    """A chunk of a document ready for indexing."""
    chunk_id: str
    content: str
    source_document: str
    section: str
    metadata: dict


class KnowledgeBaseManager:
    """
    Manages the 3GPP knowledge base:
    - Document ingestion and preprocessing
    - Semantic chunking with overlap
    - Embedding generation and vector DB indexing
    - Elasticsearch indexing for BM25 search
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunk_size = settings.rag_chunk_size
        self.chunk_overlap = settings.rag_chunk_overlap

    async def ingest_documents(self, documents_dir: str) -> dict:
        """
        Ingest all documents from a directory into the knowledge base.
        
        Returns statistics about ingestion.
        """
        stats = {"documents": 0, "chunks": 0, "errors": 0}
        
        doc_path = Path(documents_dir)
        if not doc_path.exists():
            logger.error("Documents directory not found", path=documents_dir)
            return stats

        for file_path in doc_path.rglob("*.txt"):
            try:
                chunks = self._process_document(file_path)
                await self._index_chunks(chunks)
                stats["documents"] += 1
                stats["chunks"] += len(chunks)
            except Exception as e:
                stats["errors"] += 1
                logger.error("Document ingestion failed", 
                           file=str(file_path), error=str(e))

        for file_path in doc_path.rglob("*.md"):
            try:
                chunks = self._process_document(file_path)
                await self._index_chunks(chunks)
                stats["documents"] += 1
                stats["chunks"] += len(chunks)
            except Exception as e:
                stats["errors"] += 1
                logger.error("Document ingestion failed", 
                           file=str(file_path), error=str(e))

        logger.info("Knowledge base ingestion complete", **stats)
        return stats

    def _process_document(self, file_path: Path) -> list[DocumentChunk]:
        """Process a single document into chunks."""
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Extract document metadata
        doc_name = file_path.stem
        spec_number = self._extract_spec_number(doc_name)
        
        # Semantic chunking
        sections = self._split_into_sections(content)
        
        chunks = []
        for section_title, section_content in sections:
            section_chunks = self._chunk_text(
                section_content, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            for i, chunk_text in enumerate(section_chunks):
                chunk_id = hashlib.md5(
                    f"{doc_name}:{section_title}:{i}".encode()
                ).hexdigest()
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source_document=doc_name,
                    section=section_title,
                    metadata={
                        "spec_number": spec_number,
                        "chunk_index": i,
                        "total_chunks": len(section_chunks),
                        "file_path": str(file_path),
                    }
                ))
        
        return chunks

    def _split_into_sections(self, content: str) -> list[tuple[str, str]]:
        """Split document into sections based on headings."""
        # Match common heading patterns (markdown, numbered sections)
        section_pattern = re.compile(
            r'^(?:#{1,4}\s+|(?:\d+\.)+\s+)(.+?)$',
            re.MULTILINE
        )
        
        matches = list(section_pattern.finditer(content))
        
        if not matches:
            return [("document", content)]
        
        sections = []
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[start:end].strip()
            
            if len(section_text) > 50:  # Skip very short sections
                sections.append((title, section_text))
        
        return sections if sections else [("document", content)]

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Chunk text with token-approximate splitting.
        Uses word count as proxy for tokens (~1.3 words per token average).
        """
        words = text.split()
        # Approximate tokens: 1 token ≈ 0.75 words for technical text
        words_per_chunk = int(chunk_size * 0.75)
        words_overlap = int(overlap * 0.75)
        
        if len(words) <= words_per_chunk:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - words_overlap
            
            if start >= len(words) - words_overlap:
                break
        
        return chunks

    async def _index_chunks(self, chunks: list[DocumentChunk]):
        """Index chunks in both Qdrant (dense) and Elasticsearch (sparse)."""
        if not chunks:
            return

        # Generate embeddings in batch
        texts = [c.content for c in chunks]
        embeddings = self.embedding_service.encode_batch(texts)
        
        # Index in Qdrant
        await self._index_qdrant(chunks, embeddings)
        
        # Index in Elasticsearch
        await self._index_elasticsearch(chunks)

    async def _index_qdrant(self, chunks: list[DocumentChunk], embeddings):
        """Index chunks with embeddings in Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, VectorParams, Distance

            client = QdrantClient(
                host=settings.qdrant_host, 
                port=settings.qdrant_port
            )
            
            # Ensure collection exists
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if settings.qdrant_collection not in collection_names:
                client.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
            
            # Index points
            points = [
                PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk.content,
                        "source_document": chunk.source_document,
                        "section": chunk.section,
                        "metadata": chunk.metadata,
                    }
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=batch
                )
            
            logger.debug("Indexed in Qdrant", num_chunks=len(chunks))
            
        except Exception as e:
            logger.warning("Qdrant indexing failed", error=str(e))

    async def _index_elasticsearch(self, chunks: list[DocumentChunk]):
        """Index chunks in Elasticsearch for BM25 retrieval."""
        try:
            from elasticsearch import Elasticsearch
            
            es = Elasticsearch(settings.elasticsearch_url)
            
            # Ensure index exists
            if not es.indices.exists(index=settings.elasticsearch_index):
                es.indices.create(
                    index=settings.elasticsearch_index,
                    body={
                        "mappings": {
                            "properties": {
                                "content": {"type": "text", "analyzer": "standard"},
                                "source_document": {"type": "keyword"},
                                "section": {"type": "text"},
                                "metadata": {"type": "object", "enabled": False},
                            }
                        }
                    }
                )
            
            # Bulk index
            actions = []
            for chunk in chunks:
                actions.append({"index": {"_index": settings.elasticsearch_index, "_id": chunk.chunk_id}})
                actions.append({
                    "content": chunk.content,
                    "source_document": chunk.source_document,
                    "section": chunk.section,
                    "metadata": chunk.metadata,
                })
            
            if actions:
                es.bulk(body=actions)
            
            logger.debug("Indexed in Elasticsearch", num_chunks=len(chunks))
            
        except Exception as e:
            logger.warning("Elasticsearch indexing failed", error=str(e))

    def _extract_spec_number(self, filename: str) -> Optional[str]:
        """Extract 3GPP specification number from filename."""
        match = re.search(r"(TS\s*\d{2}\.\d{3})", filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
