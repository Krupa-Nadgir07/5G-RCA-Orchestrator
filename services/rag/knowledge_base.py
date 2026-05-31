"""
Knowledge Base Manager - Handles 3GPP document ingestion, chunking, and indexing.
Supports plain-text documents, pre-chunked JSONL RAG datasets, structured
3GPP JSON knowledge files, and raw 3GPP TS specification PDFs.
"""

import hashlib
import json
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
    - JSONL RAG dataset ingestion into a dedicated Qdrant collection
    - Elasticsearch indexing for BM25 search
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunk_size = settings.rag_chunk_size
        self.chunk_overlap = settings.rag_chunk_overlap

    # -----------------------------------------------------------------
    # JSONL RAG dataset ingestion
    # -----------------------------------------------------------------

    async def ingest_jsonl(self, jsonl_path: str, recreate: bool = False) -> dict:
        """
        Ingest a pre-chunked JSONL RAG dataset into the dedicated Qdrant
        collection (``settings.qdrant_rag_collection``).

        Each line must be a JSON object with at least ``chunk_id`` and
        ``content`` fields.  Metadata, scenario_id, and chunk_type are
        preserved in the Qdrant payload.

        Returns ingestion statistics.
        """
        stats = {"file": str(jsonl_path), "chunks": 0, "errors": 0}
        file_path = Path(jsonl_path)

        if not file_path.exists():
            logger.error("JSONL file not found", path=str(file_path))
            return stats

        records: list[dict] = []
        with open(file_path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    stats["errors"] += 1
                    logger.warning("Malformed JSONL line", file=str(file_path), line=line_num)

        if not records:
            logger.warning("No records found in JSONL file", path=str(file_path))
            return stats

        logger.info("Loaded JSONL records", path=str(file_path), count=len(records))

        await self._index_jsonl_to_qdrant(records, file_path.stem, recreate=recreate)
        stats["chunks"] = len(records)
        logger.info("JSONL ingestion complete", **stats)
        return stats

    async def ingest_jsonl_directory(self, directory: str, recreate: bool = False) -> dict:
        """Ingest all ``.jsonl`` files found under *directory*."""
        stats = {"files": 0, "total_chunks": 0, "errors": 0}
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error("Directory not found", path=directory)
            return stats

        for jsonl_file in sorted(dir_path.glob("*.jsonl")):
            file_stats = await self.ingest_jsonl(str(jsonl_file), recreate=recreate and stats["files"] == 0)
            stats["files"] += 1
            stats["total_chunks"] += file_stats["chunks"]
            stats["errors"] += file_stats["errors"]
            recreate = False  # only recreate on the first file

        logger.info("JSONL directory ingestion complete", **stats)
        return stats

    # -----------------------------------------------------------------
    # Structured JSON knowledge ingestion (3GPP concept / event files)
    # -----------------------------------------------------------------

    async def ingest_json_directory(self, directory: str, recreate: bool = False) -> dict:
        """
        Ingest all ``*.json`` files under *directory* into the 3GPP knowledge
        Qdrant collection (``settings.qdrant_collection``).

        Supports two JSON schemas:
        - **Flat array**: A top-level list of concept objects (e.g.
          ``3gpp_rca_knowledge.json``). Each object is serialised into a
          human-readable text chunk with key fields expanded inline.
        - **Nested object**: A top-level dict with a ``source_document`` key
          and one or more section keys whose values are dicts or lists (e.g.
          ``handover_rca_knowledge.json``). Each section entry is serialised
          as a separate chunk.

        Returns ingestion statistics.
        """
        stats = {"files": 0, "total_chunks": 0, "errors": 0}
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error("Directory not found", path=directory)
            return stats

        for json_file in sorted(dir_path.glob("*.json")):
            try:
                file_stats = await self._ingest_json_file(json_file, recreate=recreate and stats["files"] == 0)
                stats["files"] += 1
                stats["total_chunks"] += file_stats["chunks"]
                stats["errors"] += file_stats["errors"]
                recreate = False
            except Exception as e:
                stats["errors"] += 1
                logger.error("JSON file ingestion failed", file=str(json_file), error=str(e))

        logger.info("JSON directory ingestion complete", **stats)
        return stats

    async def _ingest_json_file(self, file_path: Path, recreate: bool = False) -> dict:
        """Parse and ingest a single JSON file into the 3GPP Qdrant collection."""
        stats = {"file": str(file_path), "chunks": 0, "errors": 0}

        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        source_name = file_path.stem
        chunks: list[DocumentChunk] = []

        if isinstance(data, list):
            # Flat-array schema: each element is a 3GPP concept/event dict.
            for idx, entry in enumerate(data):
                if not isinstance(entry, dict):
                    continue
                try:
                    chunks.append(self._json_entry_to_chunk(entry, source_name, section=entry.get("domain", "knowledge"), index=idx))
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning("Failed to convert JSON entry", index=idx, error=str(e))

        elif isinstance(data, dict):
            # Nested-object schema: each top-level key (except source_document meta)
            # is treated as a section; its value (dict or list) is flattened.
            meta = data.get("source_document", {})
            spec_ref = meta.get("specification", source_name) if isinstance(meta, dict) else source_name
            skip_keys = {"source_document"}
            for section_key, section_value in data.items():
                if section_key in skip_keys:
                    continue
                if isinstance(section_value, dict):
                    for sub_key, sub_value in section_value.items():
                        entry = {"concept": f"{section_key} / {sub_key}", "spec_reference": spec_ref}
                        if isinstance(sub_value, dict):
                            entry.update(sub_value)
                        else:
                            entry["description"] = str(sub_value)
                        try:
                            chunks.append(self._json_entry_to_chunk(entry, source_name, section=section_key, index=len(chunks)))
                        except Exception as e:
                            stats["errors"] += 1
                            logger.warning("Failed to convert nested JSON entry", section=section_key, error=str(e))
                elif isinstance(section_value, list):
                    for idx2, item in enumerate(section_value):
                        entry = {"concept": f"{section_key}[{idx2}]", "spec_reference": spec_ref}
                        if isinstance(item, dict):
                            entry.update(item)
                        else:
                            entry["description"] = str(item)
                        try:
                            chunks.append(self._json_entry_to_chunk(entry, source_name, section=section_key, index=len(chunks)))
                        except Exception as e:
                            stats["errors"] += 1
                else:
                    # Scalar top-level key - include as a micro-chunk
                    chunks.append(DocumentChunk(
                        chunk_id=hashlib.md5(f"{source_name}:{section_key}".encode()).hexdigest(),
                        content=f"{section_key}: {section_value}",
                        source_document=source_name,
                        section=section_key,
                        metadata={"spec_reference": spec_ref, "chunk_index": 0, "total_chunks": 1, "file_path": str(file_path)},
                    ))
        else:
            logger.warning("Unrecognised JSON structure (not list or dict)", file=str(file_path))
            return stats

        if chunks:
            await self._index_chunks(chunks)
        stats["chunks"] = len(chunks)
        logger.info("JSON file ingestion complete", **stats)
        return stats

    def _json_entry_to_chunk(self, entry: dict, source_name: str, section: str, index: int) -> DocumentChunk:
        """Serialise a structured JSON entry dict into a human-readable DocumentChunk."""
        concept = entry.get("concept") or entry.get("name") or entry.get("event") or f"entry_{index}"
        spec_ref = entry.get("spec_reference", "")

        lines = [f"Concept: {concept}"]
        if spec_ref:
            lines.append(f"Spec Reference: {spec_ref}")

        # Include all informative text fields
        text_fields = [
            "domain", "purpose", "description", "section_reference",
            "operational_impact", "rca_relevance",
        ]
        list_fields = [
            "trigger_conditions", "failure_symptoms", "possible_root_causes",
            "recommended_actions", "related_kpis", "kpi_patterns",
            "related_events", "related_timers", "procedure_steps", "benefits",
            "diagnostic_steps",
        ]

        for field in text_fields:
            val = entry.get(field)
            if val and isinstance(val, str):
                lines.append(f"{field.replace('_', ' ').title()}: {val}")
            elif val and isinstance(val, dict):
                lines.append(f"{field.replace('_', ' ').title()}: {json.dumps(val, ensure_ascii=False)[:300]}")

        for field in list_fields:
            val = entry.get(field)
            if val and isinstance(val, list):
                lines.append(f"{field.replace('_', ' ').title()}:")
                for item in val:
                    if isinstance(item, str):
                        lines.append(f"  - {item}")
                    elif isinstance(item, dict):
                        lines.append(f"  - {json.dumps(item, ensure_ascii=False)[:200]}")

        # Catch any remaining dict/list fields we haven't enumerated
        known = set(text_fields + list_fields + ["concept", "name", "event", "spec_reference", "confidence", "domain"])
        for k, v in entry.items():
            if k in known:
                continue
            if isinstance(v, str):
                lines.append(f"{k.replace('_', ' ').title()}: {v[:400]}")
            elif isinstance(v, (int, float, bool)):
                lines.append(f"{k.replace('_', ' ').title()}: {v}")

        content = "\n".join(lines)
        chunk_id = hashlib.md5(f"{source_name}:{section}:{index}".encode()).hexdigest()
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            source_document=source_name,
            section=section,
            metadata={
                "spec_reference": spec_ref,
                "chunk_index": index,
                "total_chunks": -1,  # unknown at this point
                "file_path": source_name,
            },
        )

    # -----------------------------------------------------------------
    # PDF ingestion (3GPP Technical Specifications)
    # -----------------------------------------------------------------

    async def ingest_pdf_directory(self, directory: str, recreate: bool = False) -> dict:
        """
        Ingest all ``*.pdf`` files under *directory* into the 3GPP knowledge
        Qdrant collection (``settings.qdrant_collection``).

        Extracts text page-by-page using ``pypdf``, then applies the same
        section-aware chunking used for plain-text files. Pages with fewer
        than 80 characters (e.g. cover pages, tables-of-contents) are skipped.

        Returns ingestion statistics.
        """
        stats = {"files": 0, "total_chunks": 0, "errors": 0}
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error("Directory not found", path=directory)
            return stats

        try:
            from pypdf import PdfReader  # lazy import — optional dependency
        except ImportError:
            logger.error("pypdf not installed; run: pip install pypdf")
            return stats

        for pdf_file in sorted(dir_path.glob("*.pdf")):
            try:
                file_stats = await self._ingest_pdf_file(pdf_file, PdfReader)
                stats["files"] += 1
                stats["total_chunks"] += file_stats["chunks"]
                stats["errors"] += file_stats["errors"]
            except Exception as e:
                stats["errors"] += 1
                logger.error("PDF file ingestion failed", file=str(pdf_file), error=str(e))

        logger.info("PDF directory ingestion complete", **stats)
        return stats

    async def _ingest_pdf_file(self, file_path: Path, PdfReader) -> dict:
        """Extract text from a PDF and index as document chunks."""
        stats = {"file": str(file_path), "chunks": 0, "errors": 0}
        doc_name = file_path.stem
        spec_number = self._extract_spec_number(doc_name)

        logger.info("Extracting PDF", file=file_path.name)
        reader = PdfReader(str(file_path))

        # Accumulate all page text then split into sections
        page_texts = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                if len(text.strip()) >= 80:  # skip near-empty pages
                    page_texts.append(text)
            except Exception:
                stats["errors"] += 1

        full_text = "\n".join(page_texts)
        if not full_text.strip():
            logger.warning("No text extracted from PDF", file=file_path.name)
            return stats

        sections = self._split_into_sections(full_text)
        chunks: list[DocumentChunk] = []

        for section_title, section_content in sections:
            section_chunks = self._chunk_text(section_content, self.chunk_size, self.chunk_overlap)
            for i, chunk_text in enumerate(section_chunks):
                chunk_id = hashlib.md5(f"{doc_name}:{section_title}:{i}".encode()).hexdigest()
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
                        "source_type": "pdf",
                    },
                ))

        if chunks:
            await self._index_chunks(chunks)
        stats["chunks"] = len(chunks)
        logger.info("PDF ingestion complete", file=file_path.name, chunks=len(chunks))
        return stats

    async def _index_jsonl_to_qdrant(self, records: list[dict], source_name: str, recreate: bool = False):
        """Embed and upsert pre-chunked JSONL records into the RAG Qdrant collection."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, VectorParams, Distance

            client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            collection = settings.qdrant_rag_collection

            existing = [c.name for c in client.get_collections().collections]
            if collection in existing and recreate:
                client.delete_collection(collection)
                existing.remove(collection)

            if collection not in existing:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )

            batch_size = 64
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                texts = [r.get("content", "") for r in batch]
                embeddings = self.embedding_service.encode_batch(texts)

                points = [
                    PointStruct(
                        id=abs(hash(rec.get("chunk_id", str(i + j)))) % (2**63),
                        vector=emb.tolist(),
                        payload={
                            "content": rec.get("content", ""),
                            "source_document": source_name,
                            "chunk_type": rec.get("chunk_type", ""),
                            "scenario_id": rec.get("scenario_id", ""),
                            "section": rec.get("chunk_type", ""),
                            "chunk_id": rec.get("chunk_id", ""),
                            "metadata": rec.get("metadata", {}),
                        },
                    )
                    for j, (rec, emb) in enumerate(zip(batch, embeddings))
                ]

                client.upsert(collection_name=collection, points=points)

            logger.info("Indexed JSONL in Qdrant", collection=collection, num_chunks=len(records))

        except Exception as e:
            logger.warning("JSONL Qdrant indexing failed", error=str(e))

    # -----------------------------------------------------------------
    # Plain-text document ingestion (original)
    # -----------------------------------------------------------------

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
