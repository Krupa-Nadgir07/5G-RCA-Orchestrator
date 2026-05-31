"""
RAG Dataset Ingestion — Loads JSONL datasets into Qdrant vector database.
Pipeline: JSONL → embeddings → Qdrant (collection: 5g_rag_dataset)

Supports both rag_dataset.jsonl and rag_dataset_2.jsonl, with deduplication
by chunk_id so re-runs are idempotent.
"""

import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

settings = get_settings()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSONL_FILES = [
    DATA_DIR / "rag_dataset.jsonl",
    DATA_DIR / "rag_train_dataset.jsonl",
]
COLLECTION_NAME = settings.qdrant_rag_collection
EMBEDDING_MODEL = settings.embedding_model
BATCH_SIZE = 64


def load_jsonl(file_path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed line {line_num}: {e}")
    return records


def ensure_collection(client: QdrantClient, embedding_dim: int, recreate: bool = False):
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Deleted existing collection: {COLLECTION_NAME}")
        else:
            info = client.get_collection(COLLECTION_NAME)
            print(f"  Collection '{COLLECTION_NAME}' already exists ({info.points_count} points)")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"  Created collection: {COLLECTION_NAME} (dim={embedding_dim})")


def ingest_records(client: QdrantClient, model: SentenceTransformer, records: list[dict], source_file: str):
    """Embed and upsert a list of JSONL records into Qdrant."""
    total = len(records)
    upserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        texts = [r.get("content", "") for r in batch]

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = []
        for rec, embedding in zip(batch, embeddings):
            chunk_id = rec.get("chunk_id", "")
            point_id = abs(hash(chunk_id)) % (2**63)

            payload = {
                "content": rec.get("content", ""),
                "source_document": source_file,
                "chunk_type": rec.get("chunk_type", ""),
                "scenario_id": rec.get("scenario_id", ""),
                "section": rec.get("chunk_type", ""),
                "chunk_id": chunk_id,
                "metadata": rec.get("metadata", {}),
            }
            points.append(PointStruct(id=point_id, vector=embedding.tolist(), payload=payload))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        upserted += len(points)
        pct = round(upserted / total * 100, 1)
        print(f"    [{source_file}] {upserted}/{total} chunks ingested ({pct}%)", end="\r")

    print(f"    [{source_file}] {upserted}/{total} chunks ingested (100%)     ")
    return upserted


def main():
    recreate = "--recreate" in sys.argv

    print("=" * 60)
    print("5G RAG Dataset → Qdrant Ingestion")
    print("=" * 60)

    # -- Embedding model -------------------------------------------------------
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embedding_dim}")

    # -- Qdrant connection -----------------------------------------------------
    print(f"\nConnecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    ensure_collection(client, embedding_dim, recreate=recreate)

    # -- Ingest each JSONL file ------------------------------------------------
    grand_total = 0
    t0 = time.time()

    for jsonl_path in JSONL_FILES:
        if not jsonl_path.exists():
            print(f"\n  SKIP: {jsonl_path.name} not found")
            continue

        print(f"\n  Loading {jsonl_path.name} ...")
        records = load_jsonl(jsonl_path)
        print(f"  Loaded {len(records)} chunks")

        count = ingest_records(client, model, records, source_file=jsonl_path.stem)
        grand_total += count

    elapsed = time.time() - t0

    # -- Verify ----------------------------------------------------------------
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n{'=' * 60}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  Total chunks upserted : {grand_total}")
    print(f"  Points in collection  : {info.points_count}")
    print(f"  Collection            : {COLLECTION_NAME}")
    print(f"  Embedding model       : {EMBEDDING_MODEL}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
