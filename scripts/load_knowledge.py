"""
Knowledge Base Loader - Ingests knowledge documents into Qdrant vector database.
Pipeline: TXT → chunks → embeddings → Qdrant
"""

import hashlib
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

settings = get_settings()

# Configuration
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
COLLECTION_NAME = settings.qdrant_collection
EMBEDDING_MODEL = settings.embedding_model
CHUNK_SIZE = 300  # characters per chunk
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    sentences = text.split("\n")
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap // 5:]) if len(words) > overlap // 5 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def generate_chunk_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a chunk."""
    content = f"{source}:{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()


def main():
    print("=" * 60)
    print("5G gNB RCA Knowledge Base Loader")
    print("=" * 60)

    # Initialize embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Create or recreate collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection: {COLLECTION_NAME}")

    # Process knowledge documents
    if not KNOWLEDGE_DIR.exists():
        print(f"\nERROR: Knowledge directory not found: {KNOWLEDGE_DIR}")
        return

    all_points = []
    total_chunks = 0

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        print(f"\nProcessing: {file_path.name}")
        content = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        print(f"  → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(file_path.stem, i)
            embedding = model.encode(chunk, normalize_embeddings=True)

            point = PointStruct(
                id=total_chunks + i,
                vector=embedding.tolist(),
                payload={
                    "content": chunk,
                    "source_document": file_path.stem,
                    "section": file_path.stem.replace("_", " ").title(),
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "metadata": {
                        "file": file_path.name,
                        "category": file_path.stem,
                    },
                },
            )
            all_points.append(point)

        total_chunks += len(chunks)

    # Upload to Qdrant
    if all_points:
        # Upload in batches
        batch_size = 100
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i : i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)

        print(f"\n{'=' * 60}")
        print(f"Successfully loaded {total_chunks} chunks from {len(list(KNOWLEDGE_DIR.glob('*.txt')))} documents")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Vector dimension: {embedding_dim}")
        print(f"{'=' * 60}")
    else:
        print("\nNo documents found to process!")

    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\nVerification - Points in collection: {collection_info.points_count}")


if __name__ == "__main__":
    main()
