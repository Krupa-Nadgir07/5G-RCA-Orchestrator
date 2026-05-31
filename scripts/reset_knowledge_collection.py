"""
Reset and re-ingest Qdrant collections for the 5G RCA system.

Usage:
    python scripts/reset_knowledge_collection.py              # wipe + re-ingest ALL
    python scripts/reset_knowledge_collection.py --wipe-only  # just delete, no ingest
    python scripts/reset_knowledge_collection.py --status     # show current collection sizes

Re-ingests:
  Collection: 3gpp_knowledge
    - knowledge/*.txt          (domain knowledge text files)
    - data/3PGPP/*.json        (3GPP concept + handover RCA JSONs)

  Collection: 5g_rag_dataset
    - data/rag_dataset.jsonl   (main RCA scenario dataset)
    - data/rag_dataset_2.jsonl (supplementary RCA scenarios)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_client():
    from qdrant_client import QdrantClient
    from config.settings import get_settings
    s = get_settings()
    return QdrantClient(host=s.qdrant_host, port=s.qdrant_port), s


def cmd_status():
    """Print collection sizes."""
    client, settings = get_client()
    cols = client.get_collections().collections
    if not cols:
        print("No collections found in Qdrant.")
        return
    print(f"\n{'Collection':<30} {'Points':>10}")
    print("-" * 42)
    for col in cols:
        info = client.get_collection(col.name)
        print(f"  {col.name:<28} {info.points_count:>10,}")
    print()


def cmd_wipe(collection: str):
    """Delete a Qdrant collection."""
    client, _ = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        print(f"Collection '{collection}' does not exist — nothing to delete.")
        return
    client.delete_collection(collection)
    print(f"✓ Deleted collection '{collection}'")


def cmd_wipe_all():
    """Delete both Qdrant collections."""
    from config.settings import get_settings
    settings = get_settings()
    for col in [settings.qdrant_collection, settings.qdrant_rag_collection]:
        cmd_wipe(col)


async def cmd_reingest():
    """Wipe then re-ingest all sources into both collections."""
    from config.settings import get_settings
    from services.rag.embeddings import EmbeddingService
    from services.rag.knowledge_base import KnowledgeBaseManager

    settings = get_settings()

    print("Initialising embedding model…")
    emb = EmbeddingService()
    emb.initialize()

    kb = KnowledgeBaseManager(emb)

    # ── Collection 1: 3gpp_knowledge ──────────────────────────────────
    print("\n━━ 3GPP Knowledge Collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("[1/3] Ingesting knowledge/*.txt …")
    txt_stats = await kb.ingest_documents("knowledge")
    print(f"      → {txt_stats['chunks']} chunks from {txt_stats['documents']} files")

    print("[2/3] Ingesting data/3PGPP/*.json …")
    json_stats = await kb.ingest_json_directory("data/3PGPP")
    print(f"      → {json_stats['total_chunks']} chunks from {json_stats['files']} files")

    kb_total = txt_stats["chunks"] + json_stats["total_chunks"]
    print(f"      ✓ {kb_total} total chunks → '{settings.qdrant_collection}'")

    # ── Collection 2: 5g_rag_dataset ─────────────────────────────────
    print("\n━━ RCA Dataset Collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("[3/3] Ingesting data/*.jsonl (RCA scenario datasets) …")
    jsonl_stats = {"files": 0, "total_chunks": 0, "errors": 0}
    for file_name in ["rag_dataset.jsonl", "rag_train_dataset.jsonl"]:
        p = Path("data") / file_name
        if p.exists():
            file_stats = await kb.ingest_jsonl(str(p), recreate=(jsonl_stats["files"] == 0))
            jsonl_stats["files"] += 1
            jsonl_stats["total_chunks"] += file_stats["chunks"]
            jsonl_stats["errors"] += file_stats["errors"]
    print(f"      → {jsonl_stats['total_chunks']} chunks from {jsonl_stats['files']} files")
    print(f"      ✓ {jsonl_stats['total_chunks']} total chunks → '{settings.qdrant_rag_collection}'")

    print(f"\n✓ All done! Total chunks indexed: {kb_total + jsonl_stats['total_chunks']:,}")


def main():
    parser = argparse.ArgumentParser(description="Reset and re-ingest Qdrant collections")
    parser.add_argument("--wipe-only", action="store_true", help="Delete both collections without re-ingesting")
    parser.add_argument("--status", action="store_true", help="Show collection sizes only")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    if args.wipe_only:
        cmd_wipe_all()
        return

    # Default: wipe both, then re-ingest everything
    cmd_wipe_all()
    asyncio.run(cmd_reingest())


if __name__ == "__main__":
    main()
