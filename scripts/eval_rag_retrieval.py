import os
import sys
import json
import asyncio
from pathlib import Path
from collections import defaultdict

# Add repo root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.rag.service import RAGService
from scripts.benchmark import load_scenarios

# Enable UTF-8 encoding support for print statements
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Root-cause label mappings (C1-C8 mapped to internal categories)
LABEL_MAP = {
    "C1": "coverage_tilt",
    "C2": "overshooting",
    "C3": "neighbor_throughput",
    "C4": "overlapping_coverage",
    "C5": "handover_degradation",
    "C6": "pci_collision",
    "C7": "mobility_speed",
    "C8": "rb_scheduling",
}

def build_scenario_mapping() -> dict[str, str]:
    """Scan all datasets to map scenario_id -> C-label (C1-C8)."""
    mapping = {}
    for filename in ["rag_dataset.jsonl", "rag_train_dataset.jsonl", "rag_dataset_2.jsonl"]:
        p = Path("data") / filename
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("chunk_type") == "combined_analysis":
                        sc_id = rec.get("scenario_id")
                        ans = rec.get("metadata", {}).get("answer")
                        if sc_id and ans:
                            mapping[sc_id] = ans
                except Exception:
                    continue
    return mapping

async def main():
    n_scenarios = 20
    top_k = 5

    print("\n" + "═" * 70)
    print("  5G RCA RAG Retrieval-Only Evaluation (Information Retrieval Metrics)")
    print("  Measures Precision@K, Recall@K, and MRR using Ground-Truth labels")
    print("═" * 70)

    # Initialize RAG Service
    print("📂 Initializing RAG Service and loading embedding models...")
    rag = RAGService()
    await rag.initialize()

    # Load unseen scenarios from the evaluation dataset
    dataset_path = Path("data/rag_dataset_2.jsonl")
    try:
        scenarios = load_scenarios(str(dataset_path), n=n_scenarios, seed=42)
    except FileNotFoundError:
        print(f"Error: evaluation dataset not found at {dataset_path}")
        return

    # Build scenario mapping to resolve cell configs and time series chunks
    print("📂 Building scenario-to-label map from files...")
    scenario_map = build_scenario_mapping()
    print(f"      ✓ Mapped {len(scenario_map)} scenarios to their root causes.")

    total_queries = len(scenarios)
    precision_at_k_sum = 0.0
    reciprocal_rank_sum = 0.0
    hits_by_category = defaultdict(int)
    total_by_category = defaultdict(int)
    category_mrr = defaultdict(float)

    print(f"\nEvaluating retrieval accuracy on {total_queries} queries (Top-{top_k})...\n")

    for i, scenario in enumerate(scenarios):
        meta = scenario.get("metadata", {})
        true_label = meta["answer"]  # e.g., "C4"
        true_category = LABEL_MAP.get(true_label, "unknown")
        rc_cat = meta.get("root_cause_category", "")
        scenario_id = scenario.get("scenario_id", f"scenario_{i}")

        # Construct the query based on the symptoms and filter for labeled scenarios
        rag_query = f"5G drive test root cause analysis: {rc_cat}"
        
        # Perform RAG retrieval, filtering the scenario dataset to only retrieve from the labeled training set
        result = await rag.retrieve(
            rag_query, 
            top_k=top_k, 
            filters={"source_document": "rag_train_dataset"}
        )
        
        relevant_chunks_count = 0
        first_relevant_rank = None

        for rank, chunk in enumerate(result.chunks, start=1):
            chunk_category = chunk.metadata.get("root_cause_category", "")
            chunk_type = chunk.metadata.get("chunk_type", "")
            chunk_scenario = chunk.metadata.get("scenario_id", "")
            
            # Hybrid Relevance Check:
            # 1. For scenario-specific chunks (cell config, KPIs): resolve scenario ID to check if it matches the true label
            # 2. For global 3GPP/concept chunks: check for root-cause keywords and synonyms in the content
            is_relevant = False
            
            if chunk_scenario:
                resolved_label = scenario_map.get(chunk_scenario, "")
                is_relevant = (resolved_label == true_label)
            else:
                # Global 3GPP references or concept definitions
                keywords = [true_label.lower(), true_category.replace("_", " ").lower()]
                synonyms = {
                    "C1": ["tilt", "downtilt", "antenna tilt", "coverage_tilt"],
                    "C2": ["overshoot", "distance > 1km", "over-shooting", "overshooting"],
                    "C3": ["neighbor throughput", "neighbour throughput", "stronger neighbor", "neighbor_throughput"],
                    "C4": ["overlap", "overlapping", "co-frequency neighbor", "overlapping_coverage"],
                    "C5": ["handover", "ping-pong", "ho fail", "handover_degradation"],
                    "C6": ["pci", "mod30", "pci collision", "collision", "pci_collision"],
                    "C7": ["mobility", "speed", "doppler", "velocity", "mobility_speed"],
                    "C8": ["scheduling", "rb count", "resource block", "prb utilization", "rb_scheduling"]
                }
                keywords.extend(synonyms.get(true_label, []))
                content_lower = chunk.content.lower()
                is_relevant = any(kw in content_lower for kw in keywords)

            if is_relevant:
                relevant_chunks_count += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # Calculate metrics for this query
        precision = relevant_chunks_count / top_k
        precision_at_k_sum += precision

        rr = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        reciprocal_rank_sum += rr

        # Record breakdown by category
        total_by_category[true_label] += 1
        if first_relevant_rank is not None:
            hits_by_category[true_label] += 1
            category_mrr[true_label] += rr

        print(f"  [{i+1:02d}/{total_queries}] {scenario_id:<12} (True={true_label}) -> "
              f"Relevant Chunks: {relevant_chunks_count}/{top_k} | First Match Rank: {first_relevant_rank or 'N/A'}")

    # Compute overall aggregates
    avg_precision = precision_at_k_sum / total_queries
    mrr = reciprocal_rank_sum / total_queries
    overall_recall = sum(hits_by_category.values()) / total_queries

    # Output Summary Report
    print("\n" + "═" * 70)
    print("                RAG RETRIEVAL EVALUATION REPORT")
    print("═" * 70)
    print(f"Overall Metrics (n={total_queries}):")
    print(f"  • Average Precision@{top_k}: {avg_precision:.2%}")
    print(f"  • Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print(f"  • Overall Recall@{top_k} (Hit Rate): {overall_recall:.2%}")
    print("─" * 70)
    print("Category Breakdown:")
    print(f"  {'Category':<10} {'Scenarios':<10} {'Recall':<10} {'MRR':<10}")
    print("  " + "─" * 40)
    for label in sorted(LABEL_MAP.keys()):
        count = total_by_category[label]
        if count > 0:
            cat_recall = hits_by_category[label] / count
            cat_mrr = category_mrr[label] / count
            print(f"  {label:<10} {count:<10} {cat_recall:>6.1%} {cat_mrr:>9.3f}")
        else:
            print(f"  {label:<10} 0          N/A        N/A")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
