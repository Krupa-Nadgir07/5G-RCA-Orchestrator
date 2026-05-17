"""
RCA Demo Runner - Runs end-to-end RCA pipeline on the dummy dataset.
Shows the full multi-agent flow without needing the API server.

Usage:
    PYTHONPATH=. python scripts/run_demo.py
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from models.schemas import KPIMetrics, ProcessedEvent
from services.orchestrator.graph import OrchestratorGraph


async def run_single_event(orchestrator: OrchestratorGraph, event_data: dict) -> dict:
    """Run RCA on a single event from the dataset."""
    # Build KPI metrics from dataset
    kpis = KPIMetrics(
        sinr_db=event_data["kpis"].get("sinr_db"),
        prb_utilization_pct=event_data["kpis"].get("prb_utilization_pct"),
        bler_pct=event_data["kpis"].get("bler_pct"),
        handover_success_rate=event_data["kpis"].get("handover_success_rate"),
        interference_level_dbm=event_data["kpis"].get("interference_level_dbm"),
        rsrp_dbm=event_data["kpis"].get("rsrp_dbm"),
        rsrq_db=event_data["kpis"].get("rsrq_db"),
        connected_ues=event_data["kpis"].get("connected_ues"),
        latency_ms=event_data["kpis"].get("latency_ms"),
    )

    # Build ProcessedEvent
    event = ProcessedEvent(
        event_id=uuid4(),
        cell_id=event_data["cell_id"],
        gnb_id=event_data["gnb_id"],
        timestamp=datetime.utcnow(),
        event_type=event_data["issue"],
        kpis=kpis,
    )

    # Execute RCA pipeline
    result = await orchestrator.execute(event)

    return {
        "event_id": event_data["event_id"],
        "cell_id": event_data["cell_id"],
        "issue": event_data["issue"],
        "ground_truth": event_data["ground_truth"],
        "predicted": result.root_cause.value,
        "confidence": result.confidence,
        "specific_cause": result.specific_cause,
        "reasoning_steps": len(result.reasoning_trace),
        "latency_ms": result.latency_ms,
        "correct": result.root_cause.value == event_data["ground_truth"],
    }


async def main():
    print("=" * 70)
    print("5G gNB RCA Multi-Agent System - Demo Runner")
    print("=" * 70)

    # Load dataset
    dataset_path = Path(__file__).parent.parent / "data" / "dummy_rca_dataset.json"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"\nLoaded {len(dataset)} events from dataset")
    print("-" * 70)

    # Initialize orchestrator
    print("\nInitializing orchestrator...")
    orchestrator = OrchestratorGraph()
    
    try:
        await orchestrator.initialize()
        print("Orchestrator initialized successfully")
    except Exception as e:
        print(f"Warning: Partial initialization ({e})")
        print("Continuing with available services...")

    print("\n" + "=" * 70)
    print("Running RCA Pipeline")
    print("=" * 70)

    results = []
    correct = 0
    total = 0

    for i, event_data in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Event: {event_data['event_id']} | "
              f"Cell: {event_data['cell_id']} | Issue: {event_data['issue']}")

        try:
            result = await run_single_event(orchestrator, event_data)
            results.append(result)
            total += 1
            if result["correct"]:
                correct += 1
                status = "CORRECT"
            else:
                status = "WRONG"

            print(f"  → Predicted: {result['predicted']} | "
                  f"Truth: {result['ground_truth']} | "
                  f"Confidence: {result['confidence']:.2f} | "
                  f"Latency: {result['latency_ms']}ms | "
                  f"[{status}]")

        except Exception as e:
            print(f"  → ERROR: {str(e)[:100]}")
            results.append({
                "event_id": event_data["event_id"],
                "error": str(e)[:200],
                "correct": False,
            })
            total += 1

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Total events:  {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.1f}%")

    # Per-class accuracy
    print("\nPer-class breakdown:")
    classes = set(r.get("ground_truth", "") for r in results if "ground_truth" in r)
    for cls in sorted(classes):
        cls_results = [r for r in results if r.get("ground_truth") == cls]
        cls_correct = sum(1 for r in cls_results if r.get("correct", False))
        cls_total = len(cls_results)
        cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
        print(f"  {cls:25s}: {cls_correct}/{cls_total} ({cls_acc:.0f}%)")

    # Average confidence
    confidences = [r["confidence"] for r in results if "confidence" in r]
    if confidences:
        print(f"\nAvg confidence: {sum(confidences)/len(confidences):.2f}")

    # Average latency
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    if latencies:
        print(f"Avg latency:    {sum(latencies)/len(latencies):.0f}ms")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "rca_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    await orchestrator.inference_service.close()


if __name__ == "__main__":
    asyncio.run(main())
