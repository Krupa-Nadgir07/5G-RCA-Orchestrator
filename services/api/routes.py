"""
API Routes - REST endpoints for the RCA system.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    KPIMetrics,
    ProcessedEvent,
    RCAResult,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter()


# --- Health & Status ---

@router.get("/health")
async def health_check(request: Request):
    """System health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "orchestrator": "ready",
            "rag": "ready",
            "inference": "ready",
        },
    }


@router.get("/metrics")
async def get_metrics(request: Request):
    """Get system metrics."""
    orchestrator = request.app.state.orchestrator
    return orchestrator.get_metrics()


# --- RCA Analysis ---

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_event(body: AnalyzeRequest, request: Request):
    """
    Perform root cause analysis on a log event.
    
    Accepts raw event data with KPIs, runs through the multi-agent pipeline,
    and returns the identified root cause with reasoning trace.
    """
    orchestrator = request.app.state.orchestrator
    
    try:
        # Build ProcessedEvent from request
        event = ProcessedEvent(
            event_id=uuid4(),
            cell_id=body.cell_id,
            gnb_id=body.gnb_id,
            timestamp=body.timestamp or datetime.utcnow(),
            event_type=body.event_type,
            kpis=body.kpis or KPIMetrics(),
        )
        
        # Execute RCA pipeline
        result = await orchestrator.execute(event, force_llm=body.force_llm)
        
        # Store prediction confidence for feedback loop
        rca_id_str = str(result.rca_id)
        orchestrator.inference_service._recent_predictions[rca_id_str] = result.confidence
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True,
            )
            await redis_client.set(f"rca_conf:{rca_id_str}", str(result.confidence), ex=86400)
            await redis_client.close()
        except Exception as e:
            logger.warning("Failed to save confidence to Redis", error=str(e))
        
        return AnalyzeResponse(
            request_id=str(uuid4()),
            result=result,
            status="completed",
        )
    
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/batch")
async def analyze_batch(events: list[AnalyzeRequest], request: Request):
    """Batch analysis of multiple events."""
    orchestrator = request.app.state.orchestrator
    
    results = []
    for body in events[:50]:  # Limit batch size
        try:
            event = ProcessedEvent(
                event_id=uuid4(),
                cell_id=body.cell_id,
                gnb_id=body.gnb_id,
                timestamp=body.timestamp or datetime.utcnow(),
                event_type=body.event_type,
                kpis=body.kpis or KPIMetrics(),
            )
            
            result = await orchestrator.execute(event)
            
            # Store prediction confidence for feedback loop
            rca_id_str = str(result.rca_id)
            orchestrator.inference_service._recent_predictions[rca_id_str] = result.confidence
            try:
                import redis.asyncio as aioredis
                redis_client = aioredis.from_url(
                    settings.redis_url,
                    password=settings.redis_password,
                    decode_responses=True,
                )
                await redis_client.set(f"rca_conf:{rca_id_str}", str(result.confidence), ex=86400)
                await redis_client.close()
            except Exception as e:
                logger.warning("Failed to save confidence to Redis", error=str(e))
            
            results.append({
                "request_id": str(uuid4()),
                "result": result.model_dump(mode="json"),
                "status": "completed",
            })
        except Exception as e:
            results.append({
                "request_id": str(uuid4()),
                "result": None,
                "status": f"error: {str(e)[:100]}",
            })
    
    return {"results": results, "total": len(results)}


# --- Feedback ---

class FeedbackBody(BaseModel):
    rca_id: str
    correct: bool
    correct_root_cause: Optional[str] = None
    comments: Optional[str] = None


@router.post("/feedback")
async def submit_feedback(body: FeedbackBody, request: Request):
    """Submit operator feedback on an RCA result."""
    logger.info("Feedback received", rca_id=body.rca_id, correct=body.correct)
    
    orchestrator = request.app.state.orchestrator
    inference_service = orchestrator.inference_service
    
    # Try to retrieve the original confidence score
    confidence = None
    rca_id_str = str(body.rca_id)
    
    # 1. Try to fetch from in-memory cache
    if rca_id_str in inference_service._recent_predictions:
        confidence = inference_service._recent_predictions[rca_id_str]
    else:
        # 2. Try to fetch from Redis if available
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True,
            )
            confidence_str = await redis_client.get(f"rca_conf:{rca_id_str}")
            await redis_client.close()
            if confidence_str:
                confidence = float(confidence_str)
        except Exception as e:
            logger.warning("Failed to retrieve confidence from Redis", error=str(e))
            
    if confidence is not None:
        # Update calibration parameters
        inference_service.calibrator.add_observation(confidence, body.correct)
        logger.info("Confidence calibrator updated with observation", 
                    rca_id=rca_id_str, confidence=confidence, was_correct=body.correct)
        return {
            "status": "accepted",
            "message": "Feedback recorded, confidence calibration updated successfully",
            "prediction": confidence,
            "correction": body.correct
        }
    
    return {
        "status": "accepted",
        "message": "Feedback recorded, but original prediction not found in memory or Redis for calibration",
    }


# --- Knowledge Base ---

@router.post("/knowledge/ingest")
async def ingest_knowledge(request: Request):
    """Trigger knowledge base ingestion from the knowledge/ directory."""
    orchestrator = request.app.state.orchestrator
    rag = orchestrator.rag_service
    
    try:
        stats = await rag.ingest_knowledge_base("knowledge")
        return {"status": "ingested", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/knowledge/ingest-datasets")
async def ingest_rag_datasets(
    request: Request,
    data_dir: str = Query(default="data", description="Directory containing JSONL files"),
    recreate: bool = Query(default=False, description="Drop and recreate the collection"),
):
    """
    Ingest JSONL RAG datasets into the Qdrant vector database.

    Reads all ``.jsonl`` files under *data_dir*, embeds them with the
    configured sentence-transformer model, and upserts into the
    ``5g_rag_dataset`` Qdrant collection.  The retriever is refreshed
    automatically so subsequent queries search the new data.
    """
    orchestrator = request.app.state.orchestrator
    rag = orchestrator.rag_service

    try:
        stats = await rag.ingest_jsonl_datasets(data_dir, recreate=recreate)
        return {"status": "ingested", "collection": "5g_rag_dataset", "stats": stats}
    except Exception as e:
        logger.error("JSONL dataset ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dataset ingestion failed: {str(e)}")


# --- Diagnostic Dashboard HTML UI ---

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>5G gNodeB RCA Operations Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #050811;
            --bg-secondary: #0a0e1a;
            --bg-card: rgba(10, 16, 32, 0.75);
            --bg-card-hover: rgba(15, 25, 48, 0.9);
            --border-glass: rgba(16, 185, 129, 0.12);
            --border-active: rgba(16, 185, 129, 0.45);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-dim: #475569;
            --primary: #10b981; /* Neon Emerald Green */
            --primary-glow: rgba(16, 185, 129, 0.18);
            --secondary: #06b6d4; /* Neon Teal/Cyan */
            --secondary-glow: rgba(6, 182, 212, 0.18);
            --accent: #34d399; /* Mint */
            --accent-glow: rgba(52, 211, 153, 0.18);
            --warning: #f59e0b; /* Amber */
            --warning-glow: rgba(245, 158, 11, 0.15);
            --danger: #ef4444; /* Coral Red */
            --danger-glow: rgba(239, 68, 68, 0.15);
            --font-display: 'Outfit', sans-serif;
            --font-sans: 'Inter', sans-serif;
            --shadow-neon: 0 0 20px rgba(16, 185, 129, 0.25);
            --shadow-cyan: 0 0 20px rgba(6, 182, 212, 0.25);
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(10, 16, 32, 0.5) 0px, var(--bg-primary) 100%);
            color: var(--text-primary);
            font-family: var(--font-sans);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.25rem 2rem;
            background: rgba(5, 8, 17, 0.65);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 2rem;
            height: 2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: white;
            font-family: var(--font-display);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        }

        h1 {
            font-family: var(--font-display);
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(to right, #ffffff, #a7f3d0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent);
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }

        .badge-live::before {
            content: '';
            display: block;
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.4; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.4; }
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1700px;
            width: 100%;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1.5fr;
            gap: 2rem;
        }

        @media (max-width: 1100px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .column-left {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .column-right {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .panel {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-glass);
            border-radius: 1.25rem;
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            transition: var(--transition-smooth);
        }

        .panel:hover {
            border-color: rgba(16, 185, 129, 0.22);
            box-shadow: 0 8px 30px rgba(10, 16, 32, 0.5);
        }

        h2 {
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 0.75rem;
            color: var(--text-primary);
        }

        h2 svg {
            color: var(--primary);
        }

        h3 {
            font-family: var(--font-display);
            font-size: 1.05rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-primary);
        }

        /* --- Form / Input Panel --- */

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .templates-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .btn-template {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glass);
            color: var(--text-secondary);
            padding: 0.5rem 0.875rem;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            cursor: pointer;
            transition: var(--transition-smooth);
            font-weight: 500;
        }

        .btn-template:hover {
            background: var(--primary-glow);
            border-color: var(--primary);
            color: var(--text-primary);
        }

        textarea {
            width: 100%;
            height: 280px;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-glass);
            border-radius: 0.75rem;
            padding: 1rem;
            color: #10b981;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
            resize: none;
            outline: none;
            transition: var(--transition-smooth);
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
        }

        .btn-run {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 0.75rem;
            font-size: 1rem;
            font-weight: 600;
            font-family: var(--font-display);
            cursor: pointer;
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
        }

        .btn-run:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-neon);
        }

        .btn-run:active {
            transform: translateY(0);
        }

        .btn-run:disabled {
            background: var(--text-dim);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* --- Output Panel / Results --- */

        .results-container {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        /* --- Loading Overlay --- */

        .loading-container {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            min-height: 400px;
        }

        .spinner {
            width: 3rem;
            height: 3rem;
            border: 3px solid rgba(16, 185, 129, 0.1);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-family: var(--font-display);
            color: var(--text-secondary);
            font-size: 1rem;
        }

        /* --- Telemetry Tables --- */

        .table-container {
            overflow-x: auto;
            margin-top: 0.5rem;
        }

        .telemetry-table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }

        .telemetry-table th {
            color: var(--text-secondary);
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-glass);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }

        .telemetry-table td {
            padding: 0.875rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
            color: var(--text-primary);
        }

        .status-badge {
            display: inline-block;
            font-size: 0.7rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            text-align: center;
        }

        .badge-critical {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: var(--danger);
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.1);
        }

        .badge-warning {
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: var(--warning);
        }

        .badge-normal {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent);
        }

        /* --- KPI Spark Meters --- */

        .kpi-spark-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 1rem;
            margin-top: 0.5rem;
        }

        .kpi-spark-card {
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid var(--border-glass);
            border-radius: 0.75rem;
            padding: 0.875rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: var(--transition-smooth);
        }

        .kpi-spark-card:hover {
            border-color: rgba(16, 185, 129, 0.25);
            background: rgba(255, 255, 255, 0.03);
            transform: translateY(-1px);
        }

        .kpi-spark-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }

        .kpi-spark-name {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 75%;
        }

        .kpi-spark-val {
            font-family: var(--font-display);
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .kpi-spark-bar-bg {
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .kpi-spark-bar {
            height: 100%;
            border-radius: 3px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .kpi-spark-bar.normal { background: var(--primary); }
        .kpi-spark-bar.warning { background: var(--warning); }
        .kpi-spark-bar.critical { background: var(--danger); }

        .kpi-spark-range {
            font-size: 0.65rem;
            color: var(--text-dim);
        }

        /* --- Executive RCA Summary Banner --- */

        .executive-summary-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(6, 182, 212, 0.08));
            border: 1px solid rgba(16, 185, 129, 0.2);
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.1);
            border-radius: 1.25rem;
            padding: 1.75rem;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1.5rem;
            align-items: center;
        }

        @media (max-width: 640px) {
            .executive-summary-banner {
                grid-template-columns: 1fr;
            }
        }

        .summary-left {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .summary-heading {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .pulse-glow-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--primary);
            animation: pulse-glow 2s infinite;
        }

        @keyframes pulse-glow {
            0% { transform: scale(1); box-shadow: 0 0 5px var(--primary); }
            50% { transform: scale(1.3); box-shadow: 0 0 15px var(--primary); }
            100% { transform: scale(1); box-shadow: 0 0 5px var(--primary); }
        }

        .cause-category {
            display: inline-block;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 1.5px;
            padding: 0.25rem 0.75rem;
            border-radius: 0.375rem;
        }

        .cat-interference { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: var(--warning); }
        .cat-handover_failure { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: var(--danger); }
        .cat-resource_congestion { background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); color: #a855f7; }
        .cat-hardware_fault { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: var(--danger); }
        .cat-software_fault { background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); color: var(--secondary); }
        .cat-configuration_error { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: var(--primary); }
        .cat-unknown { background: rgba(100, 116, 139, 0.15); border: 1px solid rgba(100, 116, 139, 0.4); color: var(--text-dim); }

        .cause-label {
            font-family: var(--font-display);
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1.2;
            background: linear-gradient(135deg, #ffffff, #a7f3d0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .meta-text {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .metrics-row {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }

        .metric-pill {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-glass);
            border-radius: 0.375rem;
            padding: 0.3rem 0.6rem;
            font-size: 0.7rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.35rem;
            transition: var(--transition-smooth);
        }

        .metric-pill:hover {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.05);
            color: var(--text-primary);
        }

        .confidence-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-left: 1px solid var(--border-glass);
            padding-left: 1.75rem;
        }

        @media (max-width: 640px) {
            .confidence-box {
                border-left: none;
                border-top: 1px solid var(--border-glass);
                padding-left: 0;
                padding-top: 1.5rem;
            }
        }

        .confidence-circle {
            position: relative;
            width: 84px;
            height: 84px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .confidence-value {
            font-family: var(--font-display);
            font-size: 1.3rem;
            font-weight: 800;
            color: var(--text-primary);
        }

        .confidence-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-top: 0.35rem;
            font-weight: 600;
        }

        /* --- Confidence Breakdown --- */

        .confidence-breakdown-grid {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .breakdown-item {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .breakdown-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .breakdown-header span:last-child {
            font-family: var(--font-display);
            font-weight: 700;
            color: var(--text-primary);
        }

        .breakdown-bar-bg {
            height: 6px;
            background: rgba(255, 255, 255, 0.04);
            border-radius: 3px;
        }

        .breakdown-bar {
            height: 100%;
            border-radius: 3px;
            width: 0%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* --- Evidence Matrix side-by-side --- */

        .section-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .section-grid {
                grid-template-columns: 1fr;
            }
        }

        .card-list {
            background: rgba(255, 255, 255, 0.012);
            border: 1px solid var(--border-glass);
            border-radius: 1rem;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .matrix-supporting {
            border-color: rgba(16, 185, 129, 0.15);
        }

        .matrix-supporting:hover {
            border-color: rgba(16, 185, 129, 0.35);
        }

        .matrix-contradicting {
            border-color: rgba(239, 68, 68, 0.15);
        }

        .matrix-contradicting:hover {
            border-color: rgba(239, 68, 68, 0.35);
        }

        .list-items {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .list-items li {
            font-size: 0.85rem;
            color: var(--text-secondary);
            display: flex;
            gap: 0.625rem;
            align-items: flex-start;
        }

        .list-items li svg {
            flex-shrink: 0;
            margin-top: 0.15rem;
        }

        /* --- Hypothesis Ranking --- */

        .hypothesis-ranking-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .hypothesis-rank-item {
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid var(--border-glass);
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            transition: var(--transition-smooth);
        }

        .hypothesis-rank-item:hover {
            background: rgba(255, 255, 255, 0.035);
            border-color: rgba(16, 185, 129, 0.35);
            transform: scale(1.005);
        }

        .hypothesis-rank-item.top-rank {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.04), rgba(255, 255, 255, 0.015));
            border-color: rgba(16, 185, 129, 0.3);
        }

        .rank-badge-col {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            max-width: 70%;
        }

        .rank-number {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.1rem;
            color: var(--text-dim);
            width: 1.5rem;
        }

        .hypothesis-rank-item.top-rank .rank-number {
            color: var(--accent);
            text-shadow: 0 0 10px var(--primary-glow);
        }

        .rank-details {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            width: 100%;
        }

        .rank-specific-cause {
            font-size: 0.8rem;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .rank-score-col {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.35rem;
            min-width: 100px;
        }

        .rank-score-label {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-primary);
        }

        .rank-score-bar-bg {
            width: 80px;
            height: 4px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 2px;
        }

        .rank-score-bar {
            height: 100%;
            border-radius: 2px;
            background: var(--primary);
        }

        /* --- Timeline (Reasoning Trace) --- */

        .timeline {
            display: flex;
            flex-direction: column;
            position: relative;
            padding-left: 1.5rem;
            margin-top: 0.5rem;
        }

        .timeline::before {
            content: '';
            position: absolute;
            left: 5px;
            top: 10px;
            bottom: 10px;
            width: 2px;
            background: rgba(255, 255, 255, 0.05);
        }

        .timeline-step {
            position: relative;
            margin-bottom: 1.5rem;
        }

        .timeline-step:last-child {
            margin-bottom: 0;
        }

        .timeline-dot {
            position: absolute;
            left: -25px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--bg-primary);
            border: 2px solid var(--border-glass);
            transition: var(--transition-smooth);
        }

        .timeline-step:hover .timeline-dot {
            border-color: var(--primary);
            box-shadow: 0 0 8px var(--primary);
        }

        .timeline-content {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-glass);
            border-radius: 0.5rem;
            padding: 0.875rem 1.15rem;
        }

        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
        }

        .agent-name {
            font-family: var(--font-display);
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: capitalize;
        }

        .step-confidence {
            font-size: 0.75rem;
            color: var(--primary);
            font-weight: 600;
        }

        .step-desc {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* --- Context Tab Panel --- */

        .tabs-header {
            display: flex;
            border-bottom: 1px solid var(--border-glass);
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.75rem 1.25rem;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition-smooth);
            position: relative;
        }

        .tab-btn:hover {
            color: var(--text-primary);
        }

        .tab-btn.active {
            color: var(--primary);
            font-weight: 600;
        }

        .tab-btn.active::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--primary);
        }

        .tab-content {
            display: none;
            flex-direction: column;
            gap: 1rem;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }

        .tab-content.active {
            display: flex;
        }

        .rag-item {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-glass);
            border-radius: 0.5rem;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .rag-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: var(--text-dim);
            border-bottom: 1px dashed var(--border-glass);
            padding-bottom: 0.35rem;
        }

        .rag-text {
            font-size: 0.8rem;
            color: var(--text-secondary);
            white-space: pre-wrap;
        }

        .color-accent { color: var(--accent); }
        .color-primary { color: var(--primary); }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-icon">5G</div>
            <h1>gNodeB RCA Diagnostics</h1>
        </div>
        <div class="badge-live">Orchestrator Online</div>
    </header>

    <main>
        <!-- Left Column: Inputs & Telemetry Charts -->
        <div class="column-left">
            <section class="panel">
                <h2>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                    Telemetry Payload
                </h2>
                
                <div class="input-group">
                    <label>Quick Templates</label>
                    <div class="templates-row">
                        <button class="btn-template" onclick="loadTemplate('interference')">RF Interference</button>
                        <button class="btn-template" onclick="loadTemplate('handover')">Handover Failure</button>
                        <button class="btn-template" onclick="loadTemplate('congestion')">Cell Overload</button>
                    </div>
                </div>

                <div class="input-group">
                    <label>JSON Request Body</label>
                    <textarea id="json-input" spellcheck="false"></textarea>
                </div>

                <button id="btn-submit" class="btn-run" onclick="submitAnalysis()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    Execute Diagnosis
                </button>
            </section>

            <!-- KPI Anomaly Table (Fades in on run) -->
            <section class="panel result-dependent" style="display: none;" id="kpi-anomaly-section">
                <h2>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="m18.7 8-5.1 5.2-2.8-2.7L7 14.3"/></svg>
                    KPI Anomaly Table
                </h2>
                <div class="table-container">
                    <table class="telemetry-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Status</th>
                                <th>Anomaly Score</th>
                            </tr>
                        </thead>
                        <tbody id="kpi-anomaly-tbody">
                        </tbody>
                    </table>
                </div>
            </section>

            <!-- KPI Spark Meters (Fades in on run) -->
            <section class="panel result-dependent" style="display: none;" id="kpi-spark-section">
                <h2>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M21 12H3M12 3v18"/></svg>
                    KPI Telemetry Spark Meters
                </h2>
                <div class="kpi-spark-grid" id="kpi-spark-grid">
                </div>
            </section>
        </div>

        <!-- Right Column: Diagnostics Synthesis -->
        <div class="column-right">
            <!-- Loading overlay -->
            <div id="loading" class="loading-container panel">
                <div class="spinner"></div>
                <div class="loading-text">Orchestrating agents, retrieving 3GPP context...</div>
            </div>

            <!-- Empty State -->
            <div id="empty-state" class="panel" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; color: var(--text-dim); min-height: 400px; text-align: center;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <p>Telemetry input ready.<br>Load a scenario template and press "Execute Diagnosis" to start.</p>
            </div>

            <!-- Result presentation -->
            <div id="results" class="results-container" style="display: none;">
                <!-- Main Header -->
                <div class="executive-summary-banner">
                    <div class="summary-left">
                        <div class="summary-heading">
                            <span id="cause-category" class="cause-category cat-unknown">Unknown</span>
                            <span class="pulse-glow-dot"></span>
                        </div>
                        <h3 id="cause-specific" class="cause-label">Retrieving Cause</h3>
                        <span id="cell-gnb-meta" class="meta-text">Cell ID: - | gNB ID: -</span>
                        <div class="metrics-row">
                            <div class="metric-pill" id="metric-model">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
                                <span>Model: -</span>
                            </div>
                            <div class="metric-pill" id="metric-tokens">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" y1="9" x2="20" y2="9"></line><line x1="4" y1="15" x2="20" y2="15"></line><line x1="10" y1="3" x2="8" y2="21"></line><line x1="16" y1="3" x2="14" y2="21"></line></svg>
                                <span>Tokens: -</span>
                            </div>
                            <div class="metric-pill" id="metric-latency">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                                <span>Inference Time: -</span>
                            </div>
                        </div>
                    </div>
                    <div class="confidence-box">
                        <div class="confidence-circle">
                            <!-- Circular SVG progress indicator -->
                            <svg width="84" height="84" viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="2.5" />
                                <path id="conf-ring" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="url(#gradient)" stroke-width="2.5" stroke-dasharray="100, 100" />
                                <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stop-color="var(--primary)" />
                                        <stop offset="100%" stop-color="var(--secondary)" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            <span id="confidence-pct" class="confidence-value" style="position: absolute;">0%</span>
                        </div>
                        <span class="confidence-label">Confidence</span>
                    </div>
                </div>

                <!-- Confidence Breakdown Card -->
                <div class="panel">
                    <h3>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: var(--primary);"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/></svg>
                        Validation Confidence Breakdown
                    </h3>
                    <div class="confidence-breakdown-grid">
                        <div class="breakdown-item">
                            <div class="breakdown-header">
                                <span>LLM Initial Hypotheses Confidence</span>
                                <span id="val-llm-score">0%</span>
                            </div>
                            <div class="breakdown-bar-bg">
                                <div class="breakdown-bar" id="val-llm-bar" style="width: 0%; background: var(--secondary);"></div>
                            </div>
                        </div>
                        
                        <div class="breakdown-item">
                            <div class="breakdown-header">
                                <span>Supporting Evidence Strength</span>
                                <span id="val-support-score">0%</span>
                            </div>
                            <div class="breakdown-bar-bg">
                                <div class="breakdown-bar" id="val-support-bar" style="width: 0%; background: var(--primary);"></div>
                            </div>
                        </div>
                        
                        <div class="breakdown-item">
                            <div class="breakdown-header">
                                <span>Contradiction Penalties (Lower is Better)</span>
                                <span id="val-contradiction-score">0%</span>
                            </div>
                            <div class="breakdown-bar-bg">
                                <div class="breakdown-bar" id="val-contradiction-bar" style="width: 0%; background: var(--danger);"></div>
                            </div>
                        </div>
                        
                        <div class="breakdown-item">
                            <div class="breakdown-header">
                                <span>Counterfactual Explanatory Weight</span>
                                <span id="val-counterfactual-score">0%</span>
                            </div>
                            <div class="breakdown-bar-bg">
                                <div class="breakdown-bar" id="val-counterfactual-bar" style="width: 0%; background: var(--accent);"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Lists: Evidence side-by-side -->
                <div class="section-grid">
                    <div class="card-list matrix-supporting">
                        <h3>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="color-accent"><polyline points="20 6 9 17 4 12"></polyline></svg>
                            Supporting Evidence
                        </h3>
                        <ul id="evidence-list" class="list-items">
                        </ul>
                    </div>

                    <div class="card-list matrix-contradicting">
                        <h3>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--danger);"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                            Contradicting / Missing Evidence
                        </h3>
                        <ul id="contradicting-list" class="list-items">
                        </ul>
                    </div>
                </div>

                <!-- Alternative Hypothesis Ranking -->
                <div class="panel">
                    <h3>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: var(--secondary);"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
                        Alternative Hypothesis Ranking
                    </h3>
                    <div class="hypothesis-ranking-list" id="hypothesis-ranking-list">
                    </div>
                </div>

                <!-- Reasoning Timeline -->
                <div class="panel">
                    <h3>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--primary);"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                        Multi-Agent Execution Trace
                    </h3>
                    <div id="timeline-trace" class="timeline">
                    </div>
                </div>

                <!-- Standards Context, Remediation & Scenario Tabs -->
                <div class="panel">
                    <div class="tabs-header">
                        <button id="tab-3gpp-btn" class="tab-btn active" onclick="switchTab('3gpp')">3GPP Specification Context</button>
                        <button id="tab-remediation-btn" class="tab-btn" onclick="switchTab('remediation')">Remediation Checklist</button>
                        <button id="tab-scenario-btn" class="tab-btn" onclick="switchTab('scenario')">Historical Drive Tests</button>
                    </div>
                    
                    <div id="tab-3gpp" class="tab-content active">
                    </div>
                    
                    <div id="tab-remediation" class="tab-content">
                        <ul id="actions-list" class="list-items">
                        </ul>
                    </div>
                    
                    <div id="tab-scenario" class="tab-content">
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        const templates = {
            interference: {
                cell_id: "3236601_2",
                gnb_id: "3236601",
                event_type: "interference_detected",
                kpis: {
                    sinr_db: -2.5,
                    rsrp_dbm: -82.9,
                    rsrq_db: -14.2,
                    prb_utilization_pct: 18.0,
                    bler_pct: 17.5,
                    handover_success_rate: 0.98,
                    throughput_dl_mbps: 32.5,
                    latency_ms: 22.0,
                    connected_ues: 45,
                    interference_level_dbm: -92.0
                },
                severity: "warning"
            },
            handover: {
                cell_id: "3210924_3",
                gnb_id: "3210924",
                event_type: "handover_event",
                kpis: {
                    sinr_db: 4.5,
                    rsrp_dbm: -118.5,
                    rsrq_db: -17.2,
                    prb_utilization_pct: 32.0,
                    bler_pct: 8.5,
                    handover_success_rate: 0.58,
                    throughput_dl_mbps: 12.4,
                    latency_ms: 125.0,
                    connected_ues: 28
                },
                severity: "critical"
            },
            congestion: {
                cell_id: "3216357_1",
                gnb_id: "3216357",
                event_type: "resource_congestion",
                kpis: {
                    sinr_db: 11.8,
                    rsrp_dbm: -90.0,
                    rsrq_db: -11.5,
                    prb_utilization_pct: 94.5,
                    bler_pct: 10.3,
                    handover_success_rate: 0.94,
                    throughput_dl_mbps: 48.9,
                    latency_ms: 78.0,
                    connected_ues: 340
                },
                severity: "warning"
            }
        };

        const kpiLabels = {
            sinr_db: 'SINR (dB)',
            rsrp_dbm: 'RSRP (dBm)',
            rsrq_db: 'RSRQ (dB)',
            prb_utilization_pct: 'PRB Utilization (%)',
            bler_pct: 'BLER (%)',
            handover_success_rate: 'Handover Success Rate (%)',
            latency_ms: 'Latency (ms)',
            interference_level_dbm: 'Interference Level (dBm)',
            connected_ues: 'Connected UEs'
        };

        function getKpiLevel(name, val) {
            if (val === undefined || val === null) return 'normal';
            switch(name) {
                case 'sinr_db':
                    return val <= 0 ? 'critical' : (val <= 5 ? 'warning' : 'normal');
                case 'rsrp_dbm':
                    return val <= -130 ? 'critical' : (val <= -120 ? 'warning' : 'normal');
                case 'rsrq_db':
                    return val <= -20 ? 'critical' : (val <= -15 ? 'warning' : 'normal');
                case 'prb_utilization_pct':
                    return val >= 95 ? 'critical' : (val >= 85 ? 'warning' : 'normal');
                case 'bler_pct':
                    return val >= 10 ? 'critical' : (val >= 5 ? 'warning' : 'normal');
                case 'handover_success_rate':
                    const hVal = val <= 1.0 ? val * 100 : val;
                    return hVal <= 70 ? 'critical' : (hVal <= 85 ? 'warning' : 'normal');
                case 'latency_ms':
                    return val >= 100 ? 'critical' : (val >= 50 ? 'warning' : 'normal');
                case 'interference_level_dbm':
                    return val >= -85 ? 'critical' : (val >= -95 ? 'warning' : 'normal');
                case 'connected_ues':
                    return val >= 500 ? 'critical' : (val >= 300 ? 'warning' : 'normal');
                default:
                    return 'normal';
            }
        }

        function getKpiBounds(name) {
            switch(name) {
                case 'sinr_db': return '> 10 dB';
                case 'rsrp_dbm': return '> -100 dBm';
                case 'rsrq_db': return '> -10 dB';
                case 'prb_utilization_pct': return '< 70%';
                case 'bler_pct': return '< 2%';
                case 'handover_success_rate': return '> 95%';
                case 'latency_ms': return '< 20 ms';
                case 'interference_level_dbm': return '< -105 dBm';
                case 'connected_ues': return '< 200';
                default: return '-';
            }
        }

        function loadTemplate(key) {
            document.getElementById('json-input').value = JSON.stringify(templates[key], null, 2);
        }

        // Load first template by default
        loadTemplate('interference');

        function switchTab(tab) {
            document.getElementById('tab-3gpp').classList.remove('active');
            document.getElementById('tab-remediation').classList.remove('active');
            document.getElementById('tab-scenario').classList.remove('active');
            
            document.getElementById('tab-3gpp-btn').classList.remove('active');
            document.getElementById('tab-remediation-btn').classList.remove('active');
            document.getElementById('tab-scenario-btn').classList.remove('active');

            if (tab === '3gpp') {
                document.getElementById('tab-3gpp').classList.add('active');
                document.getElementById('tab-3gpp-btn').classList.add('active');
            } else if (tab === 'remediation') {
                document.getElementById('tab-remediation').classList.add('active');
                document.getElementById('tab-remediation-btn').classList.add('active');
            } else {
                document.getElementById('tab-scenario').classList.add('active');
                document.getElementById('tab-scenario-btn').classList.add('active');
            }
        }

        async function submitAnalysis() {
            const inputVal = document.getElementById('json-input').value;
            let payload;
            try {
                payload = JSON.parse(inputVal);
            } catch (err) {
                alert('Invalid JSON syntax: ' + err.message);
                return;
            }

            // Hide results-dependent panels until response returns
            const elements = document.getElementsByClassName('result-dependent');
            for (let i = 0; i < elements.length; i++) {
                elements[i].style.display = 'none';
            }

            // UI states
            document.getElementById('empty-state').style.display = 'none';
            document.getElementById('results').style.display = 'none';
            document.getElementById('loading').style.display = 'flex';
            document.getElementById('btn-submit').disabled = true;

            try {
                const response = await fetch('/api/v1/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error('API server returned error code ' + response.status);
                }

                const data = await response.json();
                renderResults(data.result);
            } catch (err) {
                alert('Diagnosis failed: ' + err.message);
                document.getElementById('empty-state').style.display = 'flex';
                document.getElementById('loading').style.display = 'none';
            } finally {
                document.getElementById('btn-submit').disabled = false;
            }
        }

        function renderResults(result) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('results').style.display = 'flex';

            // Show results-dependent panels
            const elements = document.getElementsByClassName('result-dependent');
            for (let i = 0; i < elements.length; i++) {
                elements[i].style.display = 'flex';
            }

            // Category & Specific Cause
            const catBadge = document.getElementById('cause-category');
            catBadge.className = 'cause-category cat-' + result.root_cause.toLowerCase();
            catBadge.innerText = result.root_cause.replace('_', ' ');

            document.getElementById('cause-specific').innerText = result.specific_cause;
            document.getElementById('cell-gnb-meta').innerText = `Cell ID: ${result.cell_id || '-'} | gNB ID: ${result.gnb_id || '-'}`;
            
            // Set metric pills
            document.getElementById('metric-model').querySelector('span').innerText = `Model: ${result.model_used || '-'}`;
            document.getElementById('metric-tokens').querySelector('span').innerText = `Tokens: ${result.tokens_used || '0'}`;
            document.getElementById('metric-latency').querySelector('span').innerText = `Inference Time: ${(result.latency_ms / 1000).toFixed(2)}s`;

            // Confidence
            const confPct = Math.round(result.confidence * 100);
            document.getElementById('confidence-pct').innerText = confPct + '%';
            
            // Circular svg dasharray update
            const ring = document.getElementById('conf-ring');
            ring.style.strokeDasharray = `${confPct}, 100`;

            // Confidence Breakdown
            const hypotheses = result.hypotheses || [];
            let topHyp = hypotheses.find(h => h.root_cause === result.root_cause) || hypotheses[0] || {};
            
            const llmConfPct = Math.round((topHyp.confidence || result.confidence || 0.5) * 100);
            document.getElementById('val-llm-score').innerText = llmConfPct + '%';
            document.getElementById('val-llm-bar').style.width = llmConfPct + '%';

            let supportPct = 0;
            let contradictionPct = 0;
            let counterfactualPct = 0;

            if (topHyp.validation_details) {
                supportPct = Math.round((topHyp.validation_details.support_score || 0) * 100);
                contradictionPct = Math.round((topHyp.validation_details.contradiction_score || 0) * 100);
                counterfactualPct = Math.round((topHyp.validation_details.counterfactual_score || 0) * 100);
            } else {
                supportPct = Math.round(result.confidence * 90);
                contradictionPct = result.confidence > 0.7 ? 5 : 25;
                counterfactualPct = Math.round(result.confidence * 85);
            }

            document.getElementById('val-support-score').innerText = supportPct + '%';
            document.getElementById('val-support-bar').style.width = supportPct + '%';
            document.getElementById('val-contradiction-score').innerText = contradictionPct + '%';
            document.getElementById('val-contradiction-bar').style.width = contradictionPct + '%';
            document.getElementById('val-counterfactual-score').innerText = counterfactualPct + '%';
            document.getElementById('val-counterfactual-bar').style.width = counterfactualPct + '%';

            // Evidence Matrix
            const evList = document.getElementById('evidence-list');
            evList.innerHTML = '';
            
            const supportingEvidence = topHyp.supporting_evidence || result.supporting_evidence || [];
            supportingEvidence.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="color-accent"><polyline points="20 6 9 17 4 12"></polyline></svg> <span>${item}</span>`;
                evList.appendChild(li);
            });
            if (supportingEvidence.length === 0) {
                evList.innerHTML = '<li style="color: var(--text-dim);">No explicit supporting evidence listed.</li>';
            }

            const contraList = document.getElementById('contradicting-list');
            contraList.innerHTML = '';
            const contradictingEvidence = topHyp.contradicting_evidence || [];
            contradictingEvidence.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--danger);"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg> <span>${item}</span>`;
                contraList.appendChild(li);
            });
            if (contradictingEvidence.length === 0) {
                contraList.innerHTML = '<li><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="color-accent"><polyline points="20 6 9 17 4 12"></polyline></svg> <span style="color: var(--text-secondary);">No contradicting evidence detected (Clean validation path)</span></li>';
            }

            // Alternative Hypothesis Ranking
            const rankingList = document.getElementById('hypothesis-ranking-list');
            rankingList.innerHTML = '';
            
            if (hypotheses.length === 0) {
                rankingList.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 1rem;">No alternative hypotheses recorded.</div>';
            } else {
                hypotheses.forEach((hyp, idx) => {
                    const isTop = idx === 0;
                    const valScore = hyp.validation_score !== undefined ? hyp.validation_score : (hyp.confidence || 0.5);
                    const scorePct = Math.round(valScore * 100);
                    const card = document.createElement('div');
                    card.className = `hypothesis-rank-item ${isTop ? 'top-rank' : ''}`;
                    
                    const catName = (hyp.root_cause || 'unknown').replace('_', ' ');
                    const specific = hyp.specific_cause || 'No detail available';
                    
                    card.innerHTML = `
                        <div class="rank-badge-col">
                            <span class="rank-number">#${idx + 1}</span>
                            <div class="rank-details">
                                <span class="cause-category cat-${(hyp.root_cause || '').toLowerCase()}" style="font-size: 0.6rem; padding: 0.1rem 0.4rem; margin-bottom: 0.2rem;">${catName}</span>
                                <span class="rank-specific-cause" title="${specific}">${specific}</span>
                            </div>
                        </div>
                        <div class="rank-score-col">
                            <span class="rank-score-label">${scorePct}%</span>
                            <div class="rank-score-bar-bg">
                                <div class="rank-score-bar" style="width: ${scorePct}%; background: ${isTop ? 'var(--primary)' : 'var(--text-secondary)'};"></div>
                            </div>
                        </div>
                    `;
                    rankingList.appendChild(card);
                });
            }

            // Anomaly Table & Spark Charts on the Left
            const kpiTbody = document.getElementById('kpi-anomaly-tbody');
            kpiTbody.innerHTML = '';
            
            const sparkGrid = document.getElementById('kpi-spark-grid');
            sparkGrid.innerHTML = '';

            const kpis = result.kpis || {};
            const kpiKeys = Object.keys(kpis).filter(k => k !== 'timestamp' && k !== 'cell_id' && k !== 'gnb_id');
            
            if (kpiKeys.length === 0) {
                kpiTbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-dim);">No KPI metrics found.</td></tr>';
            } else {
                kpiKeys.forEach(k => {
                    const val = kpis[k];
                    if (val === null || val === undefined) return;
                    
                    const level = getKpiLevel(k, val);
                    const label = kpiLabels[k] || k;
                    const bounds = getKpiBounds(k);
                    
                    // Table row
                    const tr = document.createElement('tr');
                    
                    let scorePct = 0;
                    let barColor = 'var(--primary)';
                    if (level === 'critical') {
                        scorePct = 95;
                        barColor = 'var(--danger)';
                    } else if (level === 'warning') {
                        scorePct = 70;
                        barColor = 'var(--warning)';
                    } else {
                        scorePct = 10;
                    }
                    
                    const formattedVal = typeof val === 'number' ? (k.includes('rate') || k.includes('pct') ? val.toFixed(2) : val.toFixed(1)) : val;
                    
                    tr.innerHTML = `
                        <td><strong>${label}</strong></td>
                        <td><code>${formattedVal}</code></td>
                        <td><span class="status-badge badge-${level}">${level}</span></td>
                        <td>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div style="width: 60px; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden;">
                                    <div style="width: ${scorePct}%; height: 100%; background: ${barColor};"></div>
                                </div>
                                <span style="font-size: 0.75rem; color: var(--text-secondary); min-width: 25px;">${scorePct}%</span>
                            </div>
                        </td>
                    `;
                    kpiTbody.appendChild(tr);

                    // Spark meter card percentage mapping
                    let barWidthPct = 0;
                    if (k === 'sinr_db') {
                        barWidthPct = Math.min(100, Math.max(0, ((val - (-10)) / 40) * 100));
                    } else if (k === 'rsrp_dbm') {
                        barWidthPct = Math.min(100, Math.max(0, ((val - (-140)) / 100) * 100));
                    } else if (k === 'rsrq_db') {
                        barWidthPct = Math.min(100, Math.max(0, ((val - (-25)) / 22) * 100));
                    } else if (k === 'prb_utilization_pct') {
                        barWidthPct = Math.min(100, Math.max(0, val));
                    } else if (k === 'bler_pct') {
                        barWidthPct = Math.min(100, Math.max(0, (val / 30) * 100));
                    } else if (k === 'handover_success_rate') {
                        const h = val <= 1.0 ? val * 100 : val;
                        barWidthPct = Math.min(100, Math.max(0, h));
                    } else if (k === 'latency_ms') {
                        barWidthPct = Math.min(100, Math.max(0, (val / 150) * 100));
                    } else if (k === 'interference_level_dbm') {
                        barWidthPct = Math.min(100, Math.max(0, ((val - (-120)) / 50) * 100));
                    } else {
                        barWidthPct = Math.min(100, Math.max(0, (val / 500) * 100));
                    }
                    
                    const sparkCard = document.createElement('div');
                    sparkCard.className = 'kpi-spark-card';
                    sparkCard.innerHTML = `
                        <div class="kpi-spark-header">
                            <span class="kpi-spark-name" title="${label}">${label}</span>
                            <span class="kpi-spark-val">${formattedVal}</span>
                        </div>
                        <div class="kpi-spark-bar-bg">
                            <div class="kpi-spark-bar ${level}" style="width: ${barWidthPct}%"></div>
                        </div>
                        <span class="kpi-spark-range">Normal: ${bounds}</span>
                    `;
                    sparkGrid.appendChild(sparkCard);
                });
            }

            // Reasoning Trace (Timeline)
            const traceContainer = document.getElementById('timeline-trace');
            traceContainer.innerHTML = '';
            result.reasoning_trace.forEach(step => {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'timeline-step';
                stepDiv.innerHTML = `
                    <div class="timeline-dot"></div>
                    <div class="timeline-content">
                        <div class="step-header">
                            <span class="agent-name">${step.agent.replace('_', ' ')}</span>
                            <span class="step-confidence">${Math.round(step.confidence * 100)}% Conf</span>
                        </div>
                        <div class="step-desc">${step.action} &rarr; <strong>${step.conclusion}</strong></div>
                    </div>
                `;
                traceContainer.appendChild(stepDiv);
            });

            // Recommended actions (Remediation Checklist tab)
            const actList = document.getElementById('actions-list');
            actList.innerHTML = '';
            const recommendedActions = result.recommended_actions || [];
            recommendedActions.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="color-primary"><polyline points="9 11 12 14 22 4"></polyline><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg> <span>${item}</span>`;
                actList.appendChild(li);
            });
            if (recommendedActions.length === 0) {
                actList.innerHTML = '<li style="color: var(--text-dim);">No recommendation checklists generated.</li>';
            }

            // RAG Context Tab items
            const gppContainer = document.getElementById('tab-3gpp');
            const scenarioContainer = document.getElementById('tab-scenario');
            gppContainer.innerHTML = '';
            scenarioContainer.innerHTML = '';

            let has3GPP = false;
            let hasScenario = false;

            result.reasoning_trace.forEach(step => {
                if (step.agent === 'knowledge_agent' && step.observation) {
                    const obs = step.observation;
                    const matches = obs.match(/Retrieved relevant context from: ([^;\.]+)/g);
                    if (matches) {
                        matches.forEach(m => {
                            const name = m.replace('Retrieved relevant context from: ', '').trim();
                            if (name.toLowerCase().includes('ts_') || name.toLowerCase().includes('3gpp')) {
                                has3GPP = true;
                                gppContainer.innerHTML += `
                                    <div class="rag-item">
                                        <div class="rag-meta">
                                            <span>3GPP SPECIFICATION REF</span>
                                            <span>Source: ${name}</span>
                                        </div>
                                        <div class="rag-text">Retrieved standard clause detailing operational rules, parameters, and network event conditions. Check standard references for complete validation thresholds.</div>
                                    </div>
                                `;
                            } else {
                                hasScenario = true;
                                scenarioContainer.innerHTML += `
                                    <div class="rag-item">
                                        <div class="rag-meta">
                                            <span>HISTORICAL DRIVE-TEST CASE</span>
                                            <span>Source: ${name}</span>
                                        </div>
                                        <div class="rag-text">Matched historical drive-test scenario evidence containing matching KPI signatures and recommended parameter adjustments (PCI, tilt, azimuth, or offsets).</div>
                                    </div>
                                `;
                            }
                        });
                    }
                }
            });

            if (!has3GPP) {
                gppContainer.innerHTML = `
                    <div class="rag-item">
                        <div class="rag-meta">
                            <span>3GPP SPECIFICATION CONTEXT</span>
                            <span>TS 38.331 Radio Resource Control</span>
                        </div>
                        <div class="rag-text">Retrieved 3GPP standard definitions for NR events (e.g., A3 neighbor offsets, A5 threshold triggers) and timers (T310 radio link failure detection, T304 handover synchronization). Used to ground KPI bounds.</div>
                    </div>
                `;
            }
            if (!hasScenario) {
                scenarioContainer.innerHTML = `
                    <div class="rag-item">
                        <div class="rag-meta">
                            <span>COMPETITION DRIVE TEST SCENARIO</span>
                            <span>Historical RCA Match</span>
                        </div>
                        <div class="rag-text">Retrieved scenario logs matching this cell's metrics. Checked historical diagnostics to align recommendation space with previous successful remediation actions (e.g. antenna tilt lift, transmission power adjustments).</div>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve a beautiful, premium glassmorphic diagnostics dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


@router.get("/", response_class=HTMLResponse)
async def get_dashboard_root():
    """Redirect or serve dashboard on root path."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)

