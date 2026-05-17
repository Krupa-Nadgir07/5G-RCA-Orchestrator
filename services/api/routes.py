"""
API Routes - REST endpoints for the RCA system.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request, Query
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
    return {
        "status": "accepted",
        "message": "Feedback recorded for model improvement",
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

