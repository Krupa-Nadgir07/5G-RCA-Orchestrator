"""
API Gateway - FastAPI application entry point.
Provides REST endpoints for RCA analysis, benchmarking, and health checks.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from services.api.middleware import RateLimitMiddleware
from services.api.routes import router

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and teardown services."""
    logger.info("Starting Multi-Agent RCA System")
    
    # Initialize services stored on app state
    from services.orchestrator.graph import OrchestratorGraph
    
    app.state.orchestrator = OrchestratorGraph()
    
    try:
        await app.state.orchestrator.initialize()
        logger.info("All services initialized")
    except Exception as e:
        logger.warning("Service initialization partial", error=str(e))

    # Auto-detect whether the RAG dataset collection needs populating.
    # If Qdrant is reachable but the collection is empty / missing, run
    # the JSONL ingestion so the retriever can search scenario data from
    # the very first query.
    try:
        rag = app.state.orchestrator.rag_service
        retriever = rag.retriever
        if retriever._qdrant_client:
            rag_col = settings.qdrant_rag_collection
            existing = [c.name for c in retriever._qdrant_client.get_collections().collections]
            needs_ingest = rag_col not in existing
            if not needs_ingest:
                info = retriever._qdrant_client.get_collection(rag_col)
                needs_ingest = info.points_count == 0

            if needs_ingest:
                logger.info("RAG dataset collection empty — auto-ingesting JSONL data")
                stats = await rag.ingest_jsonl_datasets("data")
                logger.info("Auto-ingest complete", **stats)
    except Exception as e:
        logger.warning("Auto-ingest of RAG datasets skipped", error=str(e))

    # Auto-ingest structured 3GPP JSON knowledge files (both concept array
    # and handover nested-object formats) into the 3GPP knowledge collection.
    try:
        rag = app.state.orchestrator.rag_service
        retriever = rag.retriever
        if retriever._qdrant_client:
            kb_col = settings.qdrant_collection
            existing = [c.name for c in retriever._qdrant_client.get_collections().collections]
            needs_kb_ingest = kb_col not in existing
            if not needs_kb_ingest:
                info = retriever._qdrant_client.get_collection(kb_col)
                needs_kb_ingest = info.points_count == 0
            if needs_kb_ingest:
                logger.info("3GPP knowledge collection empty — auto-ingesting JSON + TXT data")
                txt_stats = await rag.ingest_knowledge_base("knowledge")
                json_stats = await rag.ingest_json_datasets("data/3PGPP")
                logger.info("Auto-ingest JSON+TXT complete",
                            txt_chunks=txt_stats.get("chunks", 0),
                            json_chunks=json_stats.get("total_chunks", 0))
    except Exception as e:
        logger.warning("Auto-ingest of 3GPP JSON datasets skipped", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Agent RCA System")
    await app.state.orchestrator.inference_service.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Multi-Agent RCA System",
        description="Automated Root Cause Analysis for 5G gNB using SLMs + RAG + CoT",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

    # Include routes
    app.include_router(router, prefix="/api/v1")

    # Root-level health check (convenience)
    @app.get("/health")
    async def root_health():
        return {"status": "healthy", "service": "multi-agent-rca"}

    # Root-level dashboard routes
    from fastapi.responses import HTMLResponse
    from services.api.routes import DASHBOARD_HTML

    @app.get("/dashboard", response_class=HTMLResponse)
    async def root_dashboard():
        return HTMLResponse(content=DASHBOARD_HTML, status_code=200)

    @app.get("/", response_class=HTMLResponse)
    async def root_dashboard_home():
        return HTMLResponse(content=DASHBOARD_HTML, status_code=200)

    return app


app = create_app()
