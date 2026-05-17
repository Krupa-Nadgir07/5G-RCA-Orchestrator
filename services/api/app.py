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
        # Continue even if Qdrant/Redis not available - will use fallbacks
    
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

    return app


app = create_app()
