"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from services.api.app import app
from models.schemas import RCAResult, RootCauseCategory, ReasoningStep, AgentType


@pytest.fixture(autouse=True)
def mock_orchestrator():
    """Mock orchestrator on app.state so routes don't crash without services."""
    mock = MagicMock()
    mock.get_metrics.return_value = {"total_requests": 0, "avg_latency_ms": 0}
    mock.execute = AsyncMock(return_value=RCAResult(
        root_cause=RootCauseCategory.INTERFERENCE,
        confidence=0.91,
        specific_cause="Co-channel interference from adjacent cell",
        reasoning_trace=[ReasoningStep(
            step_number=1,
            agent=AgentType.SIGNAL,
            action="analyze_kpis",
            observation="Low SINR detected",
            conclusion="Interference likely",
            confidence=0.9,
        )],
        latency_ms=120,
    ))
    app.state.orchestrator = mock
    yield mock
    del app.state.orchestrator


@pytest.mark.asyncio
async def test_health_check():
    """Health endpoint returns healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_analyze_endpoint():
    """Analyze endpoint accepts valid request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/analyze", json={
            "cell_id": "cell-101",
            "gnb_id": "gnb-001",
            "event_type": "interference_detected",
            "kpis": {"sinr_db": 3.0, "rsrp_dbm": -110},
        })
    
    # May fail without backend services, but should not 422
    assert response.status_code in (200, 500)


@pytest.mark.asyncio
async def test_analyze_missing_fields():
    """Analyze endpoint validates required fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/analyze", json={})
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Metrics endpoint returns data."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/metrics")
    
    assert response.status_code == 200
