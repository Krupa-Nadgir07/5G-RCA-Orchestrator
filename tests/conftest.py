"""
Test configuration and shared fixtures.
"""

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from models.schemas import KPIMetrics, ProcessedEvent, RawLogEvent


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_kpis():
    """Sample KPI metrics for testing."""
    return KPIMetrics(
        sinr_db=3.0,
        rsrp_dbm=-110,
        rsrq_db=-12,
        bler_pct=8.0,
        prb_utilization_pct=85,
        throughput_dl_mbps=25,
        connected_ues=150,
        latency_ms=45,
        handover_success_rate=0.78,
    )


@pytest.fixture
def sample_event(sample_kpis):
    """Sample processed event for testing."""
    return ProcessedEvent(
        event_id=uuid4(),
        cell_id="cell-101",
        gnb_id="gnb-001",
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        event_type="interference_detected",
        kpis=sample_kpis,
        severity="warning",
    )


@pytest.fixture
def sample_raw_log():
    """Sample raw log event."""
    return RawLogEvent(
        source_id="gnb-001",
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        message="[ALARM] cell-101 SINR degradation detected: SINR=3.0dB (threshold=5dB), interference_level=-88dBm",
        metadata={"cell_id": "cell-101", "severity": "warning"},
    )


@pytest.fixture
def congestion_event():
    """Event with congestion pattern."""
    return ProcessedEvent(
        event_id=uuid4(),
        cell_id="cell-201",
        gnb_id="gnb-002",
        timestamp=datetime(2026, 1, 15, 14, 0, 0),
        event_type="resource_congestion",
        kpis=KPIMetrics(
            prb_utilization_pct=96,
            connected_ues=400,
            latency_ms=80,
            throughput_dl_mbps=10,
            sinr_db=15,
        ),
        severity="critical",
    )


@pytest.fixture
def handover_event():
    """Event with handover failure pattern."""
    return ProcessedEvent(
        event_id=uuid4(),
        cell_id="cell-301",
        gnb_id="gnb-003",
        timestamp=datetime(2026, 1, 15, 11, 15, 0),
        event_type="handover_event",
        kpis=KPIMetrics(
            handover_success_rate=0.60,
            rsrp_dbm=-125,
            rsrq_db=-17,
            sinr_db=2,
        ),
        severity="critical",
    )
