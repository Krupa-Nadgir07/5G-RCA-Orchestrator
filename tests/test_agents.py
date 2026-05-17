"""Tests for agent orchestration."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from models.schemas import KPIMetrics, ProcessedEvent, RootCauseCategory
from services.orchestrator.agents.signal_agent import SignalAnalysisAgent
from services.orchestrator.agents.validation_agent import ValidationAgent
from services.orchestrator.working_memory import WorkingMemory


class TestSignalAgent:
    """Tests for signal analysis agent."""

    def setup_method(self):
        self.agent = SignalAnalysisAgent()

    @pytest.mark.asyncio
    async def test_detects_interference(self, sample_event):
        """Signal agent should detect interference anomalies."""
        memory = WorkingMemory("test-session")
        await memory.initialize()
        
        result = await self.agent.execute(sample_event, memory)
        
        assert result is not None
        assert result.confidence > 0
        anomalies = result.content.get("anomalies", [])
        # SINR=3.0 is below threshold, should flag anomaly
        assert len(anomalies) > 0
        
        await memory.close()

    @pytest.mark.asyncio
    async def test_no_anomalies_normal_kpis(self):
        """No anomalies when KPIs are normal."""
        from datetime import datetime
        from uuid import uuid4
        
        event = ProcessedEvent(
            event_id=uuid4(),
            cell_id="cell-100",
            gnb_id="gnb-001",
            timestamp=datetime(2026, 1, 15, 10, 0, 0),
            event_type="kpi_report",
            kpis=KPIMetrics(sinr_db=20, rsrp_dbm=-85, prb_utilization_pct=40),
            severity="info",
        )
        
        memory = WorkingMemory("test-session-2")
        await memory.initialize()
        
        result = await self.agent.execute(event, memory)
        anomalies = result.content.get("anomalies", [])
        # All KPIs normal, should have few or no anomalies
        assert result.confidence <= 0.5
        
        await memory.close()

    @pytest.mark.asyncio
    async def test_detects_congestion(self, congestion_event):
        """Should detect congestion pattern."""
        memory = WorkingMemory("test-session-3")
        await memory.initialize()
        
        result = await self.agent.execute(congestion_event, memory)
        
        assert result.confidence > 0.5
        anomalies = result.content.get("anomalies", [])
        assert len(anomalies) > 0
        
        await memory.close()


class TestValidationAgent:
    """Tests for validation agent."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_validates_hypothesis(self, sample_event):
        """Validation agent processes hypotheses from memory."""
        memory = WorkingMemory("test-validation")
        await memory.initialize()
        
        # Pre-populate memory with signal and hypothesis data
        await memory.store("signal_analysis", {
            "anomalies": [{"kpi": "sinr_db", "value": 3.0, "threshold": 5.0}],
            "patterns": ["interference_congestion"],
        })
        await memory.store("hypotheses", {
            "hypotheses": [{
                "root_cause": "interference",
                "specific_cause": "Co-channel interference",
                "confidence": 0.75,
                "supporting_evidence": ["Low SINR"],
            }]
        })
        
        result = await self.agent.execute(sample_event, memory)
        
        assert result is not None
        assert result.confidence > 0
        
        await memory.close()


class TestWorkingMemory:
    """Tests for working memory."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Basic store/retrieve operations."""
        memory = WorkingMemory("test-mem")
        await memory.initialize()
        
        await memory.store("test_key", {"value": 42})
        result = await memory.retrieve("test_key")
        
        assert result is not None
        assert result["value"] == 42
        
        await memory.close()

    @pytest.mark.asyncio
    async def test_retrieve_missing_key(self):
        """Returns None for missing keys."""
        memory = WorkingMemory("test-mem-2")
        await memory.initialize()
        
        result = await memory.retrieve("nonexistent")
        assert result is None
        
        await memory.close()

    @pytest.mark.asyncio
    async def test_clear(self):
        """Clear removes all data."""
        memory = WorkingMemory("test-mem-3")
        await memory.initialize()
        
        await memory.store("key1", {"a": 1})
        await memory.clear()
        
        result = await memory.retrieve("key1")
        assert result is None
        
        await memory.close()
