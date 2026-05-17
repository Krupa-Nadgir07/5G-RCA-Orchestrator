"""Tests for preprocessing services."""

import pytest
from datetime import datetime

from services.preprocessing.log_parser import HybridLogParser
from services.preprocessing.kpi_extractor import KPIExtractor
from models.schemas import KPIMetrics


class TestHybridLogParser:
    """Tests for the hybrid log parser."""

    def setup_method(self):
        self.parser = HybridLogParser()

    def test_parse_interference_log(self):
        log = "[ALARM] cell-101 SINR degradation: SINR=2.5dB, interference_level=-88dBm"
        result = self.parser.parse(log)
        assert result is not None
        assert "sinr" in str(result).lower() or result.get("event_type") is not None

    def test_parse_handover_log(self):
        log = "2026-01-15T10:30:00 [INFO] Handover attempt cell-101 -> cell-102, RSRP=-115dBm, cause=A3_event"
        result = self.parser.parse(log)
        assert result is not None

    def test_parse_rrc_failure(self):
        log = "2026-01-15T10:30:00 [ERROR] RRC connection failure: cell=cell-101, ue_id=UE-5001, cause=radio_link_failure"
        result = self.parser.parse(log)
        assert result is not None

    def test_parse_unknown_format(self):
        log = "some random text that doesn't match any pattern"
        result = self.parser.parse(log)
        # Should still return something via Drain3 fallback
        assert result is not None

    def test_parse_empty_string(self):
        result = self.parser.parse("")
        assert result is not None


class TestKPIExtractor:
    """Tests for KPI extraction."""

    def setup_method(self):
        self.extractor = KPIExtractor()

    def test_extract_sinr(self):
        log = "SINR=3.5dB measured on cell-101"
        kpis = self.extractor.extract(log)
        assert kpis.sinr_db == pytest.approx(3.5, rel=0.1)

    def test_extract_rsrp(self):
        log = "RSRP=-115dBm reported by UE"
        kpis = self.extractor.extract(log)
        assert kpis.rsrp_dbm == pytest.approx(-115, rel=0.1)

    def test_extract_multiple_kpis(self):
        log = "KPI report: SINR=12dB, RSRP=-95dBm, BLER=2.5%, PRB_util=65%"
        kpis = self.extractor.extract(log)
        # Count non-None fields
        extracted = {k: v for k, v in kpis.model_dump().items() if v is not None}
        assert len(extracted) >= 2

    def test_extract_no_kpis(self):
        log = "System started successfully"
        kpis = self.extractor.extract(log)
        assert isinstance(kpis, KPIMetrics)
