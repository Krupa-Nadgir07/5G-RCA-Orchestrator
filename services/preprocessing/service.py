"""
Preprocessing Service - Orchestrates log parsing, KPI extraction, and feature engineering.
"""

import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import structlog

from config.settings import get_settings
from models.schemas import (
    CellContext,
    KPIMetrics,
    ProcessedEvent,
    RawLogEvent,
    Severity,
)
from services.preprocessing.kpi_extractor import KPIExtractor
from services.preprocessing.log_parser import HybridLogParser

logger = structlog.get_logger(__name__)
settings = get_settings()


class PreprocessingService:
    """
    Multi-stage preprocessing pipeline:
    Parse → Normalize → Extract KPIs → Detect Anomalies → Enrich → Emit
    """

    # Severity thresholds for anomaly detection
    SEVERITY_THRESHOLDS = {
        "sinr_db": {"critical": 0, "warning": 5},
        "prb_utilization_pct": {"critical": 95, "warning": 85},
        "bler_pct": {"critical": 10, "warning": 5},
        "handover_success_rate": {"critical": 0.7, "warning": 0.85},
        "interference_level_dbm": {"critical": -85, "warning": -95},
        "latency_ms": {"critical": 100, "warning": 50},
    }

    def __init__(self):
        self.parser = HybridLogParser()
        self.kpi_extractor = KPIExtractor()
        self._metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "avg_processing_time_ms": 0.0,
        }

    async def process(self, raw_event: RawLogEvent) -> Optional[ProcessedEvent]:
        """
        Process a single raw log event through the full pipeline.
        
        Returns ProcessedEvent or None if processing fails.
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse the raw message
            parsed = self.parser.parse(raw_event.raw_message)

            # Step 2: Extract KPIs
            kpis = self.kpi_extractor.extract(
                raw_event.raw_message, 
                parsed.parameters
            )

            # Step 3: Determine cell and gNB identifiers
            cell_id = self._extract_cell_id(raw_event, parsed.parameters)
            gnb_id = self._extract_gnb_id(raw_event, parsed.parameters)

            # Step 4: Detect anomalies and compute severity
            anomaly_scores = self._compute_anomaly_scores(kpis)
            severity = self._determine_severity(kpis, anomaly_scores)

            # Step 5: Build context
            context = self._build_context(raw_event, parsed.parameters)

            # Step 6: Construct processed event
            processed = ProcessedEvent(
                event_id=raw_event.event_id,
                cell_id=cell_id,
                gnb_id=gnb_id,
                timestamp=raw_event.timestamp,
                event_type=parsed.event_type,
                kpis=kpis,
                parsed_template=parsed.template,
                severity=severity,
                context=context,
                raw_message=raw_event.raw_message[:512],  # Truncate for storage
                anomaly_scores=anomaly_scores,
            )

            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time)
            
            logger.debug("Event processed",
                        event_id=str(processed.event_id),
                        event_type=processed.event_type,
                        severity=processed.severity.value,
                        processing_time_ms=round(processing_time, 2))

            return processed

        except Exception as e:
            self._metrics["events_failed"] += 1
            logger.error("Processing failed",
                        event_id=str(raw_event.event_id),
                        error=str(e))
            return None

    async def process_batch(self, events: list[RawLogEvent]) -> list[ProcessedEvent]:
        """Process a batch of events."""
        results = []
        for event in events:
            processed = await self.process(event)
            if processed:
                results.append(processed)
        return results

    def _extract_cell_id(self, event: RawLogEvent, params: dict) -> str:
        """Extract cell ID from event metadata or parsed parameters."""
        # Check metadata first
        if event.metadata.get("cell_id"):
            return event.metadata["cell_id"]
        # Check parsed params
        if params.get("cell_id"):
            return params["cell_id"]
        if params.get("target_cell"):
            return f"cell-{params['target_cell']}"
        # Fallback
        return event.source_id

    def _extract_gnb_id(self, event: RawLogEvent, params: dict) -> str:
        """Extract gNB ID from event metadata or source."""
        if event.metadata.get("gnb_id"):
            return event.metadata["gnb_id"]
        # Derive from source_id
        if event.source_id.startswith("gnb-"):
            return event.source_id
        return f"gnb-{event.source_id}"

    def _compute_anomaly_scores(self, kpis: KPIMetrics) -> dict[str, float]:
        """
        Compute anomaly scores for each KPI based on threshold distance.
        Score: 0.0 (normal) to 1.0 (critical anomaly)
        """
        scores = {}

        kpi_dict = kpis.model_dump(exclude_none=True)
        
        for kpi_name, thresholds in self.SEVERITY_THRESHOLDS.items():
            if kpi_name not in kpi_dict:
                continue
                
            value = kpi_dict[kpi_name]
            critical = thresholds["critical"]
            warning = thresholds["warning"]

            # Determine if lower or higher is worse
            if kpi_name in ("sinr_db", "handover_success_rate", "rrc_connection_setup_success_rate"):
                # Lower is worse
                if value <= critical:
                    scores[kpi_name] = 1.0
                elif value <= warning:
                    # Linear interpolation between warning and critical
                    scores[kpi_name] = 0.5 + 0.5 * (warning - value) / (warning - critical)
                else:
                    scores[kpi_name] = max(0, 0.5 * (warning * 1.5 - value) / (warning * 0.5))
            elif kpi_name == "interference_level_dbm":
                # Higher (less negative) is worse for interference
                if value >= critical:
                    scores[kpi_name] = 1.0
                elif value >= warning:
                    scores[kpi_name] = 0.5 + 0.5 * (value - warning) / (critical - warning)
                else:
                    scores[kpi_name] = 0.0
            else:
                # Higher is worse (prb_util, bler, latency)
                if value >= critical:
                    scores[kpi_name] = 1.0
                elif value >= warning:
                    scores[kpi_name] = 0.5 + 0.5 * (value - warning) / (critical - warning)
                else:
                    scores[kpi_name] = max(0, 0.5 * (value / warning))

        # Clamp all scores to [0, 1]
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    def _determine_severity(self, kpis: KPIMetrics, anomaly_scores: dict) -> Severity:
        """Determine overall severity from anomaly scores."""
        if not anomaly_scores:
            return Severity.INFO

        max_score = max(anomaly_scores.values()) if anomaly_scores else 0.0
        
        if max_score >= 0.8:
            return Severity.CRITICAL
        elif max_score >= 0.5:
            return Severity.WARNING
        elif max_score >= 0.2:
            return Severity.NORMAL
        else:
            return Severity.INFO

    def _build_context(self, event: RawLogEvent, params: dict) -> CellContext:
        """Build cell context from available information."""
        return CellContext(
            neighboring_cells=event.metadata.get("neighboring_cells", []),
            frequency_band=event.metadata.get("frequency_band"),
            bandwidth_mhz=event.metadata.get("bandwidth_mhz"),
        )

    def _update_metrics(self, processing_time_ms: float):
        """Update running metrics."""
        self._metrics["events_processed"] += 1
        n = self._metrics["events_processed"]
        # Running average
        self._metrics["avg_processing_time_ms"] = (
            self._metrics["avg_processing_time_ms"] * (n - 1) / n 
            + processing_time_ms / n
        )

    def get_metrics(self) -> dict:
        """Return service metrics."""
        return {
            **self._metrics,
            "parser_stats": self.parser.get_stats()
        }
