"""
Evaluation Datasets - Benchmark dataset loaders for RCA evaluation.
Supports ITU Challenge, Loghub, and synthetic datasets.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from models.schemas import KPIMetrics, ProcessedEvent, RootCauseCategory

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation sample with ground truth."""
    event: ProcessedEvent
    ground_truth_cause: RootCauseCategory
    ground_truth_specific: str
    reference_reasoning: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


class BaseDataset:
    """Base class for evaluation datasets."""
    
    name: str = "base"
    
    def load(self) -> list[EvaluationSample]:
        """Load the dataset."""
        raise NotImplementedError
    
    def sample(self, n: int) -> list[EvaluationSample]:
        """Sample n items from the dataset."""
        data = self.load()
        if len(data) <= n:
            return data
        return random.sample(data, n)


class SyntheticDataset(BaseDataset):
    """
    Synthetic dataset generated from known 5G failure patterns.
    Useful for controlled evaluation when real data is limited.
    """
    
    name = "synthetic_5g"

    # Predefined failure scenarios with KPI signatures
    SCENARIOS = [
        {
            "root_cause": RootCauseCategory.INTERFERENCE,
            "specific_cause": "Co-channel interference from neighboring cell",
            "kpis": {"sinr_db": 2.5, "interference_level_dbm": -88, "bler_pct": 8.5, "prb_utilization_pct": 75},
            "reasoning": [
                "SINR is 2.5 dB, well below the 5 dB warning threshold",
                "Interference level at -88 dBm indicates strong co-channel interference",
                "Elevated BLER (8.5%) is consistent with interference-degraded signal",
                "PRB utilization is moderate, ruling out congestion as primary cause",
            ],
            "difficulty": "easy",
        },
        {
            "root_cause": RootCauseCategory.INTERFERENCE,
            "specific_cause": "Adjacent channel interference due to poor filtering",
            "kpis": {"sinr_db": 4.0, "interference_level_dbm": -92, "bler_pct": 6.0, "throughput_dl_mbps": 25},
            "reasoning": [
                "SINR at 4 dB is borderline, suggesting moderate interference",
                "Interference level at -92 dBm is above normal baseline",
                "BLER at 6% impacts throughput but isn't critical",
                "DL throughput degraded to 25 Mbps from expected ~50 Mbps",
            ],
            "difficulty": "medium",
        },
        {
            "root_cause": RootCauseCategory.HANDOVER_FAILURE,
            "specific_cause": "Handover failure due to insufficient target cell signal",
            "kpis": {"handover_success_rate": 0.65, "rsrp_dbm": -125, "rsrq_db": -17, "sinr_db": 3},
            "reasoning": [
                "Handover success rate at 65% is critically low",
                "RSRP at -125 dBm indicates UE is at cell edge",
                "RSRQ at -17 dB confirms poor signal quality",
                "Low SINR at cell boundary causes A3 event threshold issues",
            ],
            "difficulty": "easy",
        },
        {
            "root_cause": RootCauseCategory.HANDOVER_FAILURE,
            "specific_cause": "Ping-pong handover between two cells",
            "kpis": {"handover_success_rate": 0.78, "rsrp_dbm": -108, "latency_ms": 65},
            "reasoning": [
                "Handover success rate below 85% threshold",
                "Moderate RSRP suggests UE in overlap zone between cells",
                "Increased latency due to frequent handover interruptions",
                "Pattern consistent with ping-pong behavior requiring hysteresis adjustment",
            ],
            "difficulty": "hard",
        },
        {
            "root_cause": RootCauseCategory.RESOURCE_CONGESTION,
            "specific_cause": "Cell overload due to excessive connected UEs",
            "kpis": {"prb_utilization_pct": 97, "connected_ues": 450, "latency_ms": 85, "throughput_dl_mbps": 8},
            "reasoning": [
                "PRB utilization at 97% indicates near-complete resource exhaustion",
                "450 connected UEs far exceeds typical capacity planning",
                "High latency (85ms) due to scheduling delays",
                "Low per-cell throughput confirms capacity limitation",
            ],
            "difficulty": "easy",
        },
        {
            "root_cause": RootCauseCategory.RESOURCE_CONGESTION,
            "specific_cause": "Traffic burst causing temporary congestion",
            "kpis": {"prb_utilization_pct": 92, "connected_ues": 280, "latency_ms": 55, "cqi_avg": 9},
            "reasoning": [
                "PRB utilization spike to 92% during busy period",
                "UE count is moderate but generating high data volume",
                "CQI remains healthy at 9, ruling out signal issues",
                "Latency increase is moderate, suggesting temporary congestion burst",
            ],
            "difficulty": "medium",
        },
        {
            "root_cause": RootCauseCategory.HARDWARE_FAULT,
            "specific_cause": "RRU degradation causing increased BLER",
            "kpis": {"bler_pct": 15, "sinr_db": 12, "prb_utilization_pct": 45, "throughput_dl_mbps": 15},
            "reasoning": [
                "BLER at 15% is extremely high despite good SINR (12 dB)",
                "Good SINR with high BLER strongly suggests hardware issue",
                "Low PRB utilization rules out congestion",
                "Throughput degradation inconsistent with radio conditions indicates RRU problem",
            ],
            "difficulty": "hard",
        },
        {
            "root_cause": RootCauseCategory.SOFTWARE_FAULT,
            "specific_cause": "Scheduler bug causing unfair resource allocation",
            "kpis": {"prb_utilization_pct": 55, "connected_ues": 100, "throughput_dl_mbps": 5, "latency_ms": 120},
            "reasoning": [
                "PRB utilization is moderate (55%) but throughput is very low",
                "Few UEs (100) should not cause congestion at this utilization level",
                "Extremely high latency (120ms) with available resources indicates scheduling issue",
                "Mismatch between available capacity and delivered performance points to software bug",
            ],
            "difficulty": "hard",
        },
        {
            "root_cause": RootCauseCategory.CONFIGURATION_ERROR,
            "specific_cause": "Incorrect antenna tilt causing coverage gap",
            "kpis": {"rsrp_dbm": -130, "sinr_db": -2, "handover_success_rate": 0.55, "connected_ues": 15},
            "reasoning": [
                "Very weak RSRP (-130 dBm) suggests poor coverage in area",
                "Negative SINR (-2 dB) indicates signal below noise floor",
                "Very low handover success rate due to weak serving cell signal",
                "Unusually few UEs suggests coverage gap driving users to other cells",
            ],
            "difficulty": "medium",
        },
        {
            "root_cause": RootCauseCategory.CONFIGURATION_ERROR,
            "specific_cause": "Missing neighbor relation causing handover failures to adjacent cell",
            "kpis": {"handover_success_rate": 0.60, "rsrp_dbm": -115, "rsrq_db": -14, "connected_ues": 200},
            "reasoning": [
                "Low handover success rate despite moderate signal strength",
                "RSRP at -115 dBm is in handover zone but not critically weak",
                "Normal UE count indicates coverage is present but mobility is impaired",
                "Consistent failures to specific target suggest missing NRT entry",
            ],
            "difficulty": "hard",
        },
    ]

    def load(self) -> list[EvaluationSample]:
        """Generate synthetic evaluation samples."""
        from datetime import datetime
        from uuid import uuid4
        
        samples = []
        
        for i, scenario in enumerate(self.SCENARIOS):
            kpi_data = scenario["kpis"]
            
            event = ProcessedEvent(
                event_id=uuid4(),
                cell_id=f"cell-{100 + i}",
                gnb_id=f"gnb-{i // 3 + 1:03d}",
                timestamp=datetime(2026, 1, 15, 10 + i, 30, 0),
                event_type=self._cause_to_event_type(scenario["root_cause"]),
                kpis=KPIMetrics(**kpi_data),
                severity="critical" if scenario["difficulty"] == "easy" else "warning",
            )
            
            samples.append(EvaluationSample(
                event=event,
                ground_truth_cause=scenario["root_cause"],
                ground_truth_specific=scenario["specific_cause"],
                reference_reasoning=scenario["reasoning"],
                difficulty=scenario["difficulty"],
            ))
        
        # Generate variations (augment with noise)
        augmented = []
        for sample in samples:
            for j in range(4):  # 4 variations per scenario
                augmented.append(self._augment_sample(sample, j))
        
        return samples + augmented

    def _cause_to_event_type(self, cause: RootCauseCategory) -> str:
        """Map root cause to event type."""
        mapping = {
            RootCauseCategory.INTERFERENCE: "interference_detected",
            RootCauseCategory.HANDOVER_FAILURE: "handover_event",
            RootCauseCategory.RESOURCE_CONGESTION: "resource_congestion",
            RootCauseCategory.HARDWARE_FAULT: "hardware_fault",
            RootCauseCategory.SOFTWARE_FAULT: "software_fault",
            RootCauseCategory.CONFIGURATION_ERROR: "configuration_error",
        }
        return mapping.get(cause, "unknown")

    def _augment_sample(self, sample: EvaluationSample, variation: int) -> EvaluationSample:
        """Create a noisy variation of a sample."""
        from datetime import datetime, timedelta
        from uuid import uuid4
        
        # Add noise to KPIs
        kpi_dict = sample.event.kpis.model_dump(exclude_none=True)
        noisy_kpis = {}
        for key, value in kpi_dict.items():
            if isinstance(value, (int, float)):
                noise = random.gauss(0, abs(value) * 0.1)  # 10% gaussian noise
                noisy_kpis[key] = round(value + noise, 2)
            else:
                noisy_kpis[key] = value
        
        event = ProcessedEvent(
            event_id=uuid4(),
            cell_id=f"cell-{200 + variation}",
            gnb_id=sample.event.gnb_id,
            timestamp=sample.event.timestamp + timedelta(minutes=variation * 5),
            event_type=sample.event.event_type,
            kpis=KPIMetrics(**noisy_kpis),
            severity=sample.event.severity,
        )
        
        return EvaluationSample(
            event=event,
            ground_truth_cause=sample.ground_truth_cause,
            ground_truth_specific=sample.ground_truth_specific,
            reference_reasoning=sample.reference_reasoning,
            difficulty=sample.difficulty,
        )


class ITUChallengeDataset(BaseDataset):
    """
    ITU AI/ML Challenge dataset loader.
    Expects data to be pre-downloaded to a local directory.
    """
    
    name = "itu_challenge"

    def __init__(self, data_dir: str = "data/itu_challenge"):
        self.data_dir = Path(data_dir)

    def load(self) -> list[EvaluationSample]:
        """Load ITU challenge dataset from files."""
        if not self.data_dir.exists():
            logger.warning("ITU dataset not found, using synthetic fallback",
                         path=str(self.data_dir))
            return SyntheticDataset().load()

        samples = []
        # Load from JSON files in the data directory
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                for item in data:
                    sample = self._parse_item(item)
                    if sample:
                        samples.append(sample)
            except Exception as e:
                logger.error("Failed to load dataset file", 
                           file=str(file_path), error=str(e))

        if not samples:
            logger.warning("No samples loaded from ITU dataset, using synthetic")
            return SyntheticDataset().load()

        return samples

    def _parse_item(self, item: dict) -> Optional[EvaluationSample]:
        """Parse a single dataset item into EvaluationSample."""
        try:
            from datetime import datetime
            from uuid import uuid4
            
            kpis = KPIMetrics(**{k: v for k, v in item.get("kpis", {}).items() 
                                if hasattr(KPIMetrics, k)})
            
            event = ProcessedEvent(
                event_id=uuid4(),
                cell_id=item.get("cell_id", "cell-unknown"),
                gnb_id=item.get("gnb_id", "gnb-unknown"),
                timestamp=datetime.fromisoformat(item.get("timestamp", "2026-01-01T00:00:00")),
                event_type=item.get("event_type"),
                kpis=kpis,
            )
            
            cause = item.get("root_cause", "unknown")
            try:
                root_cause = RootCauseCategory(cause)
            except ValueError:
                root_cause = RootCauseCategory.UNKNOWN
            
            return EvaluationSample(
                event=event,
                ground_truth_cause=root_cause,
                ground_truth_specific=item.get("specific_cause", ""),
                reference_reasoning=item.get("reasoning", []),
            )
        except Exception:
            return None


class LoghubDataset(BaseDataset):
    """
    Loghub telecom dataset loader.
    """
    
    name = "loghub_telecom"

    def __init__(self, data_dir: str = "data/loghub"):
        self.data_dir = Path(data_dir)

    def load(self) -> list[EvaluationSample]:
        """Load Loghub dataset."""
        if not self.data_dir.exists():
            logger.warning("Loghub dataset not found, using synthetic fallback")
            return SyntheticDataset().load()

        # Similar structure to ITU dataset
        samples = []
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                for item in data:
                    sample = self._parse_item(item)
                    if sample:
                        samples.append(sample)
            except Exception as e:
                logger.error("Failed to load Loghub file", error=str(e))

        return samples or SyntheticDataset().load()

    def _parse_item(self, item: dict) -> Optional[EvaluationSample]:
        """Parse Loghub item."""
        try:
            from datetime import datetime
            from uuid import uuid4
            
            kpis = KPIMetrics(**{k: v for k, v in item.get("kpis", {}).items()
                                if hasattr(KPIMetrics, k)})
            
            event = ProcessedEvent(
                event_id=uuid4(),
                cell_id=item.get("cell_id", "cell-unknown"),
                gnb_id=item.get("gnb_id", "gnb-unknown"),
                timestamp=datetime.fromisoformat(item.get("timestamp", "2026-01-01T00:00:00")),
                event_type=item.get("event_type"),
                kpis=kpis,
            )
            
            cause = item.get("root_cause", "unknown")
            try:
                root_cause = RootCauseCategory(cause)
            except ValueError:
                root_cause = RootCauseCategory.UNKNOWN
            
            return EvaluationSample(
                event=event,
                ground_truth_cause=root_cause,
                ground_truth_specific=item.get("specific_cause", ""),
            )
        except Exception:
            return None
