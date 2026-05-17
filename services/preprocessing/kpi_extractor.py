"""
KPI Extractor - Extracts Key Performance Indicators from parsed logs.
Handles both explicit KPI reports and KPI inference from event context.
"""

import re
from typing import Optional

import structlog

from models.schemas import KPIMetrics

logger = structlog.get_logger(__name__)


class KPIExtractor:
    """
    Extracts KPI metrics from parsed log events.
    Supports explicit extraction from KPI report messages
    and inference from various log event types.
    """

    # Regex patterns for KPI value extraction
    KPI_PATTERNS = {
        "sinr_db": [
            re.compile(r"SINR[:\s=]*(-?[\d.]+)\s*(?:dB)?", re.IGNORECASE),
            re.compile(r"sinr[_\s]*(?:value)?[:\s=]*(-?[\d.]+)", re.IGNORECASE),
        ],
        "rsrp_dbm": [
            re.compile(r"RSRP[:\s=]*(-?[\d.]+)\s*(?:dBm)?", re.IGNORECASE),
            re.compile(r"rsrp[_\s]*(?:value)?[:\s=]*(-?[\d.]+)", re.IGNORECASE),
        ],
        "rsrq_db": [
            re.compile(r"RSRQ[:\s=]*(-?[\d.]+)\s*(?:dB)?", re.IGNORECASE),
            re.compile(r"rsrq[_\s]*(?:value)?[:\s=]*(-?[\d.]+)", re.IGNORECASE),
        ],
        "prb_utilization_pct": [
            re.compile(r"PRB[_\s]*(?:utilization|usage|util)[:\s=]*([\d.]+)\s*%?", re.IGNORECASE),
            re.compile(r"(?:utilization|usage)[:\s=]*([\d.]+)\s*%", re.IGNORECASE),
        ],
        "bler_pct": [
            re.compile(r"BLER[:\s=]*([\d.]+)\s*%?", re.IGNORECASE),
            re.compile(r"block[_\s]*error[_\s]*rate[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "handover_success_rate": [
            re.compile(r"(?:handover|HO)[_\s]*success[_\s]*rate[:\s=]*([\d.]+)", re.IGNORECASE),
            re.compile(r"HO[_\s]*SR[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "rrc_connection_setup_success_rate": [
            re.compile(r"RRC[_\s]*(?:connection)?[_\s]*setup[_\s]*success[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "throughput_dl_mbps": [
            re.compile(r"(?:DL|downlink)[_\s]*(?:throughput|tput)[:\s=]*([\d.]+)\s*(?:Mbps)?", re.IGNORECASE),
            re.compile(r"throughput[_\s]*DL[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "throughput_ul_mbps": [
            re.compile(r"(?:UL|uplink)[_\s]*(?:throughput|tput)[:\s=]*([\d.]+)\s*(?:Mbps)?", re.IGNORECASE),
            re.compile(r"throughput[_\s]*UL[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "latency_ms": [
            re.compile(r"latency[:\s=]*([\d.]+)\s*(?:ms)?", re.IGNORECASE),
            re.compile(r"delay[:\s=]*([\d.]+)\s*(?:ms)?", re.IGNORECASE),
        ],
        "connected_ues": [
            re.compile(r"(?:connected)?[_\s]*UEs?[:\s=]*(\d+)", re.IGNORECASE),
            re.compile(r"num[_\s]*(?:connected)?[_\s]*users?[:\s=]*(\d+)", re.IGNORECASE),
        ],
        "cqi_avg": [
            re.compile(r"CQI[_\s]*(?:avg|average)?[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "mcs_dl_avg": [
            re.compile(r"MCS[_\s]*(?:DL)?[_\s]*(?:avg)?[:\s=]*([\d.]+)", re.IGNORECASE),
        ],
        "ta_advance_us": [
            re.compile(r"(?:TA|timing[_\s]*advance)[:\s=]*([\d.]+)\s*(?:us|μs)?", re.IGNORECASE),
        ],
        "interference_level_dbm": [
            re.compile(r"interference[_\s]*(?:level)?[:\s=]*(-?[\d.]+)\s*(?:dBm)?", re.IGNORECASE),
            re.compile(r"RSSI[:\s=]*(-?[\d.]+)\s*(?:dBm)?", re.IGNORECASE),
        ],
    }

    def extract(self, raw_message: str, parsed_params: dict[str, str] = None) -> KPIMetrics:
        """
        Extract KPI metrics from a raw message and parsed parameters.
        
        Args:
            raw_message: The original log message text
            parsed_params: Parameters extracted during parsing
            
        Returns:
            KPIMetrics with all extractable values populated
        """
        kpis = {}

        # Extract from raw message using patterns
        for kpi_name, patterns in self.KPI_PATTERNS.items():
            value = self._extract_value(raw_message, patterns)
            if value is not None:
                kpis[kpi_name] = value

        # Extract from parsed parameters
        if parsed_params:
            kpis.update(self._extract_from_params(parsed_params))

        return KPIMetrics(**kpis)

    def _extract_value(self, message: str, patterns: list[re.Pattern]) -> Optional[float]:
        """Try each pattern until a value is found."""
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                try:
                    value = float(match.group(1))
                    return value
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_from_params(self, params: dict[str, str]) -> dict[str, float]:
        """Extract KPI values from parsed parameters."""
        kpis = {}
        
        param_to_kpi = {
            "sinr": "sinr_db",
            "rsrp": "rsrp_dbm",
            "rsrq": "rsrq_db",
            "prb": "prb_utilization_pct",
            "utilization": "prb_utilization_pct",
            "bler": "bler_pct",
            "level": "interference_level_dbm",
            "current": "throughput_dl_mbps",
        }

        for param_name, value in params.items():
            kpi_name = param_to_kpi.get(param_name.lower())
            if kpi_name and value:
                try:
                    kpis[kpi_name] = float(value)
                except ValueError:
                    continue

        return kpis

    def compute_derived_kpis(self, kpis: KPIMetrics) -> dict[str, float]:
        """
        Compute derived KPIs from base metrics.
        E.g., spectral efficiency, load factor, etc.
        """
        derived = {}

        if kpis.throughput_dl_mbps and kpis.prb_utilization_pct:
            # Approximate spectral efficiency
            if kpis.prb_utilization_pct > 0:
                derived["spectral_efficiency"] = (
                    kpis.throughput_dl_mbps / (kpis.prb_utilization_pct / 100)
                )

        if kpis.connected_ues and kpis.throughput_dl_mbps:
            # Per-user throughput
            if kpis.connected_ues > 0:
                derived["per_user_throughput_mbps"] = (
                    kpis.throughput_dl_mbps / kpis.connected_ues
                )

        if kpis.handover_success_rate is not None:
            # Handover failure rate (inverse)
            derived["handover_failure_rate"] = 1.0 - kpis.handover_success_rate

        return derived
