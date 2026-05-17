"""
Log Parser - Hybrid parsing using Drain3 + regex patterns.
Parses semi-structured 5G gNB logs into structured format.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ParsedLog:
    """Result of log parsing."""
    template: str
    parameters: dict[str, str] = field(default_factory=dict)
    event_type: Optional[str] = None
    parse_method: str = "unknown"


class HybridLogParser:
    """
    Multi-strategy log parser:
    1. Regex patterns for known 5G gNB log formats (fastest, ~0.1ms)
    2. Drain3 template mining for unknown patterns (~1ms)
    3. Fallback to raw message if all fail
    """

    # Known 5G gNB log patterns
    PATTERNS = {
        "rrc_failure": re.compile(
            r"RRC connection (?P<action>failure|release|setup) for UE (?P<ue_id>0x[A-Fa-f0-9]+),?\s*"
            r"cause:\s*(?P<cause>\w+),?\s*target_cell:\s*(?P<target_cell>\d+)"
        ),
        "handover": re.compile(
            r"Handover (?P<status>initiated|completed|failed) "
            r"from cell (?P<source_cell>\w+) to cell (?P<target_cell>\w+)"
            r"(?:\s*UE:\s*(?P<ue_id>\w+))?"
        ),
        "throughput_degradation": re.compile(
            r"Throughput degradation detected.*?"
            r"cell[_\s]*(?:id)?:?\s*(?P<cell_id>\w+).*?"
            r"current:\s*(?P<current>[\d.]+)\s*(?P<unit>\w+)"
        ),
        "interference": re.compile(
            r"(?:High\s+)?[Ii]nterference (?:detected|level|alarm).*?"
            r"cell[_\s]*(?:id)?:?\s*(?P<cell_id>\w+).*?"
            r"level:\s*(?P<level>[-\d.]+)\s*dBm"
        ),
        "prb_congestion": re.compile(
            r"PRB (?:utilization|usage) (?:high|critical|alarm).*?"
            r"cell[_\s]*(?:id)?:?\s*(?P<cell_id>\w+).*?"
            r"(?:utilization|usage):\s*(?P<utilization>[\d.]+)%"
        ),
        "rlf": re.compile(
            r"Radio Link Failure (?:detected|declared).*?"
            r"UE:?\s*(?P<ue_id>\w+).*?"
            r"cell:?\s*(?P<cell_id>\w+).*?"
            r"(?:cause:?\s*(?P<cause>\w+))?"
        ),
        "beam_failure": re.compile(
            r"Beam failure (?:detected|recovery).*?"
            r"cell:?\s*(?P<cell_id>\w+).*?"
            r"beam[_\s]*(?:id)?:?\s*(?P<beam_id>\d+)"
        ),
        "sinr_alarm": re.compile(
            r"SINR (?:below threshold|alarm|low).*?"
            r"cell:?\s*(?P<cell_id>\w+).*?"
            r"SINR:?\s*(?P<sinr>[-\d.]+)\s*dB"
        ),
        "kpi_report": re.compile(
            r"KPI Report.*?cell:?\s*(?P<cell_id>\w+).*?"
            r"SINR:?\s*(?P<sinr>[-\d.]+).*?"
            r"PRB:?\s*(?P<prb>[\d.]+).*?"
            r"(?:BLER:?\s*(?P<bler>[\d.]+))?"
        ),
        "hardware_alarm": re.compile(
            r"Hardware (?:alarm|fault|error).*?"
            r"(?:component|module):?\s*(?P<component>\w+).*?"
            r"(?:severity|level):?\s*(?P<severity>\w+)"
        ),
    }

    # Event type mapping
    EVENT_TYPE_MAP = {
        "rrc_failure": "rrc_failure",
        "handover": "handover_event",
        "throughput_degradation": "throughput_degradation",
        "interference": "interference_detected",
        "prb_congestion": "resource_congestion",
        "rlf": "radio_link_failure",
        "beam_failure": "beam_failure",
        "sinr_alarm": "signal_quality_alarm",
        "kpi_report": "kpi_report",
        "hardware_alarm": "hardware_fault",
    }

    def __init__(self):
        self._drain_parser = None
        self._parse_count = 0
        self._method_stats = {"regex": 0, "drain": 0, "fallback": 0}

    def parse(self, raw_message: str) -> ParsedLog:
        """
        Parse a raw log message using multiple strategies.
        Returns structured ParsedLog with template and parameters.
        """
        self._parse_count += 1

        # Strategy 1: Regex pattern matching (fastest)
        result = self._try_regex(raw_message)
        if result:
            self._method_stats["regex"] += 1
            return result

        # Strategy 2: Drain3 template mining
        result = self._try_drain(raw_message)
        if result:
            self._method_stats["drain"] += 1
            return result

        # Strategy 3: Fallback - use raw message as template
        self._method_stats["fallback"] += 1
        return ParsedLog(
            template=raw_message[:256],
            parameters={},
            event_type=self._infer_event_type(raw_message),
            parse_method="fallback"
        )

    def _try_regex(self, message: str) -> Optional[ParsedLog]:
        """Try matching against known regex patterns."""
        for pattern_name, pattern in self.PATTERNS.items():
            match = pattern.search(message)
            if match:
                return ParsedLog(
                    template=pattern_name,
                    parameters=match.groupdict(),
                    event_type=self.EVENT_TYPE_MAP.get(pattern_name),
                    parse_method="regex"
                )
        return None

    def _try_drain(self, message: str) -> Optional[ParsedLog]:
        """Try Drain3 template mining."""
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig

            if self._drain_parser is None:
                config = TemplateMinerConfig()
                config.drain_sim_th = 0.4
                config.drain_depth = 4
                config.drain_max_children = 100
                self._drain_parser = TemplateMiner(config=config)

            result = self._drain_parser.add_log_message(message)
            if result and result["cluster_id"]:
                template = result["template_mined"]
                # Extract parameters (tokens that differ from template)
                params = self._extract_drain_params(message, template)
                return ParsedLog(
                    template=template,
                    parameters=params,
                    event_type=self._infer_event_type(message),
                    parse_method="drain"
                )
        except ImportError:
            logger.debug("Drain3 not available, skipping")
        except Exception as e:
            logger.warning("Drain3 parsing failed", error=str(e))
        return None

    def _extract_drain_params(self, message: str, template: str) -> dict[str, str]:
        """Extract parameter values from message using template."""
        params = {}
        template_tokens = template.split()
        message_tokens = message.split()

        param_idx = 0
        for i, (t_token, m_token) in enumerate(zip(template_tokens, message_tokens)):
            if t_token == "<*>":
                params[f"param_{param_idx}"] = m_token
                param_idx += 1

        return params

    def _infer_event_type(self, message: str) -> Optional[str]:
        """Infer event type from keywords in the message."""
        message_lower = message.lower()
        
        keyword_map = {
            "handover": "handover_event",
            "interference": "interference_detected",
            "throughput": "throughput_degradation",
            "prb": "resource_congestion",
            "congestion": "resource_congestion",
            "radio link failure": "radio_link_failure",
            "rlf": "radio_link_failure",
            "rrc": "rrc_failure",
            "beam failure": "beam_failure",
            "hardware": "hardware_fault",
            "sinr": "signal_quality_alarm",
            "configuration": "configuration_error",
        }

        for keyword, event_type in keyword_map.items():
            if keyword in message_lower:
                return event_type

        return None

    def get_stats(self) -> dict:
        """Return parsing statistics."""
        return {
            "total_parsed": self._parse_count,
            "method_stats": self._method_stats.copy()
        }
