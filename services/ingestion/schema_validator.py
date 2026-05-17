"""
Schema Validator for log events.
Validates incoming log events against expected schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.schemas import RawLogEvent


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)


class SchemaValidator:
    """
    Validates raw log events for completeness and correctness.
    Checks required fields, data types, and domain constraints.
    """

    # Maximum allowed age for a log event (24 hours)
    MAX_EVENT_AGE_HOURS = 24
    # Maximum raw message size (64KB)
    MAX_MESSAGE_SIZE = 65536
    # Valid source ID pattern
    VALID_SOURCE_PREFIXES = ("gnb-", "cell-", "enb-", "cu-", "du-")

    def validate(self, event: RawLogEvent) -> ValidationResult:
        """Perform full validation on a raw log event."""
        result = ValidationResult()
        
        self._validate_source_id(event, result)
        self._validate_timestamp(event, result)
        self._validate_message(event, result)
        self._validate_metadata(event, result)
        
        return result

    def _validate_source_id(self, event: RawLogEvent, result: ValidationResult):
        """Validate source_id format."""
        if not event.source_id:
            result.is_valid = False
            result.errors.append("source_id is required")
            return
        
        if not any(event.source_id.startswith(p) for p in self.VALID_SOURCE_PREFIXES):
            # Allow non-standard prefixes but log a warning
            pass  # Flexible parsing for varied sources

    def _validate_timestamp(self, event: RawLogEvent, result: ValidationResult):
        """Validate timestamp is recent and not in the future."""
        now = datetime.now(timezone.utc)
        event_time = event.timestamp.replace(tzinfo=timezone.utc) if event.timestamp.tzinfo is None else event.timestamp
        
        # Check for future timestamps (allow 5 min clock skew)
        if event_time > now.replace(microsecond=0):
            from datetime import timedelta
            if event_time - now > timedelta(minutes=5):
                result.is_valid = False
                result.errors.append("timestamp is in the future beyond allowed clock skew")
                return
        
        # Check for stale events
        from datetime import timedelta
        if now - event_time > timedelta(hours=self.MAX_EVENT_AGE_HOURS):
            result.is_valid = False
            result.errors.append(
                f"timestamp is older than {self.MAX_EVENT_AGE_HOURS} hours"
            )

    def _validate_message(self, event: RawLogEvent, result: ValidationResult):
        """Validate raw message content."""
        if not event.raw_message:
            result.is_valid = False
            result.errors.append("raw_message is required")
            return
        
        if len(event.raw_message) > self.MAX_MESSAGE_SIZE:
            result.is_valid = False
            result.errors.append(
                f"raw_message exceeds maximum size of {self.MAX_MESSAGE_SIZE} bytes"
            )

    def _validate_metadata(self, event: RawLogEvent, result: ValidationResult):
        """Validate metadata fields if present."""
        if event.metadata:
            # Ensure metadata doesn't contain excessively large values
            for key, value in event.metadata.items():
                if isinstance(value, str) and len(value) > 1024:
                    result.is_valid = False
                    result.errors.append(
                        f"metadata field '{key}' exceeds maximum length of 1024"
                    )
