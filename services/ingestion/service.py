"""
Log Ingestion Service
Accepts log events from streaming (Kafka), REST API, and batch (S3) sources.
Validates schema, applies rate limiting, and routes to preprocessing pipeline.
"""

import asyncio
import time
from typing import AsyncIterator, Optional
from uuid import UUID

import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from config.settings import get_settings
from models.schemas import RawLogEvent, Priority
from services.ingestion.schema_validator import SchemaValidator
from services.ingestion.rate_limiter import TokenBucketRateLimiter

logger = structlog.get_logger(__name__)
settings = get_settings()


class IngestionService:
    """
    Core ingestion service that handles multiple input sources
    and routes validated events to the preprocessing pipeline.
    """

    def __init__(self):
        self.validator = SchemaValidator()
        self.rate_limiter = TokenBucketRateLimiter(
            rate=10000,  # 10K events/sec
            burst=20000
        )
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._metrics = {
            "events_received": 0,
            "events_validated": 0,
            "events_rejected": 0,
            "events_published": 0,
        }

    async def start(self):
        """Initialize Kafka connections."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: v.model_dump_json().encode("utf-8"),
        )
        await self._producer.start()
        self._running = True
        logger.info("Ingestion service started", 
                   kafka_servers=settings.kafka_bootstrap_servers)

    async def stop(self):
        """Gracefully shutdown connections."""
        self._running = False
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()
        logger.info("Ingestion service stopped")

    async def ingest_single(self, event: RawLogEvent, priority: Priority = Priority.NORMAL) -> dict:
        """
        Ingest a single log event via REST API.
        Validates, rate-limits, and publishes to Kafka.
        """
        start_time = time.time()
        self._metrics["events_received"] += 1

        # Rate limiting (bypass for critical priority)
        if priority != Priority.CRITICAL:
            if not self.rate_limiter.allow(event.source_id):
                self._metrics["events_rejected"] += 1
                logger.warning("Rate limit exceeded", source_id=event.source_id)
                return {
                    "status": "rejected",
                    "reason": "rate_limit_exceeded",
                    "event_id": str(event.event_id)
                }

        # Schema validation
        validation_result = self.validator.validate(event)
        if not validation_result.is_valid:
            self._metrics["events_rejected"] += 1
            logger.warning("Validation failed", 
                         event_id=str(event.event_id),
                         errors=validation_result.errors)
            return {
                "status": "rejected",
                "reason": "validation_failed",
                "errors": validation_result.errors,
                "event_id": str(event.event_id)
            }

        self._metrics["events_validated"] += 1

        # Publish to Kafka processed topic
        await self._publish_event(event, priority)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info("Event ingested", 
                   event_id=str(event.event_id),
                   latency_ms=round(latency_ms, 2))

        return {
            "status": "accepted",
            "event_id": str(event.event_id),
            "latency_ms": round(latency_ms, 2)
        }

    async def ingest_batch(self, events: list[RawLogEvent], priority: Priority = Priority.NORMAL) -> dict:
        """Ingest a batch of log events."""
        results = {
            "total": len(events),
            "accepted": 0,
            "rejected": 0,
            "event_ids": []
        }

        for event in events:
            result = await self.ingest_single(event, priority)
            if result["status"] == "accepted":
                results["accepted"] += 1
                results["event_ids"].append(result["event_id"])
            else:
                results["rejected"] += 1

        return results

    async def consume_stream(self) -> AsyncIterator[RawLogEvent]:
        """
        Consume events from Kafka log topic.
        Implements consumer group management and offset tracking.
        """
        self._consumer = AIOKafkaConsumer(
            settings.kafka_log_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_consumer_group,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda v: RawLogEvent.model_validate_json(v),
        )
        await self._consumer.start()
        logger.info("Kafka consumer started", 
                   topic=settings.kafka_log_topic,
                   group=settings.kafka_consumer_group)

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                event = message.value
                self._metrics["events_received"] += 1
                yield event
        finally:
            await self._consumer.stop()

    async def _publish_event(self, event: RawLogEvent, priority: Priority):
        """Publish validated event to the processed topic."""
        if self._producer:
            headers = [
                ("priority", priority.value.encode("utf-8")),
                ("source_id", event.source_id.encode("utf-8")),
            ]
            await self._producer.send(
                settings.kafka_processed_topic,
                value=event,
                key=event.source_id.encode("utf-8"),
                headers=headers,
            )
            self._metrics["events_published"] += 1

    def get_metrics(self) -> dict:
        """Return current service metrics."""
        return self._metrics.copy()
