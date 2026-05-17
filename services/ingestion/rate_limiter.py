"""
Token Bucket Rate Limiter for ingestion service.
Provides per-source rate limiting with configurable burst capacity.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Bucket:
    """Token bucket for a single source."""
    tokens: float
    last_refill: float
    rate: float
    burst: float

    def consume(self) -> bool:
        """Try to consume a token. Returns True if allowed."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class TokenBucketRateLimiter:
    """
    Per-source token bucket rate limiter.
    
    Args:
        rate: Tokens per second (sustained rate)
        burst: Maximum burst capacity
    """

    def __init__(self, rate: float = 10000, burst: float = 20000):
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, Bucket] = {}
        self._lock = Lock()
        self._global_bucket = Bucket(
            tokens=burst,
            last_refill=time.time(),
            rate=rate,
            burst=burst
        )

    def allow(self, source_id: str) -> bool:
        """
        Check if a request from the given source is allowed.
        Uses both per-source and global rate limiting.
        """
        # Global rate limit check
        if not self._global_bucket.consume():
            return False

        # Per-source rate limit (10% of global)
        with self._lock:
            if source_id not in self._buckets:
                self._buckets[source_id] = Bucket(
                    tokens=self.burst * 0.1,
                    last_refill=time.time(),
                    rate=self.rate * 0.1,
                    burst=self.burst * 0.1,
                )
            return self._buckets[source_id].consume()

    def reset(self, source_id: str = None):
        """Reset rate limiter state."""
        with self._lock:
            if source_id:
                self._buckets.pop(source_id, None)
            else:
                self._buckets.clear()
                self._global_bucket = Bucket(
                    tokens=self.burst,
                    last_refill=time.time(),
                    rate=self.rate,
                    burst=self.burst
                )
