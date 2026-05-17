"""
Working Memory - Redis-backed shared state for agent collaboration.
Scoped per RCA session with TTL-based expiration.
"""

import json
from typing import Any, Optional

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class WorkingMemory:
    """
    Redis-backed shared state for inter-agent communication.
    Each RCA session gets its own memory scope with TTL expiration.
    """

    def __init__(self, session_id: str, redis_client=None):
        self.session_id = session_id
        self._redis = redis_client
        self._local_cache: dict[str, Any] = {}  # Fallback when Redis unavailable
        self.ttl = settings.redis_working_memory_ttl

    async def initialize(self):
        """Initialize Redis connection if not provided."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    settings.redis_url,
                    password=settings.redis_password,
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.debug("Working memory Redis connected", session=self.session_id)
            except Exception as e:
                logger.warning("Redis unavailable, using local cache", error=str(e))
                self._redis = None

    async def store(self, key: str, value: Any):
        """Store a value in working memory."""
        serialized = json.dumps(value, default=str)
        
        if self._redis:
            try:
                await self._redis.hset(f"wm:{self.session_id}", key, serialized)
                await self._redis.expire(f"wm:{self.session_id}", self.ttl)
            except Exception as e:
                logger.warning("Redis store failed, using local", error=str(e))
                self._local_cache[key] = value
        else:
            self._local_cache[key] = value

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from working memory."""
        if self._redis:
            try:
                data = await self._redis.hget(f"wm:{self.session_id}", key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning("Redis retrieve failed, using local", error=str(e))
        
        return self._local_cache.get(key)

    async def get_full_context(self) -> dict[str, Any]:
        """Retrieve all working memory for the session."""
        if self._redis:
            try:
                data = await self._redis.hgetall(f"wm:{self.session_id}")
                return {k: json.loads(v) for k, v in data.items()}
            except Exception as e:
                logger.warning("Redis getall failed, using local", error=str(e))
        
        return self._local_cache.copy()

    async def append_to_list(self, key: str, value: Any):
        """Append a value to a list in working memory."""
        existing = await self.retrieve(key) or []
        if isinstance(existing, list):
            existing.append(value)
        else:
            existing = [existing, value]
        await self.store(key, existing)

    async def clear(self):
        """Clear all working memory for this session."""
        if self._redis:
            try:
                await self._redis.delete(f"wm:{self.session_id}")
            except Exception:
                pass
        self._local_cache.clear()

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
