"""
Embedding Service - Manages text embedding using sentence-transformers.
"""

from typing import Optional

import numpy as np
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """
    Text embedding service using sentence-transformers (BGE-base-en-v1.5).
    Supports batch encoding with caching for repeated queries.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max_size = 10000

    def initialize(self):
        """Pre-load the embedding model into memory at startup."""
        self._load_model()

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded", model=self.model_name)
            except Exception as e:
                logger.error("Failed to load embedding model", 
                           model=self.model_name, error=str(e))
                raise

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to embedding vector."""
        if text in self._cache:
            return self._cache[text]

        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        
        # Cache management
        if len(self._cache) < self._cache_max_size:
            self._cache[text] = embedding
        
        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts in a batch."""
        # Check cache for already encoded texts
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            if text not in self._cache:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Encode uncached texts
        if uncached_texts:
            self._load_model()
            new_embeddings = self._model.encode(
                uncached_texts, 
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                if len(self._cache) < self._cache_max_size:
                    self._cache[text] = embedding

        # Assemble full result
        results = []
        uncached_idx = 0
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append(self._cache[text])
            else:
                results.append(new_embeddings[uncached_idx])
                uncached_idx += 1

        return np.array(results)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2))

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
