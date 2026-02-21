"""Backward-compatible shim for cached-embedding inference predictor imports."""

from .inference.cached_embeddings import EmbeddingCachePredictor, CachedEmbeddingPredictor

__all__ = [
    'EmbeddingCachePredictor',
    'CachedEmbeddingPredictor',
]
