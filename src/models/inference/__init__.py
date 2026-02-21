"""Inference wrapper namespace."""

from .full_graph import ChemDiseasePredictor, FullGraphPredictor
from .cached_embeddings import EmbeddingCachePredictor, CachedEmbeddingPredictor

__all__ = [
    'ChemDiseasePredictor',
    'FullGraphPredictor',
    'EmbeddingCachePredictor',
    'CachedEmbeddingPredictor',
]

