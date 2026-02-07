"""
Atomic Embedding Extraction Module

This module provides tools to extract atomic embeddings from crystal structures
using the pretrained F_nonlocal model. The embeddings encode electron cloud
shape information through spherical harmonics (L=0,1,2,3,4).

Main components:
- AtomicEmbeddingExtractor: Extract 992-dim atomic embeddings from structures
"""

from .extractor import AtomicEmbeddingExtractor

__all__ = ['AtomicEmbeddingExtractor']
