"""Pluggable feature and similarity backends for OrchardBuilder."""

from orchardlogic.cultivate.backends.family import AppExactMatchLayer
from orchardlogic.cultivate.backends.fusion import (
    SimilarityProfile,
    fuse_to_dissimilarity,
    raw_convex_fusion,
    validate_dissimilarity,
    variance_calibrated_fusion,
)
from orchardlogic.cultivate.backends.layers import (
    LayerRegistry,
    MatrixLayer,
    SimilarityLayer,
    TaxonomyCosineLayer,
    TaxonomyJsLayer,
    TfidfCosineLayer,
)
from orchardlogic.cultivate.backends.minilm import (
    MiniLMCosineLayer,
    MiniLMEmbeddingBackend,
    apply_description_transform,
    mapped_cosine,
    minilm_similarity_matrix,
    raw_cosine,
)
from orchardlogic.cultivate.backends.modernbert import ModernBERTFeatureBackend
from orchardlogic.cultivate.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    linkage_from_dissimilarity,
    linkage_from_similarity,
    similarity_to_dissimilarity,
    validate_similarity_matrix,
)
from orchardlogic.cultivate.backends.taxonomy_heads import (
    remap_head_to_label_order,
)
from orchardlogic.cultivate.backends.tfidf import TfidfEmbeddingBackend, tfidf_matrix

__all__ = [
    "AppExactMatchLayer",
    "LayerRegistry",
    "MatrixLayer",
    "MiniLMCosineLayer",
    "MiniLMEmbeddingBackend",
    "ModernBERTFeatureBackend",
    "SimilarityLayer",
    "SimilarityProfile",
    "TaxonomyCosineLayer",
    "TaxonomyJsLayer",
    "TfidfCosineLayer",
    "TfidfEmbeddingBackend",
    "apply_description_transform",
    "cosine_matrix",
    "fuse_to_dissimilarity",
    "jensen_shannon_matrix",
    "linkage_from_dissimilarity",
    "linkage_from_similarity",
    "mapped_cosine",
    "remap_head_to_label_order",
    "minilm_similarity_matrix",
    "raw_convex_fusion",
    "raw_cosine",
    "similarity_to_dissimilarity",
    "tfidf_matrix",
    "validate_dissimilarity",
    "validate_similarity_matrix",
    "variance_calibrated_fusion",
]
