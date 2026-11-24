"""Enrichers (spec section 3): callables that add titles, summaries and tags to items/nodes.

Each enricher is usable on its own: ``get_enricher("item.extract@1").enrich(docs)``.
"""

from orchardlogic.cultivate.enrichers.base import (
    ENRICHER_SCHEMA_VERSION,
    ItemEnricher,
    TreeContext,
    TreeEnricher,
    UnknownEnricherError,
    available_enrichers,
    enricher_id,
    get_enricher,
    handle_filename,
    handle_of,
    register_enricher,
)

# Importing the modules registers the built-ins.
from orchardlogic.cultivate.enrichers import (  # noqa: F401,I001
    item_extract,
    tree_contrast,
    tree_contrast2,
    tree_extractive,
    tree_extractive2,
    tree_llm,
    tree_tags,
    tree_tags2,
)

__all__ = [
    "ENRICHER_SCHEMA_VERSION",
    "ItemEnricher",
    "TreeContext",
    "TreeEnricher",
    "UnknownEnricherError",
    "available_enrichers",
    "enricher_id",
    "get_enricher",
    "handle_filename",
    "handle_of",
    "register_enricher",
]
