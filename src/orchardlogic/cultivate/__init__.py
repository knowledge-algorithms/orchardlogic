"""Orchard: persistent hierarchical trees over a fixed document corpus."""

from orchardlogic.cultivate.backends.fusion import SimilarityProfile
from orchardlogic.cultivate.builder import OrchardBuilder
from orchardlogic.cultivate.cuts import (
    build_dynamic_cut,
    cut_partition_item_ids,
    linkage_for_canonical_subtree,
    pack_canonical_tree,
    validate_dynamic_cut,
    walk_cut_json,
)
from orchardlogic.cultivate.document import Document
from orchardlogic.cultivate.exceptions import (
    CorpusMutationUnsupportedError,
    DuplicateItemIdError,
    InvalidFusionError,
    InvalidIdentityError,
    InvalidLinkageError,
    MissingOptionalDependencyError,
    OrchardError,
    UnknownLabelSetError,
    UnknownNodeError,
    UnknownTreeError,
)
from orchardlogic.cultivate.identity import (
    assign_canonical_ids,
    canonical_node_id,
    ensure_unique_item_ids,
    generate_item_id,
    membership_hash,
    validate_tree_id,
)
from orchardlogic.cultivate.labeling.bottom_up import label_bottom_up
from orchardlogic.cultivate.labeling.contrast import label_contrast
from orchardlogic.cultivate.labeling.cost import estimate_labeling
from orchardlogic.cultivate.labels import import_labels, label_intrinsic
from orchardlogic.cultivate.orchard import Orchard
from orchardlogic.cultivate.schemas import ARTIFACT_SCHEMA_VERSION, DOCUMENT_SCHEMA_VERSION
from orchardlogic.cultivate.taxonomy import (
    StubTaxonomy,
    TaxonomyModel,
)
from orchardlogic.cultivate.tree import Tree
from orchardlogic.cultivate.viz import (
    cut_to_plotly_structure,
    orchard_plotly_structures,
    validate_plotly_structure,
)

__version__ = "0.1.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DOCUMENT_SCHEMA_VERSION",
    "CorpusMutationUnsupportedError",
    "Document",
    "DuplicateItemIdError",
    "InvalidFusionError",
    "InvalidIdentityError",
    "InvalidLinkageError",
    "MissingOptionalDependencyError",
    "Orchard",
    "OrchardBuilder",
    "OrchardError",
    "SimilarityProfile",
    "StubTaxonomy",
    "TaxonomyModel",
    "Tree",
    "UnknownLabelSetError",
    "UnknownNodeError",
    "UnknownTreeError",
    "assign_canonical_ids",
    "build_dynamic_cut",
    "canonical_node_id",
    "cut_partition_item_ids",
    "cut_to_plotly_structure",
    "ensure_unique_item_ids",
    "estimate_labeling",
    "generate_item_id",
    "import_labels",
    "label_bottom_up",
    "label_contrast",
    "label_intrinsic",
    "linkage_for_canonical_subtree",
    "membership_hash",
    "orchard_plotly_structures",
    "pack_canonical_tree",
    "validate_dynamic_cut",
    "validate_plotly_structure",
    "validate_tree_id",
    "walk_cut_json",
    "__version__",
]
