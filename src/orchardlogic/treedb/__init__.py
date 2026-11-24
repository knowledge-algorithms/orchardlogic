"""TreeDB: storage and query operators for mutable hierarchies over shared items.

See docs/ARCHITECTURE.md for the contract and docs/PLAN.md for build status.
"""

__version__ = "0.0.1"

from orchardlogic.treedb.api import TreeDB  # noqa: E402

__all__ = ["TreeDB", "__version__"]
