-- TreeDB SQL convenience macros (phase P6A). Read-only: mutation is Python only (D-020, OD-10).
--
-- Installed by TreeDB(...) after the schema, the read macros and the cohort tables.
-- Idempotent (CREATE OR REPLACE). Documented, with examples, in docs/SQL_API.md.
--
-- Each macro is a thin SQL view over the same tables the Python calls read. A cultivar is
-- named by its name; a node by its UUID (text or UUID); a cohort by its handle (text or UUID).
-- Items that a cultivar does not contain are ignored here; the Python calls report them.

-- One row per cultivar, with the state_version that tree-dependent results carry.
CREATE OR REPLACE MACRO treedb_cultivars() AS TABLE
    SELECT t.id AS tree_id, t.name AS cultivar, t.state_version
    FROM treedb.tree_registry t
    ORDER BY t.name;

-- The node's kind, item (leaves) and incoming row. A root has no incoming row (NULL columns).
CREATE OR REPLACE MACRO treedb_node(_cultivar, _node) AS TABLE
    SELECT t.name AS cultivar, t.state_version, CAST(_node AS UUID) AS node_id,
           CASE WHEN b.branch_id IS NOT NULL THEN 'branch' WHEN i.item_id IS NOT NULL THEN 'leaf' ELSE 'other' END AS kind,
           i.item_id,
           v.orchard_id, v.branch_id AS parent_branch_id, v.edge_ix, v.coph_distance AS edge_coph_distance
    FROM treedb.tree_registry t
    LEFT JOIN treedb.tree_branches b ON b.tree_id = t.id AND b.branch_id = CAST(_node AS UUID)
    LEFT JOIN treedb.v_orchard v ON v.tree_id = t.id AND v.node_id = CAST(_node AS UUID)
    LEFT JOIN treedb.item_registry i ON i.item_uuid = v.node_id AND v.edge_ix = 1
    WHERE t.name = _cultivar AND (b.branch_id IS NOT NULL OR v.node_id IS NOT NULL);

-- graph.get_ancestry_paths from the node's incoming row up to its root (empty for a root).
CREATE OR REPLACE MACRO treedb_ancestry(_cultivar, _node) AS TABLE
    SELECT * FROM treedb.get_ancestry_paths((
        SELECT list(v.orchard_id)
        FROM treedb.v_orchard v JOIN treedb.tree_registry t ON t.id = v.tree_id
        WHERE t.name = _cultivar AND v.node_id = CAST(_node AS UUID)));

-- graph.get_compressed_subtree of a cohort's items in a cultivar (the raw PostgreSQL contract).
CREATE OR REPLACE MACRO treedb_subtree(_cultivar, _cohort) AS TABLE
    SELECT * FROM treedb.get_compressed_subtree((
        SELECT list(v.orchard_id)
        FROM treedb.v_orchard v
        JOIN treedb.tree_registry t ON t.id = v.tree_id
        JOIN treedb.item_registry r ON r.item_uuid = v.node_id
        JOIN treedb.cohort_member m ON m.item_id = r.item_id
        WHERE t.name = _cultivar AND v.edge_ix = 1 AND m.cohort_id = CAST(_cohort AS UUID)));

-- The members of a cohort, sorted.
CREATE OR REPLACE MACRO treedb_cohort_items(_cohort) AS TABLE
    SELECT item_id FROM treedb.cohort_items(_cohort) ORDER BY item_id;

-- A cohort's header row: kind, size, membership hash, provenance and recorded state versions.
CREATE OR REPLACE MACRO treedb_cohort(_cohort) AS TABLE
    SELECT * FROM treedb.cohort_info(_cohort);
