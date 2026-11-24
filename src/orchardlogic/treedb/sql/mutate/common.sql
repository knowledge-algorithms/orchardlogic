-- TreeDB mutation coordinator: statements shared by attach and detach (phase P3C).
--
-- Only `find_orchard_id` has a PostgreSQL counterpart (repo.py:331-351). The
-- others are TreeDB additions: the D-007 state_version bump, the read-only
-- queries the coordinator uses to report what a mutation did (D-016c), and the
-- read-only detach cycle guard detach_cycle_rows_in_changed_trees (CYC-3,
-- G6 D-038; kept here so that detach_node.sql holds only the ported dt:
-- statements).
-- Statements are separated by `-- name: <step>` markers (treedb.mutate._load_sql).

-- name: find_orchard_id
-- repo.py:337-345  SELECT orchard_id FROM graph.v_orchard
--                  WHERE node_id = :node_id AND tree_id = :tree_id LIMIT 1
-- Literal, including LIMIT 1 without ORDER BY (only ambiguous when a node has
-- two parents in one tree, which the D-016b strict path prevents).
SELECT orchard_id
FROM treedb.v_orchard
WHERE node_id = CAST($node_id AS UUID)
  AND tree_id = CAST($tree_id AS UUID)
LIMIT 1;

-- name: bump_state_version
-- D-007: +1 per committed structural mutation. Runs inside the mutation's
-- transaction, so a rollback also undoes it. state_version is not indexed, so
-- the UPDATE is unaffected by DuckDB's index rules (UQ-4).
UPDATE treedb.tree_registry
SET state_version = state_version + 1
WHERE id = CAST($tree_id AS UUID)
RETURNING state_version, name, root_branch_id;

-- name: tree_roots
-- Branches of one tree that no orchard row OF THAT TREE names as its node:
-- the roots of its components. Same definition as P2B's
-- tools/pg_reference/canonical.py tree_structure "roots".
SELECT b.branch_id
FROM treedb.tree_branches b
WHERE b.tree_id = CAST($tree_id AS UUID)
  AND NOT EXISTS (SELECT 1 FROM treedb.v_orchard v
                  WHERE v.tree_id = b.tree_id AND v.node_id = b.branch_id)
ORDER BY b.branch_id;

-- name: tree_items
-- Registered items that hang anywhere in one tree (any component).
SELECT DISTINCT i.item_id
FROM treedb.v_orchard v
JOIN treedb.item_registry i ON i.item_uuid = v.node_id
WHERE v.tree_id = CAST($tree_id AS UUID)
ORDER BY i.item_id;

-- name: trees_of_branches
-- TreeDB reporting: the trees owning the branches of the rows the dt:38-41
-- rewire touched (its RETURNING branch_id). A tree other than the target's is
-- collateral (GRAFT-1): its state_version is bumped too (D-007).
SELECT DISTINCT b.tree_id
FROM treedb.tree_branches b
WHERE list_contains(CAST($branch_ids AS UUID[]), b.branch_id)
ORDER BY b.tree_id;

-- name: tree_name
SELECT name, state_version, root_branch_id
FROM treedb.tree_registry
WHERE id = CAST($tree_id AS UUID);

-- name: detach_cycle_rows_in_changed_trees
-- TreeDB guard CYC-3 (G6 D-038, amending G3 D-024 Q5; a divergence;
-- docs/porting/P3C.md CYC-3). No PostgreSQL counterpart: dt:5-44 has no
-- ancestry check, and the dt:38-41 rewire (no tree filter, GRAFT-1) can close
-- a loop in another tree (tests/mutate/test_detach_cycle_pin.py), which PG
-- commits.
-- Run after rewire_parent and before COMMIT, on BOTH D-016b paths (detach has
-- no strict switch; the check is not part of D-016b). Read-only.
-- $tree_ids: every tree whose rows this call changed (the detached tree and
-- each collateral tree of the rewire). The parent relation is the tree-scoped
-- one of get_ancestry_paths (gap:48-51) and of CYC-1: within tree T, the parent
-- of node X is the branch_id of a v_orchard row of T whose node_id is X.
-- ancestors(tree_id, start_node, node): node is an ancestor of start_node in
-- tree_id. The anchor starts only from rows whose node is itself a branch with
-- children in that tree (a leaf is never on a cycle). UNION (not UNION ALL)
-- keeps a visited set of (tree, start, node) triples, which is finite, so the
-- recursion terminates even on rows that are already cyclic (as CYC-1).
-- Result: every row of those trees that lies on a cycle: the row (B, S) is on
-- a cycle exactly when S = B or S is an ancestor of B in the same tree. Empty
-- result: every checked tree is acyclic.
WITH RECURSIVE
checked AS (
    SELECT v.tree_id, v.orchard_id, v.branch_id, v.node_id, v.edge_ix
    FROM treedb.v_orchard v
    WHERE list_contains(CAST($tree_ids AS UUID[]), v.tree_id)
),
ancestors(tree_id, start_node, node) AS (
    SELECT c.tree_id, c.node_id, c.branch_id
    FROM checked c
    WHERE EXISTS (SELECT 1 FROM checked p
                  WHERE p.tree_id = c.tree_id AND p.branch_id = c.node_id)
    UNION
    SELECT a.tree_id, a.start_node, c.branch_id
    FROM ancestors a
    JOIN checked c ON c.tree_id = a.tree_id AND c.node_id = a.node
)
SELECT c.tree_id, r.name, c.orchard_id, c.branch_id, c.node_id, c.edge_ix
FROM checked c
JOIN treedb.tree_registry r ON r.id = c.tree_id
WHERE c.node_id = c.branch_id
   OR EXISTS (SELECT 1 FROM ancestors a
              WHERE a.tree_id = c.tree_id
                AND a.start_node = c.branch_id
                AND a.node = c.node_id)
ORDER BY r.name, c.tree_id, c.branch_id, c.node_id;
