-- TreeDB port of graph.orchard_detach_node (phase P3C), including the 019 fix.
--
-- Installed text (D-002, D-009), cited as `dt:<line>`:
--   reference/pg_installed/graph/function/orchard_detach_node_uuid.sql
-- Identical body: reference/graph-engine/migrations/019_fix_detach_edge_ix.py:37-83
-- (014:318-362 is the pre-019 body; 019 adds v_sibling_edge_ix, dt:9, :21-22, :40).
-- PG signature (dt:1): orchard_detach_node(p_target_orchard_id uuid) RETURNS integer
--
-- The plpgsql control flow (two IF ... RAISE blocks, dt:17-19 and dt:28-31)
-- and the plpgsql variables (dt:5-10) live in treedb.mutate.detach_node, which
-- runs these statements in order inside ONE DuckDB transaction. Statements
-- are separated by `-- name: <step>` markers (read by treedb.mutate._load_sql).
-- Divergences are in docs/porting/P3C.md (IDs in brackets).
-- The TreeDB detach cycle guard (CYC-3, G6 D-038), run after rewire_parent and
-- before COMMIT, is common.detach_cycle_rows_in_changed_trees (no dt: line).

-- name: select_target
-- dt:12-15  SELECT branch_id, node_id INTO v_branch_id, v_target_node_id
--           FROM graph.v_orchard WHERE orchard_id = p_target_orchard_id;
-- plpgsql SELECT INTO (not STRICT): first row, or NULLs when there is none.
-- orchard_id is the orchard PK, so there is at most one row.
SELECT branch_id, node_id, tree_id
FROM treedb.v_orchard
WHERE orchard_id = CAST($target_orchard_id AS UUID);

-- name: select_sibling
-- dt:21-26  SELECT node_id, edge_ix INTO v_sibling_node_id, v_sibling_edge_ix
--           FROM graph.v_orchard WHERE branch_id = v_branch_id
--           AND orchard_id <> p_target_orchard_id LIMIT 1;
-- Literal, including LIMIT 1 without ORDER BY: attach and detach only ever
-- leave binary branches, so there is exactly one candidate [DT-2].
SELECT node_id, edge_ix
FROM treedb.v_orchard
WHERE branch_id = CAST($v_branch_id AS UUID)
  AND orchard_id <> CAST($target_orchard_id AS UUID)
LIMIT 1;

-- name: cascade_delete_orchard
-- CASCADE of dt:33-34. PG: orchard.branch_id REFERENCES tree_branches(branch_id)
-- ON DELETE CASCADE (pgi:table/orchard.sql:7, 014:116), so deleting the branch
-- also deletes its child rows: the target row and the sibling row.
-- DuckDB has no cascade and TreeDB declares no FK (FK-1, CASCADE-1), so the
-- cascade is this explicit DELETE, run FIRST (children before parent) [DT-1].
DELETE FROM treedb.orchard
WHERE branch_id = CAST($v_branch_id AS UUID);

-- name: delete_branch
-- dt:33-34  DELETE FROM graph.tree_branches WHERE branch_id = v_branch_id;
-- dt:36     GET DIAGNOSTICS v_rowcount = ROW_COUNT;   -- the DELETE's count
-- The count is the tree_branches rows only; PG does not count cascaded rows.
DELETE FROM treedb.tree_branches
WHERE branch_id = CAST($v_branch_id AS UUID);

-- name: rewire_parent
-- dt:38-41  UPDATE graph.orchard SET node_id = v_sibling_node_id,
--           edge_ix = v_sibling_edge_ix WHERE node_id = v_branch_id;
-- Literal SET and WHERE, including the 019 edge_ix fix and the absence of a
-- tree filter: a row of ANOTHER tree that names the deleted branch (an
-- edge_ix 3 graft) is rewired too, and its edge_ix 3 is overwritten by the
-- sibling's edge_ix, as in PG [GRAFT-1].
-- TreeDB addition: RETURNING id, branch_id (reporting only; the rows written
-- are PG's). The coordinator derives the affected trees from branch_id, bumps
-- every affected tree's state_version (D-007) and reports the other trees as
-- collateral [GRAFT-1, SV-1]. DuckDB 2.0 rejects subqueries in RETURNING
-- (test_GRAFT_1_reproducer_returning_rejects_subquery), so the tree lookup is
-- the separate read common.trees_of_branches.
-- The D-016c split is NOT "0 rows here": it is "the deleted branch had no
-- parent row in its own tree" (roots_before), which also holds when a graft
-- row in another tree names it [SPLIT-1].
-- Updates node_id, a member of UNIQUE (branch_id, node_id), in place: DuckDB
-- 2.0 accepts this inside the transaction [UPD-1].
UPDATE treedb.orchard
SET node_id = CAST($v_sibling_node_id AS UUID),
    edge_ix = CAST($v_sibling_edge_ix AS INTEGER)
WHERE node_id = CAST($v_branch_id AS UUID)
RETURNING id, branch_id;
