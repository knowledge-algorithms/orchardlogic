-- TreeDB port of graph.orchard_attach_node (phase P3C).
--
-- Installed text (D-002, D-009), cited as `at:<line>`:
--   reference/pg_installed/graph/function/orchard_attach_node_uuid_uuid_integer_numeric.sql
-- Identical body: reference/graph-engine/migrations/014_orchard_schema.py:251-310.
-- PG signature (at:1): orchard_attach_node(p_new_node_id uuid, p_target_orchard_id uuid,
--                                          p_new_edge_ix integer, p_coph_distance numeric)
--                      RETURNS integer
--
-- The plpgsql body has NO control flow (no IF, no RAISE): it is a straight
-- sequence of SQL statements. Each statement below is one plpgsql statement,
-- in the same order, run by treedb.mutate.attach_node inside ONE DuckDB
-- transaction (plpgsql runs its body inside the caller's transaction).
-- Statements are separated by `-- name: <step>` markers (read by
-- treedb.mutate._load_sql). Parameters are DuckDB named parameters `$name`.
-- Divergences are in docs/porting/P3C.md (IDs in brackets).
--
-- Statements marked "TreeDB" have no PostgreSQL counterpart; they are run by
-- the CYC-1 cycle guard (both paths, G3 D-024 Q7), the D-016b strict path,
-- the item registration of G3 D-024 Q4 (only when an item_id is passed) or
-- the coordinator's reporting. None of them writes a graph row.

-- name: drop_new_linkage
-- at:8   DROP TABLE IF EXISTS new_linkage;
-- Qualified temp.main so it can never drop a permanent table (CO-3, [AT-1]).
DROP TABLE IF EXISTS temp.main.new_linkage;

-- name: create_new_linkage
-- at:9-14  CREATE TEMP TABLE new_linkage AS SELECT p_new_node_id AS new_node_id,
--          p_target_orchard_id AS target_orchard_id, p_new_edge_ix::int AS new_edge_ix,
--          p_coph_distance AS target_coph;
-- The plpgsql parameters become CASTs of bound parameters (PG binds uuid,
-- uuid, integer, numeric at the call). target_coph is DOUBLE, NOT DuckDB
-- NUMERIC (= DECIMAL(18,3), which would truncate): the only reader is
-- `target_coph::double precision` (at:30), and numeric -> float8 of the
-- caller's exact decimal gives the caller's double [AT-2, CO-4, D-012].
CREATE TEMP TABLE new_linkage AS
SELECT
    CAST($new_node_id AS UUID)          AS new_node_id,
    CAST($target_orchard_id AS UUID)    AS target_orchard_id,
    CAST($new_edge_ix AS INTEGER)       AS new_edge_ix,
    CAST($target_coph AS DOUBLE)        AS target_coph;

-- name: drop_target_node_row
-- at:16  DROP TABLE IF EXISTS target_node_row;
DROP TABLE IF EXISTS temp.main.target_node_row;

-- name: create_target_node_row
-- at:17-32  literal, apart from the schema/temp qualification.
-- gen_random_uuid() is evaluated once per row of a materialised CTAS, as in
-- PG; working_row has at most one row (orchard_id is the orchard PK) [AT-3].
-- `coph_distance - 0.001`: PG float8 - numeric literal resolves to float8 -
-- float8; DuckDB DOUBLE - DECIMAL(4,3) is DOUBLE - DOUBLE [AT-4].
-- No rounding here: attach distances are NOT rounded (D-012 A, CO-4).
CREATE TEMP TABLE target_node_row AS
WITH working_row AS (
    SELECT *
    FROM temp.main.new_linkage nl
    INNER JOIN treedb.v_orchard o ON nl.target_orchard_id = o.orchard_id
)
SELECT
    tree_id,
    gen_random_uuid() AS new_branch_id,
    node_id,
    edge_ix,
    CASE WHEN target_coph IS NULL
        THEN coph_distance - 0.001
        ELSE target_coph::double precision
    END AS coph_distance
FROM working_row;

-- name: cycle_subject_is_ancestor
-- TreeDB guard CYC-1 (G3 D-024 Q7, a divergence; docs/porting/P3C.md CYC-1).
-- PG: no ancestry check anywhere in at:8-52; attaching a branch above a row
-- of its own subtree commits a cycle and returns 2.
-- After at:34-49 the target row's parent branch P names the new branch NB,
-- and NB names the subject S. S becomes its own ancestor exactly when S is P
-- or an ancestor of P in the target's tree (the same tree-scoped parent
-- relation as get_ancestry_paths, gap:48-51). Read-only, run on BOTH the
-- strict and the strict=False path, before any write. Counts the matches.
-- UNION (not UNION ALL) keeps a set of visited nodes, so this query also
-- terminates on a cycle that already exists in the stored rows.
WITH RECURSIVE ancestors(tree_id, node) AS (
    SELECT v.tree_id, v.branch_id
    FROM treedb.v_orchard v
    JOIN temp.main.new_linkage nl ON v.orchard_id = nl.target_orchard_id
    UNION
    SELECT v.tree_id, v.branch_id
    FROM treedb.v_orchard v
    JOIN ancestors a ON v.node_id = a.node AND v.tree_id = a.tree_id
)
SELECT count(*)
FROM ancestors
WHERE node = (SELECT new_node_id FROM temp.main.new_linkage);

-- name: item_lookup
-- TreeDB (G3 D-024 Q4, only when the call passes an item_id): the registry
-- rows that already name this item id or the subject UUID. PG has no item
-- registry (P3C INT-1). The coordinator compares them with the request and
-- raises ItemConflictError on any difference, before any write.
SELECT item_id, item_uuid, source_ref
FROM treedb.item_registry
WHERE item_id = CAST($item_id AS VARCHAR)
   OR item_uuid = (SELECT new_node_id FROM temp.main.new_linkage)
ORDER BY item_id;

-- name: item_register
-- TreeDB (G3 D-024 Q4, only when the call passes an item_id): register a new
-- item in the attach's own transaction (a rollback removes it again). Only
-- when the target row exists: the PG-faithful unknown-target no-op
-- (strict=False, D-016b) writes nothing, the registry included. RETURNING
-- gives the new item ids for AttachResult.new_items (empty when the item was
-- already registered).
INSERT INTO treedb.item_registry (item_id, item_uuid, source_ref)
SELECT CAST($item_id AS VARCHAR), nl.new_node_id, CAST($source_ref AS VARCHAR)
FROM temp.main.new_linkage nl
WHERE EXISTS (SELECT 1 FROM temp.main.target_node_row)
  AND NOT EXISTS (SELECT 1 FROM treedb.item_registry i WHERE i.item_id = CAST($item_id AS VARCHAR))
RETURNING item_id;

-- name: strict_target_rows
-- TreeDB (D-016b strict path only): how many target rows were found.
-- PG: an unknown target gives 0 rows here and the function returns 0.
SELECT count(*) FROM temp.main.target_node_row;

-- name: strict_subject_rows
-- TreeDB (D-016b strict path only): orchard rows of the target's tree whose
-- node is already the subject (the subject already has a parent there).
-- PG: no check; the subject gets a second parent.
SELECT count(*)
FROM treedb.v_orchard v
WHERE v.tree_id IN (SELECT tree_id FROM temp.main.target_node_row)
  AND v.node_id = (SELECT new_node_id FROM temp.main.new_linkage);

-- name: insert_branch
-- at:34-35  INSERT INTO graph.tree_branches (tree_id, branch_id)
--           SELECT DISTINCT tree_id, new_branch_id FROM target_node_row;
-- graft_id and created_at come from the column defaults, as in PG; created_at
-- defaults to store.UTC_NOW_SQL (D-017 B, TS-1).
INSERT INTO treedb.tree_branches (tree_id, branch_id)
SELECT DISTINCT tree_id, new_branch_id FROM temp.main.target_node_row;

-- name: rewire_target
-- at:37-40  UPDATE graph.orchard SET node_id = (SELECT DISTINCT new_branch_id FROM target_node_row),
--           edge_ix = 2 WHERE id = (SELECT DISTINCT target_orchard_id FROM new_linkage);
-- Literal. Updates node_id, a member of UNIQUE (branch_id, node_id), in place:
-- DuckDB 2.0 accepts this inside the transaction (UQ-1, [UPD-1]).
UPDATE treedb.orchard
SET node_id = (SELECT DISTINCT new_branch_id FROM temp.main.target_node_row),
    edge_ix = 2
WHERE id = (SELECT DISTINCT target_orchard_id FROM temp.main.new_linkage);

-- name: add_subject_row
-- at:42-45  INSERT INTO target_node_row SELECT tree_id, new_branch_id, new_node_id,
--           new_edge_ix, coph_distance FROM new_linkage nl, target_node_row tn;
-- The SELECT reads target_node_row as it was before this INSERT (PG and
-- DuckDB alike) [AT-5].
INSERT INTO temp.main.target_node_row
SELECT
    tree_id, new_branch_id, new_node_id, new_edge_ix, coph_distance
FROM temp.main.new_linkage nl, temp.main.target_node_row tn;

-- name: insert_rows
-- at:47-49  INSERT INTO graph.orchard (branch_id, node_id, edge_ix, coph_distance)
--           SELECT new_branch_id, node_id, edge_ix, coph_distance FROM target_node_row;
-- at:51-52  GET DIAGNOSTICS v_rowcount = ROW_COUNT; RETURN v_rowcount;
-- DuckDB returns the inserted row count as the INSERT's result (CO-5).
INSERT INTO treedb.orchard (branch_id, node_id, edge_ix, coph_distance)
SELECT new_branch_id, node_id, edge_ix, coph_distance
FROM temp.main.target_node_row;

-- name: report_new_branch
-- TreeDB (reporting only): the tree and the branch this call created.
SELECT DISTINCT tree_id, new_branch_id FROM temp.main.target_node_row;
