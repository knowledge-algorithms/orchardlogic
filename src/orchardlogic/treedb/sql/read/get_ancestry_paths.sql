-- TreeDB read operations (phase P3B): get_ancestry_paths.
--
-- Port of the INSTALLED PostgreSQL function (D-002, D-009):
--   reference/pg_installed/graph/function/get_ancestry_paths_uuid_arr.sql:1-65
--   (cited below as "gap:<line>"); identical in migration
--   reference/graph-engine/migrations/015_orchard_phase2.py:48-125.
-- Porting notes: docs/porting/P3B.md (IDs in brackets).
--
-- PG `LANGUAGE sql RETURNS TABLE(...)` function -> DuckDB table macro [RT-2].
-- A DuckDB table macro takes its output types from its query, so every
-- declared PG output type (gap:2) is written as an explicit CAST.
--
-- The macro must stay idempotent (CREATE OR REPLACE): treedb.read.install()
-- runs this file whenever the installed read SQL is missing or outdated.

CREATE SCHEMA IF NOT EXISTS treedb;

-- gap:1-3  CREATE FUNCTION graph.get_ancestry_paths(_orchard_ids uuid[])
--          RETURNS TABLE(orchard_id uuid, tree_id uuid, branch_id uuid,
--          node_id uuid, edge_ix integer, coph_distance double precision,
--          depth integer, start_orchard_id uuid, start_node_id uuid)
CREATE OR REPLACE MACRO treedb.get_ancestry_paths(_orchard_ids) AS TABLE
    WITH RECURSIVE                                                  -- gap:16
    start_children AS (                                             -- gap:17-19
        -- PG uuid[] -> DuckDB UUID[] LIST; unnest yields one row per element,
        -- duplicates and NULLs included, as PG unnest does [ARR-1].
        SELECT unnest(CAST(_orchard_ids AS UUID[])) AS orchard_id
    )

    , descendants AS (                                              -- gap:21
        SELECT                                                      -- gap:22-34
            r.orchard_id,
            r.tree_id,
            r.branch_id,
            r.node_id,
            r.edge_ix,
            r.coph_distance,
            0                   AS depth,
            -- TreeDB [CYC-2]: the depth bound, the number of v_orchard rows,
            -- counted once per call (this anchor's uncorrelated subquery) and
            -- carried down the recursion. Internal only: the final SELECT
            -- below lists the PG columns and does not return it.
            (SELECT count(*) FROM treedb.v_orchard) AS depth_bound,
            sc.orchard_id       AS start_orchard_id,
            r.node_id           AS start_node_id
        FROM treedb.v_orchard AS r
        JOIN start_children AS sc
            ON r.orchard_id = sc.orchard_id

        UNION ALL                                                   -- gap:36

        SELECT                                                      -- gap:38-51
            r.orchard_id,
            r.tree_id,
            r.branch_id,
            r.node_id,
            r.edge_ix,
            r.coph_distance,
            -- gap:45  d.depth + 1 AS depth, plus the TreeDB depth guard [CYC-2]
            -- (G3 D-024 Q7, docs/porting/P3B.md). On rows without a cycle a
            -- chain from a start row visits distinct rows, so depth stays
            -- below the number of v_orchard rows and the CASE always yields
            -- d.depth + 1: output unchanged. PG has no guard and loops
            -- forever on a cycle; here the chain that reaches that bound
            -- raises (DuckDB error() aborts the query) instead of hanging.
            CASE
                WHEN d.depth + 1 < d.depth_bound
                    THEN d.depth + 1
                ELSE error('treedb.get_ancestry_paths: ancestry cycle: the chain from start row '
                           || CAST(d.start_orchard_id AS VARCHAR)
                           || ' reached depth ' || CAST(d.depth + 1 AS VARCHAR)
                           || ', the number of orchard rows, so it repeats a row (CYC-2)')
            END                 AS depth,
            d.depth_bound,
            d.start_orchard_id,
            d.start_node_id
        FROM treedb.v_orchard AS r
        JOIN descendants AS d
            ON r.node_id = d.branch_id
           AND r.tree_id = d.tree_id
    )

    SELECT                                                          -- gap:54-64
        CAST(orchard_id       AS UUID)    AS orchard_id,
        CAST(tree_id          AS UUID)    AS tree_id,
        CAST(branch_id        AS UUID)    AS branch_id,
        CAST(node_id          AS UUID)    AS node_id,
        CAST(edge_ix          AS INTEGER) AS edge_ix,
        CAST(coph_distance    AS DOUBLE)  AS coph_distance,
        CAST(depth            AS INTEGER) AS depth,
        CAST(start_orchard_id AS UUID)    AS start_orchard_id,
        CAST(start_node_id    AS UUID)    AS start_node_id
    FROM descendants
    -- PG has no ORDER BY (gap:54-64): rows come in the order the recursive
    -- CTE produces them, observed breadth-first (depth 0 rows first) and
    -- stable within one state but unspecified. TreeDB fixes a total order
    -- that refines the observed one [DET-1, proposed with OD-1 at G3]:
    -- depth first, then the start node (an item UUID for leaf starts, so the
    -- order is stable across reloads for leaf cohorts), then the start row.
    ORDER BY depth, start_node_id, start_orchard_id;
