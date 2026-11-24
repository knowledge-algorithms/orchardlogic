-- TreeDB read operations (phase P3B): get_compressed_subtree and its stages.
--
-- Port of the INSTALLED PostgreSQL function (D-002, D-009):
--   reference/pg_installed/graph/function/get_compressed_subtree_uuid_arr.sql:1-121
--   (cited below as "gcs:<line>"); identical in migration
--   reference/graph-engine/migrations/015_orchard_phase2.py:133-267.
-- Requires treedb.get_ancestry_paths (get_ancestry_paths.sql).
-- Porting notes: docs/porting/P3B.md (IDs in brackets).
--
-- The PG function is ONE statement with the CTE chain
--   doc_list -> paths -> full_linkage -> compressed_linkage
--            -> node_ix, branch_ix -> indexed_linkage -> final SELECT.
-- Architecture §8 asks for the intermediate stages to be inspectable, so the
-- chain is split into stacked table macros, one per stage. Each macro keeps
-- the PG CTE names and text; the next stage reads the previous macro under
-- the same CTE name, so the composed statement is the PG statement [STAGE-1]:
--
--   stage 1  treedb.cs_paths(ids)               doc_list, paths            (path closure)
--   stage 2  treedb.cs_full_linkage(ids)        full_linkage               (grouping by contributing leaves: attached_docs)
--   stage 3  treedb.cs_compressed_linkage(ids)  compressed_linkage         (unary-chain compression)
--   stage 4  treedb.cs_indexed_linkage(ids)     node_ix, branch_ix,
--                                               indexed_linkage            (Z-like numbering)
--   final    treedb.get_compressed_subtree(ids) final SELECT (gcs:106-120) (l/r pivot, raw contract INV-14)
--
-- Determinism [DET-2..DET-5]. PG leaves four orders open (no tie-break):
-- FIRST_VALUE chain ends (gcs:61-64), leaf row_number (gcs:75), l/r
-- row_number with no ORDER BY (gcs:99) and the final ORDER BY coph_distance
-- (gcs:120). P2B measured that the results vary across reloads
-- (docs/PG_BASELINE.md §5, §7, §9). TreeDB appends tie-break keys AFTER the
-- PG keys, so every DuckDB result is one of the results PG's own ordering
-- admits; it never reorders rows that PG orders. The specific tie-break rules
-- are a PROPOSAL for OD-1 at G3 (D-015), isolated in the lines marked DET-n.
--
-- All macros are idempotent (CREATE OR REPLACE); treedb.read.install() runs
-- this file whenever the installed read SQL is missing or outdated.

CREATE SCHEMA IF NOT EXISTS treedb;

-- ---------------------------------------------------------------------------
-- Stage 1: path closure.  gcs:19-29
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO treedb.cs_paths(_orchard_ids) AS TABLE
    WITH
    doc_list AS (                                                   -- gcs:20-22
        SELECT unnest(CAST(_orchard_ids AS UUID[])) AS orchard_id
    )

    , paths AS (                                                    -- gcs:24-29
        SELECT *
        FROM treedb.get_ancestry_paths(
            -- PG array_agg(DISTINCT orchard_id::uuid); NULL for an empty
            -- input, which unnest turns into zero rows in both systems [ARR-1].
            (SELECT array_agg(DISTINCT CAST(orchard_id AS UUID)) FROM doc_list)
        ) sb
    )

    SELECT * FROM paths;

-- ---------------------------------------------------------------------------
-- Stage 2: grouping of edges by contributing leaves (attached_docs).  gcs:31-45
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO treedb.cs_full_linkage(_orchard_ids) AS TABLE
    WITH
    paths AS (
        SELECT * FROM treedb.cs_paths(_orchard_ids)
    )

    -- Gather every intermediate link from each starting node up to the root.
    -- Groups by branch_id so we can see all node_ids that feed into each branch.
    , full_linkage AS (                                             -- gcs:33-45
        SELECT
            tree_id,
            branch_id,
            node_id,
            coph_distance,
            edge_ix,
            -- PG array_agg(... ORDER BY uuid): DuckDB orders UUIDs by their
            -- 16 bytes, as PG uuid_cmp does (memcmp) [UUID-1].
            array_agg(start_node_id    ORDER BY start_node_id)    AS attached_docs,
            array_agg(start_orchard_id ORDER BY start_orchard_id) AS attached_grafts,
            count(start_node_id)                                   AS doc_count,
            -- TreeDB addition [DET-2]: the smallest path depth of this edge.
            -- Inside one compressed chain it rises by exactly 1 per edge going
            -- up, so it orders a chain's edges when their heights tie (F4).
            -- Not part of the PG column list; stage 3 drops it again.
            min(depth)                                             AS min_depth
        FROM paths
        GROUP BY tree_id, branch_id, coph_distance, node_id, edge_ix
    )

    SELECT * FROM full_linkage;

-- ---------------------------------------------------------------------------
-- Stage 3: unary-chain compression.  gcs:47-67
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO treedb.cs_compressed_linkage(_orchard_ids) AS TABLE
    WITH
    full_linkage AS (
        SELECT * FROM treedb.cs_full_linkage(_orchard_ids)
    )

    -- Compress runs of single-child branches so every row in compressed_linkage
    -- represents a direct logical link between two subtree participants.
    , compressed_linkage AS (                                       -- gcs:49-67
        SELECT DISTINCT
            tree_id,
            top_branch      AS branch_id,
            bottom_node     AS node_id,
            maxcoph         AS coph_distance,
            bottom_edge_ix  AS edge_ix,
            attached_docs,
            attached_grafts,
            doc_count
        FROM (
            SELECT *,
                -- gcs:61-64. PG orders only by coph_distance, so a chain whose
                -- heights tie has no defined end (PG_BASELINE §7, F4).
                -- [DET-2] appends min_depth: lowest edge = bottom, highest = top.
                FIRST_VALUE(node_id)   OVER (PARTITION BY attached_docs ORDER BY coph_distance,      min_depth)      AS bottom_node,
                FIRST_VALUE(branch_id) OVER (PARTITION BY attached_docs ORDER BY coph_distance DESC, min_depth DESC) AS top_branch,
                max(coph_distance)     OVER (PARTITION BY attached_docs)                                             AS maxcoph,
                FIRST_VALUE(edge_ix)   OVER (PARTITION BY attached_docs ORDER BY coph_distance,      min_depth)      AS bottom_edge_ix
            FROM full_linkage
        ) fl
    )

    SELECT * FROM compressed_linkage;

-- ---------------------------------------------------------------------------
-- Stage 4: Z-like numbering.  gcs:69-104
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO treedb.cs_indexed_linkage(_orchard_ids) AS TABLE
    WITH
    compressed_linkage AS (
        SELECT * FROM treedb.cs_compressed_linkage(_orchard_ids)
    )

    -- Assign stable integer indices to leaf nodes (doc number) and branch nodes
    -- (numbered above the max leaf index) so the result can be fed directly into
    -- scipy.cluster.hierarchy.dendrogram or similar.
    , node_ix AS (                                                  -- gcs:72-78
        SELECT
            node_id,
            -- gcs:75 row_number() OVER (ORDER BY coph_distance): 1-based.
            -- [DET-3] appends node_id (a leaf's item UUID, stable across
            -- reloads) to order leaves whose heights tie.
            row_number() OVER (ORDER BY coph_distance, node_id) AS node_ix
        FROM compressed_linkage
        WHERE edge_ix = 1
    )

    , branch_ix AS (                                                -- gcs:80-89
        SELECT
            branch_id,
            dense_rank() OVER (ORDER BY coph_distance)
                + (SELECT max(node_ix) FROM node_ix) AS branch_ix
        FROM (
            SELECT DISTINCT branch_id, coph_distance
            FROM compressed_linkage
        ) sq1
    )

    , indexed_linkage AS (                                          -- gcs:91-104
        SELECT
            cl.*,
            bx.branch_ix,
            CASE
                WHEN dx.node_ix  IS NOT NULL THEN dx.node_ix
                ELSE nx.branch_ix
            END                                                         AS node_ix,
            -- gcs:99 row_number() OVER (PARTITION BY cl.tree_id, cl.branch_id)
            -- has NO ORDER BY, so PG's left/right is arbitrary (PG_BASELINE §5).
            -- [DET-4] orders children by their index (smaller left; NULL index
            -- last), then by attached_docs (disjoint between siblings).
            row_number() OVER (
                PARTITION BY cl.tree_id, cl.branch_id
                ORDER BY CASE
                             WHEN dx.node_ix  IS NOT NULL THEN dx.node_ix
                             ELSE nx.branch_ix
                         END NULLS LAST,
                         cl.attached_docs
            )                                                           AS lr
        FROM compressed_linkage cl
        JOIN branch_ix bx ON cl.branch_id = bx.branch_id
        LEFT JOIN node_ix  dx ON cl.node_id  = dx.node_id
        LEFT JOIN branch_ix nx ON cl.node_id  = nx.branch_id
    )

    SELECT * FROM indexed_linkage;

-- ---------------------------------------------------------------------------
-- Final: the raw PG contract (INV-14).  gcs:1-3, gcs:106-120
--   RETURNS TABLE(tree_id uuid, branch_id uuid, branch_ix bigint,
--   l_node_ix bigint, r_node_ix bigint, l_node_id uuid, r_node_id uuid,
--   l_edge_ix integer, r_edge_ix integer, doc_count bigint,
--   coph_distance double precision)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE MACRO treedb.get_compressed_subtree(_orchard_ids) AS TABLE
    WITH
    indexed_linkage AS (
        SELECT * FROM treedb.cs_indexed_linkage(_orchard_ids)
    )

    SELECT                                                          -- gcs:106-117
        CAST(tree_id   AS UUID)   AS tree_id,
        CAST(branch_id AS UUID)   AS branch_id,
        CAST(branch_ix AS BIGINT) AS branch_ix,
        CAST(min(CASE WHEN lr = 1 THEN node_ix ELSE NULL END) AS BIGINT)    AS l_node_ix,
        CAST(min(CASE WHEN lr = 2 THEN node_ix ELSE NULL END) AS BIGINT)    AS r_node_ix,
        -- PG takes min() of node_id::text and casts back (uuid has no min in
        -- PG). Kept literally; at most one row per side, so it is exact.
        CAST(min(CASE WHEN lr = 1 THEN CAST(node_id AS VARCHAR) ELSE NULL END) AS UUID) AS l_node_id,
        CAST(min(CASE WHEN lr = 2 THEN CAST(node_id AS VARCHAR) ELSE NULL END) AS UUID) AS r_node_id,
        CAST(min(CASE WHEN lr = 1 THEN edge_ix ELSE NULL END) AS INTEGER)   AS l_edge_ix,
        CAST(min(CASE WHEN lr = 2 THEN edge_ix ELSE NULL END) AS INTEGER)   AS r_edge_ix,
        -- PG sum(bigint) is numeric, coerced to the declared bigint; DuckDB
        -- sum(BIGINT) is HUGEINT, cast to BIGINT [INT-1].
        CAST(sum(doc_count) AS BIGINT)                                      AS doc_count,
        CAST(coph_distance AS DOUBLE)                                       AS coph_distance
    FROM indexed_linkage il
    GROUP BY tree_id, branch_id, branch_ix, coph_distance           -- gcs:119
    -- gcs:120 ORDER BY coph_distance: rows of equal height have no order in
    -- PG. [DET-5] appends: unary (stem) rows after binary rows of the same
    -- height, then the left child's attached_docs (unique per row).
    ORDER BY coph_distance,
             (min(CASE WHEN lr = 2 THEN 1 ELSE NULL END) IS NULL),
             min(CASE WHEN lr = 1 THEN attached_docs ELSE NULL END);
