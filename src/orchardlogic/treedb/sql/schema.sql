-- TreeDB DuckDB schema (phase P3A): Orchard storage and import staging.
--
-- Port of the INSTALLED PostgreSQL graph.* Orchard objects (D-002, D-009):
--   reference/pg_installed/graph/table/{orchard_edge_types,tree_registry,tree_z,tree_branches,orchard}.sql
--   reference/pg_installed/graph/view/{v_tree_z,v_tree_map,v_orchard}.sql
-- Migrations cited for history: reference/graph-engine/migrations/014, 016, 017, 018.
-- Schema `treedb`, same table, column and view names as graph.* (D-006).
-- Every divergence is recorded in docs/PORTING_NOTES.md with the test that proves it.
--
-- This file must stay idempotent: store.install_schema() runs it on every open.
--   * tables:  CREATE TABLE IF NOT EXISTS
--   * views:   CREATE OR REPLACE VIEW
--   * seeds:   INSERT ... WHERE NOT EXISTS
-- The whole file is executed as one multi-statement script.
--
-- Timestamps (PORTING_NOTES "TS-1", D-017 option B, G2 2026-11-09): PG
-- `timestamp with time zone DEFAULT now()` columns are UTC `TIMESTAMP` here,
-- defaulting to `CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP)`: the transaction
-- start (as PG now(), NOW-1) as a UTC wall-clock value, independent of the
-- session TimeZone. No column is time-zone aware, so fetching any row into
-- Python never needs pytz.
--
-- Foreign keys: NONE are declared (PORTING_NOTES "FK-1"). PostgreSQL declares
-- ON DELETE CASCADE / ON DELETE SET NULL FKs, which DuckDB cannot express, and
-- DuckDB's plain FK check rejects the child-then-parent deletes of an atomic
-- detach and the UPDATE of any indexed column of a referenced row. The
-- references are checked instead by treedb.store.integrity_violations() and
-- the tests in tests/ingest/.

CREATE SCHEMA IF NOT EXISTS treedb;

-- ---------------------------------------------------------------------------
-- treedb.orchard_edge_types
-- PG: reference/pg_installed/graph/table/orchard_edge_types.sql:1-6 (014:56-61);
--     seed 014:64-71. No key in PG, so no key here either.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS treedb.orchard_edge_types (
    ix          INTEGER,
    etype       TEXT,
    source_path TEXT,
    notes       TEXT
);

-- PG seeds with `ON CONFLICT DO NOTHING` (014:70), which never conflicts on a
-- keyless table, so re-running 014 duplicates the rows. DuckDB rejects
-- ON CONFLICT without a key (P2A probe on-conflict-no-unique). TreeDB seeds
-- the same three rows once (PORTING_NOTES "ET-1").
INSERT INTO treedb.orchard_edge_types (ix, etype, source_path, notes)
SELECT v.ix, v.etype, v.source_path, v.notes
FROM (VALUES
    (1, 'leaf',            CAST(NULL AS TEXT), 'Leaf node — direct document attachment'),
    (2, 'internal_branch', CAST(NULL AS TEXT), 'Internal branch node — derived from Z-linkage clustering'),
    (3, 'external_graft',  CAST(NULL AS TEXT), 'External graft — attachment from a different branch or vault')
) AS v(ix, etype, source_path, notes)
WHERE NOT EXISTS (SELECT 1 FROM treedb.orchard_edge_types);

-- ---------------------------------------------------------------------------
-- treedb.tree_registry  (the cultivar registry)
-- PG: reference/pg_installed/graph/table/tree_registry.sql:1-13
--     (014:74-78, 016:50-60, 017:62-72).
-- Columns 1-5 are the PG columns in PG order. Columns 6+ are TreeDB additions
-- (PORTING_NOTES "TR-2"). Retention of vault_id / graph_version_id /
-- parent_tree_id: PORTING_NOTES "TR-1".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS treedb.tree_registry (
    id               UUID        NOT NULL DEFAULT gen_random_uuid(),
    vault_id         UUID,                       -- PG: NOT NULL REFERENCES public.vaults(id). Nullable, no FK (TR-1)
    created_at       TIMESTAMP   NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),  -- PG: timestamp with time zone DEFAULT now() (TS-1, D-017 B)
    graph_version_id UUID,                       -- PG 016: FK ON DELETE SET NULL. Kept nullable, no FK (TR-1)
    parent_tree_id   UUID,                       -- PG 017: self-FK ON DELETE SET NULL. Kept nullable, no FK (TR-1)
    -- TreeDB additions (TR-2)
    name             TEXT        NOT NULL,       -- INV-2 human cultivar name (Orchard tree_id string)
    root_branch_id   UUID        NOT NULL DEFAULT gen_random_uuid(),  -- import-time root id (ROOT-1, OD-3)
    state_version    BIGINT      NOT NULL DEFAULT 0,                  -- D-007 (Proposed): 0 = as imported
    z_sha256         TEXT,                       -- original-import provenance: hash of the imported Z
    item_order       VARCHAR[],                  -- original-import provenance: ordered leaf item_ids
    source           TEXT,                       -- original-import provenance: source artifact (JSON text)
    CONSTRAINT tree_registry_pkey PRIMARY KEY (id),
    CONSTRAINT tree_registry_name_key UNIQUE (name)
);
-- PG partial indexes ix_tree_registry_graph_version_id (016:58-60) and
-- ix_tree_registry_parent_tree_id (017:70-72) are not created: DuckDB has no
-- partial indexes (P2A probe partial-index) and they are performance-only
-- (PORTING_NOTES "TR-3").

-- ---------------------------------------------------------------------------
-- treedb.tree_z  (staging; the original imported Z is kept here)
-- PG: reference/pg_installed/graph/table/tree_z.sql:1-17 (014:85-101, after 018:65-82).
-- UUID columns are created as UUID directly (018 converted them in place).
-- FLOAT8 -> DOUBLE, INT8 -> BIGINT, BOOL -> BOOLEAN.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS treedb.tree_z (
    branch_id      BIGINT  NOT NULL,
    l_node         DOUBLE,
    r_node         DOUBLE,
    coph_distance  DOUBLE,
    doc_count      DOUBLE,
    l_doc_id       UUID,
    r_doc_id       UUID,
    tree_id        UUID    NOT NULL,
    l_leaf         BOOLEAN NOT NULL DEFAULT true,
    r_leaf         BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT tree_z_pkey PRIMARY KEY (tree_id, branch_id),
    CONSTRAINT tree_z_unique_l_doc_id UNIQUE (tree_id, l_doc_id),
    CONSTRAINT tree_z_unique_l_node   UNIQUE (tree_id, l_node),
    CONSTRAINT tree_z_unique_r_doc_id UNIQUE (tree_id, r_doc_id),
    CONSTRAINT tree_z_unique_r_node   UNIQUE (tree_id, r_node)
);

-- ---------------------------------------------------------------------------
-- treedb.tree_branches
-- PG: reference/pg_installed/graph/table/tree_branches.sql:1-9 (014:105-110).
-- PG FK tree_id -> tree_registry(id) ON DELETE CASCADE is not declared (FK-1).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS treedb.tree_branches (
    branch_id  UUID        NOT NULL DEFAULT gen_random_uuid(),
    tree_id    UUID        NOT NULL,
    graft_id   UUID        NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP   NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),  -- PG: timestamp with time zone DEFAULT now() (TS-1, D-017 B)
    CONSTRAINT tree_branches_graft_id_key UNIQUE (graft_id),
    CONSTRAINT tree_branches_pkey PRIMARY KEY (branch_id)
);

-- ---------------------------------------------------------------------------
-- treedb.orchard
-- PG: reference/pg_installed/graph/table/orchard.sql:1-10 (014:114-121).
-- PG FK branch_id -> tree_branches(branch_id) ON DELETE CASCADE is not
-- declared (FK-1): with it, an atomic detach is impossible in DuckDB 2.0
-- (P2A probe detach-one-txn).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS treedb.orchard (
    id            UUID    NOT NULL DEFAULT gen_random_uuid(),
    branch_id     UUID    NOT NULL,
    node_id       UUID    NOT NULL,
    edge_ix       INTEGER NOT NULL,
    coph_distance DOUBLE  NOT NULL,
    CONSTRAINT orchard_branch_node_key UNIQUE (branch_id, node_id),
    CONSTRAINT orchard_pkey PRIMARY KEY (id)
);

-- ---------------------------------------------------------------------------
-- TreeDB additions (not in graph.*): items registry, labels, cards.
-- PORTING_NOTES "IR-1", "LC-1".
-- ---------------------------------------------------------------------------

-- Shared leaf identities across cultivars (Arch §2 "Source of truth" item 3).
-- item_uuid is the CONTRACTS UUID: uuid5(FIXTURE_NAMESPACE, 'item:' || item_id) (INV-5).
CREATE TABLE IF NOT EXISTS treedb.item_registry (
    item_id    TEXT NOT NULL,
    item_uuid  UUID NOT NULL,
    source_ref TEXT,
    CONSTRAINT item_registry_pkey PRIMARY KEY (item_id),
    CONSTRAINT item_registry_item_uuid_key UNIQUE (item_uuid)
);

-- Labels keyed by node identity (tree_id, node_id). node_id is the node's
-- orchard identity: a branch_id for internal nodes and the root, the item
-- UUID for leaves. source_node_key is the Orchard source node key (the
-- member_ key at import, INV-3 / INV-11), kept as a separate column.
CREATE TABLE IF NOT EXISTS treedb.node_labels (
    tree_id         UUID NOT NULL,
    node_id         UUID NOT NULL,
    label_set       TEXT NOT NULL,
    label           TEXT NOT NULL,
    source_node_key TEXT NOT NULL,
    CONSTRAINT node_labels_pkey PRIMARY KEY (tree_id, node_id, label_set)
);

-- Cards keyed by node identity; card is JSON text (canonical_json).
CREATE TABLE IF NOT EXISTS treedb.node_cards (
    tree_id         UUID NOT NULL,
    node_id         UUID NOT NULL,
    card            TEXT NOT NULL,
    source_node_key TEXT NOT NULL,
    CONSTRAINT node_cards_pkey PRIMARY KEY (tree_id, node_id)
);

-- Documents (D-027, schema v3): the text a leaf item stands for. One row per
-- item. item_id names a registered item (checked by
-- store.integrity_violations, FK-1 style: no FK declared). Deliberately
-- minimal; "we can adapt as needed" (user, 2026-11-09). The source artifact's
-- other fields (for example `source`, `schema_version`) are not kept.
-- metadata is a JSON object written as canonical JSON text.
CREATE TABLE IF NOT EXISTS treedb.documents (
    item_id  TEXT NOT NULL,
    title    TEXT,
    text     TEXT NOT NULL,
    metadata JSON,
    CONSTRAINT documents_pkey PRIMARY KEY (item_id)
);

-- ---------------------------------------------------------------------------
-- treedb.v_tree_z
-- PG: reference/pg_installed/graph/view/v_tree_z.sql:1-35 (018:90-125).
-- Literal port. `(- 1::bigint)` binds as in PG (P2A probe neg-cast-precedence);
-- DOUBLE -> BIGINT casts round half to even as PG float8 -> int8 does
-- (P2A probe double-to-bigint-half); the inputs are integral anyway.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW treedb.v_tree_z AS
SELECT
    branch_id,
    l_node::bigint                  AS l_node,
    r_node::bigint                  AS r_node,
    coph_distance,
    doc_count::bigint               AS doc_count,
    l_doc_id,
    r_doc_id,
    tree_id,
    n_docs::bigint                  AS n_docs,
    CASE
        WHEN l_node < n_docs
            THEN l_node * (- 1::bigint)::double precision - 1::bigint::double precision
        ELSE l_node - n_docs
    END::bigint                     AS l_mod,
    CASE
        WHEN r_node < n_docs
            THEN r_node * (- 1::bigint)::double precision - 1::bigint::double precision
        ELSE r_node - n_docs
    END::bigint                     AS r_mod,
    CASE WHEN l_node < n_docs THEN true  ELSE false END AS l_leaf,
    CASE WHEN r_node < n_docs THEN true  ELSE false END AS r_leaf
FROM (
    SELECT
        tree_z.branch_id,
        tree_z.l_node,
        tree_z.r_node,
        tree_z.coph_distance,
        tree_z.doc_count,
        tree_z.l_doc_id,
        tree_z.r_doc_id,
        tree_z.tree_id,
        max(tree_z.doc_count) OVER (PARTITION BY tree_z.tree_id) AS n_docs
    FROM treedb.tree_z
) tz;

-- ---------------------------------------------------------------------------
-- treedb.v_tree_map
-- PG: reference/pg_installed/graph/view/v_tree_map.sql:1-42 (018:135-187).
-- Literal port EXCEPT the root UNION branch's doc_id (PG line 33:
-- `gen_random_uuid() AS doc_id`, VOLATILE, regenerated on every read; OD-3).
-- DuckDB reads the root id stored at import in tree_registry.root_branch_id
-- (deliberate divergence, PORTING_NOTES "ROOT-1"). LEFT JOIN keeps the root
-- row for every staged tree, as in PG; its doc_id is NULL only for a staged
-- tree without a registry row, which TreeDB's import never produces.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW treedb.v_tree_map AS
SELECT
    tree_id,
    doc_id,
    z_id,
    leaf,
    branch_id,
    z_mod,
    b_side,
    coph_distance,
    doc_count
FROM (
    SELECT
        z.tree_id,
        z.l_doc_id          AS doc_id,
        z.l_node            AS z_id,
        z.l_leaf            AS leaf,
        z.branch_id,
        z.l_mod             AS z_mod,
        'l'::text           AS b_side,
        z.coph_distance,
        z.doc_count
    FROM treedb.v_tree_z z

    UNION

    SELECT
        z.tree_id,
        z.r_doc_id          AS doc_id,
        z.r_node            AS z_id,
        z.r_leaf            AS leaf,
        z.branch_id,
        z.r_mod             AS z_mod,
        'r'::text           AS b_side,
        z.coph_distance,
        z.doc_count
    FROM treedb.v_tree_z z

    UNION

    SELECT
        v_tree_z.tree_id,
        r.root_branch_id                                             AS doc_id,
        max(GREATEST(v_tree_z.l_node, v_tree_z.r_node)) + 1         AS z_id,
        false                                                        AS leaf,
        max(v_tree_z.branch_id) + 1                                  AS branch_id,
        max(GREATEST(v_tree_z.l_mod, v_tree_z.r_mod)) + 1           AS z_mod,
        'root'::text                                                 AS b_side,
        max(v_tree_z.coph_distance) * 1.05::double precision         AS coph_distance,
        max(v_tree_z.doc_count)                                      AS doc_count
    FROM treedb.v_tree_z
    LEFT JOIN treedb.tree_registry r ON r.id = v_tree_z.tree_id
    GROUP BY v_tree_z.tree_id, r.root_branch_id
) sq;

-- ---------------------------------------------------------------------------
-- treedb.v_orchard
-- PG: reference/pg_installed/graph/view/v_orchard.sql:1-11 (014:226-238).
-- Literal port.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW treedb.v_orchard AS
SELECT
    b.graft_id,
    o.branch_id,
    o.node_id,
    o.edge_ix,
    o.coph_distance,
    b.tree_id,
    o.id        AS orchard_id,
    r.vault_id
FROM treedb.orchard o
JOIN treedb.tree_branches b USING (branch_id)
JOIN treedb.tree_registry  r ON r.id = b.tree_id;

-- ===========================================================================
-- Schema v4 (docs/DATA_ORCHARD_V4.md section 5, amendments A1, A4). All
-- additive: a v2 or v3 file is migrated in place by install_schema. No FKs
-- (FK-1); references are checked by store.integrity_violations. Never use a
-- reserved word as a column name (A1; the DDL is probed against DuckDB 2.0 in
-- tests/orchard_db/test_schema_v4.py).
-- ===========================================================================

CREATE TABLE IF NOT EXISTS treedb.perspective (
    perspective_id TEXT PRIMARY KEY,
    name TEXT NOT NULL, version TEXT NOT NULL,
    record JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    CONSTRAINT perspective_name_version_key UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS treedb.enricher (
    enricher_id TEXT PRIMARY KEY,
    name TEXT NOT NULL, version TEXT NOT NULL,
    scope TEXT NOT NULL,
    method TEXT NOT NULL,
    record JSON NOT NULL,
    code_sha256 TEXT, code_link TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    CONSTRAINT enricher_name_version_key UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS treedb.cultivar_artifact (
    artifact_sha256 TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    manifest JSON NOT NULL,
    orchard_manifest JSON NOT NULL,
    perspective_id TEXT NOT NULL,
    enricher_ids VARCHAR[] NOT NULL,
    cultivate_version TEXT,
    source_path TEXT,
    imported_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP)
);

CREATE TABLE IF NOT EXISTS treedb.cultivar_tree (
    tree_id UUID PRIMARY KEY,
    artifact_sha256 TEXT NOT NULL,
    artifact_tree_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS treedb.cultivar_alias (
    alias TEXT PRIMARY KEY,
    tree_id UUID NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP)
);

-- A4: node_key (opaque after import) -> orchard node identity.
CREATE TABLE IF NOT EXISTS treedb.cultivar_node (
    tree_id UUID NOT NULL, node_key TEXT NOT NULL, node_id UUID NOT NULL,
    PRIMARY KEY (tree_id, node_key)
);

CREATE TABLE IF NOT EXISTS treedb.vector (
    space TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec FLOAT[] NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    PRIMARY KEY (space, content_sha256)
);

CREATE TABLE IF NOT EXISTS treedb.item_vector (
    item_id TEXT NOT NULL, space TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY (item_id, space)
);

CREATE TABLE IF NOT EXISTS treedb.item_enrichment (
    item_id TEXT NOT NULL, enricher_id TEXT NOT NULL,
    output JSON NOT NULL,
    input_sha256 TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    PRIMARY KEY (item_id, enricher_id)
);

CREATE TABLE IF NOT EXISTS treedb.node_enrichment (
    tree_id UUID NOT NULL, node_key TEXT NOT NULL, enricher_id TEXT NOT NULL,
    output JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    PRIMARY KEY (tree_id, node_key, enricher_id)
);

CREATE TABLE IF NOT EXISTS treedb.cut (
    tree_id UUID NOT NULL, cut_name TEXT NOT NULL,
    settings JSON NOT NULL,
    revision BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    PRIMARY KEY (tree_id, cut_name)
);

CREATE TABLE IF NOT EXISTS treedb.cut_node (
    tree_id UUID NOT NULL, cut_name TEXT NOT NULL, node_key TEXT NOT NULL,
    parent_key TEXT, kind TEXT NOT NULL,
    depth INTEGER NOT NULL, position INTEGER NOT NULL,
    item_id TEXT, descendant_count INTEGER NOT NULL,
    path VARCHAR[] NOT NULL,
    title TEXT, summary TEXT, tags VARCHAR[],
    PRIMARY KEY (tree_id, cut_name, node_key)
);

CREATE TABLE IF NOT EXISTS treedb.ledger (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    occurred_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    session_id TEXT, actor TEXT, action TEXT NOT NULL,
    tree_id UUID, node_key TEXT, item_id TEXT, cohort_id TEXT,
    payload JSON
);

CREATE TABLE IF NOT EXISTS treedb.checkpoint (
    namespace TEXT NOT NULL, key TEXT NOT NULL,
    payload JSON NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CAST(now() AT TIME ZONE 'UTC' AS TIMESTAMP),
    PRIMARY KEY (namespace, key)
);

CREATE OR REPLACE VIEW treedb.v_item_time AS
SELECT item_id, TRY_CAST(metadata->>'captured_at' AS TIMESTAMP) AS captured_at
FROM treedb.documents;

CREATE OR REPLACE VIEW treedb.v_cut_items AS
SELECT c.tree_id, c.cut_name, c.item_id, c.node_key, c.parent_key, c.path, t.captured_at
FROM treedb.cut_node c
LEFT JOIN treedb.v_item_time t ON t.item_id = c.item_id
WHERE c.kind = 'item';
