-- TreeDB cohort store (phase P4A). TreeDB-native: no PostgreSQL predecessor.
--
-- A cohort is an immutable bounded set of item_ids plus a handle (INV-17).
-- Rows are only ever INSERTed; treedb.cohort has no UPDATE or DELETE path in
-- the API (DuckDB has no triggers, so immutability is an API rule, tested in
-- tests/cohort/test_cohort_store.py).
--
-- Idempotent: cohort.install(con) runs this on every open.
--   * tables / sequence: IF NOT EXISTS;  macros: CREATE OR REPLACE.
-- No foreign keys (same reason as schema.sql FK-1). Item references are
-- checked in cohort.py (unknown ids raise) and by cohort.integrity_violations.
-- Requires treedb.item_registry and treedb.tree_registry (store.install_schema).

CREATE SCHEMA IF NOT EXISTS treedb;

-- Monotone ordering key (not wall-clock). Survives close and reopen of a file.
CREATE SEQUENCE IF NOT EXISTS treedb.cohort_seq START 1;

CREATE TABLE IF NOT EXISTS treedb.cohort (
    cohort_id               UUID    NOT NULL,
    -- SHA-256 hex of the sorted item_ids (see cohort.membership_hash)
    membership_hash         TEXT    NOT NULL,
    size                    BIGINT  NOT NULL,
    kind                    TEXT    NOT NULL,
    base_cohort_id          UUID,             -- left operand / Execute base; NULL for root cohorts
    operands                JSON    NOT NULL, -- kind-specific provenance (object)
    expression              JSON,             -- compiled query; filled by P5B
    cultivar_state_versions JSON    NOT NULL, -- {tree_id: {"name":..., "state_version":...}} at creation (D-007)
    created_seq             BIGINT  NOT NULL DEFAULT nextval('treedb.cohort_seq'),
    CONSTRAINT cohort_pkey PRIMARY KEY (cohort_id),
    CONSTRAINT cohort_created_seq_key UNIQUE (created_seq),
    CONSTRAINT cohort_kind_check CHECK (kind IN
        ('all_items', 'from_ids', 'union', 'intersect', 'difference', 'execute', 'external'))
);

CREATE TABLE IF NOT EXISTS treedb.cohort_member (
    cohort_id UUID NOT NULL,
    item_id   TEXT NOT NULL,
    CONSTRAINT cohort_member_pkey PRIMARY KEY (cohort_id, item_id)
);

-- Plain-SQL access: SELECT * FROM treedb.cohort_items('<cohort uuid>')
-- The argument may be a UUID or its text form.
CREATE OR REPLACE MACRO treedb.cohort_items(cid) AS TABLE
    SELECT item_id FROM treedb.cohort_member WHERE cohort_id = CAST(cid AS UUID);

-- Cohort header row.
CREATE OR REPLACE MACRO treedb.cohort_info(cid) AS TABLE
    SELECT * FROM treedb.cohort WHERE cohort_id = CAST(cid AS UUID);
