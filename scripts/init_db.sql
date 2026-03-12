-- RAG Firewall — PostgreSQL Audit Schema
-- Run automatically on first container start via docker-entrypoint-initdb.d

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search on chunk content

-- ── Enums ─────────────────────────────────────────────────────────────────────
CREATE TYPE firewall_decision AS ENUM (
    'PASS',         -- Chunk passed all stages, sent to LLM
    'QUARANTINE',   -- Chunk flagged, held for human review
    'BLOCK',        -- Chunk hard-blocked, never reaches LLM
    'PENDING'       -- Awaiting human review decision
);

CREATE TYPE stage_name AS ENUM (
    'injection_classifier',
    'trust_scorer',
    'consistency_checker',
    'provenance_validator',
    'output_guard'
);

-- ── Ingestion Events ──────────────────────────────────────────────────────────
CREATE TABLE ingestion_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_uri      TEXT,                         -- File path or URL
    source_hash     TEXT NOT NULL,                -- SHA-256 of raw content
    document_id     TEXT NOT NULL,                -- Qdrant point ID (UUID string)
    chunk_index     INTEGER NOT NULL,
    chunk_count     INTEGER NOT NULL,
    byte_size       INTEGER,
    media_type      TEXT,
    metadata        JSONB DEFAULT '{}',
    ingest_duration_ms INTEGER
);

CREATE INDEX idx_ingestion_source_hash ON ingestion_events (source_hash);
CREATE INDEX idx_ingestion_document_id ON ingestion_events (document_id);
CREATE INDEX idx_ingestion_created_at  ON ingestion_events (created_at DESC);

-- ── Retrieval Events ──────────────────────────────────────────────────────────
-- One row per chunk evaluated during a query
CREATE TABLE retrieval_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_id            UUID NOT NULL,            -- Groups all chunks from one query
    chunk_id            TEXT NOT NULL,            -- Qdrant point ID
    source_uri          TEXT,
    retrieval_score     FLOAT NOT NULL,           -- Similarity score from Qdrant
    final_decision      firewall_decision NOT NULL,
    composite_risk      FLOAT,                   -- Aggregate risk score [0,1]
    chunk_content       TEXT,                    -- Stored if audit.log_chunk_content=true
    metadata            JSONB DEFAULT '{}'
);

CREATE INDEX idx_retrieval_query_id    ON retrieval_events (query_id);
CREATE INDEX idx_retrieval_created_at  ON retrieval_events (created_at DESC);
CREATE INDEX idx_retrieval_decision    ON retrieval_events (final_decision);
CREATE INDEX idx_retrieval_chunk_id    ON retrieval_events (chunk_id);

-- ── Stage Scores ──────────────────────────────────────────────────────────────
-- One row per stage per retrieval event
CREATE TABLE stage_scores (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    retrieval_event_id  UUID NOT NULL REFERENCES retrieval_events(id) ON DELETE CASCADE,
    stage               stage_name NOT NULL,
    score               FLOAT,                   -- Stage-specific score [0,1]
    flagged             BOOLEAN NOT NULL DEFAULT FALSE,
    evidence            TEXT,                    -- Human-readable reason
    latency_ms          INTEGER,
    raw_output          JSONB DEFAULT '{}'       -- Full model output for debugging
);

CREATE INDEX idx_stage_scores_retrieval ON stage_scores (retrieval_event_id);
CREATE INDEX idx_stage_scores_flagged   ON stage_scores (flagged) WHERE flagged = TRUE;

-- ── Query Events ──────────────────────────────────────────────────────────────
-- One row per user query (parent of retrieval_events)
CREATE TABLE query_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_text          TEXT NOT NULL,
    query_hash          TEXT NOT NULL,           -- SHA-256 for dedup / pattern analysis
    client_id           TEXT,                    -- API key prefix or session ID
    chunks_retrieved    INTEGER,
    chunks_passed       INTEGER,
    chunks_blocked      INTEGER,
    chunks_quarantined  INTEGER,
    total_latency_ms    INTEGER,
    llm_response_id     TEXT,                    -- Opaque ID if caller provides it
    metadata            JSONB DEFAULT '{}'
);

CREATE INDEX idx_query_created_at  ON query_events (created_at DESC);
CREATE INDEX idx_query_client_id   ON query_events (client_id);
CREATE INDEX idx_query_hash        ON query_events (query_hash);

-- ── Quarantine Queue ──────────────────────────────────────────────────────────
CREATE TABLE quarantine_queue (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ,
    retrieval_event_id  UUID REFERENCES retrieval_events(id),
    chunk_id            TEXT NOT NULL,
    reason              TEXT NOT NULL,
    risk_score          FLOAT NOT NULL,
    stage               stage_name NOT NULL,
    status              firewall_decision NOT NULL DEFAULT 'PENDING',
    reviewer_notes      TEXT,
    reviewed_by         TEXT
);

CREATE INDEX idx_quarantine_status     ON quarantine_queue (status) WHERE status = 'PENDING';
CREATE INDEX idx_quarantine_created_at ON quarantine_queue (created_at DESC);

-- ── Aggregate Stats View ──────────────────────────────────────────────────────
CREATE VIEW daily_firewall_stats AS
SELECT
    date_trunc('day', created_at)::DATE AS day,
    COUNT(*)                             AS total_queries,
    SUM(chunks_retrieved)                AS total_chunks_retrieved,
    SUM(chunks_passed)                   AS total_chunks_passed,
    SUM(chunks_blocked)                  AS total_chunks_blocked,
    SUM(chunks_quarantined)              AS total_chunks_quarantined,
    ROUND(AVG(total_latency_ms))         AS avg_latency_ms,
    ROUND(
        100.0 * SUM(chunks_blocked) / NULLIF(SUM(chunks_retrieved), 0), 2
    )                                    AS block_rate_pct
FROM query_events
GROUP BY 1
ORDER BY 1 DESC;

-- ── Comments ──────────────────────────────────────────────────────────────────
COMMENT ON TABLE ingestion_events  IS 'Audit trail for every document chunk ingested into the vector store';
COMMENT ON TABLE retrieval_events  IS 'Per-chunk firewall decisions made during query processing';
COMMENT ON TABLE stage_scores      IS 'Individual stage scores for each retrieval event';
COMMENT ON TABLE query_events      IS 'One record per user query; parent of retrieval_events';
COMMENT ON TABLE quarantine_queue  IS 'Human review queue for quarantined chunks';
