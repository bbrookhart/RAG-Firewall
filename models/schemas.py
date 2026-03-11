"""
models/schemas.py
─────────────────
Pydantic v2 schemas shared across API, firewall, and storage layers.
Single source of truth for all data shapes.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class FirewallDecision(str, Enum):
    PASS = "PASS"
    QUARANTINE = "QUARANTINE"
    BLOCK = "BLOCK"
    PENDING = "PENDING"


class StageName(str, Enum):
    INJECTION_CLASSIFIER = "injection_classifier"
    TRUST_SCORER = "trust_scorer"
    CONSISTENCY_CHECKER = "consistency_checker"
    PROVENANCE_VALIDATOR = "provenance_validator"
    OUTPUT_GUARD = "output_guard"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


# ── Document Ingestion ────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Caller-supplied metadata attached to every ingested document."""
    source_uri: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    content_type: str = "text/plain"
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request body for POST /ingest/document"""
    content: str = Field(..., min_length=10, description="Raw document text")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> "IngestRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class IngestResponse(BaseModel):
    """Response from POST /ingest/document"""
    document_id: str
    chunks_created: int
    chunks_quarantined: int
    chunks_failed: int
    source_hash: str
    status: DocumentStatus
    duration_ms: int


# ── Chunk Representation ──────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata stored alongside each vector in Qdrant."""
    document_id: str
    chunk_index: int
    source_uri: str | None = None
    source_hash: str
    content_hash: str
    title: str | None = None
    published_at: datetime | None = None
    trust_level: float = 1.0          # Pre-assigned baseline trust [0,1]
    tags: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store, before firewall scoring."""
    chunk_id: str
    content: str
    score: float                       # Similarity score from Qdrant [0,1]
    metadata: ChunkMetadata


# ── Firewall Stage Results ────────────────────────────────────────────────────

class InjectionScore(BaseModel):
    """Output from Stage 1: Injection Classifier."""
    probability: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    pattern_matched: str | None = None   # Which regex fired, if any
    model_used: str = "heuristic"
    evidence: str | None = None
    latency_ms: int = 0


class TrustScore(BaseModel):
    """Output from Stage 2: Trust Scorer (stub in Phase 1)."""
    source_authority: float = 1.0
    freshness: float = 1.0
    corpus_agreement: float = 1.0
    retrieval_rank: float = 1.0
    composite: float = 1.0
    flagged: bool = False
    latency_ms: int = 0


class ConsistencyScore(BaseModel):
    """Output from Stage 3: Consistency Checker (stub in Phase 1)."""
    contradiction_detected: bool = False
    contradiction_confidence: float = 0.0
    conflicting_chunk_id: str | None = None
    latency_ms: int = 0


class ProvenanceScore(BaseModel):
    """Output from Stage 4: Provenance Validator (stub in Phase 1)."""
    domain_trusted: bool = True
    hash_verified: bool = False
    flagged: bool = False
    latency_ms: int = 0


class OutputGuardResult(BaseModel):
    """Output from Stage 5: Output Guard (stub in Phase 1)."""
    grounded: bool = True
    citation_valid: bool = True
    instruction_leak: bool = False
    pii_detected: bool = False
    approved: bool = True
    latency_ms: int = 0


class StageSummary(BaseModel):
    """Rolled-up summary of a single stage for a single chunk."""
    stage: StageName
    score: float | None = None
    flagged: bool = False
    evidence: str | None = None
    latency_ms: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


# ── Scored Chunk (post-firewall) ──────────────────────────────────────────────

class ScoredChunk(BaseModel):
    """A chunk after all active firewall stages have been applied."""
    chunk_id: str
    content: str
    retrieval_score: float
    metadata: ChunkMetadata

    # Stage results
    injection: InjectionScore
    trust: TrustScore
    consistency: ConsistencyScore
    provenance: ProvenanceScore

    # Final verdict
    composite_risk: float = Field(..., ge=0.0, le=1.0)
    decision: FirewallDecision
    blocking_stage: StageName | None = None   # Which stage triggered block/quarantine
    stage_summaries: list[StageSummary] = Field(default_factory=list)

    @property
    def safe(self) -> bool:
        return self.decision == FirewallDecision.PASS


# ── Query API ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    trust_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    include_audit: bool = Field(default=False, description="Return full audit trail in response")
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Qdrant metadata filter to pre-filter candidates"
    )


class FirewallStats(BaseModel):
    """Per-query firewall statistics."""
    chunks_retrieved: int
    chunks_passed: int
    chunks_blocked: int
    chunks_quarantined: int
    pipeline_latency_ms: int
    injection_detections: int


class QueryResponse(BaseModel):
    """Response from POST /query"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    safe_chunks: list[ScoredChunk]
    firewall_stats: FirewallStats
    audit_trail: list[ScoredChunk] | None = None   # Only if include_audit=True


# ── Audit API ─────────────────────────────────────────────────────────────────

class AuditQueryFilter(BaseModel):
    """Filter parameters for GET /audit/queries"""
    start_date: datetime | None = None
    end_date: datetime | None = None
    decision: FirewallDecision | None = None
    client_id: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class AuditRecord(BaseModel):
    """One audit record returned from the audit API."""
    query_id: str
    created_at: datetime
    query_hash: str
    chunks_retrieved: int
    chunks_passed: int
    chunks_blocked: int
    chunks_quarantined: int
    total_latency_ms: int
    metadata: dict[str, Any]


class QuarantineRecord(BaseModel):
    """A record in the human-review quarantine queue."""
    id: str
    created_at: datetime
    chunk_id: str
    reason: str
    risk_score: float
    stage: StageName
    status: FirewallDecision
    chunk_content: str | None = None
    reviewer_notes: str | None = None


class QuarantineReviewRequest(BaseModel):
    """Body for PATCH /audit/quarantine/{id}"""
    decision: FirewallDecision   # PASS or BLOCK
    reviewer_notes: str | None = None
    reviewed_by: str | None = None


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    name: str
    status: str   # "ok" | "degraded" | "down"
    latency_ms: int | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: list[ComponentHealth]
    uptime_seconds: float
