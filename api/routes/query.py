"""
api/routes/query.py
───────────────────
POST /query   — Retrieve chunks through the firewall and return safe context.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, status

from api.dependencies import AuditDBDep, AuthDep, PipelineDep, VectorStoreDep
from firewall.pipeline import FirewallPipeline
from models.schemas import (
    FirewallDecision,
    FirewallStats,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
)
from storage.audit_db import AuditDB
from storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve chunks through the RAG Firewall",
    description=(
        "Embeds the query, retrieves candidate chunks from Qdrant, "
        "runs every active firewall stage, and returns only the chunks "
        "that passed all checks. Blocked and quarantined chunks are "
        "logged to the audit database."
    ),
)
async def protected_query(
    body: QueryRequest,
    _auth: Annotated[str, AuthDep],
    vector_store: Annotated[VectorStore, VectorStoreDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
    pipeline: Annotated[FirewallPipeline, PipelineDep],
) -> QueryResponse:
    t_start = time.monotonic()
    query_id = str(uuid.uuid4())

    logger.info("query.start", query_id=query_id, top_k=body.top_k)

    # ── Retrieve candidates from vector store ─────────────────────────────────
    raw_chunks = await vector_store.search(
        query=body.query,
        top_k=body.top_k,
        metadata_filter=body.metadata_filter,
    )

    # ── Run firewall pipeline on each chunk ───────────────────────────────────
    scored_chunks: list[ScoredChunk] = []
    for chunk in raw_chunks:
        scored = await pipeline.score_chunk(chunk)
        scored_chunks.append(scored)

    # ── Partition results ─────────────────────────────────────────────────────
    safe_chunks = [c for c in scored_chunks if c.decision == FirewallDecision.PASS]
    blocked = [c for c in scored_chunks if c.decision == FirewallDecision.BLOCK]
    quarantined = [c for c in scored_chunks if c.decision == FirewallDecision.QUARANTINE]

    duration_ms = int((time.monotonic() - t_start) * 1000)

    # ── Audit log ─────────────────────────────────────────────────────────────
    await audit_db.log_query(
        query_id=query_id,
        query_text=body.query,
        client_id=None,
        chunks_retrieved=len(raw_chunks),
        chunks_passed=len(safe_chunks),
        chunks_blocked=len(blocked),
        chunks_quarantined=len(quarantined),
        total_latency_ms=duration_ms,
    )

    for chunk in blocked + quarantined:
        await audit_db.log_retrieval_event(query_id, chunk)

    # Log quarantined chunks to quarantine queue for human review
    for chunk in quarantined:
        await audit_db.log_quarantine(chunk, query_id=query_id)

    stats = FirewallStats(
        chunks_retrieved=len(raw_chunks),
        chunks_passed=len(safe_chunks),
        chunks_blocked=len(blocked),
        chunks_quarantined=len(quarantined),
        pipeline_latency_ms=duration_ms,
        injection_detections=sum(1 for c in scored_chunks if c.injection.flagged),
    )

    logger.info(
        "query.complete",
        query_id=query_id,
        retrieved=len(raw_chunks),
        passed=len(safe_chunks),
        blocked=len(blocked),
        quarantined=len(quarantined),
        duration_ms=duration_ms,
    )

    return QueryResponse(
        query_id=query_id,
        query=body.query,
        safe_chunks=safe_chunks,
        firewall_stats=stats,
        audit_trail=scored_chunks if body.include_audit else None,
    )
