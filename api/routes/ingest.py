"""
api/routes/ingest.py
────────────────────
POST /ingest/document  — Chunk, embed, pre-scan, and store a document.
POST /ingest/batch     — Bulk ingestion (up to 20 documents).
DELETE /ingest/{document_id} — Remove all chunks for a document.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from api.config import settings
from api.dependencies import AuditDBDep, AuthDep, PipelineDep, VectorStoreDep
from firewall.pipeline import FirewallPipeline
from models.schemas import (
    ChunkMetadata,
    DocumentStatus,
    FirewallDecision,
    IngestRequest,
    IngestResponse,
    RetrievedChunk,
)
from storage.audit_db import AuditDB
from storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple token-approximation chunker (character-based for Phase 1)."""
    # 1 token ≈ 4 chars — chunk_size is in tokens
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += char_size - char_overlap

    return chunks


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/document",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single document",
)
async def ingest_document(
    body: IngestRequest,
    _auth: Annotated[str, AuthDep],
    vector_store: Annotated[VectorStore, VectorStoreDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
    pipeline: Annotated[FirewallPipeline, PipelineDep],
) -> IngestResponse:
    t_start = time.monotonic()

    # Guard: max document size
    max_bytes = settings.max_document_size_mb * 1024 * 1024
    if len(body.content.encode()) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Document exceeds maximum size of {settings.max_document_size_mb}MB",
        )

    document_id = str(uuid.uuid4())
    source_hash = hashlib.sha256(body.content.encode()).hexdigest()

    logger.info(
        "ingest.start",
        document_id=document_id,
        source_uri=body.metadata.source_uri,
        source_hash=source_hash[:16],
    )

    # Chunk
    raw_chunks = _split_text(body.content, body.chunk_size, body.chunk_overlap)
    if not raw_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document produced no usable chunks after splitting",
        )

    chunks_created = 0
    chunks_quarantined = 0
    chunks_failed = 0

    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

        chunk_meta = ChunkMetadata(
            document_id=document_id,
            chunk_index=idx,
            source_uri=body.metadata.source_uri,
            source_hash=source_hash,
            content_hash=content_hash,
            title=body.metadata.title,
            published_at=body.metadata.published_at,
            trust_level=1.0,
            tags=body.metadata.tags,
        )

        # Pre-scan chunk through injection classifier before ingestion
        fake_chunk = RetrievedChunk(
            chunk_id=chunk_id,
            content=chunk_text,
            score=1.0,
            metadata=chunk_meta,
        )

        try:
            scored = await pipeline.score_chunk(fake_chunk)
        except Exception as exc:
            logger.error("ingest.score_error", chunk_id=chunk_id, error=str(exc))
            chunks_failed += 1
            continue

        if scored.decision == FirewallDecision.BLOCK:
            logger.warning(
                "ingest.chunk_blocked",
                document_id=document_id,
                chunk_index=idx,
                risk_score=scored.composite_risk,
                stage=scored.blocking_stage,
            )
            # Store in quarantine collection for review
            await vector_store.store_quarantined(scored)
            await audit_db.log_quarantine(scored, query_id=None)
            chunks_quarantined += 1
            continue

        if scored.decision == FirewallDecision.QUARANTINE:
            await vector_store.store_quarantined(scored)
            await audit_db.log_quarantine(scored, query_id=None)
            chunks_quarantined += 1
            continue

        # Safe chunk — embed and store
        try:
            await vector_store.upsert_chunk(chunk_id, chunk_text, chunk_meta)
            chunks_created += 1
        except Exception as exc:
            logger.error("ingest.upsert_error", chunk_id=chunk_id, error=str(exc))
            chunks_failed += 1

    # Audit the ingestion event
    duration_ms = int((time.monotonic() - t_start) * 1000)
    await audit_db.log_ingestion(
        document_id=document_id,
        source_uri=body.metadata.source_uri,
        source_hash=source_hash,
        chunk_count=len(raw_chunks),
        chunks_created=chunks_created,
        duration_ms=duration_ms,
    )

    final_status = (
        DocumentStatus.INDEXED if chunks_created > 0
        else DocumentStatus.QUARANTINED if chunks_quarantined > 0
        else DocumentStatus.FAILED
    )

    logger.info(
        "ingest.complete",
        document_id=document_id,
        chunks_created=chunks_created,
        chunks_quarantined=chunks_quarantined,
        chunks_failed=chunks_failed,
        duration_ms=duration_ms,
    )

    return IngestResponse(
        document_id=document_id,
        chunks_created=chunks_created,
        chunks_quarantined=chunks_quarantined,
        chunks_failed=chunks_failed,
        source_hash=source_hash,
        status=final_status,
        duration_ms=duration_ms,
    )


@router.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch ingest up to 20 documents",
)
async def ingest_batch(
    documents: list[IngestRequest],
    _auth: Annotated[str, AuthDep],
    vector_store: Annotated[VectorStore, VectorStoreDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
    pipeline: Annotated[FirewallPipeline, PipelineDep],
) -> dict:
    if len(documents) > 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size cannot exceed 20 documents",
        )

    results = []
    for doc in documents:
        result = await ingest_document(doc, "batch", vector_store, audit_db, pipeline)
        results.append(result.model_dump())

    return {
        "processed": len(results),
        "results": results,
    }


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove all chunks for a document",
)
async def delete_document(
    document_id: str,
    _auth: Annotated[str, AuthDep],
    vector_store: Annotated[VectorStore, VectorStoreDep],
) -> dict:
    deleted = await vector_store.delete_document(document_id)
    logger.info("ingest.deleted", document_id=document_id, chunks_deleted=deleted)
    return {"document_id": document_id, "chunks_deleted": deleted}
