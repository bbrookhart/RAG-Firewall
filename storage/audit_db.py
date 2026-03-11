"""
storage/audit_db.py
────────────────────
Async PostgreSQL audit logging via SQLAlchemy 2.0 + asyncpg.

Every retrieval decision, query event, stage score, and quarantine
action is durably recorded here for compliance, review, and analysis.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from models.schemas import (
    AuditRecord,
    FirewallDecision,
    QuarantineRecord,
    ScoredChunk,
    StageName,
)

logger = structlog.get_logger(__name__)


class AuditDB:
    """Thin async wrapper around PostgreSQL for audit persistence."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._engine = create_async_engine(
            dsn,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Verify connectivity."""
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("audit_db.connected")

    async def close(self) -> None:
        await self._engine.dispose()

    async def ping(self) -> None:
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    # ── Ingestion Logging ─────────────────────────────────────────────────────

    async def log_ingestion(
        self,
        document_id: str,
        source_uri: str | None,
        source_hash: str,
        chunk_count: int,
        chunks_created: int,
        duration_ms: int,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO ingestion_events
                        (id, document_id, source_uri, source_hash,
                         chunk_index, chunk_count, ingest_duration_ms)
                    VALUES
                        (:id, :document_id, :source_uri, :source_hash,
                         0, :chunk_count, :duration_ms)
                """),
                {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "source_uri": source_uri,
                    "source_hash": source_hash,
                    "chunk_count": chunk_count,
                    "duration_ms": duration_ms,
                },
            )
            await session.commit()

    # ── Query Logging ─────────────────────────────────────────────────────────

    async def log_query(
        self,
        query_id: str,
        query_text: str,
        client_id: str | None,
        chunks_retrieved: int,
        chunks_passed: int,
        chunks_blocked: int,
        chunks_quarantined: int,
        total_latency_ms: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO query_events
                        (id, query_text, query_hash, client_id,
                         chunks_retrieved, chunks_passed, chunks_blocked,
                         chunks_quarantined, total_latency_ms, metadata)
                    VALUES
                        (:id, :query_text, :query_hash, :client_id,
                         :chunks_retrieved, :chunks_passed, :chunks_blocked,
                         :chunks_quarantined, :total_latency_ms, :metadata::jsonb)
                """),
                {
                    "id": query_id,
                    "query_text": query_text,
                    "query_hash": query_hash,
                    "client_id": client_id,
                    "chunks_retrieved": chunks_retrieved,
                    "chunks_passed": chunks_passed,
                    "chunks_blocked": chunks_blocked,
                    "chunks_quarantined": chunks_quarantined,
                    "total_latency_ms": total_latency_ms,
                    "metadata": str(metadata or {}),
                },
            )
            await session.commit()

    # ── Retrieval Event Logging ───────────────────────────────────────────────

    async def log_retrieval_event(self, query_id: str, scored: ScoredChunk) -> str:
        """Log a single chunk's firewall decision. Returns retrieval_event_id."""
        event_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            # Main retrieval event
            await session.execute(
                text("""
                    INSERT INTO retrieval_events
                        (id, query_id, chunk_id, source_uri, retrieval_score,
                         final_decision, composite_risk, chunk_content)
                    VALUES
                        (:id, :query_id, :chunk_id, :source_uri, :retrieval_score,
                         :final_decision, :composite_risk, :chunk_content)
                """),
                {
                    "id": event_id,
                    "query_id": query_id,
                    "chunk_id": scored.chunk_id,
                    "source_uri": scored.metadata.source_uri,
                    "retrieval_score": scored.retrieval_score,
                    "final_decision": scored.decision.value,
                    "composite_risk": scored.composite_risk,
                    "chunk_content": scored.content[:4000],  # cap storage
                },
            )

            # Stage scores
            for summary in scored.stage_summaries:
                await session.execute(
                    text("""
                        INSERT INTO stage_scores
                            (id, retrieval_event_id, stage, score, flagged, evidence, latency_ms)
                        VALUES
                            (:id, :retrieval_event_id, :stage, :score, :flagged, :evidence, :latency_ms)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "retrieval_event_id": event_id,
                        "stage": summary.stage.value,
                        "score": summary.score,
                        "flagged": summary.flagged,
                        "evidence": summary.evidence,
                        "latency_ms": summary.latency_ms,
                    },
                )
            await session.commit()
        return event_id

    # ── Quarantine ────────────────────────────────────────────────────────────

    async def log_quarantine(
        self,
        scored: ScoredChunk,
        query_id: str | None,
    ) -> str:
        """Add a chunk to the quarantine queue."""
        record_id = str(uuid.uuid4())
        reason = (
            scored.injection.evidence
            or f"Stage {scored.blocking_stage} flagged this chunk"
            if scored.blocking_stage
            else "Risk threshold exceeded"
        )
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO quarantine_queue
                        (id, chunk_id, reason, risk_score, stage, status)
                    VALUES
                        (:id, :chunk_id, :reason, :risk_score, :stage, 'PENDING')
                """),
                {
                    "id": record_id,
                    "chunk_id": scored.chunk_id,
                    "reason": reason or "Unknown",
                    "risk_score": scored.composite_risk,
                    "stage": (
                        scored.blocking_stage.value
                        if scored.blocking_stage
                        else StageName.INJECTION_CLASSIFIER.value
                    ),
                },
            )
            await session.commit()
        return record_id

    async def resolve_quarantine(
        self,
        record_id: str,
        decision: FirewallDecision,
        reviewer_notes: str | None = None,
        reviewed_by: str | None = None,
    ) -> QuarantineRecord | None:
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    UPDATE quarantine_queue
                    SET status = :status,
                        resolved_at = NOW(),
                        reviewer_notes = :notes,
                        reviewed_by = :reviewed_by
                    WHERE id = :id
                """),
                {
                    "id": record_id,
                    "status": decision.value,
                    "notes": reviewer_notes,
                    "reviewed_by": reviewed_by,
                },
            )
            await session.commit()

            row = await session.execute(
                text("SELECT * FROM quarantine_queue WHERE id = :id"),
                {"id": record_id},
            )
            rec = row.mappings().first()
            if not rec:
                return None
            return QuarantineRecord(
                id=str(rec["id"]),
                created_at=rec["created_at"],
                chunk_id=rec["chunk_id"],
                reason=rec["reason"],
                risk_score=rec["risk_score"],
                stage=StageName(rec["stage"]),
                status=FirewallDecision(rec["status"]),
                reviewer_notes=rec["reviewer_notes"],
            )

    # ── Query API ─────────────────────────────────────────────────────────────

    async def list_queries(
        self,
        limit: int = 50,
        offset: int = 0,
        decision: FirewallDecision | None = None,
        client_id: str | None = None,
    ) -> list[AuditRecord]:
        filters = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if client_id:
            filters.append("client_id = :client_id")
            params["client_id"] = client_id

        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        async with self._session_factory() as session:
            rows = await session.execute(
                text(f"""
                    SELECT id, created_at, query_hash, chunks_retrieved,
                           chunks_passed, chunks_blocked, chunks_quarantined,
                           total_latency_ms, metadata
                    FROM query_events
                    {where}
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                params,
            )
            records = []
            for r in rows.mappings():
                records.append(AuditRecord(
                    query_id=str(r["id"]),
                    created_at=r["created_at"],
                    query_hash=r["query_hash"],
                    chunks_retrieved=r["chunks_retrieved"] or 0,
                    chunks_passed=r["chunks_passed"] or 0,
                    chunks_blocked=r["chunks_blocked"] or 0,
                    chunks_quarantined=r["chunks_quarantined"] or 0,
                    total_latency_ms=r["total_latency_ms"] or 0,
                    metadata=r["metadata"] or {},
                ))
            return records

    async def get_query(self, query_id: str) -> AuditRecord | None:
        async with self._session_factory() as session:
            row = await session.execute(
                text("SELECT * FROM query_events WHERE id = :id"),
                {"id": query_id},
            )
            r = row.mappings().first()
            if not r:
                return None
            return AuditRecord(
                query_id=str(r["id"]),
                created_at=r["created_at"],
                query_hash=r["query_hash"],
                chunks_retrieved=r["chunks_retrieved"] or 0,
                chunks_passed=r["chunks_passed"] or 0,
                chunks_blocked=r["chunks_blocked"] or 0,
                chunks_quarantined=r["chunks_quarantined"] or 0,
                total_latency_ms=r["total_latency_ms"] or 0,
                metadata=r["metadata"] or {},
            )

    async def list_quarantine(
        self,
        limit: int = 50,
        offset: int = 0,
        status: FirewallDecision | None = None,
    ) -> list[QuarantineRecord]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        where = ""
        if status:
            where = "WHERE status = :status"
            params["status"] = status.value

        async with self._session_factory() as session:
            rows = await session.execute(
                text(f"""
                    SELECT * FROM quarantine_queue
                    {where}
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                params,
            )
            records = []
            for r in rows.mappings():
                records.append(QuarantineRecord(
                    id=str(r["id"]),
                    created_at=r["created_at"],
                    chunk_id=r["chunk_id"],
                    reason=r["reason"],
                    risk_score=r["risk_score"],
                    stage=StageName(r["stage"]),
                    status=FirewallDecision(r["status"]),
                    reviewer_notes=r.get("reviewer_notes"),
                ))
            return records

    async def daily_stats(self, days: int = 7) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            rows = await session.execute(
                text("""
                    SELECT
                        day::TEXT,
                        total_queries,
                        total_chunks_retrieved,
                        total_chunks_passed,
                        total_chunks_blocked,
                        total_chunks_quarantined,
                        avg_latency_ms,
                        block_rate_pct
                    FROM daily_firewall_stats
                    WHERE day >= CURRENT_DATE - INTERVAL ':days days'
                    ORDER BY day DESC
                """.replace(":days", str(days))),
            )
            return [dict(r) for r in rows.mappings()]
