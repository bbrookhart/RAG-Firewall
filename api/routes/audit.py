"""
api/routes/audit.py
───────────────────
GET  /audit/queries           — Paginated query audit log.
GET  /audit/queries/{id}      — Single query with all chunk decisions.
GET  /audit/quarantine        — Pending human-review queue.
PATCH /audit/quarantine/{id}  — Approve or block a quarantined chunk.
GET  /audit/stats             — Aggregate daily firewall statistics.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import AuditDBDep, AuthDep
from models.schemas import (
    AuditQueryFilter,
    AuditRecord,
    FirewallDecision,
    QuarantineRecord,
    QuarantineReviewRequest,
)
from storage.audit_db import AuditDB

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/queries",
    response_model=list[AuditRecord],
    summary="List query audit records",
)
async def list_queries(
    _auth: Annotated[str, AuthDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    decision: FirewallDecision | None = Query(default=None),
    client_id: str | None = Query(default=None),
) -> list[AuditRecord]:
    records = await audit_db.list_queries(
        limit=limit,
        offset=offset,
        decision=decision,
        client_id=client_id,
    )
    return records


@router.get(
    "/queries/{query_id}",
    response_model=AuditRecord,
    summary="Get a single query audit record",
)
async def get_query(
    query_id: str,
    _auth: Annotated[str, AuthDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
) -> AuditRecord:
    record = await audit_db.get_query(query_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    return record


@router.get(
    "/quarantine",
    response_model=list[QuarantineRecord],
    summary="List chunks pending human review",
)
async def list_quarantine(
    _auth: Annotated[str, AuthDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status_filter: FirewallDecision | None = Query(default=FirewallDecision.PENDING, alias="status"),
) -> list[QuarantineRecord]:
    return await audit_db.list_quarantine(limit=limit, offset=offset, status=status_filter)


@router.patch(
    "/quarantine/{record_id}",
    response_model=QuarantineRecord,
    summary="Approve or reject a quarantined chunk",
)
async def review_quarantine(
    record_id: str,
    body: QuarantineReviewRequest,
    _auth: Annotated[str, AuthDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
) -> QuarantineRecord:
    if body.decision not in (FirewallDecision.PASS, FirewallDecision.BLOCK):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="decision must be PASS or BLOCK",
        )
    record = await audit_db.resolve_quarantine(
        record_id=record_id,
        decision=body.decision,
        reviewer_notes=body.reviewer_notes,
        reviewed_by=body.reviewed_by,
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"Quarantine record {record_id} not found")

    logger.info(
        "quarantine.resolved",
        record_id=record_id,
        decision=body.decision,
        reviewed_by=body.reviewed_by,
    )
    return record


@router.get(
    "/stats",
    summary="Daily aggregate firewall statistics",
)
async def firewall_stats(
    _auth: Annotated[str, AuthDep],
    audit_db: Annotated[AuditDB, AuditDBDep],
    days: int = Query(default=7, ge=1, le=90),
) -> list[dict]:
    return await audit_db.daily_stats(days=days)
