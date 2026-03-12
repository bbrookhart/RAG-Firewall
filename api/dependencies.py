"""
api/dependencies.py
───────────────────
FastAPI dependency functions injected into route handlers.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from api.config import settings
from firewall.pipeline import FirewallPipeline
from storage.audit_db import AuditDB
from storage.vector_store import VectorStore

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Auth ──────────────────────────────────────────────────────────────────────

async def require_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    """Validate the X-API-Key header.  Skip check in development mode."""
    if settings.env == "development":
        return "dev"
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ── Resource accessors ────────────────────────────────────────────────────────

def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_audit_db(request: Request) -> AuditDB:
    return request.app.state.audit_db


def get_pipeline(request: Request) -> FirewallPipeline:
    return request.app.state.pipeline


# ── Convenience bundles ───────────────────────────────────────────────────────

VectorStoreDep = Depends(get_vector_store)
AuditDBDep = Depends(get_audit_db)
PipelineDep = Depends(get_pipeline)
AuthDep = Depends(require_api_key)
