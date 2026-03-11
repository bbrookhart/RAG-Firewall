"""
api/main.py
───────────
FastAPI application factory.  All startup/shutdown lifecycle hooks live here.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.logging_config import configure_logging
from api.middleware.rate_limiter import RateLimitMiddleware
from api.routes import audit, ingest, query
from storage.audit_db import AuditDB
from storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)

_START_TIME = time.monotonic()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize heavy resources once at startup; clean up at shutdown."""
    configure_logging(settings.log_level)
    logger.info("rag_firewall.startup", version=settings.version, env=settings.env)

    # Vector store
    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        quarantine_collection=settings.qdrant_quarantine_collection,
        embedding_dim=settings.embedding_dim,
    )
    await vector_store.initialize()
    app.state.vector_store = vector_store
    logger.info("rag_firewall.vector_store.ready", collection=settings.qdrant_collection)

    # Audit database
    audit_db = AuditDB(dsn=settings.postgres_dsn)
    await audit_db.initialize()
    app.state.audit_db = audit_db
    logger.info("rag_firewall.audit_db.ready")

    # Firewall pipeline — lazy import to avoid circular deps
    from firewall.pipeline import FirewallPipeline
    pipeline = FirewallPipeline.from_yaml(settings.config_path)
    await pipeline.initialize()
    app.state.pipeline = pipeline
    logger.info("rag_firewall.pipeline.ready", stages=pipeline.active_stage_names)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("rag_firewall.shutdown")
    await audit_db.close()
    await vector_store.close()
    await pipeline.close()


# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Firewall",
        description=(
            "Adversarial-grade defense layer for Retrieval-Augmented Generation systems. "
            "Scores retrieved chunks for injection risk, trust, consistency, and provenance "
            "before they reach your LLM."
        ),
        version=settings.version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ingest.router,  prefix="/ingest",  tags=["Ingestion"])
    app.include_router(query.router,   prefix="/query",   tags=["Query"])
    app.include_router(audit.router,   prefix="/audit",   tags=["Audit"])

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health(request: Request):
        from models.schemas import ComponentHealth, HealthResponse

        components: list[ComponentHealth] = []

        # Qdrant
        try:
            vs: VectorStore = request.app.state.vector_store
            t0 = time.monotonic()
            await vs.ping()
            components.append(ComponentHealth(
                name="qdrant", status="ok",
                latency_ms=int((time.monotonic() - t0) * 1000)
            ))
        except Exception as exc:
            components.append(ComponentHealth(name="qdrant", status="down", detail=str(exc)))

        # PostgreSQL
        try:
            db: AuditDB = request.app.state.audit_db
            t0 = time.monotonic()
            await db.ping()
            components.append(ComponentHealth(
                name="postgres", status="ok",
                latency_ms=int((time.monotonic() - t0) * 1000)
            ))
        except Exception as exc:
            components.append(ComponentHealth(name="postgres", status="down", detail=str(exc)))

        overall = "ok" if all(c.status == "ok" for c in components) else "degraded"
        return HealthResponse(
            status=overall,
            version=settings.version,
            components=components,
            uptime_seconds=round(time.monotonic() - _START_TIME, 2),
        )

    # ── Global Exception Handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
