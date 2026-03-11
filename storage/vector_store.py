"""
storage/vector_store.py
────────────────────────
Qdrant vector store integration.

Responsibilities:
  - Initialize collections (chunks + quarantine) with correct schema
  - Embed text using sentence-transformers
  - Upsert, search, delete chunks
  - Store quarantined chunks in isolated collection
"""

from __future__ import annotations

import asyncio
import time
import uuid
from functools import lru_cache
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from models.schemas import ChunkMetadata, RetrievedChunk, ScoredChunk

logger = structlog.get_logger(__name__)

# ── Qdrant payload field names ────────────────────────────────────────────────
FIELD_CONTENT = "content"
FIELD_DOCUMENT_ID = "document_id"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_SOURCE_URI = "source_uri"
FIELD_SOURCE_HASH = "source_hash"
FIELD_CONTENT_HASH = "content_hash"
FIELD_TITLE = "title"
FIELD_TRUST_LEVEL = "trust_level"
FIELD_TAGS = "tags"
FIELD_INGESTED_AT = "ingested_at"
FIELD_PUBLISHED_AT = "published_at"


class VectorStore:
    """Async Qdrant client wrapper with embedding support."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        collection: str = "ragfw_chunks",
        quarantine_collection: str = "ragfw_quarantine",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ) -> None:
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection = collection
        self.quarantine_collection = quarantine_collection
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim

        self._client: AsyncQdrantClient | None = None
        self._embedder: SentenceTransformer | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Connect to Qdrant and ensure collections exist."""
        self._client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            timeout=30,
        )

        # Load embedding model (CPU; run in thread pool to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        self._embedder = await loop.run_in_executor(
            None, lambda: SentenceTransformer(self.embedding_model_name)
        )
        logger.info("vector_store.embedder_loaded", model=self.embedding_model_name)

        await self._ensure_collection(self.collection)
        await self._ensure_collection(self.quarantine_collection)
        logger.info("vector_store.initialized", collection=self.collection)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def ping(self) -> None:
        """Health check — raises on failure."""
        assert self._client is not None
        await self._client.get_collections()

    # ── Collection Management ─────────────────────────────────────────────────

    async def _ensure_collection(self, name: str) -> None:
        """Create collection if it doesn't exist."""
        assert self._client is not None
        try:
            await self._client.get_collection(name)
            logger.debug("vector_store.collection_exists", collection=name)
        except (UnexpectedResponse, Exception):
            await self._client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
            )

            # Create payload indexes for fast metadata filtering
            for field, schema_type in [
                (FIELD_DOCUMENT_ID, qmodels.PayloadSchemaType.KEYWORD),
                (FIELD_SOURCE_HASH, qmodels.PayloadSchemaType.KEYWORD),
                (FIELD_TRUST_LEVEL, qmodels.PayloadSchemaType.FLOAT),
                (FIELD_TAGS, qmodels.PayloadSchemaType.KEYWORD),
            ]:
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema_type,
                )
            logger.info("vector_store.collection_created", collection=name)

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        """Embed a single text string. Runs in thread pool."""
        assert self._embedder is not None
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(
            None, lambda: self._embedder.encode(text, normalize_embeddings=True).tolist()
        )
        return vector

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Runs in thread pool."""
        assert self._embedder is not None
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            lambda: self._embedder.encode(
                texts, normalize_embeddings=True, batch_size=32
            ).tolist(),
        )
        return vectors

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def upsert_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: ChunkMetadata,
    ) -> None:
        """Embed and store a chunk in the main collection."""
        assert self._client is not None

        vector = await self._embed(content)
        payload = {
            FIELD_CONTENT: content,
            FIELD_DOCUMENT_ID: metadata.document_id,
            FIELD_CHUNK_INDEX: metadata.chunk_index,
            FIELD_SOURCE_URI: metadata.source_uri,
            FIELD_SOURCE_HASH: metadata.source_hash,
            FIELD_CONTENT_HASH: metadata.content_hash,
            FIELD_TITLE: metadata.title,
            FIELD_TRUST_LEVEL: metadata.trust_level,
            FIELD_TAGS: metadata.tags,
            FIELD_INGESTED_AT: metadata.ingested_at.isoformat(),
            FIELD_PUBLISHED_AT: (
                metadata.published_at.isoformat() if metadata.published_at else None
            ),
        }

        await self._client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def store_quarantined(self, scored: ScoredChunk) -> None:
        """Store a flagged chunk in the quarantine collection."""
        assert self._client is not None

        vector = await self._embed(scored.content)
        payload = {
            FIELD_CONTENT: scored.content,
            FIELD_DOCUMENT_ID: scored.metadata.document_id,
            FIELD_CHUNK_INDEX: scored.metadata.chunk_index,
            FIELD_SOURCE_URI: scored.metadata.source_uri,
            FIELD_SOURCE_HASH: scored.metadata.source_hash,
            FIELD_CONTENT_HASH: scored.metadata.content_hash,
            "decision": scored.decision.value,
            "composite_risk": scored.composite_risk,
            "blocking_stage": scored.blocking_stage.value if scored.blocking_stage else None,
            "injection_probability": scored.injection.probability,
            "injection_pattern": scored.injection.pattern_matched,
        }

        await self._client.upsert(
            collection_name=self.quarantine_collection,
            points=[
                qmodels.PointStruct(
                    id=scored.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed query and return top-k similar chunks."""
        assert self._client is not None

        vector = await self._embed(query)

        # Convert caller-supplied filter dict to Qdrant Filter if provided
        qdrant_filter: qmodels.Filter | None = None
        if metadata_filter:
            conditions = []
            for field, value in metadata_filter.items():
                if isinstance(value, list):
                    conditions.append(
                        qmodels.FieldCondition(
                            key=field,
                            match=qmodels.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        qmodels.FieldCondition(
                            key=field,
                            match=qmodels.MatchValue(value=value),
                        )
                    )
            qdrant_filter = qmodels.Filter(must=conditions)

        results = await self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            meta = ChunkMetadata(
                document_id=payload.get(FIELD_DOCUMENT_ID, "unknown"),
                chunk_index=payload.get(FIELD_CHUNK_INDEX, 0),
                source_uri=payload.get(FIELD_SOURCE_URI),
                source_hash=payload.get(FIELD_SOURCE_HASH, ""),
                content_hash=payload.get(FIELD_CONTENT_HASH, ""),
                title=payload.get(FIELD_TITLE),
                trust_level=float(payload.get(FIELD_TRUST_LEVEL, 1.0)),
                tags=payload.get(FIELD_TAGS, []),
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(hit.id),
                    content=payload.get(FIELD_CONTENT, ""),
                    score=float(hit.score),
                    metadata=meta,
                )
            )

        return chunks

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        assert self._client is not None

        # Qdrant doesn't return delete counts directly; scroll first
        scroll_results, _ = await self._client.scroll(
            collection_name=self.collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=FIELD_DOCUMENT_ID,
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=10_000,
            with_payload=False,
        )
        ids = [str(r.id) for r in scroll_results]

        if ids:
            await self._client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(points=ids),
            )

        return len(ids)

    async def collection_stats(self) -> dict[str, Any]:
        """Return basic collection statistics."""
        assert self._client is not None
        info = await self._client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
