"""
firewall/pipeline.py
─────────────────────
FirewallPipeline — orchestrates all active stages for each chunk.

Design decisions:
  - Each stage is optional and toggled via config yaml.
  - Stages run sequentially; first BLOCK decision short-circuits.
  - QUARANTINE decisions accumulate (chunk still runs remaining stages).
  - Composite risk score = max risk signal across all stages.
  - Full audit data is attached to every ScoredChunk for downstream logging.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog
import yaml

from firewall.stages.consistency_checker import ConsistencyChecker
from firewall.stages.injection_classifier import InjectionClassifier
from firewall.stages.output_guard import OutputGuard
from firewall.stages.provenance_validator import ProvenanceValidator
from firewall.stages.trust_scorer import TrustScorer
from models.schemas import (
    ConsistencyScore,
    FirewallDecision,
    InjectionScore,
    OutputGuardResult,
    ProvenanceScore,
    RetrievedChunk,
    ScoredChunk,
    StageName,
    StageSummary,
    TrustScore,
)

logger = structlog.get_logger(__name__)


class FirewallPipeline:
    """
    Runs a retrieved chunk through all active defense stages and
    returns a ScoredChunk with a final firewall decision.
    """

    def __init__(
        self,
        injection_classifier: InjectionClassifier,
        trust_scorer: TrustScorer,
        consistency_checker: ConsistencyChecker,
        provenance_validator: ProvenanceValidator,
        output_guard: OutputGuard,
        hard_block_injections: bool = True,
        trust_threshold: float = 0.75,
    ) -> None:
        self.injection_classifier = injection_classifier
        self.trust_scorer = trust_scorer
        self.consistency_checker = consistency_checker
        self.provenance_validator = provenance_validator
        self.output_guard = output_guard
        self.hard_block_injections = hard_block_injections
        self.trust_threshold = trust_threshold

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: Path) -> "FirewallPipeline":
        """Load pipeline from a YAML configuration file."""
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        pipeline_cfg = cfg.get("pipeline", {})
        hard_block = pipeline_cfg.get("hard_block_injections", True)
        trust_threshold = pipeline_cfg.get("trust_threshold", 0.75)

        return cls(
            injection_classifier=InjectionClassifier.from_config(
                cfg.get("injection_classifier", {})
            ),
            trust_scorer=TrustScorer.from_config(
                cfg.get("trust_scorer", {})
            ),
            consistency_checker=ConsistencyChecker.from_config(
                cfg.get("consistency_checker", {})
            ),
            provenance_validator=ProvenanceValidator.from_config(
                cfg.get("provenance_validator", {})
            ),
            output_guard=OutputGuard.from_config(
                cfg.get("output_guard", {})
            ),
            hard_block_injections=hard_block,
            trust_threshold=trust_threshold,
        )

    @classmethod
    def from_defaults(cls) -> "FirewallPipeline":
        """Create a pipeline with default settings (useful for testing)."""
        return cls(
            injection_classifier=InjectionClassifier(),
            trust_scorer=TrustScorer(),
            consistency_checker=ConsistencyChecker(),
            provenance_validator=ProvenanceValidator(),
            output_guard=OutputGuard(),
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Warm up all active stages."""
        await self.injection_classifier.initialize()
        await self.trust_scorer.initialize()
        await self.consistency_checker.initialize()
        await self.provenance_validator.initialize()
        await self.output_guard.initialize()
        logger.info("pipeline.initialized", stages=self.active_stage_names)

    async def close(self) -> None:
        await self.injection_classifier.close()
        await self.trust_scorer.close()
        await self.consistency_checker.close()
        await self.provenance_validator.close()
        await self.output_guard.close()

    @property
    def active_stage_names(self) -> list[str]:
        active = [StageName.INJECTION_CLASSIFIER.value]  # always active
        if self.trust_scorer.enabled:
            active.append(StageName.TRUST_SCORER.value)
        if self.consistency_checker.enabled:
            active.append(StageName.CONSISTENCY_CHECKER.value)
        if self.provenance_validator.enabled:
            active.append(StageName.PROVENANCE_VALIDATOR.value)
        if self.output_guard.enabled:
            active.append(StageName.OUTPUT_GUARD.value)
        return active

    # ── Core Scoring ──────────────────────────────────────────────────────────

    async def score_chunk(
        self,
        chunk: RetrievedChunk,
        sibling_chunks: list[RetrievedChunk] | None = None,
    ) -> ScoredChunk:
        """
        Run a single chunk through all active stages.
        Returns a ScoredChunk with decision and full audit trail.
        """
        t0 = time.monotonic()
        summaries: list[StageSummary] = []
        decision = FirewallDecision.PASS
        blocking_stage: StageName | None = None
        risk_scores: list[float] = []

        # ── Stage 1: Injection Classifier (always runs) ───────────────────────
        injection: InjectionScore = await self.injection_classifier.score(chunk.content)
        summaries.append(StageSummary(
            stage=StageName.INJECTION_CLASSIFIER,
            score=injection.probability,
            flagged=injection.flagged,
            evidence=injection.evidence,
            latency_ms=injection.latency_ms,
            raw=injection.model_dump(),
        ))
        risk_scores.append(injection.probability)

        if injection.flagged:
            stage_decision = (
                FirewallDecision.BLOCK if self.hard_block_injections
                else FirewallDecision.QUARANTINE
            )
            if decision == FirewallDecision.PASS:
                decision = stage_decision
                blocking_stage = StageName.INJECTION_CLASSIFIER
            logger.warning(
                "pipeline.injection_detected",
                chunk_id=chunk.chunk_id,
                probability=injection.probability,
                pattern=injection.pattern_matched,
                decision=stage_decision,
            )
            # Short-circuit on BLOCK
            if stage_decision == FirewallDecision.BLOCK:
                return self._build_scored_chunk(
                    chunk, injection,
                    TrustScore(), ConsistencyScore(), ProvenanceScore(),
                    risk_scores, decision, blocking_stage, summaries,
                )

        # ── Stage 2: Trust Scorer ─────────────────────────────────────────────
        trust: TrustScore = await self.trust_scorer.score(chunk)
        summaries.append(StageSummary(
            stage=StageName.TRUST_SCORER,
            score=trust.composite,
            flagged=trust.flagged,
            latency_ms=trust.latency_ms,
            raw=trust.model_dump(),
        ))
        risk_scores.append(1.0 - trust.composite)

        if trust.flagged and decision == FirewallDecision.PASS:
            decision = FirewallDecision.QUARANTINE
            blocking_stage = StageName.TRUST_SCORER

        # ── Stage 3: Consistency Checker ──────────────────────────────────────
        consistency: ConsistencyScore = await self.consistency_checker.score(
            chunk, sibling_chunks
        )
        summaries.append(StageSummary(
            stage=StageName.CONSISTENCY_CHECKER,
            score=consistency.contradiction_confidence,
            flagged=consistency.contradiction_detected,
            latency_ms=consistency.latency_ms,
            raw=consistency.model_dump(),
        ))
        if consistency.contradiction_detected:
            risk_scores.append(consistency.contradiction_confidence)
            if decision == FirewallDecision.PASS:
                decision = FirewallDecision.QUARANTINE
                blocking_stage = StageName.CONSISTENCY_CHECKER

        # ── Stage 4: Provenance Validator ─────────────────────────────────────
        provenance: ProvenanceScore = await self.provenance_validator.score(chunk)
        summaries.append(StageSummary(
            stage=StageName.PROVENANCE_VALIDATOR,
            score=0.0 if provenance.flagged else 1.0,
            flagged=provenance.flagged,
            latency_ms=provenance.latency_ms,
            raw=provenance.model_dump(),
        ))
        if provenance.flagged and decision == FirewallDecision.PASS:
            decision = FirewallDecision.QUARANTINE
            blocking_stage = StageName.PROVENANCE_VALIDATOR

        composite_risk = max(risk_scores) if risk_scores else 0.0

        scored = self._build_scored_chunk(
            chunk, injection, trust, consistency, provenance,
            risk_scores, decision, blocking_stage, summaries,
        )

        logger.debug(
            "pipeline.chunk_scored",
            chunk_id=chunk.chunk_id,
            decision=decision,
            composite_risk=round(composite_risk, 3),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        return scored

    def _build_scored_chunk(
        self,
        chunk: RetrievedChunk,
        injection: InjectionScore,
        trust: TrustScore,
        consistency: ConsistencyScore,
        provenance: ProvenanceScore,
        risk_scores: list[float],
        decision: FirewallDecision,
        blocking_stage: StageName | None,
        summaries: list[StageSummary],
    ) -> ScoredChunk:
        composite_risk = max(risk_scores) if risk_scores else 0.0
        return ScoredChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            retrieval_score=chunk.score,
            metadata=chunk.metadata,
            injection=injection,
            trust=trust,
            consistency=consistency,
            provenance=provenance,
            composite_risk=composite_risk,
            decision=decision,
            blocking_stage=blocking_stage,
            stage_summaries=summaries,
        )
