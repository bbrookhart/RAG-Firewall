"""
firewall/stages/injection_classifier.py
────────────────────────────────────────
Stage 1 — Injection Classifier

Two-layer detection:
  Layer A: Fast regex pre-filter (zero latency, high recall on known patterns)
  Layer B: Transformer classifier (DeBERTa-v3 or HF-compatible model)

The classifier is loaded lazily on first use and optionally exported to ONNX
for sub-15ms latency.  Falls back gracefully to regex-only if the model cannot
be loaded (useful in CI/CD or resource-constrained environments).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from models.schemas import InjectionScore

logger = structlog.get_logger(__name__)

# ── Heuristic Patterns ────────────────────────────────────────────────────────
# Ordered from highest to lowest signal strength.
# Each tuple: (pattern_name, compiled_regex)

_RAW_PATTERNS: list[tuple[str, str]] = [
    # Role / system override
    ("system_override",
     r"(?i)(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|earlier)"
     r"\s+(instructions?|prompts?|directives?|rules?|context)"),

    ("role_hijack",
     r"(?i)you\s+are\s+now\s+(a\s+|an\s+)?(different|new|another|unrestricted)?\s*"
     r"(ai|assistant|model|gpt|llm|bot|chatbot)"),

    ("act_as",
     r"(?i)(act\s+as|pretend\s+(you\s+are|to\s+be)|roleplay\s+as|behave\s+as)\s+"
     r"(a\s+|an\s+)?(?!user|customer|analyst)"),  # negative lookahead for benign uses

    # Special token injection
    ("special_tokens",
     r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"
     r"|\[INST\]|\[/INST\]|\[SYSTEM\]|<SYS>|</SYS>|<s>|</s>)"),

    # Credential / secret exfiltration
    ("exfiltration",
     r"(?i)(reveal|print|output|show|display|repeat|tell me)\s+"
     r"(your\s+)?(system\s+prompt|instructions?|api\s+key|secret|password|token)"),

    # Jailbreak phrases
    ("jailbreak",
     r"(?i)(jailbreak|bypass|circumvent|override|unlock|disable)\s+"
     r"(your\s+)?(safety|filter|guardrail|restriction|policy|alignment)"),

    # JSON/YAML system message injection
    ("json_system_inject",
     r'(?i)```\s*(json|yaml)?\s*\{[\s\S]*?"role"\s*:\s*"system"'),

    # DAN-style prompts
    ("dan_style",
     r"(?i)(do\s+anything\s+now|DAN\s+mode|developer\s+mode|god\s+mode|"
     r"unrestricted\s+mode|no\s+filter\s+mode)"),

    # Delimiter manipulation
    ("delimiter_escape",
     r'(?i)("""|\'\'\').{0,50}(system|instruction|prompt)'),

    # Translation obfuscation attempt
    ("translate_inject",
     r"(?i)translate\s+(the\s+)?following.{0,100}(ignore|override|system)"),

    # URL/link injection
    ("url_inject",
     r"(?i)fetch\s+(from\s+|the\s+)?(url|http|https|ftp)"),
]

COMPILED_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for name, pattern in _RAW_PATTERNS
]


def _heuristic_scan(text: str) -> tuple[bool, str | None, float]:
    """
    Fast regex pre-filter.
    Returns (flagged, pattern_name, risk_score).
    Risk score is 1.0 if any pattern matches, else 0.0.
    """
    for name, pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return True, name, 1.0
    return False, None, 0.0


# ── Transformer Classifier ────────────────────────────────────────────────────

class _TransformerClassifier:
    """
    Wraps a HuggingFace text-classification pipeline.
    Loaded lazily; falls back to None if unavailable.
    """

    def __init__(self, model_id: str, onnx_path: Path | None = None) -> None:
        self.model_id = model_id
        self.onnx_path = onnx_path
        self._pipeline: Any = None
        self._available: bool | None = None   # None = not yet attempted

    def _load(self) -> bool:
        """Attempt to load the model.  Returns True on success."""
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_id,
                top_k=None,
                truncation=True,
                max_length=512,
                device=-1,   # CPU; change to 0 for CUDA
            )
            logger.info("injection_classifier.model_loaded", model_id=self.model_id)
            return True
        except Exception as exc:
            logger.warning(
                "injection_classifier.model_unavailable",
                model_id=self.model_id,
                error=str(exc),
                fallback="heuristic_only",
            )
            return False

    def score(self, text: str) -> float | None:
        """Return injection probability [0,1] or None if model unavailable."""
        if self._available is None:
            self._available = self._load()
        if not self._available or self._pipeline is None:
            return None

        try:
            results: list[dict] = self._pipeline(text[:2000])[0]  # type: ignore
            # Label names vary by model — look for INJECTION / LABEL_1 / positive
            for item in results:
                label = str(item.get("label", "")).upper()
                if any(k in label for k in ("INJECT", "LABEL_1", "POSITIVE", "MALICIOUS")):
                    return float(item["score"])
            # If no positive label found, return 1 - highest negative score
            return 1.0 - max(float(r["score"]) for r in results)
        except Exception as exc:
            logger.warning("injection_classifier.inference_error", error=str(exc))
            return None


# ── Public Stage Class ────────────────────────────────────────────────────────

@dataclass
class InjectionClassifier:
    """
    Stage 1: Two-layer injection detection.

    Configuration keys (from config yaml):
      model_id         str   HuggingFace model ID
      threshold        float Score above which a chunk is flagged
      hard_block       bool  True = BLOCK, False = QUARANTINE
      use_model        bool  Whether to attempt loading the transformer
    """

    model_id: str = "deepset/deberta-v3-base-injection"
    threshold: float = 0.80
    hard_block: bool = True
    use_model: bool = True
    onnx_path: Path | None = None

    _classifier: _TransformerClassifier = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._classifier = _TransformerClassifier(
            model_id=self.model_id,
            onnx_path=self.onnx_path,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "InjectionClassifier":
        onnx_raw = cfg.get("onnx_path")
        return cls(
            model_id=cfg.get("model_id", "deepset/deberta-v3-base-injection"),
            threshold=float(cfg.get("threshold", 0.80)),
            hard_block=bool(cfg.get("hard_block", True)),
            use_model=bool(cfg.get("use_model", True)),
            onnx_path=Path(onnx_raw) if onnx_raw else None,
        )

    async def score(self, text: str) -> InjectionScore:
        """
        Score a single chunk of text.
        Always runs heuristics; conditionally runs the transformer.
        """
        t0 = time.monotonic()

        # Layer A — heuristics
        heuristic_flagged, pattern_name, heuristic_score = _heuristic_scan(text)

        if heuristic_flagged:
            # Short-circuit: don't call the model if regex already caught it
            return InjectionScore(
                probability=1.0,
                flagged=True,
                pattern_matched=pattern_name,
                model_used="heuristic",
                evidence=f"Matched pattern: {pattern_name}",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )

        # Layer B — transformer
        model_prob: float | None = None
        model_used = "heuristic"

        if self.use_model:
            model_prob = self._classifier.score(text)
            if model_prob is not None:
                model_used = self.model_id

        # Combine: if model gave a result, use it; else default to 0.0 (heuristic clean)
        final_prob = model_prob if model_prob is not None else 0.0
        flagged = final_prob >= self.threshold

        return InjectionScore(
            probability=final_prob,
            flagged=flagged,
            pattern_matched=None,
            model_used=model_used,
            evidence=(
                f"Model score {final_prob:.3f} ≥ threshold {self.threshold}"
                if flagged and model_prob is not None
                else None
            ),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    async def initialize(self) -> None:
        """Warm up the model (optional — called at pipeline startup)."""
        if self.use_model:
            _ = self._classifier.score("warm-up text")

    async def close(self) -> None:
        pass
