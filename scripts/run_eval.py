#!/usr/bin/env python3
"""
scripts/run_eval.py
────────────────────
Run the adversarial evaluation suite against the firewall pipeline.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --no-model      # heuristics only (no HF download)
    python scripts/run_eval.py --output results/eval_$(date +%Y%m%d).json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import evaluate_pipeline
from firewall.pipeline import FirewallPipeline
from firewall.stages.injection_classifier import InjectionClassifier


async def main(use_model: bool, output_path: Path | None) -> None:
    print("\n🔥 RAG Firewall — Adversarial Evaluation Suite")
    print(f"   Mode: {'transformer + heuristic' if use_model else 'heuristic only'}\n")

    pipeline = FirewallPipeline.from_defaults()
    # Override injection classifier if requested
    if not use_model:
        pipeline.injection_classifier = InjectionClassifier(use_model=False)

    await pipeline.initialize()

    result = await evaluate_pipeline(pipeline)
    print(result.summary())

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "heuristic_only" if not use_model else "heuristic+model",
            "attack_success_rate": result.attack_success_rate,
            "block_rate": result.block_rate,
            "false_positive_rate": result.false_positive_rate,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "mean_latency_ms": result.mean_latency_ms,
            "total_attacks": result.total_attacks,
            "total_benign": result.total_benign,
            "missed_attacks": result.missed_attacks,
            "false_positives": result.false_positives,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n📄 Report saved to {output_path}")

    await pipeline.close()

    # Exit non-zero if targets missed
    if result.attack_success_rate > 0.02:
        print(f"\n❌ FAIL: ASR {result.attack_success_rate:.1%} exceeds 2% target")
        sys.exit(1)
    if result.false_positive_rate > 0.03:
        print(f"\n❌ FAIL: FPR {result.false_positive_rate:.1%} exceeds 3% target")
        sys.exit(1)
    print("\n✅ All targets met.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Firewall evaluation runner")
    parser.add_argument("--no-model", action="store_true", help="Disable transformer classifier")
    parser.add_argument("--output", type=Path, default=None, help="Save JSON report to path")
    args = parser.parse_args()

    asyncio.run(main(use_model=not args.no_model, output_path=args.output))
