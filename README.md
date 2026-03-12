<div align="center">

```
██████╗  █████╗  ██████╗    ███████╗██╗██████╗ ███████╗██╗    ██╗ █████╗ ██╗     ██╗
██╔══██╗██╔══██╗██╔════╝    ██╔════╝██║██╔══██╗██╔════╝██║    ██║██╔══██╗██║     ██║
██████╔╝███████║██║  ███╗   █████╗  ██║██████╔╝█████╗  ██║ █╗ ██║███████║██║     ██║
██╔══██╗██╔══██║██║   ██║   ██╔══╝  ██║██╔══██╗██╔══╝  ██║███╗██║██╔══██║██║     ██║
██║  ██║██║  ██║╚██████╔╝   ██║     ██║██║  ██║███████╗╚███╔███╔╝██║  ██║███████╗███████╗
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝╚══════╝
```

# 🔥 RAG Firewall

### **Adversarial-Grade Defense Layer for Retrieval-Augmented Generation Systems**

*Stops indirect prompt injection, poisoned knowledge, and retrieval manipulation — before they reach your LLM.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-1a1a2e?style=for-the-badge&logo=python&logoColor=4fc3f7)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-1a1a2e?style=for-the-badge&logo=fastapi&logoColor=4fc3f7)](https://fastapi.tiangolo.com)
[![OWASP](https://img.shields.io/badge/OWASP-Agentic_Top_10-1a1a2e?style=for-the-badge&logoColor=e94560)](https://owasp.org)
[![NIST](https://img.shields.io/badge/NIST-GenAI_Profile-1a1a2e?style=for-the-badge&logoColor=e94560)](https://airc.nist.gov)
[![Status](https://img.shields.io/badge/Status-Active_Research-1a1a2e?style=for-the-badge&logo=statuspage&logoColor=4ade80)](https://github.com)

<br/>

> **"RAG jamming works without knowing the target embedding model or LLM — and existing safety metrics don't capture that."**
> — *USENIX Security 2025*

</div>

---

## ⚡ The Problem Space

Modern RAG pipelines are **structurally vulnerable**. They ingest documents from external sources, embed them into vector stores, and pass retrieved chunks directly to LLMs — with no semantic security layer in between. This creates three critical attack surfaces:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNPROTECTED RAG PIPELINE                         │
│                                                                     │
│  User Query ──► Embedding ──► Vector DB ──► Chunks ──► LLM ──► ✗  │
│                                   ▲                                 │
│                         Poisoned documents,                         │
│                         injected instructions,                      │
│                         contradiction attacks enter here            │
└─────────────────────────────────────────────────────────────────────┘
```

| Attack Class | Description | CVSS Equivalent |
|---|---|---|
| **Indirect Prompt Injection** | Malicious instructions embedded in retrieved docs | Critical |
| **RAG Jamming** | Blocker docs that suppress legitimate retrieval | High |
| **Knowledge Poisoning** | Contradictory facts inserted to manipulate outputs | High |
| **Provenance Spoofing** | Fake authoritative sources in the corpus | Medium–High |
| **Context Overflow Attacks** | Flooding context with adversarial content | Medium |

---

## 🛡️ What RAG Firewall Does

RAG Firewall inserts a **multi-stage inspection pipeline** between your retrieval system and your LLM. Every chunk that touches your model has been scored, validated, and logged.

```
                           ╔══════════════════════════════╗
                           ║       RAG FIREWALL           ║
                           ║                              ║
  Retrieved   ┌───────────►║  [1] Injection Classifier   ║
  Chunks      │            ║         ↓                    ║
  ────────────┤            ║  [2] Trust Scorer            ║──► Safe Chunks
              │            ║         ↓                    ║       to LLM
              │            ║  [3] Consistency Checker     ║
              │            ║         ↓                    ║──► Audit Log
              │            ║  [4] Provenance Validator    ║
              │            ║         ↓                    ║──► Quarantine
              │            ║  [5] Output Policy Guard     ║
              │            ╚══════════════════════════════╝
              │
        ┌─────▼──────────────────────┐
        │   Quarantine / Block /     │
        │   Flag for human review    │
        └────────────────────────────┘
```

---

## 🏗️ Architecture

```
rag-firewall/
│
├── 📡 api/                          # FastAPI application layer
│   ├── routes/
│   │   ├── ingest.py               # Document ingestion endpoints
│   │   ├── query.py                # Protected query interface
│   │   └── audit.py               # Audit log retrieval
│   └── middleware/
│       └── rate_limiter.py
│
├── 🔥 firewall/                     # Core defense engine
│   ├── pipeline.py                 # Orchestrates all scoring stages
│   ├── stages/
│   │   ├── injection_classifier.py # Stage 1: Prompt injection detection
│   │   ├── trust_scorer.py         # Stage 2: Source & content trust
│   │   ├── consistency_checker.py  # Stage 3: Inter-chunk contradiction
│   │   ├── provenance_validator.py # Stage 4: Source verification
│   │   └── output_guard.py         # Stage 5: Post-generation policy
│   └── models/
│       ├── embedder.py             # Shared embedding utilities
│       └── classifiers.py          # Lightweight ONNX classifiers
│
├── 🗄️ storage/                      # Data persistence
│   ├── vector_store.py             # Qdrant integration
│   ├── audit_db.py                 # PostgreSQL audit logging
│   └── quarantine.py               # Isolated chunk storage
│
├── 📊 evaluation/                   # Benchmarking & red-teaming
│   ├── attack_suite/
│   │   ├── injection_attacks.py
│   │   ├── rag_jamming.py
│   │   └── poisoning_attacks.py
│   └── metrics.py                  # ASR, precision, groundedness
│
├── 🖥️ dashboard/                    # Monitoring UI (Streamlit)
│   └── app.py
│
├── docker-compose.yml
├── pyproject.toml
└── tests/
```

---

## 🔬 Defense Model Components

### Stage 1 — Injection Classifier
```python
# Detects prompt injection patterns in retrieved chunks
# Architecture: Fine-tuned DeBERTa-v3-small (ONNX export)
# Latency target: <15ms per chunk

score = injection_classifier.score(chunk)
# Returns: InjectionScore(probability=0.97, pattern="override_instruction",
#                          evidence="[SYSTEM] Ignore previous...")
```
Trained on synthetic + real injection datasets. Detects instruction overrides, role-hijacking, delimiter attacks, and multilingual obfuscation variants.

---

### Stage 2 — Retrieval Trust Scorer
```python
# Multi-factor trust scoring per chunk
TrustScore(
    source_authority  = 0.88,   # Domain / publisher reputation
    freshness         = 0.72,   # Time decay on knowledge
    corpus_agreement  = 0.61,   # Agreement with trusted segments
    retrieval_rank    = 0.90,   # Similarity score from vector DB
    composite         = 0.78    # Weighted aggregate
)
```
Trust thresholds are configurable per deployment context. Low-trust chunks trigger quarantine or require human approval before reaching the LLM.

---

### Stage 3 — Consistency Checker
Detects inter-chunk contradictions using a semantic NLI model. Critical for catching **knowledge poisoning**, where attacker-inserted documents contradict established facts.

```
Chunk A (trusted):  "Product X requires Python 3.9+"
Chunk B (suspect):  "Product X is compatible with Python 2.7"
                          ↓
            ContradictionAlert(severity=HIGH,
                               pair=(A, B),
                               confidence=0.94)
```

---

### Stage 4 — Provenance Validator
```
┌─────────────────────────────────────────────────────────────┐
│  PROVENANCE CHAIN VALIDATION                                │
│                                                             │
│  Claimed Source → Verified Against → Trusted Registry      │
│       ↓                                                     │
│  Hash Check → Signature Verify → Domain Reputation → ✓/✗  │
└─────────────────────────────────────────────────────────────┘
```

---

### Stage 5 — Output Policy Guard
Post-generation enforcement. Validates that the LLM's response is **grounded** in the approved corpus segments and does not leak system instructions.

```python
PolicyResult(
    grounded          = True,
    citation_valid    = True,
    instruction_leak  = False,
    pii_detected      = False,
    approved          = True
)
```

---

## 📈 Evaluation Framework

| Metric | Description | Target |
|---|---|---|
| **Attack Success Rate (ASR)** | % of injections that reach the LLM | < 2% |
| **Retrieval Precision (post-poisoning)** | Relevant chunks after corpus poisoning | > 92% |
| **False Positive Rate** | Benign docs blocked | < 3% |
| **Answer Groundedness** | Responses traceable to trusted corpus | > 95% |
| **Citation Validity** | Citations pointing to approved segments | > 98% |
| **Pipeline Latency (p95)** | End-to-end overhead vs. unprotected RAG | < 80ms |

---

## 🚀 Quickstart

### Prerequisites
```bash
python >= 3.11
docker & docker-compose
8GB RAM (for local embedding models)
```

### Launch with Docker
```bash
git clone https://github.com/yourusername/rag-firewall
cd rag-firewall

# Start all services (Qdrant, PostgreSQL, API, Dashboard)
docker-compose up -d

# Verify health
curl http://localhost:8000/health
```

### Install for Development
```bash
pip install -e ".[dev]"

# Run the test suite
pytest tests/ -v

# Launch the monitoring dashboard
streamlit run dashboard/app.py
```

### Protect Your First RAG Query
```python
from rag_firewall import SecureRAGPipeline

pipeline = SecureRAGPipeline.from_config("config/default.yaml")

# Ingest documents with pre-scan
pipeline.ingest("path/to/your/documents/")

# Query — all 5 firewall stages run automatically
result = pipeline.query(
    "What is our refund policy?",
    trust_threshold=0.75,
    block_injections=True
)

print(result.answer)          # Grounded, safe response
print(result.audit_trail)     # Full provenance chain
print(result.blocked_chunks)  # What the firewall caught
```

---

## 🧪 Red Team Test Suite

```bash
# Run the full adversarial evaluation suite
python -m evaluation.run_attacks \
    --suite all \
    --corpus data/test_corpus/ \
    --injection-rate 0.15 \
    --poison-rate 0.10 \
    --report reports/adversarial_eval.json
```

Includes reproduced attack patterns from:
- USENIX Security 2025 RAG-jamming methodology
- OWASP LLM Top 10 indirect injection scenarios
- Custom blocker-document and context-flooding variants

---

## 🗺️ Roadmap

```
Phase 1 — Core Pipeline (Weeks 1–3)
  ✅ Injection Classifier (DeBERTa-v3 fine-tuned)
  ✅ Qdrant integration + metadata filtering
  ✅ FastAPI skeleton + audit logging (PostgreSQL)
  ✅ Docker Compose stack

Phase 2 — Trust & Consistency (Weeks 4–6)
  🔲 Trust Scorer with configurable policy thresholds
  🔲 NLI-based Consistency Checker
  🔲 Provenance Validator + domain reputation lookup
  🔲 Quarantine store + human review API

Phase 3 — Output Guard & Evaluation (Weeks 7–9)
  🔲 Post-generation grounding checker
  🔲 Citation validity enforcement
  🔲 Full adversarial evaluation suite
  🔲 Streamlit monitoring dashboard

Phase 4 — Hardening & Integrations (Weeks 10–12)
  🔲 LangChain + LlamaIndex adapter layers
  🔲 Uncertainty estimation (conformal prediction)
  🔲 Multi-agent workflow support (OWASP agentic guidance)
  🔲 Published benchmark results
```

---

## 📚 Research Foundation

This project implements defenses grounded in current peer-reviewed research and security standards:

| Source | Relevance |
|---|---|
| **USENIX Security 2025** — RAG Jamming | Black-box corpus poisoning; motivates Stages 1–3 |
| **NIST AI 600-1 / GenAI Cybersecurity Profile** | AI analytics, triage, and human-validation requirements |
| **OWASP Agentic AI Top 10 (2025)** | Indirect injection, workflow hijacking defense patterns |
| **Survey: LLM Extraction Defenses** | Model protection, data privacy, prompt-targeted strategies |
| **Robustness of Open-Weight Safeguards** | Adversarial fine-tuning durability; motivates Stage 5 |

---

## 🤝 Contributing

Contributions welcome — especially:
- New attack vectors for the red team suite
- Integration adapters (LangChain, LlamaIndex, Haystack)
- Improved classifier training data
- Benchmarks on production corpora

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

---

## ⚖️ License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**Built as part of ongoing research into agentic AI security.**

*If this project helps you defend a production RAG system, please star ⭐ and share your findings.*

[![Star History Chart](https://img.shields.io/github/stars/yourusername/rag-firewall?style=social)](https://github.com/yourusername/rag-firewall)

</div>
