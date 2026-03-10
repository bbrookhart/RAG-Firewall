# Contributing to RAG Firewall

Thanks for helping make RAG systems safer. Contributions most needed:

## High-Value Areas

- **New attack vectors** in `evaluation/attack_suite/` — document source, severity, and expected behaviour
- **Integration adapters** — LangChain `BaseRetriever`, LlamaIndex `NodePostprocessor`
- **Classifier improvements** — better training data, ONNX export scripts
- **Benchmarks on real corpora** — privacy-safe, reproducible

## Setup

```bash
git clone https://github.com/yourusername/rag-firewall
cd rag-firewall
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v                     # Full suite
pytest tests/ -k "injection" -v      # Specific tests
python scripts/run_eval.py --no-model  # Adversarial eval (no GPU needed)
```

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] New attack patterns have corresponding tests
- [ ] Docstrings on all public functions
- [ ] No new dependencies without discussion
- [ ] `ruff check .` passes

## Code Style

- `ruff` for linting and formatting
- `mypy --strict` for type checking (we aim for full type coverage)
- Async-first: all I/O must be async

## Reporting Vulnerabilities

For security issues in the firewall itself, please email security@yourdomain.com rather than opening a public issue.
