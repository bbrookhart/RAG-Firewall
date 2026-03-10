# ─── Base ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependencies ─────────────────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml .
RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

# ─── API target ───────────────────────────────────────────────────────────────
FROM deps AS api

COPY . .

RUN addgroup --system ragfw && adduser --system --group ragfw
RUN chown -R ragfw:ragfw /app
USER ragfw

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ─── Dashboard target ─────────────────────────────────────────────────────────
FROM deps AS dashboard

RUN pip install -e ".[dashboard]"

COPY . .

RUN addgroup --system ragfw && adduser --system --group ragfw
RUN chown -R ragfw:ragfw /app
USER ragfw

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
