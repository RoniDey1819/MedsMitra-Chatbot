# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# MedsMitra Chatbot — backend container
#
# This image runs only backend/app.py (the FastAPI RAG service). The widget
# (widget/chatbot-widget.js) and the demo site (widget/MedsMitra/) are static
# files meant to be hosted separately (e.g. Netlify, Vercel, GitHub Pages, or
# your existing pharmacy website) — they are not part of this image.
#
# Build (run from the repository root, where this Dockerfile lives):
#   docker build -t medsmitra-backend .
#
# Run:
#   docker run -d -p 8000:8000 --env-file backend/.env --name medsmitra medsmitra-backend
#
# The sentence-transformers embedding model (~90MB) is pre-downloaded during
# the build so the container doesn't need internet access or a slow cold
# start on first request.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# which keeps container logs flowing in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System packages needed to build psycopg2 and other compiled wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached unless
# requirements.txt changes (keeps rebuilds fast during development).
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so the image is
# self-contained and startup doesn't depend on internet access.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy backend source. medicines.csv is included for reference/local use with
# load_data.py, but the running API reads medicine data from Supabase, not
# from this file directly.
COPY backend/ ./

# Run as a non-root user for defense-in-depth.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uses the /health endpoint already defined in app.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker: app.py loads the embedding model and, if enabled, schedules
# a background crawl job in-process — multiple workers would duplicate both.
# Scale horizontally with multiple containers behind a load balancer instead
# of multiple uvicorn workers in one container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]