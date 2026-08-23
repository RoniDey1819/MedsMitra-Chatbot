# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# MedsMitra Chatbot - backend container
#
# This image runs only backend/app.py (the FastAPI RAG service). The widget
# (widget/chatbot-widget.js) and the demo site (widget/MedsMitra/) are static
# files meant to be hosted separately (e.g. Netlify, Vercel, GitHub Pages, or
# your existing pharmacy website) - they are not part of this image.
#
# Build (run from the repository root, where this Dockerfile lives):
#   docker build -t medsmitra-backend .
#
# Run:
#   docker run -d -p 8000:8000 --env-file backend/.env --name medsmitra medsmitra-backend
#
# Embeddings are computed by Hugging Face's hosted Inference API (see
# backend/hf_embedder.py) rather than a local model, so this image has no
# large ML dependencies to download or bake in - it builds fast and stays
# small, which matters most on resource-limited hosts like Render's free
# tier where local CPU-bound embedding was previously the bottleneck.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# which keeps container logs flowing in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages needed to build psycopg2 and other compiled wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached unless
# requirements.txt changes (keeps rebuilds fast during development).
#
# NOTE: as of the HF-Inference-API embedding change, requirements.txt no
# longer includes sentence-transformers/torch - embeddings are computed by
# a hosted HF endpoint instead of locally, so this image stays small and
# builds fast. If you re-enable local embeddings, see requirements-local.txt
# and re-add the torch pre-download step that used to live here.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

# Copy backend source. medicines.csv is included for reference/local use with
# load_data.py, but the running API reads medicine data from Supabase, not
# from this file directly.
COPY backend/ ./

# Run as a non-root user for defense-in-depth.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uses the /health endpoint already defined in app.py. Uses $PORT if set
# (e.g. by Render), falling back to 8000 for local docker/docker-compose use.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Single worker: app.py loads the embedding model and, if enabled, schedules
# a background crawl job in-process - multiple workers would duplicate both.
# Scale horizontally with multiple containers behind a load balancer instead
# of multiple uvicorn workers in one container.
#
# Binds to $PORT so this works unchanged on platforms (Render, etc.) that
# inject their own port, and falls back to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]