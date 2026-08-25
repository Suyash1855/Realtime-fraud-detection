# ─── Stage 1: build the frontend ─────────────────────────────────────────────
FROM node:20-alpine AS web

WORKDIR /web
# Copy manifests first so the dependency layer caches independently of source.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-fund --no-audit 2>/dev/null || npm install --no-fund --no-audit

COPY web/ ./
RUN npm run build


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
# Python 3.11 rather than the 3.9 used locally: it is faster, and the codebase
# already uses PEP 604 annotations that 3.9 needs workarounds for.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Only the serving dependencies. No MLflow, no Kafka, no Redis, no SHAP package,
# and no pyarrow: the bundle is frozen at build time, explanations use XGBoost's
# own pred_contribs, and the demo data is gzipped CSV that pandas reads with the
# stdlib. Dropping pyarrow alone removes ~105 MB of compiled Arrow libraries.
RUN pip install --no-cache-dir \
      "fastapi==0.128.8" \
      "uvicorn[standard]>=0.30" \
      "pandas==2.3.3" \
      "numpy==2.0.2" \
      "xgboost==2.1.4" \
      "prometheus-fastapi-instrumentator==7.1.0"

# Application code
COPY app/ ./app/
COPY features.py ./

# The frozen serving bundle and the demo dataset. Both are build artifacts:
#   python scripts/export_for_deploy.py
#   python scripts/build_demo_data.py
COPY deploy_bundle/ ./deploy_bundle/
COPY demo_data/ ./demo_data/

# Built frontend from stage 1
COPY --from=web /web/dist ./web/dist

# Run unprivileged.
RUN useradd --create-home --shell /bin/false appuser && chown -R appuser /app
USER appuser

ENV BUNDLE_DIR=/app/deploy_bundle \
    DEMO_DATA_DIR=/app/demo_data \
    STATIC_DIR=/app/web/dist \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=4s --start-period=25s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3).status==200 else 1)"

# One worker on purpose. Each worker would load its own copy of the model and
# precompute its own scores, and simulation sessions live in process memory, so a
# second worker would double the RAM and split sessions unpredictably.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
