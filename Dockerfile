# ==============================================================================
# MedTrust V1 — Multi-Stage Production Dockerfile
# Builds the React frontend SPA bundle and containerizes the Python inference engine.
# ==============================================================================

# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /build

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Runtime Environment
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MEDTRUST_ENV=production

WORKDIR /app

# Install minimal OS runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy backend application source, model inference modules, and assets
COPY src/ ./src/
COPY backend/ ./backend/
COPY outputs/ ./outputs/
COPY checkpoints/ ./checkpoints/
COPY test_images/ ./test_images/

# Copy compiled frontend SPA bundle from builder stage
COPY --from=frontend-builder /build/dist ./frontend/dist

# Create unprivileged production user and assign ownership
RUN groupadd -r medtrust && useradd -r -g medtrust -s /bin/false medtrust \
    && mkdir -p /app/backend /app/data \
    && chown -R medtrust:medtrust /app

USER medtrust

EXPOSE 8000

# Automated container liveness healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/live || exit 1

CMD ["python", "backend/server.py"]

