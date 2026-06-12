# ============================================================
# Dockerfile — Travel ML Project
# ============================================================
# Multi-stage build for a lean production image
#
# Build:
#   docker build -t travel-ml-project:latest .
#
# Run (Flask API):
#   docker run -p 5000:5000 travel-ml-project:latest
#
# Run (Streamlit):
#   docker run -p 8501:8501 -e APP_MODE=streamlit travel-ml-project:latest
# ============================================================

# ── Stage 1: Builder — install dependencies ──────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/packages --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime — minimal production image ──────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy project code
COPY --chown=appuser:appuser . .

# ── Environment variables ────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    DEBUG=false \
    APP_MODE=api

# ── Expose ports ─────────────────────────────────────────────
EXPOSE 5000
EXPOSE 8501

# ── Entrypoint script ────────────────────────────────────────
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
