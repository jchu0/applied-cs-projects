# RAG Baseline System - Production Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir -e ".[full]"

# Create data directories
RUN mkdir -p /app/data/tenants /app/data/logs /app/data/usage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "ragbaseline.api:app", "--host", "0.0.0.0", "--port", "8000"]
