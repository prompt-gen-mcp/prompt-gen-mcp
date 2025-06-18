FROM python:3.11-slim

LABEL maintainer="PromptGen.ai <support@promptgen.ai>"
LABEL description="Advanced Prompt Engineering MCP Server with Self-RAG"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ src/
COPY mcp_server.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcp
RUN chown -R mcp:mcp /app
USER mcp

# Create data directory for vector storage
RUN mkdir -p /app/data

# Expose port (if needed for health checks)
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV QDRANT_PATH=/app/data/qdrant
ENV TECHNIQUES_ENDPOINT=https://api.promptgen.ai/techniques

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the MCP server
CMD ["python", "mcp_server.py"] 