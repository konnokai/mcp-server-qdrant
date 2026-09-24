FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install the mcp-server-qdrant package from this source tree
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache-dir .

# Expose the default port for Streamable HTTP transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# opencode v2 會對 URL 直接 POST，SSE 模式的 /sse 只收 GET 會回 405，所以改走 /mcp/
CMD ["mcp-server-qdrant", "--transport", "streamable-http"]
