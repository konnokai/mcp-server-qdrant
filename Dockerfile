FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install the mcp-server-qdrant package from this source tree
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# 照 uv.lock 的版本裝。不鎖的話會裝到新版 onnxruntime，
# 它不接受 HF cache 用 symlink 指到 blobs 的 model.onnx_data，大於 2 GB 的模型會載入失敗
RUN uv export --locked --no-dev --no-emit-project --no-hashes -o requirements.txt \
    && uv pip install --system --no-cache-dir -r requirements.txt . \
    && rm requirements.txt

# Expose the default port for Streamable HTTP transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# opencode v2 會對 URL 直接 POST，SSE 模式的 /sse 只收 GET 會回 405，所以改走 /mcp/
CMD ["mcp-server-qdrant", "--transport", "streamable-http"]
