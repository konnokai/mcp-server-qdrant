import inspect

import pytest

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry
from mcp_server_qdrant.settings import QdrantSettings, ToolSettings


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0]

    def get_vector_name(self) -> str:
        return "fake"

    def get_vector_size(self) -> int:
        return 1


@pytest.fixture
def embedding_provider():
    return FakeEmbeddingProvider()


async def create_tools(embedding_provider: EmbeddingProvider) -> dict:
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider=embedding_provider,
    )
    return await server.get_tools()


@pytest.mark.asyncio
async def test_write_mode_exposes_all_tools(embedding_provider):
    """Test that all memory tools are available in write mode."""
    tools = await create_tools(embedding_provider)

    assert set(tools) == {
        "qdrant-find",
        "qdrant-store",
        "qdrant-edit",
        "qdrant-delete",
    }


@pytest.mark.asyncio
async def test_read_only_mode_only_exposes_find_tool(monkeypatch, embedding_provider):
    """Test that all mutating tools are hidden in read-only mode."""
    monkeypatch.setenv("QDRANT_READ_ONLY", "1")

    tools = await create_tools(embedding_provider)

    assert set(tools) == {"qdrant-find"}


@pytest.mark.asyncio
async def test_default_collection_removes_collection_parameter(
    monkeypatch, embedding_provider
):
    """Test that the default collection is bound for every tool."""
    monkeypatch.setenv("COLLECTION_NAME", "memories")

    tools = await create_tools(embedding_provider)

    for tool in tools.values():
        assert "collection_name" not in inspect.signature(tool.fn).parameters


@pytest.mark.asyncio
async def test_without_default_collection_requires_collection_parameter(
    monkeypatch, embedding_provider
):
    """Test that every tool exposes collection_name without a default."""
    monkeypatch.delenv("COLLECTION_NAME", raising=False)

    tools = await create_tools(embedding_provider)

    for tool in tools.values():
        assert "collection_name" in inspect.signature(tool.fn).parameters


def test_format_entry_includes_point_id(embedding_provider):
    """Test that search results expose the ID needed by mutation tools."""
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider=embedding_provider,
    )

    formatted = server.format_entry(
        Entry(id="12345678123456781234567812345678", content="Memory")
    )

    assert "<id>12345678123456781234567812345678</id>" in formatted


@pytest.mark.asyncio
async def test_mutation_tools_require_point_id(embedding_provider):
    """Test that edit and delete expose an exact point ID parameter."""
    tools = await create_tools(embedding_provider)

    assert "point_id" in inspect.signature(tools["qdrant-edit"].fn).parameters
    assert "point_id" in inspect.signature(tools["qdrant-delete"].fn).parameters
