import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.fixture
async def embedding_provider():
    """Fixture to provide a FastEmbed embedding provider."""
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def qdrant_connector(embedding_provider):
    """Fixture to provide a QdrantConnector with in-memory Qdrant client."""
    # Use a random collection name to avoid conflicts between tests
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Create connector with in-memory Qdrant
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )

    yield connector


@pytest.mark.asyncio
async def test_store_and_search(qdrant_connector):
    """Test storing an entry and then searching for it."""
    # Store a test entry
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "importance": "high"},
    )
    point_id = await qdrant_connector.store(test_entry)

    # Search for the entry
    results = await qdrant_connector.search("fox jumps")

    # Verify results
    assert len(results) == 1
    assert results[0].id == point_id
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_search_empty_collection(qdrant_connector):
    """Test searching in an empty collection."""
    # Search in an empty collection
    results = await qdrant_connector.search("test query")

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_entries(qdrant_connector):
    """Test storing and searching multiple entries."""
    # Store multiple entries
    entries = [
        Entry(
            content="Python is a programming language",
            metadata={"topic": "programming"},
        ),
        Entry(content="The Eiffel Tower is in Paris", metadata={"topic": "landmarks"}),
        Entry(content="Machine learning is a subset of AI", metadata={"topic": "AI"}),
    ]

    for entry in entries:
        await qdrant_connector.store(entry)

    # Search for programming-related entries
    programming_results = await qdrant_connector.search("Python programming")
    assert len(programming_results) > 0
    assert any("Python" in result.content for result in programming_results)

    # Search for landmark-related entries
    landmark_results = await qdrant_connector.search("Eiffel Tower Paris")
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.content for result in landmark_results)

    # Search for AI-related entries
    ai_results = await qdrant_connector.search(
        "artificial intelligence machine learning"
    )
    assert len(ai_results) > 0
    assert any("machine learning" in result.content.lower() for result in ai_results)


@pytest.mark.asyncio
async def test_ensure_collection_exists(qdrant_connector):
    """Test that the collection is created if it doesn't exist."""
    # The collection shouldn't exist yet
    assert not await qdrant_connector._client.collection_exists(
        qdrant_connector._default_collection_name
    )

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await qdrant_connector.store(test_entry)

    # Now the collection should exist
    assert await qdrant_connector._client.collection_exists(
        qdrant_connector._default_collection_name
    )


@pytest.mark.asyncio
async def test_metadata_handling(qdrant_connector):
    """Test that metadata is properly stored and retrieved."""
    # Store entries with different metadata
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await qdrant_connector.store(
        Entry(content="Content with structured metadata", metadata=metadata1)
    )
    await qdrant_connector.store(
        Entry(content="Content with list in metadata", metadata=metadata2)
    )

    # Search and verify metadata is preserved
    results = await qdrant_connector.search("metadata")

    assert len(results) == 2

    # Check that both metadata objects are present in the results
    found_metadata1 = False
    found_metadata2 = False

    for result in results:
        if result.metadata.get("source") == "book":
            assert result.metadata.get("author") == "Jane Doe"
            assert result.metadata.get("year") == 2023
            found_metadata1 = True
        elif result.metadata.get("source") == "article":
            assert "science" in result.metadata.get("tags", [])
            assert "research" in result.metadata.get("tags", [])
            found_metadata2 = True

    assert found_metadata1
    assert found_metadata2


@pytest.mark.asyncio
async def test_entry_without_metadata(qdrant_connector):
    """Test storing and retrieving entries without metadata."""
    # Store an entry without metadata
    await qdrant_connector.store(Entry(content="Entry without metadata"))

    # Search and verify
    results = await qdrant_connector.search("without metadata")

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None


@pytest.mark.asyncio
async def test_custom_collection_store_and_search(qdrant_connector):
    """Test storing and searching in a custom collection."""
    # Define a custom collection name
    custom_collection = f"custom_collection_{uuid.uuid4().hex}"

    # Store a test entry in the custom collection
    test_entry = Entry(
        content="This is stored in a custom collection",
        metadata={"custom": True},
    )
    await qdrant_connector.store(test_entry, collection_name=custom_collection)

    # Search in the custom collection
    results = await qdrant_connector.search(
        "custom collection", collection_name=custom_collection
    )

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata

    # Verify the entry is not in the default collection
    default_results = await qdrant_connector.search("custom collection")
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_multiple_collections(qdrant_connector):
    """Test using multiple collections with the same connector."""
    # Define two custom collection names
    collection_a = f"collection_a_{uuid.uuid4().hex}"
    collection_b = f"collection_b_{uuid.uuid4().hex}"

    # Store entries in different collections
    entry_a = Entry(
        content="This belongs to collection A", metadata={"collection": "A"}
    )
    entry_b = Entry(
        content="This belongs to collection B", metadata={"collection": "B"}
    )
    entry_default = Entry(content="This belongs to the default collection")

    await qdrant_connector.store(entry_a, collection_name=collection_a)
    await qdrant_connector.store(entry_b, collection_name=collection_b)
    await qdrant_connector.store(entry_default)

    # Search in collection A
    results_a = await qdrant_connector.search("belongs", collection_name=collection_a)
    assert len(results_a) == 1
    assert results_a[0].content == entry_a.content

    # Search in collection B
    results_b = await qdrant_connector.search("belongs", collection_name=collection_b)
    assert len(results_b) == 1
    assert results_b[0].content == entry_b.content

    # Search in default collection
    results_default = await qdrant_connector.search("belongs")
    assert len(results_default) == 1
    assert results_default[0].content == entry_default.content


@pytest.mark.asyncio
async def test_nonexistent_collection_search(qdrant_connector):
    """Test searching in a collection that doesn't exist."""
    # Search in a collection that doesn't exist
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await qdrant_connector.search(
        "test query", collection_name=nonexistent_collection
    )

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_edit_updates_entry_by_id(qdrant_connector):
    """Test editing content and metadata without changing the point ID."""
    point_id = await qdrant_connector.store(
        Entry(
            content="Paris is the capital of France",
            metadata={"source": "atlas", "version": 1},
        )
    )

    updated = await qdrant_connector.edit(
        point_id,
        Entry(
            content="Paris is the capital and largest city of France",
            metadata={"source": "atlas", "version": 2},
        ),
    )

    assert updated is not None
    assert updated.id == point_id
    assert updated.content == "Paris is the capital and largest city of France"
    assert updated.metadata == {"source": "atlas", "version": 2}

    results = await qdrant_connector.search("largest city of France")
    assert len(results) == 1
    assert results[0] == updated

    count = await qdrant_connector._client.count(
        collection_name=qdrant_connector._default_collection_name,
        exact=True,
    )
    assert count.count == 1


@pytest.mark.asyncio
async def test_edit_preserves_metadata_when_omitted(qdrant_connector):
    """Test editing content while preserving existing metadata."""
    metadata = {"source": "encyclopedia", "verified": True}
    point_id = await qdrant_connector.store(
        Entry(content="Mercury is closest to the Sun", metadata=metadata)
    )

    updated = await qdrant_connector.edit(
        point_id,
        Entry(content="Mercury is the smallest planet in the Solar System"),
    )

    assert updated is not None
    assert updated.id == point_id
    assert updated.metadata == metadata


@pytest.mark.asyncio
async def test_edit_missing_point_or_collection_returns_none(qdrant_connector):
    """Test editing does not insert a point when the target does not exist."""
    missing_id = uuid.uuid4().hex
    missing_collection = f"missing_{uuid.uuid4().hex}"

    assert await qdrant_connector.edit(missing_id, Entry(content="Replacement")) is None
    assert (
        await qdrant_connector.edit(
            missing_id,
            Entry(content="Replacement"),
            collection_name=missing_collection,
        )
        is None
    )
    assert not await qdrant_connector._client.collection_exists(missing_collection)


@pytest.mark.asyncio
async def test_delete_removes_only_requested_id(qdrant_connector):
    """Test deleting one point by ID preserves other entries."""
    delete_id = await qdrant_connector.store(
        Entry(content="This memory should be deleted", metadata={"temporary": True})
    )
    keep_id = await qdrant_connector.store(
        Entry(content="This memory should remain", metadata={"temporary": False})
    )

    deleted = await qdrant_connector.delete(delete_id)

    assert deleted is not None
    assert deleted.id == delete_id
    assert deleted.content == "This memory should be deleted"
    assert deleted.metadata == {"temporary": True}

    remaining = await qdrant_connector.search("memory should remain")
    assert len(remaining) == 1
    assert remaining[0].id == keep_id
    assert await qdrant_connector.delete(delete_id) is None


@pytest.mark.asyncio
async def test_edit_and_delete_in_custom_collection(qdrant_connector):
    """Test ID-based mutations respect an explicitly selected collection."""
    custom_collection = f"custom_{uuid.uuid4().hex}"
    point_id = await qdrant_connector.store(
        Entry(content="Original custom memory"),
        collection_name=custom_collection,
    )

    updated = await qdrant_connector.edit(
        point_id,
        Entry(content="Updated custom memory"),
        collection_name=custom_collection,
    )
    assert updated is not None
    assert updated.id == point_id

    deleted = await qdrant_connector.delete(
        point_id,
        collection_name=custom_collection,
    )
    assert deleted == updated
    assert (
        await qdrant_connector.search(
            "custom memory", collection_name=custom_collection
        )
        == []
    )


@pytest.mark.asyncio
async def test_delete_missing_collection_returns_none(qdrant_connector):
    """Test deleting from a nonexistent collection is a no-op."""
    missing_collection = f"missing_{uuid.uuid4().hex}"

    deleted = await qdrant_connector.delete(
        uuid.uuid4().hex,
        collection_name=missing_collection,
    )

    assert deleted is None
    assert not await qdrant_connector._client.collection_exists(missing_collection)
