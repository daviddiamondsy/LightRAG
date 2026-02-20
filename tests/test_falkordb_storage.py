"""
Integration tests for FalkorDBStorage.

Requires a running FalkorDB instance (default: localhost:6379).
Run with:
    FALKORDB_HOST=localhost FALKORDB_PORT=6379 \
    python -m pytest tests/test_falkordb_storage.py --run-integration -v
"""

import asyncio
import os
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


@pytest.fixture
def falkordb_config():
    return {
        "FALKORDB_HOST": os.environ.get("FALKORDB_HOST", "localhost"),
        "FALKORDB_PORT": os.environ.get("FALKORDB_PORT", "6379"),
    }


@pytest.fixture
async def storage(falkordb_config):
    """Create and initialize a FalkorDBStorage instance for testing."""
    for key, value in falkordb_config.items():
        os.environ[key] = value

    from lightrag.kg.falkordb_impl import FalkorDBStorage

    store = FalkorDBStorage(
        namespace="test_falkordb",
        global_config={},
        embedding_func=None,
        workspace="pytest",
    )
    await store.initialize()
    yield store
    await store.drop()
    await store.finalize()


@pytest.mark.asyncio
async def test_upsert_and_has_node(storage):
    node_data = {
        "entity_id": "Alice",
        "entity_type": "Person",
        "description": "A test person",
    }
    await storage.upsert_node("Alice", node_data)
    assert await storage.has_node("Alice") is True
    assert await storage.has_node("NonExistent") is False


@pytest.mark.asyncio
async def test_get_node(storage):
    node_data = {
        "entity_id": "Bob",
        "entity_type": "Person",
        "description": "Another test person",
    }
    await storage.upsert_node("Bob", node_data)
    result = await storage.get_node("Bob")
    assert result is not None
    assert result["entity_id"] == "Bob"
    assert result["entity_type"] == "Person"


@pytest.mark.asyncio
async def test_upsert_and_has_edge(storage):
    await storage.upsert_node("Alice", {"entity_id": "Alice", "entity_type": "Person"})
    await storage.upsert_node("Bob", {"entity_id": "Bob", "entity_type": "Person"})
    edge_data = {"weight": 1.0, "description": "knows", "keywords": "friend"}
    await storage.upsert_edge("Alice", "Bob", edge_data)
    assert await storage.has_edge("Alice", "Bob") is True
    assert await storage.has_edge("Alice", "NonExistent") is False


@pytest.mark.asyncio
async def test_get_edge(storage):
    await storage.upsert_node("Carol", {"entity_id": "Carol", "entity_type": "Person"})
    await storage.upsert_node("Dave", {"entity_id": "Dave", "entity_type": "Person"})
    edge_data = {"weight": 2.5, "description": "works with", "keywords": "colleague"}
    await storage.upsert_edge("Carol", "Dave", edge_data)
    result = await storage.get_edge("Carol", "Dave")
    assert result is not None
    assert float(result["weight"]) == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_node_degree(storage):
    await storage.upsert_node("Hub", {"entity_id": "Hub", "entity_type": "Concept"})
    await storage.upsert_node("Spoke1", {"entity_id": "Spoke1", "entity_type": "Concept"})
    await storage.upsert_node("Spoke2", {"entity_id": "Spoke2", "entity_type": "Concept"})
    await storage.upsert_edge("Hub", "Spoke1", {"weight": 1.0})
    await storage.upsert_edge("Hub", "Spoke2", {"weight": 1.0})
    degree = await storage.node_degree("Hub")
    assert degree >= 2


@pytest.mark.asyncio
async def test_get_node_edges(storage):
    await storage.upsert_node("X", {"entity_id": "X", "entity_type": "Thing"})
    await storage.upsert_node("Y", {"entity_id": "Y", "entity_type": "Thing"})
    await storage.upsert_edge("X", "Y", {"weight": 1.0})
    edges = await storage.get_node_edges("X")
    assert edges is not None
    assert len(edges) >= 1
    entity_ids = {eid for pair in edges for eid in pair}
    assert "X" in entity_ids or "Y" in entity_ids


@pytest.mark.asyncio
async def test_search_labels(storage):
    await storage.upsert_node(
        "Artificial Intelligence",
        {"entity_id": "Artificial Intelligence", "entity_type": "Concept"},
    )
    await storage.upsert_node(
        "Artificial Neural Network",
        {"entity_id": "Artificial Neural Network", "entity_type": "Concept"},
    )
    await asyncio.sleep(0.5)
    results = await storage.search_labels("Artificial", limit=10)
    assert isinstance(results, list)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_get_all_labels(storage):
    await storage.upsert_node("LabelA", {"entity_id": "LabelA", "entity_type": "Test"})
    await storage.upsert_node("LabelB", {"entity_id": "LabelB", "entity_type": "Test"})
    labels = await storage.get_all_labels()
    assert "LabelA" in labels
    assert "LabelB" in labels


@pytest.mark.asyncio
async def test_delete_node(storage):
    await storage.upsert_node("ToDelete", {"entity_id": "ToDelete", "entity_type": "Temp"})
    assert await storage.has_node("ToDelete") is True
    await storage.delete_node("ToDelete")
    assert await storage.has_node("ToDelete") is False


@pytest.mark.asyncio
async def test_remove_edges(storage):
    await storage.upsert_node("E1", {"entity_id": "E1", "entity_type": "Thing"})
    await storage.upsert_node("E2", {"entity_id": "E2", "entity_type": "Thing"})
    await storage.upsert_edge("E1", "E2", {"weight": 1.0})
    assert await storage.has_edge("E1", "E2") is True
    await storage.remove_edges([("E1", "E2")])
    assert await storage.has_edge("E1", "E2") is False


@pytest.mark.asyncio
async def test_get_knowledge_graph(storage):
    await storage.upsert_node("Root", {"entity_id": "Root", "entity_type": "Concept"})
    await storage.upsert_node("Child1", {"entity_id": "Child1", "entity_type": "Concept"})
    await storage.upsert_node("Child2", {"entity_id": "Child2", "entity_type": "Concept"})
    await storage.upsert_edge("Root", "Child1", {"weight": 1.0})
    await storage.upsert_edge("Root", "Child2", {"weight": 1.0})
    kg = await storage.get_knowledge_graph("Root", max_depth=2)
    node_ids = {n.id for n in kg.nodes}
    assert "Root" in node_ids
    assert len(kg.edges) >= 2


@pytest.mark.asyncio
async def test_drop(storage):
    await storage.upsert_node("DropMe", {"entity_id": "DropMe", "entity_type": "Temp"})
    result = await storage.drop()
    assert result["status"] == "success"
    assert await storage.has_node("DropMe") is False


@pytest.mark.asyncio
async def test_get_popular_labels(storage):
    await storage.upsert_node("Popular", {"entity_id": "Popular", "entity_type": "Concept"})
    await storage.upsert_node("Leaf1", {"entity_id": "Leaf1", "entity_type": "Concept"})
    await storage.upsert_node("Leaf2", {"entity_id": "Leaf2", "entity_type": "Concept"})
    await storage.upsert_edge("Popular", "Leaf1", {"weight": 1.0})
    await storage.upsert_edge("Popular", "Leaf2", {"weight": 1.0})
    labels = await storage.get_popular_labels(limit=10)
    assert isinstance(labels, list)
    assert "Popular" in labels
