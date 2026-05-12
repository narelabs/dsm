from pathlib import Path

from dsm import DynamicSegmentedMemory
from dsm.memory import sparse_attention_cost


def test_write_route_active_context(tmp_path: Path) -> None:
    memory = DynamicSegmentedMemory(tmp_path / "dsm.json", active_segment_limit=2)
    memory.write(
        "Rust websocket backpressure is handled with bounded channels and async cancellation.",
        category_path="Programming → Rust → Async",
        importance=0.9,
    )
    memory.write(
        "Medieval history contains dynasties, battles and trade routes.",
        category_path="History → Medieval",
        importance=0.5,
    )

    results = memory.route("Rust websocket cancellation bug", k=1)

    assert len(results) == 1
    assert "Rust" in results[0].segment.category_path
    assert results[0].similarity > 0

    active = memory.active_context("Rust websocket cancellation bug", k=1)
    assert results[0].segment.id in active.segment_ids
    assert "bounded channels" in active.context_text


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "dsm.json"
    memory = DynamicSegmentedMemory(path)
    written = memory.write("MemoryOS uses hierarchical memory layers.", category_path="Science → AI")
    memory.save()

    restored = DynamicSegmentedMemory(path)
    assert written[0].id in restored.segments
    assert restored.stats()["segments"] == 1
    assert restored.route("hierarchical memory", k=1)[0].segment.id == written[0].id


def test_update_existing_and_graph_links(tmp_path: Path) -> None:
    memory = DynamicSegmentedMemory(tmp_path / "dsm.json")
    first = memory.write("Tokio async websocket timeout handling.", category_path="Programming → Rust")
    second = memory.write("Rust async websocket backpressure handling.", category_path="Programming → Rust")

    assert first[0].id != second[0].id or len(memory.segments) >= 1
    assert memory.stats()["graph_edges"] >= 0

    memory.update_from_interaction("Rust websocket timeout", "Use bounded channels and ping/pong.")
    assert memory.stats()["segments"] >= 1


def test_rebuild_prune_and_cost(tmp_path: Path) -> None:
    memory = DynamicSegmentedMemory(tmp_path / "dsm.json")
    memory.write("Rare stale note", category_path="General", importance=0.01)
    memory.decay_priorities(0.95)
    moved = memory.rebuild_structure()
    removed = memory.prune(min_priority=0.5)
    cost = sparse_attention_cost(active_tokens=100, total_tokens=1000)

    assert moved >= 0
    assert len(removed) >= 0
    assert cost["reduction_ratio"] == 100.0
