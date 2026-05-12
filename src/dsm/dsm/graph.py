from __future__ import annotations

from collections import deque


class MemoryGraph:
    """Undirected semantic graph between memory segments."""

    def __init__(self, edges: dict[str, set[str]] | None = None):
        self.edges: dict[str, set[str]] = edges or {}

    def add_node(self, node_id: str) -> None:
        self.edges.setdefault(node_id, set())

    def add_edge(self, left: str, right: str) -> None:
        if left == right:
            return
        self.add_node(left)
        self.add_node(right)
        self.edges[left].add(right)
        self.edges[right].add(left)

    def remove_node(self, node_id: str) -> None:
        neighbors = self.edges.pop(node_id, set())
        for neighbor in neighbors:
            if neighbor in self.edges:
                self.edges[neighbor].discard(node_id)

    def neighbors(self, node_id: str) -> set[str]:
        return set(self.edges.get(node_id, set()))

    def expand(self, seed_ids: list[str], max_hops: int = 2, limit: int = 32) -> dict[str, int]:
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seed_ids)

        while queue and len(distances) < limit:
            node_id, distance = queue.popleft()
            if node_id in distances or distance > max_hops:
                continue
            distances[node_id] = distance
            if distance == max_hops:
                continue
            for neighbor in sorted(self.edges.get(node_id, set())):
                if neighbor not in distances:
                    queue.append((neighbor, distance + 1))

        return distances

    def to_dict(self) -> dict[str, list[str]]:
        return {node_id: sorted(neighbors) for node_id, neighbors in sorted(self.edges.items())}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "MemoryGraph":
        return cls({str(node_id): {str(n) for n in neighbors} for node_id, neighbors in data.items()})
