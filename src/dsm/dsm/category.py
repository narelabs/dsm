from __future__ import annotations

from nare3.dsm.embedding import EmbeddingModel, cosine, mean_embedding, top_terms
from nare3.dsm.models import CategoryNode, MemorySegment, now_ts


DEFAULT_TOPICS = {
    "Programming": {
        "python",
        "rust",
        "javascript",
        "typescript",
        "async",
        "websocket",
        "api",
        "code",
        "bug",
        "test",
        "cli",
        "database",
    },
    "Science": {
        "physics",
        "biology",
        "math",
        "research",
        "theory",
        "experiment",
        "model",
    },
    "Personal": {"profile", "preference", "habit", "goal", "schedule", "name"},
    "History": {"history", "past", "timeline", "event", "date"},
    "Business": {"market", "customer", "sales", "product", "finance", "strategy"},
}


class CategoryTree:
    """Hierarchical topic tree T used by DSM routing."""

    def __init__(self, embedding_model: EmbeddingModel, root: CategoryNode | None = None):
        self.embedding_model = embedding_model
        self.root = root or CategoryNode("Memory", (), [0.0] * embedding_model.dim)
        if not root:
            self._seed_defaults()

    def ensure_path(self, path: tuple[str, ...]) -> CategoryNode:
        node = self.root
        for name in path:
            node = node.add_child(name, self.embedding_model.encode(name))
        node.updated_at = now_ts()
        return node

    def add_segment(self, segment: MemorySegment) -> None:
        node = self.ensure_path(segment.category_path)
        node.segment_ids.add(segment.id)
        node.updated_at = now_ts()

    def remove_segment(self, segment: MemorySegment) -> None:
        node = self.root.find(segment.category_path)
        if node:
            node.segment_ids.discard(segment.id)
            node.updated_at = now_ts()

    def choose_path(self, text: str, max_depth: int = 3, threshold: float = 0.24) -> tuple[str, ...]:
        query_embedding = self.embedding_model.encode(text)
        node = self.root
        path: list[str] = []

        while node.children and len(path) < max_depth:
            scored = [
                (cosine(query_embedding, child.embedding), name, child)
                for name, child in node.children.items()
            ]
            scored.sort(reverse=True, key=lambda item: item[0])
            score, name, child = scored[0]
            if score < threshold:
                break
            path.append(name)
            node = child

        if not path:
            path = [self._coarse_category(text)]

        existing = self.root.find(tuple(path))
        terms = [term.title() for term in top_terms(text, limit=2)]
        if terms and len(path) < max_depth:
            child_names = set(existing.children if existing else ())
            term = next((candidate for candidate in terms if candidate not in child_names), terms[0])
            if existing is None or not existing.children:
                path.append(term)

        return tuple(path[:max_depth])

    def refresh_embeddings(self, segments: dict[str, MemorySegment]) -> None:
        for node in reversed(self.root.walk()):
            vectors = [segments[sid].embedding for sid in node.segment_ids if sid in segments]
            vectors.extend(child.embedding for child in node.children.values())
            if vectors:
                node.embedding = mean_embedding(vectors, self.embedding_model.dim)
                node.updated_at = now_ts()

    def route_categories(
        self,
        query_embedding: list[float],
        beam_width: int = 3,
        max_depth: int = 4,
    ) -> list[tuple[CategoryNode, float]]:
        frontier: list[tuple[CategoryNode, float]] = [(self.root, 0.0)]
        visited: dict[tuple[str, ...], tuple[CategoryNode, float]] = {(): (self.root, 0.0)}

        for _ in range(max_depth):
            candidates: list[tuple[CategoryNode, float]] = []
            for node, parent_score in frontier:
                for child in node.children.values():
                    score = cosine(query_embedding, child.embedding)
                    blended = 0.75 * score + 0.25 * parent_score
                    candidates.append((child, blended))
            if not candidates:
                break
            candidates.sort(reverse=True, key=lambda item: item[1])
            frontier = candidates[:beam_width]
            for node, score in frontier:
                old = visited.get(node.path)
                if old is None or score > old[1]:
                    visited[node.path] = (node, score)

        ranked = list(visited.values())
        ranked.sort(reverse=True, key=lambda item: item[1])
        return ranked

    def rebuild_dynamic_clusters(
        self,
        segments: dict[str, MemorySegment],
        similarity_threshold: float = 0.72,
    ) -> int:
        moved = 0
        for segment in segments.values():
            current = self.root.find(segment.category_path)
            candidates = [
                node
                for node in self.root.walk()
                if node.path and node.path != segment.category_path
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda node: cosine(segment.embedding, node.embedding))
            if cosine(segment.embedding, best.embedding) >= similarity_threshold:
                if current:
                    current.segment_ids.discard(segment.id)
                segment.category_path = best.path
                best.segment_ids.add(segment.id)
                moved += 1
        if moved:
            self.refresh_embeddings(segments)
        return moved

    def to_dict(self) -> dict:
        return self.root.to_dict()

    @classmethod
    def from_dict(cls, embedding_model: EmbeddingModel, data: dict) -> "CategoryTree":
        return cls(embedding_model, CategoryNode.from_dict(data))

    def _seed_defaults(self) -> None:
        for name, terms in DEFAULT_TOPICS.items():
            parent = self.root.add_child(name, self.embedding_model.encode(" ".join([name, *terms])))
            for term in sorted(list(terms))[:6]:
                parent.add_child(term.title(), self.embedding_model.encode(f"{name} {term}"))

    def _coarse_category(self, text: str) -> str:
        tokens = set(top_terms(text, limit=16))
        best_name = "General"
        best_overlap = 0
        for name, terms in DEFAULT_TOPICS.items():
            overlap = len(tokens & terms)
            if overlap > best_overlap:
                best_name = name
                best_overlap = overlap
        return best_name
