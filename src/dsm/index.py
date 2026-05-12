from __future__ import annotations

from dsm.embedding import cosine
from dsm.models import MemorySegment


class SegmentIndex:
    """ANN-friendly segment index; exact cosine fallback without heavy dependencies."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []

    def rebuild(self, segments: dict[str, MemorySegment]) -> None:
        self._ids = []
        self._vectors = []
        for segment_id, segment in segments.items():
            self._ids.append(segment_id)
            self._vectors.append(segment.embedding)

    def search(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        scored = [
            (segment_id, cosine(query_embedding, embedding))
            for segment_id, embedding in zip(self._ids, self._vectors)
        ]
        scored.sort(reverse=True, key=lambda item: item[1])
        return scored[: max(0, k)]

    @property
    def size(self) -> int:
        return len(self._ids)
