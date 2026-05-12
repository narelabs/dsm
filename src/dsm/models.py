from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def now_ts() -> float:
    return time.time()


@dataclass(slots=True)
class PriorityVector:
    relevance: float = 0.5
    importance: float = 0.5
    recency: float = 1.0
    frequency: float = 0.0

    def clamp(self) -> None:
        self.relevance = _clamp01(self.relevance)
        self.importance = _clamp01(self.importance)
        self.recency = _clamp01(self.recency)
        self.frequency = _clamp01(self.frequency)

    def total(self, similarity: float, age_seconds: float) -> float:
        recency_decay = 1.0 / (1.0 + max(age_seconds, 0.0) / 86_400.0)
        recency_score = 0.5 * self.recency + 0.5 * recency_decay
        score = (
            0.42 * similarity
            + 0.22 * self.relevance
            + 0.18 * self.importance
            + 0.10 * recency_score
            + 0.08 * self.frequency
        )
        return _clamp01(score)

    def touch(self, similarity: float, boost: float = 0.08) -> None:
        self.relevance = max(self.relevance, _clamp01(similarity))
        self.recency = 1.0
        self.frequency = _clamp01(self.frequency + boost)
        self.clamp()

    def decay(self, amount: float = 0.02) -> None:
        self.recency = _clamp01(self.recency - amount)
        self.relevance = _clamp01(self.relevance - amount * 0.4)

    def to_dict(self) -> dict[str, float]:
        return {
            "relevance": self.relevance,
            "importance": self.importance,
            "recency": self.recency,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PriorityVector":
        if not data:
            return cls()
        item = cls(
            relevance=float(data.get("relevance", 0.5)),
            importance=float(data.get("importance", 0.5)),
            recency=float(data.get("recency", 1.0)),
            frequency=float(data.get("frequency", 0.0)),
        )
        item.clamp()
        return item


@dataclass(slots=True)
class MemorySegment:
    text: str
    description: str
    category_path: tuple[str, ...]
    embedding: list[float]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    links: set[str] = field(default_factory=set)
    priorities: PriorityVector = field(default_factory=PriorityVector)
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    last_accessed_at: float = field(default_factory=now_ts)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.text.split()))

    def touch(self, similarity: float) -> None:
        self.last_accessed_at = now_ts()
        self.access_count += 1
        self.priorities.touch(similarity)

    def update_text(self, text: str, description: str, embedding: list[float]) -> None:
        self.text = text
        self.description = description
        self.embedding = embedding
        self.updated_at = now_ts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "description": self.description,
            "category_path": list(self.category_path),
            "embedding": self.embedding,
            "links": sorted(self.links),
            "priorities": self.priorities.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorySegment":
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            description=str(data.get("description", "")),
            category_path=tuple(str(p) for p in data.get("category_path", ["General"])),
            embedding=[float(v) for v in data.get("embedding", [])],
            links={str(v) for v in data.get("links", [])},
            priorities=PriorityVector.from_dict(data.get("priorities")),
            created_at=float(data.get("created_at", now_ts())),
            updated_at=float(data.get("updated_at", now_ts())),
            last_accessed_at=float(data.get("last_accessed_at", now_ts())),
            access_count=int(data.get("access_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class CategoryNode:
    name: str
    path: tuple[str, ...]
    embedding: list[float]
    children: dict[str, "CategoryNode"] = field(default_factory=dict)
    segment_ids: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    def add_child(self, name: str, embedding: list[float]) -> "CategoryNode":
        if name not in self.children:
            self.children[name] = CategoryNode(name=name, path=self.path + (name,), embedding=embedding)
        return self.children[name]

    def find(self, path: tuple[str, ...]) -> "CategoryNode | None":
        if not path:
            return self
        head, *tail = path
        child = self.children.get(head)
        if child is None:
            return None
        return child.find(tuple(tail))

    def walk(self) -> list["CategoryNode"]:
        nodes = [self]
        for child in self.children.values():
            nodes.extend(child.walk())
        return nodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": list(self.path),
            "embedding": self.embedding,
            "segment_ids": sorted(self.segment_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "children": {name: child.to_dict() for name, child in sorted(self.children.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoryNode":
        node = cls(
            name=str(data.get("name", "Memory")),
            path=tuple(str(p) for p in data.get("path", ())),
            embedding=[float(v) for v in data.get("embedding", [])],
            segment_ids={str(v) for v in data.get("segment_ids", [])},
            created_at=float(data.get("created_at", now_ts())),
            updated_at=float(data.get("updated_at", now_ts())),
        )
        node.children = {
            str(name): cls.from_dict(child) for name, child in data.get("children", {}).items()
        }
        return node


@dataclass(slots=True)
class RouteResult:
    segment: MemorySegment
    similarity: float
    priority_score: float
    graph_distance: int
    total_score: float
    category_score: float = 0.0


@dataclass(slots=True)
class ActiveContext:
    query: str
    selected: list[RouteResult]
    context_text: str
    token_budget: int
    estimated_tokens: int
    global_summary: str

    @property
    def segment_ids(self) -> list[str]:
        return [item.segment.id for item in self.selected]


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
