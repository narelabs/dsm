from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from dsm.category import CategoryTree
from dsm.embedding import EmbeddingModel, HashEmbeddingModel, cosine, top_terms
from dsm.graph import MemoryGraph
from dsm.index import SegmentIndex
from dsm.models import ActiveContext, MemorySegment, PriorityVector, RouteResult, now_ts
from dsm.storage import JsonStorage


class DynamicSegmentedMemory:
    """Dynamic Segmented Memory engine.

    DSM state is ({S_i}, T, G): segments, hierarchy and graph. Queries route through
    categories, segment embeddings, graph expansion and priority scoring, then build
    a bounded active context instead of exposing the whole memory to the model.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        embedding_model: EmbeddingModel | None = None,
        segment_token_limit: int = 20_000,
        active_segment_limit: int = 5,
        active_token_budget: int = 100_000,
    ):
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.segment_token_limit = segment_token_limit
        self.active_segment_limit = active_segment_limit
        self.active_token_budget = active_token_budget
        self.storage = JsonStorage(storage_path or Path(".dsm") / "memory.json")

        self.segments: dict[str, MemorySegment] = {}
        self.categories = CategoryTree(self.embedding_model)
        self.graph = MemoryGraph()
        self.index = SegmentIndex()

        if self.storage.exists():
            self.load()

    def write(
        self,
        text: str,
        *,
        description: str | None = None,
        category_path: tuple[str, ...] | list[str] | str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        update_existing: bool = True,
        link_related: bool = True,
    ) -> list[MemorySegment]:
        chunks = chunk_text(text, self.segment_token_limit)
        written: list[MemorySegment] = []

        for chunk in chunks:
            chunk_description = description or summarize(chunk)
            path = normalize_path(category_path) if category_path else self.categories.choose_path(chunk)
            embedding = self.embedding_model.encode(f"{chunk_description}\n{chunk}")
            existing = self._find_existing(embedding) if update_existing else None

            if existing:
                merged = merge_text(existing.text, chunk, self.segment_token_limit)
                existing.update_text(merged, summarize(merged), self.embedding_model.encode(merged))
                existing.priorities.importance = max(existing.priorities.importance, importance)
                existing.priorities.touch(cosine(embedding, existing.embedding), boost=0.05)
                existing.metadata.update(metadata or {})
                segment = existing
            else:
                segment = MemorySegment(
                    text=chunk,
                    description=chunk_description,
                    category_path=tuple(path),
                    embedding=embedding,
                    priorities=PriorityVector(importance=importance),
                    metadata=metadata or {},
                )
                self.segments[segment.id] = segment
                self.categories.add_segment(segment)
                self.graph.add_node(segment.id)

            written.append(segment)

        if link_related:
            self._link_related(written)

        self._rebuild_index()
        self.categories.refresh_embeddings(self.segments)
        return written

    def route(
        self,
        query: str,
        *,
        k: int | None = None,
        category_beam: int = 4,
        graph_hops: int = 2,
        similarity_floor: float = -1.0,
    ) -> list[RouteResult]:
        if not self.segments:
            return []

        query_embedding = self.embedding_model.encode(query)
        category_routes = self.categories.route_categories(query_embedding, beam_width=category_beam)
        category_scores = {node.path: score for node, score in category_routes}
        candidate_ids: set[str] = set()

        for node, _ in category_routes:
            candidate_ids.update(node.segment_ids)
            for child in node.children.values():
                candidate_ids.update(child.segment_ids)

        direct_hits = self.index.search(query_embedding, max(k or self.active_segment_limit, 16))
        candidate_ids.update(segment_id for segment_id, score in direct_hits if score >= similarity_floor)

        seed_ids = [segment_id for segment_id, _ in direct_hits[: max(1, self.active_segment_limit)]]
        graph_distances = self.graph.expand(seed_ids, max_hops=graph_hops)
        candidate_ids.update(graph_distances)

        scored: list[RouteResult] = []
        current_time = now_ts()
        for segment_id in candidate_ids:
            segment = self.segments.get(segment_id)
            if not segment:
                continue

            similarity = cosine(query_embedding, segment.embedding)
            category_score = best_category_score(segment.category_path, category_scores)
            graph_distance = graph_distances.get(segment_id, 99)
            graph_bonus = 0.0 if graph_distance == 99 else 0.08 / (graph_distance + 1)
            age_seconds = current_time - segment.last_accessed_at
            priority_score = segment.priorities.total(similarity, age_seconds)
            total = (
                0.58 * similarity
                + 0.24 * priority_score
                + 0.12 * category_score
                + graph_bonus
            )
            scored.append(
                RouteResult(
                    segment=segment,
                    similarity=similarity,
                    priority_score=priority_score,
                    graph_distance=graph_distance,
                    total_score=total,
                    category_score=category_score,
                )
            )

        scored.sort(reverse=True, key=lambda item: item.total_score)
        selected = scored[: max(0, k or self.active_segment_limit)]
        for item in selected:
            item.segment.touch(item.similarity)
        return selected

    def active_context(
        self,
        query: str,
        *,
        k: int | None = None,
        token_budget: int | None = None,
    ) -> ActiveContext:
        budget = token_budget or self.active_token_budget
        selected: list[RouteResult] = []
        used = 0
        for item in self.route(query, k=k):
            cost = item.segment.estimated_tokens
            if selected and used + cost > budget:
                continue
            selected.append(item)
            used += cost

        global_summary = self._global_summary(selected)
        parts = [
            f"QUERY:\n{query}",
            f"GLOBAL SUMMARY TOKENS:\n{global_summary}",
        ]
        for index, item in enumerate(selected, start=1):
            segment = item.segment
            path = " → ".join(segment.category_path)
            parts.append(
                "\n".join(
                    [
                        f"[SEGMENT {index}] id={segment.id}",
                        f"category={path}",
                        f"description={segment.description}",
                        (
                            "scores="
                            f"sim:{item.similarity:.3f} "
                            f"priority:{item.priority_score:.3f} "
                            f"category:{item.category_score:.3f} "
                            f"total:{item.total_score:.3f}"
                        ),
                        "text:",
                        segment.text,
                    ]
                )
            )

        return ActiveContext(
            query=query,
            selected=selected,
            context_text="\n\n---\n\n".join(parts),
            token_budget=budget,
            estimated_tokens=used + len(query.split()) + len(global_summary.split()),
            global_summary=global_summary,
        )

    def update_from_interaction(
        self,
        query: str,
        answer: str,
        *,
        importance: float = 0.6,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemorySegment]:
        routed = self.route(query, k=1)
        if routed and routed[0].similarity >= 0.62:
            target = routed[0].segment
            text = merge_text(target.text, f"Query: {query}\nAnswer: {answer}", self.segment_token_limit)
            target.update_text(text, summarize(text), self.embedding_model.encode(text))
            target.priorities.importance = max(target.priorities.importance, importance)
            target.priorities.touch(routed[0].similarity)
            target.metadata.update(metadata or {})
            self._rebuild_index()
            self.categories.refresh_embeddings(self.segments)
            return [target]

        return self.write(
            f"Query: {query}\nAnswer: {answer}",
            description=summarize(query),
            importance=importance,
            metadata=metadata,
        )

    def rebuild_structure(self) -> int:
        moved = self.categories.rebuild_dynamic_clusters(self.segments)
        self._link_related(list(self.segments.values()))
        self._rebuild_index()
        return moved

    def prune(
        self,
        *,
        max_segments: int | None = None,
        min_priority: float = 0.08,
        archive_path: str | Path | None = None,
    ) -> list[MemorySegment]:
        current_time = now_ts()
        ranked = sorted(
            self.segments.values(),
            key=lambda segment: segment.priorities.total(0.0, current_time - segment.last_accessed_at),
        )
        remove_ids: set[str] = set()

        for segment in ranked:
            score = segment.priorities.total(0.0, current_time - segment.last_accessed_at)
            if score < min_priority:
                remove_ids.add(segment.id)

        if max_segments is not None and len(self.segments) - len(remove_ids) > max_segments:
            needed = len(self.segments) - len(remove_ids) - max_segments
            for segment in ranked:
                if needed <= 0:
                    break
                if segment.id not in remove_ids:
                    remove_ids.add(segment.id)
                    needed -= 1

        removed = [self.segments[segment_id] for segment_id in remove_ids if segment_id in self.segments]
        if archive_path and removed:
            archive = JsonStorage(archive_path)
            archive.save({"segments": [segment.to_dict() for segment in removed]})

        for segment in removed:
            self.categories.remove_segment(segment)
            self.graph.remove_node(segment.id)
            del self.segments[segment.id]

        if removed:
            self._rebuild_index()
            self.categories.refresh_embeddings(self.segments)
        return removed

    def decay_priorities(self, amount: float = 0.02) -> None:
        for segment in self.segments.values():
            segment.priorities.decay(amount)

    def save(self) -> None:
        self.storage.save(
            {
                "version": 1,
                "segment_token_limit": self.segment_token_limit,
                "active_segment_limit": self.active_segment_limit,
                "active_token_budget": self.active_token_budget,
                "embedding_dim": self.embedding_model.dim,
                "segments": [segment.to_dict() for segment in self.segments.values()],
                "categories": self.categories.to_dict(),
                "graph": self.graph.to_dict(),
            }
        )

    def load(self) -> None:
        data = self.storage.load()
        self.segment_token_limit = int(data.get("segment_token_limit", self.segment_token_limit))
        self.active_segment_limit = int(data.get("active_segment_limit", self.active_segment_limit))
        self.active_token_budget = int(data.get("active_token_budget", self.active_token_budget))
        self.segments = {
            segment.id: segment
            for segment in (
                MemorySegment.from_dict(item) for item in data.get("segments", [])
            )
        }
        if data.get("categories"):
            self.categories = CategoryTree.from_dict(self.embedding_model, data["categories"])
        else:
            self.categories = CategoryTree(self.embedding_model)
            for segment in self.segments.values():
                self.categories.add_segment(segment)
        self.graph = MemoryGraph.from_dict(data.get("graph", {}))
        for segment in self.segments.values():
            self.graph.add_node(segment.id)
            for linked_id in list(segment.links):
                if linked_id in self.segments:
                    self.graph.add_edge(segment.id, linked_id)
        self._rebuild_index()
        self.categories.refresh_embeddings(self.segments)

    def stats(self) -> dict[str, Any]:
        nodes = self.categories.root.walk()
        edge_count = sum(len(v) for v in self.graph.edges.values()) // 2
        tokens = sum(segment.estimated_tokens for segment in self.segments.values())
        return {
            "segments": len(self.segments),
            "categories": len(nodes),
            "graph_edges": edge_count,
            "estimated_tokens": tokens,
            "index_size": self.index.size,
            "segment_token_limit": self.segment_token_limit,
            "active_segment_limit": self.active_segment_limit,
            "active_token_budget": self.active_token_budget,
        }

    def _find_existing(self, embedding: list[float], threshold: float = 0.84) -> MemorySegment | None:
        if not self.segments:
            return None
        hits = self.index.search(embedding, 1)
        if not hits:
            return None
        segment_id, score = hits[0]
        if score >= threshold:
            return self.segments.get(segment_id)
        return None

    def _link_related(self, segments: list[MemorySegment], threshold: float = 0.58, max_links: int = 4) -> None:
        for segment in segments:
            self.graph.add_node(segment.id)
            candidates = [
                other
                for other in self.segments.values()
                if other.id != segment.id
            ]
            scored = [
                (cosine(segment.embedding, other.embedding), other)
                for other in candidates
                if share_terms(segment.text, other.text)
                or same_parent(segment.category_path, other.category_path)
                or cosine(segment.embedding, other.embedding) >= threshold
            ]
            scored.sort(reverse=True, key=lambda item: item[0])
            for _, other in scored[:max_links]:
                self.graph.add_edge(segment.id, other.id)
                segment.links.add(other.id)
                other.links.add(segment.id)

    def _rebuild_index(self) -> None:
        self.index.rebuild(self.segments)

    def _global_summary(self, selected: list[RouteResult]) -> str:
        if not selected:
            return "No active memory segments selected."
        lines = []
        for item in selected:
            segment = item.segment
            lines.append(
                f"- {' → '.join(segment.category_path)}: {segment.description} "
                f"(score={item.total_score:.3f})"
            )
        return "\n".join(lines)


def chunk_text(text: str, token_limit: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if token_limit <= 0:
        raise ValueError("token_limit must be positive")
    return [" ".join(words[i : i + token_limit]) for i in range(0, len(words), token_limit)]


def summarize(text: str, max_words: int = 24) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    words = compact.split()
    if not words:
        return "Empty segment"
    if len(words) <= max_words:
        return compact
    terms = top_terms(compact, limit=4)
    prefix = " ".join(words[:max_words])
    if terms:
        return f"{prefix}… [{', '.join(terms)}]"
    return f"{prefix}…"


def merge_text(existing: str, addition: str, token_limit: int) -> str:
    old = existing.strip()
    new = addition.strip()
    if not old:
        merged = new
    elif new in old:
        merged = old
    else:
        merged = f"{old}\n\n{new}"
    words = merged.split()
    if len(words) <= token_limit:
        return merged
    return " ".join(words[-token_limit:])


def normalize_path(path: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(path, str):
        parts = re.split(r"[>/→|]+", path)
    else:
        parts = [str(part) for part in path]
    cleaned = tuple(part.strip() for part in parts if part and part.strip())
    return cleaned or ("General",)


def best_category_score(path: tuple[str, ...], category_scores: dict[tuple[str, ...], float]) -> float:
    best = 0.0
    for i in range(len(path) + 1):
        prefix = path[:i]
        best = max(best, category_scores.get(prefix, 0.0))
    return best


def share_terms(left: str, right: str) -> bool:
    left_terms = set(top_terms(left, limit=8))
    right_terms = set(top_terms(right, limit=8))
    return bool(left_terms & right_terms)


def same_parent(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if not left or not right:
        return False
    depth = min(len(left), len(right), 2)
    return left[:depth] == right[:depth]


def sparse_attention_cost(active_tokens: int, total_tokens: int) -> dict[str, float]:
    active = max(0, active_tokens)
    total = max(active, total_tokens)
    dense = float(total * total)
    sparse = float(active * active)
    return {
        "dense_attention_ops": dense,
        "active_attention_ops": sparse,
        "reduction_ratio": math.inf if sparse == 0 else dense / sparse,
    }
