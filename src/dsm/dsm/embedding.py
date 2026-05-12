from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_+#.-]+", re.UNICODE)


class EmbeddingModel(Protocol):
    dim: int

    def encode(self, text: str) -> list[float]:
        """Return a normalized vector for text."""


class HashEmbeddingModel:
    """Deterministic local semantic-ish encoder with no network dependency."""

    def __init__(self, dim: int = 384):
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            primary = int.from_bytes(digest[:4], "big") % self.dim
            secondary = int.from_bytes(digest[4:8], "big") % self.dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 16) / 16.0
            vector[primary] += sign * weight
            vector[secondary] += sign * 0.35 * weight

            for ngram in character_ngrams(token):
                nd = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(nd[:4], "big") % self.dim
                vector[idx] += 0.08 if nd[4] % 2 == 0 else -0.08

        return normalize(vector)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def character_ngrams(token: str, n: int = 3) -> Iterable[str]:
    if len(token) <= n:
        yield token
        return
    for i in range(len(token) - n + 1):
        yield token[i : i + n]


def normalize(vector: Iterable[float]) -> list[float]:
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    left = list(a)
    right = list(b)
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(x * y for x, y in zip(left, right)))


def mean_embedding(vectors: Iterable[Iterable[float]], dim: int) -> list[float]:
    acc = [0.0] * dim
    count = 0
    for vector in vectors:
        values = list(vector)
        if len(values) != dim:
            continue
        count += 1
        for i, value in enumerate(values):
            acc[i] += float(value)
    if count == 0:
        return acc
    return normalize(value / count for value in acc)


def top_terms(text: str, limit: int = 3) -> list[str]:
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]
