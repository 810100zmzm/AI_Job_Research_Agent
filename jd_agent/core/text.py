"""Small text and vector helpers shared by memory and knowledge retrieval."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text; Chinese tokens include adjacent bigrams."""
    raw = _TOKEN_RE.findall(str(text or "").casefold())
    tokens = list(raw)
    for index in range(len(raw) - 1):
        left, right = raw[index], raw[index + 1]
        if len(left) == 1 and len(right) == 1 and "\u4e00" <= left <= "\u9fff" and "\u4e00" <= right <= "\u9fff":
            tokens.append(left + right)
    return tokens


def lexical_overlap(query: str, text: str) -> float:
    """Return a lightweight overlap score in the range [0, 1]."""
    query_tokens = Counter(tokenize(query))
    text_tokens = Counter(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    shared = sum(min(count, text_tokens.get(token, 0)) for token, count in query_tokens.items())
    return shared / max(1, sum(query_tokens.values()))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def hash_embedding(text: str, dimensions: int = 384) -> List[float]:
    """Deterministic local embedding used when no remote embedding model is configured."""
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        sign = 1.0 if value & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def unique_tokens(texts: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    return counts
