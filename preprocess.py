"""Text normalization and tokenization for BM25 search."""
from __future__ import annotations

import re

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "due",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)

_USES_PREFIX_RE = re.compile(
    r"Treatment and prevention of |Treatment of ",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_uses(text: str) -> str:
    cleaned = _USES_PREFIX_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def split_side_effects(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if "no common side effects" in raw.lower():
        return ["No common side effects reported"]
    parts = re.split(r"(?<=[a-z])(?=[A-Z])|(?<=\))(?=[A-Z])", raw)
    cleaned = [p.strip() for p in parts if p.strip()]
    if len(cleaned) <= 1:
        cleaned = [p.strip() for p in raw.split() if p.strip()]
        if len(cleaned) > 6:
            cleaned = re.findall(
                r"[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+in\s+[a-z]+)?",
                raw,
            ) or [raw]
    return cleaned


def review_score(excellent: int, average: int, poor: int) -> int:
    return excellent * 2 + average - poor
