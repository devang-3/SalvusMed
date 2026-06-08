"""Parse composition strings and build match signatures."""
from __future__ import annotations

import re

_COMPONENT_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")


def normalize_salt(name: str) -> str:
    return " ".join(name.lower().split())


def normalize_strength(strength: str) -> str:
    return strength.lower().replace(" ", "")


def parse_component(text: str) -> tuple[str, str] | None:
    text = text.strip()
    if not text:
        return None
    match = _COMPONENT_RE.match(text)
    if not match:
        return (normalize_salt(text), "")
    return (normalize_salt(match.group(1)), normalize_strength(match.group(2)))


def composition_signature(composition: str) -> str:
    parts: list[tuple[str, str]] = []
    for chunk in (composition or "").split("+"):
        parsed = parse_component(chunk)
        if parsed:
            parts.append(parsed)
    if not parts:
        return ""
    parts.sort()
    return "|".join(f"{salt}:{strength}" for salt, strength in parts)
