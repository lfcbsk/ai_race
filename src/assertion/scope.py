from __future__ import annotations

import re

from .schemas import AssertionCue


_HARD_BOUNDARY = re.compile(r"[.!?;\n]")
_CONTRAST_BOUNDARY = re.compile(
    r"\b(?:nhưng|tuy nhiên|song|dù vậy|mặc dù|trong khi)\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def cue_scope(
    text: str,
    cue: AssertionCue,
    *,
    max_scope_characters: int = 180,
) -> tuple[int, int]:
    """Return the forward clause influenced by an assertion cue."""
    if cue.direction == "backward":
        scope_start = max(0, cue.start - max_scope_characters)
        prefix = text[scope_start:cue.start]
        boundaries = [match.end() for match in _HARD_BOUNDARY.finditer(prefix)]
        boundaries.extend(match.end() for match in _CONTRAST_BOUNDARY.finditer(prefix))
        if boundaries:
            scope_start += max(boundaries)
        return scope_start, cue.start

    scope_start = cue.end
    scope_end = min(len(text), cue.end + max_scope_characters)
    tail = text[scope_start:scope_end]
    boundaries = [match.start() for match in _HARD_BOUNDARY.finditer(tail)]
    boundaries.extend(match.start() for match in _CONTRAST_BOUNDARY.finditer(tail))
    if boundaries:
        scope_end = scope_start + min(boundaries)
    return scope_start, scope_end


def cue_applies_to_span(
    text: str,
    cue: AssertionCue,
    entity_start: int,
    entity_end: int,
) -> tuple[bool, tuple[int, int], float]:
    scope = cue_scope(text, cue)
    applies = scope[0] <= entity_start and entity_end <= scope[1]
    if not applies:
        return False, scope, 0.0

    distance = (
        max(0, cue.start - entity_end)
        if cue.direction == "backward"
        else max(0, entity_start - cue.end)
    )
    distance_factor = max(0.55, 1.0 - distance / 240.0)
    return True, scope, round(cue.confidence * distance_factor, 4)
