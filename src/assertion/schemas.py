from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AssertionLabel = Literal["isNegated", "isHistorical", "isFamily"]


@dataclass(frozen=True)
class AssertionCue:
    label: AssertionLabel
    text: str
    start: int
    end: int
    confidence: float
    direction: Literal["forward", "backward"] = "forward"


@dataclass(frozen=True)
class AssertionEvidence:
    label: AssertionLabel
    cue: str
    cue_position: tuple[int, int]
    scope: tuple[int, int]
    confidence: float


@dataclass
class AssertionResult:
    document_id: str
    entity_index: int
    assertions: list[AssertionLabel] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "rule_scope"
    evidence: list[AssertionEvidence] = field(default_factory=list)
    needs_verification: bool = False
