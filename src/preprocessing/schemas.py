from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import (
    ASSERTION_ENTITY_TYPES,
    CANDIDATE_ENTITY_TYPES,
)


@dataclass
class EntityAnnotation:
    text: str
    entity_type: str
    start: int
    end: int

    assertions: list[str] = field(
        default_factory=list
    )
    candidates: list[str] = field(
        default_factory=list
    )

    # Metadata nội bộ, không đưa vào submission.
    confidence: float | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def position(self) -> list[int]:
        return [self.start, self.end]

    def to_dict(
        self,
        *,
        include_internal: bool = False,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "text": self.text,
            "type": self.entity_type,
            "position": self.position,
        }

        if (
            self.entity_type
            in ASSERTION_ENTITY_TYPES
        ):
            output["assertions"] = list(
                self.assertions
            )

        if (
            self.entity_type
            in CANDIDATE_ENTITY_TYPES
        ):
            output["candidates"] = list(
                self.candidates
            )

        if include_internal:
            output["confidence"] = self.confidence
            output["source"] = self.source
            output["metadata"] = dict(
                self.metadata
            )

        return output


@dataclass
class MedicalDocument:
    document_id: str
    raw_text: str

    normalized_text: str | None = None

    normalized_to_raw: list[int] = field(
        default_factory=list
    )

    entities: list[EntityAnnotation] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
        *,
        include_internal: bool = False,
    ) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "normalized_to_raw": (
                self.normalized_to_raw
            ),
            "entities": [
                entity.to_dict(
                    include_internal=include_internal
                )
                for entity in self.entities
            ],
            "metadata": self.metadata,
        }