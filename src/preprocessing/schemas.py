from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    @property
    def position(self) -> list[int]:
        return [self.start, self.end]

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "text": self.text,
            "type": self.entity_type,
            "position": self.position,
        }

        if self.entity_type in {
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
            "THUỐC",
        }:
            output["assertions"] = self.assertions

        if self.entity_type in {
            "CHẨN_ĐOÁN",
            "THUỐC",
        }:
            output["candidates"] = self.candidates

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "normalized_to_raw": (
                self.normalized_to_raw
            ),
            "entities": [
                entity.to_dict()
                for entity in self.entities
            ],
            "metadata": self.metadata,
        }