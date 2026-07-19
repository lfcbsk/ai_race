from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


OntologyName = Literal["ICD10", "RXNORM"]


@dataclass(frozen=True)
class OntologyEntry:
    code: str
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    ontology: OntologyName = "ICD10"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        return " ".join((self.name, *self.aliases, self.description)).strip()


@dataclass(frozen=True)
class RetrievalHit:
    code: str
    score: float
    rank: int
    source: str


@dataclass
class LinkingCandidate:
    entry: OntologyEntry
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    reranker_score: float | None = None
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.entry.code,
            "name": self.entry.name,
            "ontology": self.entry.ontology,
            "retrieval_scores": self.retrieval_scores,
            "fused_score": round(self.fused_score, 6),
            "reranker_score": self.reranker_score,
            "final_score": round(self.final_score, 6),
        }


@dataclass
class LinkingResult:
    document_id: str
    entity_index: int
    mention: str
    ontology: OntologyName
    candidates: list[LinkingCandidate] = field(default_factory=list)
    selected_code: str | None = None
    confidence: float = 0.0
    method: str = "not_found"
    needs_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "entity_index": self.entity_index,
            "mention": self.mention,
            "ontology": self.ontology,
            "selected_code": self.selected_code,
            "confidence": round(self.confidence, 6),
            "method": self.method,
            "needs_review": self.needs_review,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

