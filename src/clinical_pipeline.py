from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.assertion import AssertionDetector, AssertionResult
from src.linking import (
    HybridEntityLinker,
    LinkingResult,
    MedicalEntityLinker,
    load_icd_entries,
    load_rxnorm_entries,
)
from src.ner import (
    DEFAULT_LABELS,
    GLiNERModel,
    RawEntityPrediction,
    postprocess_predictions,
    predict_document,
)
from src.preprocessing import EntityAnnotation, MedicalDocument, NormalizedDocument
from src.validation_output import (
    OutputValidationResult,
    serialize_competition_output,
    validate_competition_output,
)


@dataclass
class ClinicalPipelineResult:
    document: MedicalDocument
    normalized: NormalizedDocument
    raw_ner_predictions: list[RawEntityPrediction] = field(default_factory=list)
    entities: list[EntityAnnotation] = field(default_factory=list)
    assertions: list[AssertionResult] = field(default_factory=list)
    links: list[LinkingResult] = field(default_factory=list)
    validation: OutputValidationResult = field(default_factory=OutputValidationResult)

    def competition_output(self, *, strict: bool = True) -> dict[str, Any]:
        return serialize_competition_output(
            self.document, self.entities, strict=strict
        )


class ClinicalNLPPipeline:
    """Self-hosted NER → assertion → ontology linking orchestration."""

    def __init__(
        self,
        ner_model: GLiNERModel,
        *,
        assertion_detector: AssertionDetector | None = None,
        entity_linker: MedicalEntityLinker | None = None,
        labels: list[str] | None = None,
        ner_threshold: float = 0.5,
        min_ner_confidence: float = 0.3,
        normalize_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.ner_model = ner_model
        self.assertion_detector = assertion_detector or AssertionDetector()
        self.entity_linker = entity_linker or MedicalEntityLinker()
        self.labels = labels or list(DEFAULT_LABELS)
        self.ner_threshold = ner_threshold
        self.min_ner_confidence = min_ner_confidence
        self.normalize_kwargs = dict(normalize_kwargs or {})

    def process(self, document: MedicalDocument) -> ClinicalPipelineResult:
        normalized, raw_predictions = predict_document(
            document,
            self.ner_model,
            labels=self.labels,
            threshold=self.ner_threshold,
            normalize_kwargs=self.normalize_kwargs,
        )
        entities = postprocess_predictions(
            raw_predictions,
            normalized,
            min_confidence=self.min_ner_confidence,
        )
        entities, assertion_results = self.assertion_detector.annotate(
            document, entities
        )
        entities, linking_results = self.entity_linker.link_entities(
            document, entities
        )
        validation = validate_competition_output(document, entities)
        return ClinicalPipelineResult(
            document=document,
            normalized=normalized,
            raw_ner_predictions=raw_predictions,
            entities=entities,
            assertions=assertion_results,
            links=linking_results,
            validation=validation,
        )

    def process_many(
        self, documents: Iterable[MedicalDocument]
    ) -> list[ClinicalPipelineResult]:
        return [self.process(document) for document in documents]


def build_default_medical_linker(
    *,
    icd_path: str | Path = "work/icd_mapping_final.json",
    drug_path: str | Path = "work/drug_mapping_final.json",
) -> MedicalEntityLinker:
    return MedicalEntityLinker(
        icd_linker=HybridEntityLinker(load_icd_entries(icd_path)),
        drug_linker=HybridEntityLinker(load_rxnorm_entries(drug_path)),
    )

