from __future__ import annotations

from dataclasses import replace

from src.preprocessing import EntityAnnotation, MedicalDocument

from .rules import find_assertion_cues
from .schemas import AssertionEvidence, AssertionLabel, AssertionResult
from .scope import cue_applies_to_span
from .verifier import AssertionVerifier


class AssertionDetector:
    supported_entity_types = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}

    def __init__(
        self,
        verifier: AssertionVerifier | None = None,
        *,
        verify_below: float = 0.82,
    ) -> None:
        self.verifier = verifier
        self.verify_below = verify_below

    def detect_entity(
        self,
        document: MedicalDocument,
        entity: EntityAnnotation,
        entity_index: int,
    ) -> AssertionResult:
        if entity.entity_type not in self.supported_entity_types:
            return AssertionResult(
                document_id=document.document_id,
                entity_index=entity_index,
            )
        evidence: list[AssertionEvidence] = []
        confidence_by_label: dict[AssertionLabel, float] = {}
        for cue in find_assertion_cues(document.raw_text):
            applies, scope, confidence = cue_applies_to_span(
                document.raw_text, cue, entity.start, entity.end
            )
            if not applies:
                continue
            confidence_by_label[cue.label] = max(
                confidence_by_label.get(cue.label, 0.0), confidence
            )
            evidence.append(
                AssertionEvidence(
                    label=cue.label,
                    cue=cue.text,
                    cue_position=(cue.start, cue.end),
                    scope=scope,
                    confidence=confidence,
                )
            )

        assertions = sorted(confidence_by_label)
        confidence = (
            min(confidence_by_label.values()) if confidence_by_label else 1.0
        )
        needs_verification = bool(assertions) and confidence < self.verify_below
        result = AssertionResult(
            document_id=document.document_id,
            entity_index=entity_index,
            assertions=assertions,
            confidence=confidence,
            evidence=evidence,
            needs_verification=needs_verification,
        )
        if needs_verification and self.verifier is not None:
            labels, verified_confidence = self.verifier.verify(
                document.raw_text, entity, result
            )
            result.assertions = labels
            result.confidence = verified_confidence
            result.source = "local_verifier"
            result.needs_verification = False
        return result

    def annotate(
        self,
        document: MedicalDocument,
        entities: list[EntityAnnotation] | None = None,
    ) -> tuple[list[EntityAnnotation], list[AssertionResult]]:
        source_entities = document.entities if entities is None else entities
        annotated: list[EntityAnnotation] = []
        results: list[AssertionResult] = []
        for index, entity in enumerate(source_entities):
            result = self.detect_entity(document, entity, index)
            results.append(result)
            annotated.append(replace(entity, assertions=list(result.assertions)))
        return annotated, results
