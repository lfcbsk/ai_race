from __future__ import annotations

from typing import Any

from src.preprocessing import EntityAnnotation, MedicalDocument

from .validator import validate_competition_output


def serialize_competition_output(
    document: MedicalDocument,
    entities: list[EntityAnnotation],
    *,
    include_text: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    validation = validate_competition_output(document, entities)
    if strict and not validation.valid:
        raise ValueError("; ".join(validation.errors))
    output: dict[str, Any] = {
        "note_id": document.document_id,
        "entities": [entity.to_dict() for entity in entities],
    }
    if include_text:
        output["text"] = document.raw_text
    if not validation.valid or validation.warnings:
        output["validation"] = {
            "valid": validation.valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
        }
    return output

