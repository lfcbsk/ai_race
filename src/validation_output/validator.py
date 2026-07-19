from __future__ import annotations

from dataclasses import dataclass, field

from src.preprocessing import EntityAnnotation, MedicalDocument


ALLOWED_ENTITY_TYPES = {
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
}
ALLOWED_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}


@dataclass
class OutputValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_competition_output(
    document: MedicalDocument,
    entities: list[EntityAnnotation],
) -> OutputValidationResult:
    result = OutputValidationResult()
    seen: set[tuple[int, int, str]] = set()
    for index, entity in enumerate(entities):
        key = (entity.start, entity.end, entity.entity_type)
        if key in seen:
            result.errors.append(f"Entity {index}: duplicate {key}.")
        seen.add(key)
        if entity.entity_type not in ALLOWED_ENTITY_TYPES:
            result.errors.append(
                f"Entity {index}: type không hợp lệ {entity.entity_type!r}."
            )
        if not 0 <= entity.start < entity.end <= len(document.raw_text):
            result.errors.append(f"Entity {index}: span ngoài raw text.")
            continue
        actual = document.raw_text[entity.start : entity.end]
        if actual != entity.text:
            result.errors.append(
                f"Entity {index}: text không khớp span; {entity.text!r} != {actual!r}."
            )
        unknown_assertions = set(entity.assertions) - ALLOWED_ASSERTIONS
        if unknown_assertions:
            result.errors.append(
                f"Entity {index}: assertion không hợp lệ {sorted(unknown_assertions)}."
            )
        if entity.entity_type not in {"CHẨN_ĐOÁN", "THUỐC"} and entity.candidates:
            result.warnings.append(
                f"Entity {index}: candidates sẽ bị bỏ vì type không hỗ trợ linking."
            )
    result.valid = not result.errors
    return result

