from __future__ import annotations
import hashlib
import re
from typing import Any

try:
    from scripts.common import (
        ASSERTION_ENTITY_TYPES,
        CANDIDATE_ENTITY_TYPES,
        CaseSpec,
        FINAL_DIR,
        GENERATED_DIR,
        MARKER_PATTERN,
        PROCESSED_DIR,
        VALID_ASSERTIONS,
        VALID_ENTITY_TYPES,
        case_spec_from_dict,
        entity_section_map,
        ensure_directories,
        iter_entities,
        read_jsonl,
        validate_case_spec,
        validate_markers,
        validate_rendered_sample,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    # Support direct execution: python scripts/validate_medical.py
    from common import (
        ASSERTION_ENTITY_TYPES,
        CANDIDATE_ENTITY_TYPES,
        CaseSpec,
        FINAL_DIR,
        GENERATED_DIR,
        MARKER_PATTERN,
        PROCESSED_DIR,
        VALID_ASSERTIONS,
        VALID_ENTITY_TYPES,
        case_spec_from_dict,
        entity_section_map,
        ensure_directories,
        iter_entities,
        read_jsonl,
        validate_case_spec,
        validate_markers,
        validate_rendered_sample,
        write_json,
        write_jsonl,
    )


NEGATION_CUES = {
    "không",
    "không có",
    "không ghi nhận",
    "phủ nhận",
    "chưa từng",
    "không xuất hiện",
}

FAMILY_CUES = {
    "bố",
    "mẹ",
    "cha",
    "gia đình",
    "người nhà",
    "anh trai",
    "chị gái",
}

HISTORICAL_CUES = {
    "tiền sử",
    "trước đây",
    "đã từng",
    "trước nhập viện",
    "trong quá khứ",
}


def build_entity_spec_map(
    case_spec,
) -> dict[str, Any]:
    return {
        entity.entity_id: entity
        for entity in iter_entities(
            case_spec
        )
    }


def strip_markers_and_align(
    marked_text: str,
    case_spec,
) -> tuple[str, list[dict[str, Any]]]:
    entity_specs = build_entity_spec_map(
        case_spec
    )

    clean_parts: list[str] = []
    entities: list[dict[str, Any]] = []

    source_cursor = 0
    clean_cursor = 0

    for match in MARKER_PATTERN.finditer(
        marked_text
    ):
        prefix = marked_text[
            source_cursor:match.start()
        ]

        clean_parts.append(prefix)
        clean_cursor += len(prefix)

        marker_id = match.group(1)
        surface_text = match.group(2)

        start = clean_cursor
        end = start + len(surface_text)

        clean_parts.append(surface_text)
        clean_cursor = end

        if marker_id.startswith("E"):
            spec = entity_specs[marker_id]

            entity = {
                "entity_id": marker_id,
                "text": surface_text,
                "type": spec.entity_type,
                "position": [start, end],
            }

            if (
                spec.entity_type
                in ASSERTION_ENTITY_TYPES
            ):
                entity["assertions"] = list(
                    spec.assertions
                )

            if (
                spec.entity_type
                in CANDIDATE_ENTITY_TYPES
            ):
                entity["candidates"] = list(
                    spec.candidates
                )

            entities.append(entity)

        source_cursor = match.end()

    clean_parts.append(
        marked_text[source_cursor:]
    )

    clean_text = "".join(clean_parts)

    entities.sort(
        key=lambda entity: (
            entity["position"][0],
            entity["position"][1],
        )
    )

    return clean_text, entities


def validate_output_schema(
    text: str,
    entities: list[dict[str, Any]],
    *,
    rxnorm_ids: set[str],
    icd10_ids: set[str],
) -> list[str]:
    errors: list[str] = []

    occupied: list[tuple[int, int]] = []

    for index, entity in enumerate(entities):
        prefix = f"Entity {index}"

        entity_type = entity.get("type")
        position = entity.get("position")
        entity_text = entity.get("text")

        if entity_type not in VALID_ENTITY_TYPES:
            errors.append(
                f"{prefix}: type sai"
            )
            continue

        if (
            not isinstance(position, list)
            or len(position) != 2
        ):
            errors.append(
                f"{prefix}: position sai"
            )
            continue

        start, end = position

        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
        ):
            errors.append(
                f"{prefix}: position ngoài text"
            )
            continue

        if text[start:end] != entity_text:
            errors.append(
                f"{prefix}: text-position mismatch"
            )

        if any(
            start < old_end
            and end > old_start
            for old_start, old_end
            in occupied
        ):
            errors.append(
                f"{prefix}: overlap entity"
            )

        occupied.append((start, end))

        if entity_type in ASSERTION_ENTITY_TYPES:
            assertions = entity.get(
                "assertions"
            )

            if not isinstance(assertions, list):
                errors.append(
                    f"{prefix}: thiếu assertions"
                )
            else:
                invalid = (
                    set(assertions)
                    - VALID_ASSERTIONS
                )

                if invalid:
                    errors.append(
                        f"{prefix}: assertions sai "
                        f"{sorted(invalid)}"
                    )

        if entity_type in CANDIDATE_ENTITY_TYPES:
            candidates = entity.get(
                "candidates"
            )

            if (
                not isinstance(candidates, list)
                or not candidates
            ):
                errors.append(
                    f"{prefix}: thiếu candidates"
                )
                continue

            if entity_type == "THUỐC":
                unknown = (
                    set(candidates)
                    - rxnorm_ids
                )
            else:
                unknown = (
                    set(candidates)
                    - icd10_ids
                )

            if unknown:
                errors.append(
                    f"{prefix}: candidate không tồn tại "
                    f"{sorted(unknown)}"
                )

    return errors


def assertion_quality_warnings(
    text: str,
    entity: dict[str, Any],
    section,
) -> list[str]:
    warnings: list[str] = []

    assertions = entity.get(
        "assertions",
        [],
    )

    start, end = entity["position"]

    window = text[
        max(0, start - 120):
        min(len(text), end + 80)
    ].lower()

    if (
        "isNegated" in assertions
        and not any(
            cue in window
            for cue in NEGATION_CUES
        )
    ):
        warnings.append(
            "Không tìm thấy cue phủ định gần entity."
        )

    if (
        "isFamily" in assertions
        and not any(
            cue in window
            for cue in FAMILY_CUES
        )
    ):
        warnings.append(
            "Không tìm thấy family cue gần entity."
        )

    if "isHistorical" in assertions:
        section_is_historical = (
            section.temporal_scope
            == "historical"
        )

        local_historical = any(
            cue in window
            for cue in HISTORICAL_CUES
        )

        if not (
            section_is_historical
            or local_historical
        ):
            warnings.append(
                "Không tìm thấy historical cue "
                "hoặc historical section."
            )

    return warnings


def clean_submission_entity(
    entity: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in entity.items()
        if key != "entity_id"
    }


def sample_split(
    sample: dict[str, Any],
    *,
    validation_ratio: float = 0.20,
) -> str:
    concept_ids = sorted(
        {
            candidate
            for entity in sample["entities"]
            for candidate in entity.get(
                "candidates",
                [],
            )
        }
    )

    if concept_ids:
        signature = "|".join(concept_ids)
    else:
        signature = "|".join(
            sorted(
                entity["text"]
                for entity in sample[
                    "entities"
                ]
            )
        )

    digest = hashlib.sha1(
        signature.encode("utf-8")
    ).hexdigest()

    bucket = int(
        digest[:8],
        16,
    ) % 10000

    return (
        "val"
        if bucket
        < int(validation_ratio * 10000)
        else "train"
    )


def export_gliner(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for sample in samples:
        output.append(
            {
                "note_id": sample["note_id"],
                "text": sample["text"],
                "entities": [
                    {
                        "start": (
                            entity["position"][0]
                        ),
                        "end": (
                            entity["position"][1]
                        ),
                        "label": entity["type"],
                    }
                    for entity in sample[
                        "entities"
                    ]
                ],
            }
        )

    return output


def convert_assertion_markers(
    marked_section: str,
) -> str:
    # Negative markers chỉ được bỏ marker.
    marked_section = re.sub(
        r"\[\[(N\d+)\]\]"
        r"(.*?)"
        r"\[\[/\1\]\]",
        lambda match: match.group(2),
        marked_section,
        flags=re.DOTALL,
    )

    # Entity marker đổi sang <E0>...</E0>.
    marked_section = re.sub(
        r"\[\[(E\d+)\]\]"
        r"(.*?)"
        r"\[\[/\1\]\]",
        lambda match: (
            f"<{match.group(1)}>"
            f"{match.group(2)}"
            f"</{match.group(1)}>"
        ),
        marked_section,
        flags=re.DOTALL,
    )

    return marked_section


def export_assertion_records(
    case_specs: dict[str, Any],
    marked_notes: dict[str, Any],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for case_id in selected_ids:
        case_spec = case_specs[case_id]
        note = marked_notes[case_id]

        section_by_id = {
            section.section_id: section
            for section in (
                case_spec.sections
            )
        }

        for rendered_section in note[
            "sections"
        ]:
            section_id = rendered_section[
                "section_id"
            ]

            section = section_by_id[
                section_id
            ]

            entities = [
                entity
                for block in section.blocks
                for entity in block.entities
                if entity.entity_type
                in ASSERTION_ENTITY_TYPES
            ]

            if not entities:
                continue

            records.append(
                {
                    "case_id": case_id,
                    "section_id": section_id,
                    "section_title": (
                        section.title
                    ),
                    "context": (
                        convert_assertion_markers(
                            rendered_section[
                                "marked_text"
                            ]
                        )
                    ),
                    "entities": [
                        {
                            "id": (
                                entity.entity_id
                            ),
                            "type": (
                                entity.entity_type
                            ),
                            "assertions": (
                                entity.assertions
                            ),
                        }
                        for entity in entities
                    ],
                }
            )

    return records


def run_validate_export() -> None:
    ensure_directories()

    raw_case_specs = read_jsonl(
        GENERATED_DIR / "case_specs.jsonl"
    )

    raw_marked_notes = read_jsonl(
        GENERATED_DIR / "marked_notes.jsonl"
    )

    case_specs = {
        raw["case_id"]: (
            case_spec_from_dict(raw)
        )
        for raw in raw_case_specs
    }

    marked_notes = {
        raw["case_id"]: raw
        for raw in raw_marked_notes
    }

    rxnorm_ids = {
        record["concept_id"]
        for record in read_jsonl(
            PROCESSED_DIR
            / "rxnorm_concepts.jsonl"
        )
    }

    icd10_ids = {
        record["concept_id"]
        for record in read_jsonl(
            PROCESSED_DIR
            / "icd10_concepts.jsonl"
        )
    }

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for case_id, case_spec in (
        case_specs.items()
    ):
        note = marked_notes.get(case_id)

        if note is None:
            rejected.append(
                {
                    "case_id": case_id,
                    "errors": [
                        "Không tìm thấy marked note."
                    ],
                }
            )
            continue

        errors = validate_case_spec(
            case_spec
        )

        errors.extend(
            validate_markers(
                note["marked_text"],
                case_spec,
            )
        )

        section_spec_by_id = {
            section.section_id: section
            for section in case_spec.sections
        }
        rendered_section_ids: set[str] = set()

        for rendered_section in note.get(
            "sections", []
        ):
            section_id = rendered_section.get(
                "section_id"
            )
            section = section_spec_by_id.get(
                section_id
            )

            if section is None:
                errors.append(
                    f"Section không tồn tại: {section_id!r}"
                )
                continue

            rendered_section_ids.add(section_id)
            mini_case = CaseSpec(
                case_id=case_spec.case_id,
                document_profile=(
                    case_spec.document_profile
                ),
                structure_style=(
                    case_spec.structure_style
                ),
                noise_profile=case_spec.noise_profile,
                sections=[section],
            )
            section_errors = validate_rendered_sample(
                rendered_section.get(
                    "marked_text", ""
                ),
                mini_case,
            )
            errors.extend(
                f"{section_id}: {error}"
                for error in section_errors
            )

        missing_sections = (
            set(section_spec_by_id)
            - rendered_section_ids
        )
        if missing_sections:
            errors.append(
                "Thiếu rendered sections: "
                f"{sorted(missing_sections)}"
            )

        if errors:
            rejected.append(
                {
                    "case_id": case_id,
                    "errors": errors,
                }
            )
            continue

        text, internal_entities = (
            strip_markers_and_align(
                note["marked_text"],
                case_spec,
            )
        )

        schema_errors = validate_output_schema(
            text,
            internal_entities,
            rxnorm_ids=rxnorm_ids,
            icd10_ids=icd10_ids,
        )

        if schema_errors:
            rejected.append(
                {
                    "case_id": case_id,
                    "errors": schema_errors,
                }
            )
            continue

        section_map = entity_section_map(
            case_spec
        )

        warnings: list[
            dict[str, Any]
        ] = []

        for entity in internal_entities:
            entity_warnings = (
                assertion_quality_warnings(
                    text,
                    entity,
                    section_map[
                        entity["entity_id"]
                    ],
                )
            )

            if entity_warnings:
                warnings.append(
                    {
                        "entity_id": (
                            entity["entity_id"]
                        ),
                        "warnings": (
                            entity_warnings
                        ),
                    }
                )

        clean_entities = [
            clean_submission_entity(entity)
            for entity in internal_entities
        ]

        accepted.append(
            {
                "note_id": case_id,
                "text": text,
                "entities": clean_entities,
                "document_profile": (
                    case_spec.document_profile
                ),
                "noise_profile": (
                    case_spec.noise_profile
                ),
                "quality_warnings": warnings,
            }
        )

    write_jsonl(
        GENERATED_DIR
        / "accepted_samples.jsonl",
        accepted,
    )

    write_jsonl(
        GENERATED_DIR
        / "rejected_samples.jsonl",
        rejected,
    )

    split_by_id = {
        sample["note_id"]: sample_split(
            sample
        )
        for sample in accepted
    }

    train_samples = [
        sample
        for sample in accepted
        if split_by_id[sample["note_id"]]
        == "train"
    ]

    val_samples = [
        sample
        for sample in accepted
        if split_by_id[sample["note_id"]]
        == "val"
    ]

    write_jsonl(
        FINAL_DIR / "end_to_end_train.jsonl",
        train_samples,
    )
    write_jsonl(
        FINAL_DIR / "end_to_end_val.jsonl",
        val_samples,
    )

    write_jsonl(
        FINAL_DIR / "gliner_train.jsonl",
        export_gliner(train_samples),
    )
    write_jsonl(
        FINAL_DIR / "gliner_val.jsonl",
        export_gliner(val_samples),
    )

    train_ids = {
        sample["note_id"]
        for sample in train_samples
    }

    val_ids = {
        sample["note_id"]
        for sample in val_samples
    }

    write_jsonl(
        FINAL_DIR
        / "assertion_train.jsonl",
        export_assertion_records(
            case_specs,
            marked_notes,
            train_ids,
        ),
    )

    write_jsonl(
        FINAL_DIR
        / "assertion_val.jsonl",
        export_assertion_records(
            case_specs,
            marked_notes,
            val_ids,
        ),
    )

    write_json(
        FINAL_DIR / "split_manifest.json",
        {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "train": len(train_samples),
            "val": len(val_samples),
            "train_ratio": (
                round(
                    len(train_samples)
                    / max(1, len(accepted)),
                    4,
                )
            ),
            "validation_ratio": (
                round(
                    len(val_samples)
                    / max(1, len(accepted)),
                    4,
                )
            ),
        },
    )

    print("\n=== VALIDATE + EXPORT COMPLETE ===")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")


def main() -> None:
    run_validate_export()


if __name__ == "__main__":
    main()