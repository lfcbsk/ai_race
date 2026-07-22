from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "raw"

SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
PROCESSED_DIR = SYNTHETIC_DIR / "processed"
CATALOG_DIR = SYNTHETIC_DIR / "catalogs"
GENERATED_DIR = SYNTHETIC_DIR / "generated"
FINAL_DIR = SYNTHETIC_DIR / "final"


def ensure_directories() -> None:
    for directory in (
        PROCESSED_DIR,
        CATALOG_DIR,
        GENERATED_DIR,
        FINAL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET SCHEMA
# ============================================================

VALID_ENTITY_TYPES = {
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
}

ASSERTION_ENTITY_TYPES = {
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
}

CANDIDATE_ENTITY_TYPES = {
    "CHẨN_ĐOÁN",
    "THUỐC",
}

VALID_ASSERTIONS = {
    "isNegated",
    "isHistorical",
    "isFamily",
}


# ============================================================
# NORMALIZATION
# ============================================================

NULL_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
}


def clean_text(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in NULL_VALUES:
        return None

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()

    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("µg", "mcg")
    )

    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n.,;:!?()[]{}\"'")

    return text


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)

    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )


# ============================================================
# JSON UTILITIES
# ============================================================

def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=indent,
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại {path}, dòng {line_number}"
                ) from exc

    return records


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# CASE SPEC SCHEMAS
# ============================================================

@dataclass
class EntitySpec:
    entity_id: str
    surface_text: str
    entity_type: str

    assertions: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    surface_id: str | None = None
    concept_id: str | None = None

    @property
    def marked_text(self) -> str:
        return (
            f"[[{self.entity_id}]]"
            f"{self.surface_text}"
            f"[[/{self.entity_id}]]"
        )


@dataclass
class NegativeSpec:
    negative_id: str
    surface_text: str
    negative_type: str

    @property
    def marked_text(self) -> str:
        return (
            f"[[{self.negative_id}]]"
            f"{self.surface_text}"
            f"[[/{self.negative_id}]]"
        )


@dataclass
class BlockSpec:
    block_id: str
    scenario_id: str

    entities: list[EntitySpec]
    hard_negatives: list[NegativeSpec] = field(
        default_factory=list
    )
    instructions: list[str] = field(
        default_factory=list
    )


@dataclass
class SectionSpec:
    section_id: str
    section_type: str
    title: str

    temporal_scope: str
    subject_scope: str
    render_style: str

    blocks: list[BlockSpec]


@dataclass
class CaseSpec:
    case_id: str
    document_profile: str
    structure_style: str
    noise_profile: str

    sections: list[SectionSpec]
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def case_spec_from_dict(
    payload: dict[str, Any],
) -> CaseSpec:
    sections: list[SectionSpec] = []

    for raw_section in payload["sections"]:
        blocks: list[BlockSpec] = []

        for raw_block in raw_section["blocks"]:
            entities = [
                EntitySpec(**entity)
                for entity in raw_block.get(
                    "entities",
                    [],
                )
            ]

            negatives = [
                NegativeSpec(**negative)
                for negative in raw_block.get(
                    "hard_negatives",
                    [],
                )
            ]

            blocks.append(
                BlockSpec(
                    block_id=raw_block["block_id"],
                    scenario_id=raw_block["scenario_id"],
                    entities=entities,
                    hard_negatives=negatives,
                    instructions=raw_block.get(
                        "instructions",
                        [],
                    ),
                )
            )

        sections.append(
            SectionSpec(
                section_id=raw_section["section_id"],
                section_type=raw_section[
                    "section_type"
                ],
                title=raw_section["title"],
                temporal_scope=raw_section[
                    "temporal_scope"
                ],
                subject_scope=raw_section[
                    "subject_scope"
                ],
                render_style=raw_section[
                    "render_style"
                ],
                blocks=blocks,
            )
        )

    return CaseSpec(
        case_id=payload["case_id"],
        document_profile=payload[
            "document_profile"
        ],
        structure_style=payload[
            "structure_style"
        ],
        noise_profile=payload["noise_profile"],
        sections=sections,
        metadata=payload.get("metadata", {}),
    )


def iter_entities(
    case_spec: CaseSpec,
) -> Iterator[EntitySpec]:
    for section in case_spec.sections:
        for block in section.blocks:
            yield from block.entities


def iter_negatives(
    case_spec: CaseSpec,
) -> Iterator[NegativeSpec]:
    for section in case_spec.sections:
        for block in section.blocks:
            yield from block.hard_negatives


def entity_section_map(
    case_spec: CaseSpec,
) -> dict[str, SectionSpec]:
    mapping: dict[str, SectionSpec] = {}

    for section in case_spec.sections:
        for block in section.blocks:
            for entity in block.entities:
                mapping[entity.entity_id] = section

    return mapping


# ============================================================
# MARKERS
# ============================================================

MARKER_PATTERN = re.compile(
    r"\[\[([EN]\d+)\]\]"
    r"(.*?)"
    r"\[\[/\1\]\]",
    re.DOTALL,
)


def expected_marker_texts(
    case_spec: CaseSpec,
) -> dict[str, str]:
    expected: dict[str, str] = {}

    for entity in iter_entities(case_spec):
        expected[entity.entity_id] = (
            entity.surface_text
        )

    for negative in iter_negatives(case_spec):
        expected[negative.negative_id] = (
            negative.surface_text
        )

    return expected


def validate_markers(
    marked_text: str,
    case_spec: CaseSpec,
) -> list[str]:
    expected = expected_marker_texts(case_spec)

    found: dict[str, list[str]] = {}

    for match in MARKER_PATTERN.finditer(
        marked_text
    ):
        marker_id = match.group(1)
        marker_text = match.group(2)

        found.setdefault(
            marker_id,
            [],
        ).append(marker_text)

    errors: list[str] = []

    for marker_id, expected_text in expected.items():
        occurrences = found.get(
            marker_id,
            [],
        )

        if len(occurrences) != 1:
            errors.append(
                f"{marker_id}: cần đúng 1 marker, "
                f"nhưng tìm thấy {len(occurrences)}"
            )
            continue

        if occurrences[0] != expected_text:
            errors.append(
                f"{marker_id}: text bị thay đổi: "
                f"{occurrences[0]!r} != "
                f"{expected_text!r}"
            )

    unknown = set(found) - set(expected)

    if unknown:
        errors.append(
            f"Xuất hiện marker không tồn tại: "
            f"{sorted(unknown)}"
        )

    return errors


def validate_case_spec(
    case_spec: CaseSpec,
) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()

    for entity in iter_entities(case_spec):
        if entity.entity_id in ids:
            errors.append(
                f"Trùng entity_id: {entity.entity_id}"
            )

        ids.add(entity.entity_id)

        if entity.entity_type not in (
            VALID_ENTITY_TYPES
        ):
            errors.append(
                f"{entity.entity_id}: type không hợp lệ "
                f"{entity.entity_type!r}"
            )

        invalid_assertions = (
            set(entity.assertions)
            - VALID_ASSERTIONS
        )

        if invalid_assertions:
            errors.append(
                f"{entity.entity_id}: assertion sai "
                f"{sorted(invalid_assertions)}"
            )

        if (
            entity.entity_type
            in CANDIDATE_ENTITY_TYPES
            and not entity.candidates
        ):
            errors.append(
                f"{entity.entity_id}: thiếu candidate"
            )

        if (
            entity.entity_type
            not in CANDIDATE_ENTITY_TYPES
            and entity.candidates
        ):
            errors.append(
                f"{entity.entity_id}: không được có "
                f"candidate"
            )

    for negative in iter_negatives(case_spec):
        if negative.negative_id in ids:
            errors.append(
                f"Trùng marker ID: "
                f"{negative.negative_id}"
            )

        ids.add(negative.negative_id)

    return errors