from __future__ import annotations

import json
import os
import re
from collections import Counter
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

RAW_DIR = ROOT_DIR / "data" / "raw"

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


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise ValueError(
            f"Thiếu biến môi trường bắt buộc: {name}"
        )

    return value


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} phải là số nguyên, nhận được "
            f"{raw_value!r}."
        ) from exc


def env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} phải là số, nhận được "
            f"{raw_value!r}."
        ) from exc


@dataclass(frozen=True)
class QwenConfig:
    base_url: str
    model: str
    api_key: str

    timeout_seconds: int
    max_tokens: int

    temperature: float
    repair_temperature: float
    top_p: float


def get_qwen_config() -> QwenConfig:
    base_url = require_env(
        "QWEN_BASE_URL"
    ).rstrip("/")

    model = require_env(
        "QWEN_MODEL"
    )

    api_key = require_env(
        "QWEN_API_KEY"
    )

    timeout_seconds = env_int(
        "QWEN_TIMEOUT_SECONDS",
        300,
    )

    max_tokens = env_int(
        "QWEN_MAX_TOKENS",
        900,
    )

    temperature = env_float(
        "QWEN_TEMPERATURE",
        0.35,
    )

    repair_temperature = env_float(
        "QWEN_REPAIR_TEMPERATURE",
        0.10,
    )

    top_p = env_float(
        "QWEN_TOP_P",
        0.90,
    )

    if timeout_seconds <= 0:
        raise ValueError(
            "QWEN_TIMEOUT_SECONDS phải lớn hơn 0."
        )

    if max_tokens <= 0:
        raise ValueError(
            "QWEN_MAX_TOKENS phải lớn hơn 0."
        )

    if not 0 <= temperature <= 2:
        raise ValueError(
            "QWEN_TEMPERATURE phải nằm trong [0, 2]."
        )

    if not 0 <= repair_temperature <= 2:
        raise ValueError(
            "QWEN_REPAIR_TEMPERATURE "
            "phải nằm trong [0, 2]."
        )

    if not 0 < top_p <= 1:
        raise ValueError(
            "QWEN_TOP_P phải nằm trong (0, 1]."
        )

    return QwenConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        repair_temperature=(
            repair_temperature
        ),
        top_p=top_p,
    )

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




PLACEHOLDER_PATTERN = re.compile(
    r"<<([EN]\d+)>>"
)

NEGATION_CUES = (
    "không ghi nhận",
    "không có",
    "không thấy",
    "không xuất hiện",
    "chưa ghi nhận",
    "phủ nhận",
    "không mắc",
    "không bị",
    "không",
)

FAMILY_CUES = (
    "tiền sử gia đình",
    "người nhà",
    "gia đình",
    "bố",
    "mẹ",
    "cha",
    "anh trai",
    "chị gái",
)

HISTORICAL_CUES = (
    "tiền sử",
    "trước đây",
    "đã từng",
    "từng",
    "trước nhập viện",
    "trong quá khứ",
)

CONTRAST_CUES = (
    "nhưng",
    "tuy nhiên",
    "song",
    "trái lại",
)

BANNED_OUTPUT_TERMS = (
    "các entity",
    "entity là",
    "hard-negative",
    "hard negative",
    "scenario_id",
    "subject_scope",
    "temporal_scope",
    "surface_text",
    "assertion",
    "section_spec",
    "case spec",
    "marker",
)


def placeholder_for(marker_id: str) -> str:
    return f"<<{marker_id}>>"


def expected_placeholders(
    case_spec: CaseSpec,
) -> dict[str, str]:
    return expected_marker_texts(case_spec)


def _contains_surface(
    text: str,
    surface: str,
) -> bool:
    collapsed_text = re.sub(r"\s+", " ", text)
    collapsed_surface = re.sub(r"\s+", " ", surface)

    if not collapsed_surface:
        return False

    pattern = re.compile(
        rf"(?<!\w){re.escape(collapsed_surface)}(?!\w)",
        re.IGNORECASE,
    )
    return pattern.search(collapsed_text) is not None


def validate_placeholder_output(
    draft_text: str,
    case_spec: CaseSpec,
) -> list[str]:
    expected = expected_placeholders(case_spec)
    found = Counter(
        match.group(1)
        for match in PLACEHOLDER_PATTERN.finditer(
            draft_text
        )
    )

    errors: list[str] = []

    for marker_id in expected:
        count = found.get(marker_id, 0)
        if count != 1:
            errors.append(
                f"{marker_id}: cần đúng 1 placeholder, "
                f"nhưng tìm thấy {count}"
            )

    unknown = set(found) - set(expected)
    if unknown:
        errors.append(
            f"Placeholder không tồn tại: {sorted(unknown)}"
        )

    if "[[" in draft_text or "]]" in draft_text:
        errors.append(
            "Draft không được tự tạo marker [[...]]."
        )

    without_placeholders = PLACEHOLDER_PATTERN.sub(
        " ",
        draft_text,
    )

    for marker_id, surface in expected.items():
        if _contains_surface(
            without_placeholders,
            surface,
        ):
            errors.append(
                f"{marker_id}: surface xuất hiện ngoài placeholder"
            )

    lowered = draft_text.lower()
    for term in BANNED_OUTPUT_TERMS:
        if term in lowered:
            errors.append(
                f"Prompt leakage: {term!r}"
            )

    if "```" in draft_text:
        errors.append(
            "Output không được chứa markdown fence."
        )

    return errors


def expand_placeholders(
    draft_text: str,
    case_spec: CaseSpec,
) -> str:
    expected = expected_placeholders(case_spec)

    def replace(match: re.Match[str]) -> str:
        marker_id = match.group(1)
        surface = expected[marker_id]
        return (
            f"[[{marker_id}]]"
            f"{surface}"
            f"[[/{marker_id}]]"
        )

    return PLACEHOLDER_PATTERN.sub(
        replace,
        draft_text,
    )


def strip_markers_with_positions(
    marked_text: str,
) -> tuple[str, dict[str, tuple[int, int]]]:
    clean_parts: list[str] = []
    positions: dict[str, tuple[int, int]] = {}
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
        surface = match.group(2)
        start = clean_cursor
        end = start + len(surface)
        clean_parts.append(surface)
        clean_cursor = end

        if marker_id.startswith("E"):
            positions[marker_id] = (start, end)

        source_cursor = match.end()

    clean_parts.append(
        marked_text[source_cursor:]
    )

    return "".join(clean_parts), positions



def _has_cue(
    text: str,
    cues: tuple[str, ...],
) -> bool:
    return any(
        re.search(
            rf"(?<!\w){re.escape(cue)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for cue in cues
    )


def _scope_prefix(
    text: str,
    start: int,
) -> str:
    left = max(
        text.rfind("\n", 0, start),
        text.rfind(".", 0, start),
        text.rfind(";", 0, start),
        text.rfind(":", 0, start),
    )
    prefix = text[left + 1:start].lower()

    contrast_positions = [
        prefix.rfind(cue)
        for cue in CONTRAST_CUES
    ]
    last_contrast = max(contrast_positions)

    if last_contrast >= 0:
        prefix = prefix[last_contrast:]

    return prefix


def _local_window(
    text: str,
    start: int,
    end: int,
) -> str:
    return text[
        max(0, start - 180):
        min(len(text), end + 100)
    ].lower()


def validate_semantic_assertions(
    marked_text: str,
    case_spec: CaseSpec,
) -> list[str]:
    clean_text, positions = (
        strip_markers_with_positions(
            marked_text
        )
    )
    errors: list[str] = []

    section_ranges: dict[str, tuple[int, int]] = {}
    search_cursor = 0
    section_starts: list[tuple[SectionSpec, int]] = []

    for section in case_spec.sections:
        title_index = clean_text.find(
            section.title,
            search_cursor,
        )
        if title_index < 0:
            title_index = search_cursor
        section_starts.append((section, title_index))
        search_cursor = title_index + len(section.title)

    for index, (section, section_start) in enumerate(
        section_starts
    ):
        section_end = (
            section_starts[index + 1][1]
            if index + 1 < len(section_starts)
            else len(clean_text)
        )
        section_ranges[section.section_id] = (
            section_start,
            section_end,
        )

    for section in case_spec.sections:
        section_start, section_end = section_ranges[
            section.section_id
        ]
        for block in section.blocks:
            for entity in block.entities:
                if (
                    entity.entity_type
                    not in ASSERTION_ENTITY_TYPES
                ):
                    continue

                position = positions.get(
                    entity.entity_id
                )
                if position is None:
                    continue

                start, end = position
                prefix = _scope_prefix(
                    clean_text,
                    start,
                )
                window = clean_text[
                    max(section_start, start - 180):
                    min(section_end, end + 100)
                ].lower()
                assertions = set(
                    entity.assertions
                )

                has_negation = _has_cue(
                    prefix,
                    NEGATION_CUES,
                )

                if "isNegated" in assertions:
                    if not has_negation:
                        errors.append(
                            f"{entity.entity_id}: thiếu cue phủ định "
                            f"trong cùng scope"
                        )
                elif has_negation:
                    errors.append(
                        f"{entity.entity_id}: entity dương tính "
                        f"nằm dưới scope phủ định"
                    )

                has_family = _has_cue(
                    window,
                    FAMILY_CUES,
                )

                if "isFamily" in assertions:
                    if not has_family:
                        errors.append(
                            f"{entity.entity_id}: thiếu family cue"
                        )
                elif (
                    has_family
                    and section.subject_scope == "patient"
                ):
                    errors.append(
                        f"{entity.entity_id}: patient entity "
                        f"nằm trong ngữ cảnh family"
                    )

                has_historical = _has_cue(
                    window,
                    HISTORICAL_CUES,
                )

                if "isHistorical" in assertions:
                    if not (
                        has_historical
                        or section.temporal_scope
                        == "historical"
                    ):
                        errors.append(
                            f"{entity.entity_id}: thiếu historical cue"
                        )
                elif (
                    section.temporal_scope == "current"
                    and has_historical
                    and block.scenario_id
                    not in {"mixed_polarity"}
                ):
                    errors.append(
                        f"{entity.entity_id}: current entity "
                        f"nằm trong ngữ cảnh historical"
                    )

    return errors


def validate_surface_occurrences(
    marked_text: str,
    case_spec: CaseSpec,
) -> list[str]:
    clean_text, _ = strip_markers_with_positions(
        marked_text
    )
    expected = expected_marker_texts(case_spec)
    grouped: dict[str, list[str]] = {}

    for marker_id, surface in expected.items():
        grouped.setdefault(surface, []).append(
            marker_id
        )

    errors: list[str] = []

    for surface, marker_ids in grouped.items():
        collapsed_text = re.sub(
            r"\s+", " ", clean_text
        )
        collapsed_surface = re.sub(
            r"\s+", " ", surface
        )
        pattern = re.compile(
            rf"(?<!\w){re.escape(collapsed_surface)}(?!\w)",
            re.IGNORECASE,
        )
        actual = len(pattern.findall(
            collapsed_text
        ))
        expected_count = len(marker_ids)

        if actual != expected_count:
            errors.append(
                f"Surface {surface!r}: cần xuất hiện "
                f"{expected_count} lần, tìm thấy {actual}"
            )

    return errors


def validate_render_quality(
    marked_text: str,
    case_spec: CaseSpec,
) -> list[str]:
    errors: list[str] = []
    lowered = marked_text.lower()

    for term in BANNED_OUTPUT_TERMS:
        if term in lowered:
            errors.append(
                f"Prompt leakage: {term!r}"
            )

    if PLACEHOLDER_PATTERN.search(marked_text):
        errors.append(
            "Còn placeholder chưa được expand."
        )

    for section in case_spec.sections:
        duplicated_title = (
            f"{section.title}\n{section.title}"
        )
        if duplicated_title in marked_text:
            errors.append(
                f"{section.section_id}: tiêu đề bị lặp"
            )

    return errors


def validate_rendered_sample(
    marked_text: str,
    case_spec: CaseSpec,
) -> list[str]:
    errors = validate_markers(
        marked_text,
        case_spec,
    )
    errors.extend(
        validate_render_quality(
            marked_text,
            case_spec,
        )
    )
    errors.extend(
        validate_surface_occurrences(
            marked_text,
            case_spec,
        )
    )
    errors.extend(
        validate_semantic_assertions(
            marked_text,
            case_spec,
        )
    )
    return errors


def validate_case_spec(
    case_spec: CaseSpec,
) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    normalized_surfaces: dict[str, str] = {}

    scenario_expected = {
        "patient_current_positive": None,
        "patient_current_negated": {"isNegated"},
        "coordinated_negation": {"isNegated"},
        "patient_historical": {"isHistorical"},
        "patient_historical_negated": {
            "isNegated",
            "isHistorical",
        },
        "family_current": {"isFamily"},
        "family_historical": {
            "isFamily",
            "isHistorical",
        },
        "home_medications": {"isHistorical"},
        "active_medications": set(),
        "laboratory_results": set(),
    }

    for section in case_spec.sections:
        for block in section.blocks:
            for entity_index, entity in enumerate(
                block.entities
            ):
                if entity.entity_id in ids:
                    errors.append(
                        f"Trùng entity_id: {entity.entity_id}"
                    )

                ids.add(entity.entity_id)

                surface = (entity.surface_text or "").strip()

                if not surface:
                    errors.append(
                        f"{entity.entity_id}: surface rỗng"
                    )
                elif any(
                    token in surface
                    for token in ("[[", "]]", "<<", ">>")
                ):
                    errors.append(
                        f"{entity.entity_id}: surface chứa marker"
                    )

                normalized = normalize_for_matching(surface)

                if (
                    normalized
                    and entity.entity_type
                    != "KẾT_QUẢ_XÉT_NGHIỆM"
                ):
                    previous = normalized_surfaces.get(normalized)
                    if previous is not None:
                        errors.append(
                            f"Surface trùng trong case: "
                            f"{previous} và {entity.entity_id} "
                            f"đều là {surface!r}"
                        )
                    else:
                        normalized_surfaces[normalized] = (
                            entity.entity_id
                        )

                if entity.entity_type not in VALID_ENTITY_TYPES:
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
                    not in ASSERTION_ENTITY_TYPES
                    and entity.assertions
                ):
                    errors.append(
                        f"{entity.entity_id}: type "
                        f"{entity.entity_type} không được có assertion"
                    )

                if (
                    entity.entity_type in CANDIDATE_ENTITY_TYPES
                    and not entity.candidates
                ):
                    errors.append(
                        f"{entity.entity_id}: thiếu candidate"
                    )

                if (
                    entity.entity_type not in CANDIDATE_ENTITY_TYPES
                    and entity.candidates
                ):
                    errors.append(
                        f"{entity.entity_id}: không được có candidate"
                    )

                if entity.entity_type == "THUỐC":
                    if any(
                        not str(candidate).isdigit()
                        for candidate in entity.candidates
                    ):
                        errors.append(
                            f"{entity.entity_id}: RxNorm ID phải là số"
                        )

                if entity.entity_type == "CHẨN_ĐOÁN":
                    if any(
                        re.fullmatch(
                            r"[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?",
                            str(candidate).upper(),
                        )
                        is None
                        for candidate in entity.candidates
                    ):
                        errors.append(
                            f"{entity.entity_id}: ICD-10 ID sai định dạng"
                        )

                assertions = set(entity.assertions)

                if section.temporal_scope == "historical":
                    if (
                        entity.entity_type in ASSERTION_ENTITY_TYPES
                        and "isHistorical" not in assertions
                    ):
                        errors.append(
                            f"{entity.entity_id}: section historical "
                            f"nhưng thiếu isHistorical"
                        )

                if section.temporal_scope == "current":
                    if "isHistorical" in assertions:
                        errors.append(
                            f"{entity.entity_id}: section current "
                            f"nhưng có isHistorical"
                        )

                if section.subject_scope == "patient":
                    if "isFamily" in assertions:
                        errors.append(
                            f"{entity.entity_id}: section patient "
                            f"nhưng có isFamily"
                        )

                expected = scenario_expected.get(
                    block.scenario_id
                )

                if block.scenario_id == "mixed_polarity":
                    expected_for_entity = (
                        {"isNegated"}
                        if entity_index == 0
                        else set()
                    )
                    if assertions != expected_for_entity:
                        errors.append(
                            f"{entity.entity_id}: mixed_polarity "
                            f"cần {sorted(expected_for_entity)}, "
                            f"nhận {sorted(assertions)}"
                        )
                elif expected is not None:
                    if assertions != expected:
                        errors.append(
                            f"{entity.entity_id}: scenario "
                            f"{block.scenario_id} cần "
                            f"{sorted(expected)}, nhận "
                            f"{sorted(assertions)}"
                        )

            for negative in block.hard_negatives:
                if negative.negative_id in ids:
                    errors.append(
                        f"Trùng marker ID: {negative.negative_id}"
                    )

                ids.add(negative.negative_id)

                negative_surface = (
                    negative.surface_text or ""
                ).strip()

                if not negative_surface:
                    errors.append(
                        f"{negative.negative_id}: hard negative rỗng"
                    )

                normalized = normalize_for_matching(
                    negative_surface
                )
                if normalized in normalized_surfaces:
                    errors.append(
                        f"{negative.negative_id}: trùng surface với "
                        f"{normalized_surfaces[normalized]}"
                    )

    return errors