from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schemas import EntityAnnotation, MedicalDocument
from .constants import ENTITY_TYPE_SET
from .schemas import (
    EntityAnnotation,
    MedicalDocument,
)


SUPPORTED_TEXT_FIELDS = ("text", "raw_text", "content", "document", "note")
SUPPORTED_ID_FIELDS = ("note_id", "document_id", "id", "record_id")
SUPPORTED_ENTITY_FIELDS = ("entities", "labels", "annotations")
SUPPORTED_SUFFIXES = {".txt", ".json", ".jsonl"}


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic data whose inputs and targets remain in separate collections."""

    contents: list[dict[str, Any]]
    labels: list[dict[str, Any]]


def synthetic_to_documents(data: SyntheticData) -> list[MedicalDocument]:
    """Create labeled documents for training while source collections stay separate."""
    if len(data.contents) != len(data.labels):
        raise ValueError("Synthetic contents và labels không cùng số lượng.")
    documents: list[MedicalDocument] = []
    for record_number, (content, label) in enumerate(
        zip(data.contents, data.labels), start=1
    ):
        content_id = _extract_document_id(content, "")
        label_id = _extract_document_id(label, "")
        if not content_id or content_id != label_id:
            raise ValueError(
                f"Synthetic pair {record_number} lệch ID: {content_id!r} != {label_id!r}."
            )
        raw_text = _extract_text(content, record_number)
        entities = _extract_entities(label, raw_text, content_id)
        documents.append(
            MedicalDocument(
                document_id=content_id,
                raw_text=raw_text,
                entities=entities,
                metadata={"source_format": "synthetic_split", "has_labels": True},
            )
        )
    return documents


def _print_warning(message: str) -> None:
    """Print a warning without failing on a limited Windows console encoding."""
    output = f"[WARNING] {message}"
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_output = output.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe_output)


def _ensure_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return file_path


def _find_first_field(
    record: dict[str, Any], candidate_fields: tuple[str, ...]
) -> Any | None:
    for field_name in candidate_fields:
        if field_name in record:
            return record[field_name]
    return None


def _read_json(path: Path) -> list[dict[str, Any]]:
    """Read a regular JSON file containing one object or an array of objects."""
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON không hợp lệ trong {path}: {exc}") from exc

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(record, dict) for record in payload):
            raise TypeError(f"Mảng JSON trong {path} chỉ được chứa object.")
        return payload
    raise TypeError(f"{path} phải chứa một JSON object hoặc mảng JSON object.")


def _iter_jsonl_records(
    path: Path, *, strict: bool = True
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` from a JSON Lines file."""
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("mỗi dòng phải là một JSON object")
            except (json.JSONDecodeError, TypeError) as exc:
                message = f"Lỗi đọc {path}, dòng {line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                _print_warning(message)
                continue
            yield line_number, record


def _read_records(
    path: str | Path, *, strict: bool = True
) -> list[tuple[int, dict[str, Any]]]:
    """Read records while preserving their source index/line number."""
    file_path = _ensure_file(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return list(enumerate(_read_json(file_path), start=1))
    if suffix == ".jsonl":
        return list(_iter_jsonl_records(file_path, strict=strict))
    raise ValueError(f"Chỉ hỗ trợ record file .json hoặc .jsonl, nhận được: {suffix}")


def _extract_text(record: dict[str, Any], record_number: int) -> str:
    text = _find_first_field(record, SUPPORTED_TEXT_FIELDS)
    if text is None:
        raise ValueError(
            f"Record {record_number}: không tìm thấy trường văn bản; "
            f"hỗ trợ {SUPPORTED_TEXT_FIELDS}."
        )
    if not isinstance(text, str):
        raise TypeError(
            f"Record {record_number}: trường văn bản phải là string, "
            f"nhận được {type(text).__name__}."
        )
    if not text.strip():
        raise ValueError(f"Record {record_number}: văn bản rỗng.")
    return text


def _extract_document_id(record: dict[str, Any], default_id: str) -> str:
    document_id = _find_first_field(record, SUPPORTED_ID_FIELDS)
    return default_id if document_id is None else str(document_id)


def _parse_entity(
    entity: Any,
    raw_text: str,
    document_id: str,
    entity_index: int,
) -> EntityAnnotation:
    if not isinstance(entity, dict):
        raise TypeError(f"{document_id}, entity {entity_index}: phải là object.")

    missing = {"text", "type", "position"} - entity.keys()
    if missing:
        raise ValueError(
            f"{document_id}, entity {entity_index}: thiếu field {sorted(missing)}."
        )

    entity_text = entity["text"]
    entity_type = entity["type"]
    if entity_type not in ENTITY_TYPE_SET:
        raise ValueError(
            f"{document_id}, entity {entity_index}: "
            f"type không hợp lệ {entity_type!r}."
        )
    position = entity["position"]
    if not isinstance(entity_text, str) or not isinstance(entity_type, str):
        raise TypeError(
            f"{document_id}, entity {entity_index}: text và type phải là string."
        )
    if (
        not isinstance(position, (list, tuple))
        or len(position) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in position)
    ):
        raise ValueError(
            f"{document_id}, entity {entity_index}: position phải có dạng [start, end]."
        )

    start, end = position
    if not 0 <= start < end <= len(raw_text):
        raise ValueError(
            f"{document_id}, entity {entity_index}: position {list(position)} "
            f"nằm ngoài văn bản dài {len(raw_text)}."
        )
    actual_text = raw_text[start:end]
    if actual_text != entity_text:
        raise ValueError(
            f"{document_id}, entity {entity_index}: span không khớp; "
            f"label={entity_text!r}, text[{start}:{end}]={actual_text!r}."
        )

    assertions = entity.get("assertions", [])
    candidates = entity.get("candidates", [])
    if not isinstance(assertions, list) or not isinstance(candidates, list):
        raise TypeError(
            f"{document_id}, entity {entity_index}: assertions và candidates phải là list."
        )
    return EntityAnnotation(
        text=entity_text,
        entity_type=entity_type,
        start=start,
        end=end,
        assertions=[
            str(value)
            for value in assertions
        ],
        candidates=[
            str(value)
            for value in candidates
        ],
        source="gold",
    )


def _extract_entities(
    record: dict[str, Any], raw_text: str, document_id: str
) -> list[EntityAnnotation]:
    raw_entities = _find_first_field(record, SUPPORTED_ENTITY_FIELDS)
    if raw_entities is None:
        return []
    if not isinstance(raw_entities, list):
        raise TypeError(f"{document_id}: entities/labels/annotations phải là list.")
    entities = [
        _parse_entity(
            entity,
            raw_text,
            document_id,
            index,
        )
        for index, entity
        in enumerate(raw_entities)
    ]

    entities.sort(
        key=lambda item: (
            item.start,
            item.end,
            item.entity_type,
        )
    )

    seen: set[tuple[int, int, str]] = set()

    for index, entity in enumerate(entities):
        key = (
            entity.start,
            entity.end,
            entity.entity_type,
        )

        if key in seen:
            raise ValueError(
                f"{document_id}: entity bị trùng "
                f"{key}."
            )

        seen.add(key)

        if index == 0:
            continue

        previous = entities[index - 1]

        if entity.start < previous.end:
            raise ValueError(
                f"{document_id}: entity overlap: "
                f"{previous.text!r} "
                f"[{previous.start}, {previous.end}) "
                f"và {entity.text!r} "
                f"[{entity.start}, {entity.end})."
            )

    return entities


def _record_to_document(
    record: dict[str, Any], source_path: Path, record_number: int
) -> MedicalDocument:
    raw_text = _extract_text(record, record_number)
    document_id = _extract_document_id(
        record, f"{source_path.stem}_{record_number:06d}"
    )
    entities = _extract_entities(record, raw_text, document_id)
    known_fields = {
        *SUPPORTED_TEXT_FIELDS,
        *SUPPORTED_ID_FIELDS,
        *SUPPORTED_ENTITY_FIELDS,
    }
    metadata = {key: value for key, value in record.items() if key not in known_fields}
    metadata.update(
        {
            "source_path": str(source_path),
            "source_format": source_path.suffix.lower().lstrip("."),
            "source_record": record_number,
            "has_labels": bool(entities),
        }
    )
    return MedicalDocument(
        document_id=document_id,
        raw_text=raw_text,
        entities=entities,
        metadata=metadata,
    )


def load_synthetic_data(
    contents_path: str | Path,
    labels_path: str | Path,
    *,
    strict: bool = True,
) -> SyntheticData:
    """Load synthetic content and labels without merging them.

    Both files may be regular JSON or JSONL. Records are preserved as dictionaries,
    so the caller can build the fine-tuning input and target independently.
    ``strict`` only controls malformed JSONL lines and mismatched ``note_id`` values.
    """
    content_records = [
        record for _, record in _read_records(contents_path, strict=strict)
    ]
    label_records = [
        record for _, record in _read_records(labels_path, strict=strict)
    ]

    def index_records(
        records: list[dict[str, Any]], kind: str
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        ordered_ids: list[str] = []
        records_by_id: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(records, start=1):
            value = _find_first_field(record, SUPPORTED_ID_FIELDS)
            if value is None:
                raise ValueError(f"{kind} record {index} không có ID.")
            record_id = str(value)
            if not record_id.strip():
                raise ValueError(f"{kind} record {index} có ID rỗng.")
            if record_id in records_by_id:
                raise ValueError(f"ID bị trùng trong {kind}: {record_id}")
            ordered_ids.append(record_id)
            records_by_id[record_id] = record
        return ordered_ids, records_by_id

    content_order, contents_by_id = index_records(content_records, "contents")
    _, labels_by_id = index_records(label_records, "labels")
    content_ids = set(contents_by_id)
    label_ids = set(labels_by_id)
    common_ids = content_ids & label_ids

    if content_ids != label_ids:
        missing_labels = sorted(content_ids - label_ids)
        missing_contents = sorted(label_ids - content_ids)
        message = (
            "Content và label không khớp ID; "
            f"thiếu label={missing_labels[:10]}, thiếu content={missing_contents[:10]}."
        )
        if strict:
            raise ValueError(message)
        _print_warning(message)

    aligned_contents: list[dict[str, Any]] = []
    aligned_labels: list[dict[str, Any]] = []
    for record_number, record_id in enumerate(content_order, start=1):
        if record_id not in common_ids:
            continue

        content_record = contents_by_id[record_id]
        label_record = labels_by_id[record_id]
        try:
            raw_text = _extract_text(content_record, record_number)
            # Validate the target independently without adding it to the content.
            _extract_entities(label_record, raw_text, record_id)
        except (TypeError, ValueError) as exc:
            message = f"Synthetic record {record_id} không hợp lệ: {exc}"
            if strict:
                raise ValueError(message) from exc
            _print_warning(message)
            continue

        aligned_contents.append(content_record)
        aligned_labels.append(label_record)

    return SyntheticData(contents=aligned_contents, labels=aligned_labels)


def load_documents(
    path: str | Path,
    *,
    strict: bool = True,
    document_id: str | None = None,
) -> list[MedicalDocument]:
    """Load medical documents from TXT, regular JSON, or JSONL.

    - TXT represents one unlabeled document.
    - JSON contains one object or an array of objects.
    - JSONL contains one object per non-empty line.

    Synthetic data stored in separate content/label files must be loaded with
    :func:`load_synthetic_data`; this function never merges those two collections.
    """
    file_path = _ensure_file(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Định dạng không hỗ trợ: {suffix}. Chỉ hỗ trợ .txt, .json và .jsonl."
        )

    if suffix == ".txt":
        raw_text = file_path.read_text(encoding="utf-8-sig")
        if not raw_text.strip():
            raise ValueError(f"File TXT rỗng: {file_path}")
        return [
            MedicalDocument(
                document_id=document_id or file_path.stem,
                raw_text=raw_text,
                entities=[],
                metadata={
                    "source_path": str(file_path),
                    "source_format": "txt",
                    "has_labels": False,
                },
            )
        ]

    if document_id is not None:
        raise ValueError("document_id chỉ dùng khi đọc file .txt.")

    documents: list[MedicalDocument] = []
    for record_number, record in _read_records(file_path, strict=strict):
        try:
            documents.append(
                _record_to_document(record, file_path, record_number)
            )
        except (TypeError, ValueError) as exc:
            message = f"Lỗi đọc {file_path}, record {record_number}: {exc}"
            if strict:
                raise ValueError(message) from exc
            _print_warning(message)
    return documents
