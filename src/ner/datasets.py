from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict

from src.preprocessing import EntityAnnotation, MedicalDocument, normalize_text


class GLiNERSample(TypedDict):
    """Training sample accepted by GLiNER."""

    tokenized_text: list[str]
    ner: list[list[int | str]]


@dataclass(frozen=True)
class DatasetError:
    document_id: str
    code: str
    message: str
    entity_index: int | None = None
    entity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "code": self.code,
            "message": self.message,
            "entity_index": self.entity_index,
            "entity": self.entity,
        }


class NERDatasetError(ValueError):
    """Raised when a document cannot be represented safely as GLiNER data."""

    def __init__(self, error: DatasetError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class DatasetBuildResult:
    samples: list[GLiNERSample] = field(default_factory=list)
    errors: list[DatasetError] = field(default_factory=list)


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int

@dataclass(frozen=True)
class _MappedEntity:
    token_start: int
    token_end: int
    label: str
    entity_index: int

# Unicode words/numbers and standalone punctuation. Whitespace is not a token.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
DEFAULT_MAX_TOKENS = 320
DEFAULT_TOKEN_OVERLAP = 64

def _tokenize_with_offsets(text: str) -> list[_Token]:
    return [
        _Token(match.group(0), match.start(), match.end())
        for match in _TOKEN_PATTERN.finditer(text)
    ]

def _build_token_windows(
    token_count: int,
    *,
    max_tokens: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if max_tokens <= 0:
        raise ValueError(
            "max_tokens phải lớn hơn 0."
        )

    if not 0 <= overlap < max_tokens:
        raise ValueError(
            "overlap phải thỏa "
            "0 <= overlap < max_tokens."
        )

    if token_count <= max_tokens:
        return [(0, token_count)]

    step = max_tokens - overlap
    windows: list[tuple[int, int]] = []

    start = 0

    while start < token_count:
        end = min(
            start + max_tokens,
            token_count,
        )

        windows.append((start, end))

        if end == token_count:
            break

        start += step

    return windows

def _entity_dict(entity: EntityAnnotation) -> dict[str, Any]:
    return {
        "text": entity.text,
        "type": entity.entity_type,
        "position": [entity.start, entity.end],
    }


def _fail(
    document: MedicalDocument,
    code: str,
    message: str,
    *,
    entity_index: int | None = None,
    entity: EntityAnnotation | None = None,
) -> None:
    raise NERDatasetError(
        DatasetError(
            document_id=document.document_id,
            code=code,
            message=message,
            entity_index=entity_index,
            entity=_entity_dict(entity) if entity is not None else None,
        )
    )


def _char_span_to_token_span(
    tokens: list[_Token], start: int, end: int
) -> tuple[int, int] | None:
    """Map an exclusive character span to an inclusive GLiNER token span."""
    start_token = next(
        (index for index, token in enumerate(tokens) if token.start == start),
        None,
    )
    end_token = next(
        (index for index, token in enumerate(tokens) if token.end == end),
        None,
    )
    if (
        start_token is None
        or end_token is None
        or start_token > end_token
    ):
        return None

    return start_token, end_token


def convert_document_to_ner_samples(
        document: MedicalDocument,
        *,
        normalize_kwargs: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        token_overlap: int = DEFAULT_TOKEN_OVERLAP,
    ) -> list[GLiNERSample]:
    """Convert one labeled ``MedicalDocument`` to GLiNER training format.

    Entity offsets in ``MedicalDocument`` are interpreted against ``raw_text``.
    The document and every entity text are normalized with the same options.
    Conversion fails if an entity disappears, changes unexpectedly, or does not
    align exactly with the word/punctuation tokens required by GLiNER.
    """
    if not isinstance(document, MedicalDocument):
        raise TypeError("document phải là MedicalDocument.")

    options = dict(normalize_kwargs or {})
    normalized = normalize_text(document.raw_text, **options)
    if not normalized.normalized_text:
        _fail(document, "empty_normalized_text", "Văn bản rỗng sau normalize.")

    tokens = _tokenize_with_offsets(normalized.normalized_text)
    if not tokens:
        _fail(document, "no_tokens", "Không tạo được token từ văn bản normalized.")

    mapped_entities: list[_MappedEntity] = []
    seen_entities: set[
        tuple[int, int, str]
    ] = set()

    for entity_index, entity in enumerate(document.entities):
        if not isinstance(entity, EntityAnnotation):
            _fail(
                document,
                "invalid_entity_type",
                f"Entity {entity_index} không phải EntityAnnotation.",
                entity_index=entity_index,
            )
        if not isinstance(entity.text, str) or not entity.text:
            _fail(
                document,
                "invalid_entity_text",
                f"Entity {entity_index} có text rỗng hoặc không phải string.",
                entity_index=entity_index,
                entity=entity,
            )
        if not isinstance(entity.entity_type, str) or not entity.entity_type.strip():
            _fail(
                document,
                "invalid_entity_label",
                f"Entity {entity_index} có label rỗng hoặc không phải string.",
                entity_index=entity_index,
                entity=entity,
            )
        if not all(
            isinstance(offset, int) and not isinstance(offset, bool)
            for offset in (entity.start, entity.end)
        ):
            _fail(
                document,
                "invalid_raw_span",
                f"Entity {entity_index} có offset không phải số nguyên.",
                entity_index=entity_index,
                entity=entity,
            )
        if not 0 <= entity.start < entity.end <= len(document.raw_text):
            _fail(
                document,
                "raw_span_out_of_bounds",
                f"Entity {entity_index} có raw span ngoài văn bản: "
                f"[{entity.start}, {entity.end}).",
                entity_index=entity_index,
                entity=entity,
            )

        raw_entity_text = document.raw_text[entity.start : entity.end]
        if raw_entity_text != entity.text:
            _fail(
                document,
                "raw_text_mismatch",
                f"Entity {entity_index} không khớp raw span: "
                f"label={entity.text!r}, raw={raw_entity_text!r}.",
                entity_index=entity_index,
                entity=entity,
            )

        normalized_span = normalized.raw_span_to_normalized(
            entity.start, entity.end
        )
        if normalized_span is None:
            _fail(
                document,
                "entity_removed_by_normalization",
                f"Entity {entity_index} bị loại hoàn toàn sau normalize.",
                entity_index=entity_index,
                entity=entity,
            )

        normalized_start, normalized_end = normalized_span
        actual_text = normalized.normalized_text[
            normalized_start:normalized_end
        ]
        expected_text = normalize_text(entity.text, **options).normalized_text
        if actual_text != expected_text:
            _fail(
                document,
                "normalized_text_mismatch",
                f"Entity {entity_index} không khớp sau normalize: "
                f"expected={expected_text!r}, actual={actual_text!r}.",
                entity_index=entity_index,
                entity=entity,
            )

        token_span = _char_span_to_token_span(
            tokens, normalized_start, normalized_end
        )
        if token_span is None:
            _fail(
                document,
                "token_alignment_error",
                f"Entity {entity_index} không trùng ranh giới token GLiNER: "
                f"normalized span=[{normalized_start}, {normalized_end}).",
                entity_index=entity_index,
                entity=entity,
            )

        token_start, token_end = token_span
        key = (token_start, token_end, entity.entity_type)
        if key in seen_entities:
            _fail(
                document,
                "duplicate_entity",
                f"Entity {entity_index} bị trùng token span và label: {key}.",
                entity_index=entity_index,
                entity=entity,
            )
        seen_entities.add(key)
        mapped_entities.append(
                            _MappedEntity(
                                token_start=token_start,
                                token_end=token_end,
                                label=entity.entity_type,
                                entity_index=entity_index,
                            )
                        )

    windows = _build_token_windows(
        len(tokens),
        max_tokens=max_tokens,
        overlap=token_overlap,
    )

    samples: list[GLiNERSample] = []
    covered_entities: set[int] = set()

    for window_start, window_end in windows:
        chunk_entities: list[
            list[int | str]
        ] = []

        for mapped in mapped_entities:
            if not (
                window_start
                <= mapped.token_start
                and mapped.token_end
                < window_end
            ):
                continue

            chunk_entities.append(
                [
                    mapped.token_start
                    - window_start,
                    mapped.token_end
                    - window_start,
                    mapped.label,
                ]
            )

            covered_entities.add(
                mapped.entity_index
            )

        # Giai đoạn đầu chỉ train chunk có entity.
        if not chunk_entities:
            continue

        chunk_entities.sort(
            key=lambda item: (
                int(item[0]),
                int(item[1]),
                str(item[2]),
            )
        )

        samples.append(
            {
                "tokenized_text": [
                    token.text
                    for token in tokens[
                        window_start:window_end
                    ]
                ],
                "ner": chunk_entities,
            }
        )

    missing_entities = {
        mapped.entity_index
        for mapped in mapped_entities
    } - covered_entities

    if missing_entities:
        _fail(
            document,
            "entity_not_covered",
            (
                "Một số entity không nằm "
                "trong training window: "
                f"{sorted(missing_entities)}"
            ),
        )

    if not samples:
        _fail(
            document,
            "no_training_chunks",
            (
                "Không tạo được training chunk "
                "có entity."
            ),
        )

    return samples


def _write_error_log(path: str | Path, errors: list[DatasetError]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error.to_dict(), ensure_ascii=False) + "\n")


def build_gliner_dataset(
    documents: Iterable[MedicalDocument],
    *,
    on_error: Literal["raise", "skip"]
        = "skip",
    error_log_path:
        str | Path | None = None,
    normalize_kwargs:
        dict[str, Any] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_overlap: int = (
        DEFAULT_TOKEN_OVERLAP
    ),
) -> DatasetBuildResult:
    """Convert documents, optionally skipping and logging invalid samples.

    ``on_error='raise'`` stops at the first invalid sample. ``on_error='skip'``
    excludes the whole invalid document and records the reason in ``errors``.
    When ``error_log_path`` is provided, collected errors are written as JSONL.
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error phải là 'raise' hoặc 'skip'.")

    result = DatasetBuildResult()
    for document_index, document in enumerate(documents):
        try:
            result.samples.extend(
                convert_document_to_ner_samples(
                    document,
                    normalize_kwargs=normalize_kwargs,
                    max_tokens=max_tokens,
                    token_overlap=token_overlap,
                )
            )
        except NERDatasetError as exc:
            result.errors.append(exc.error)
            if on_error == "raise":
                if error_log_path is not None:
                    _write_error_log(error_log_path, result.errors)
                raise
        except (TypeError, ValueError) as exc:
            error = DatasetError(
                document_id=getattr(document, "document_id", f"index-{document_index}"),
                code="invalid_document",
                message=str(exc),
            )
            result.errors.append(error)
            if on_error == "raise":
                if error_log_path is not None:
                    _write_error_log(error_log_path, result.errors)
                raise NERDatasetError(error) from exc

    if error_log_path is not None:
        _write_error_log(error_log_path, result.errors)
    return result


