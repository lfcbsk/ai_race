from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from src.preprocessing import MedicalDocument, NormalizedDocument, normalize_text
from src.preprocessing.constants import (
    ENTITY_TYPES,
    NER_NORMALIZE_KWARGS,
)

DEFAULT_LABELS = list(ENTITY_TYPES)

# Model GLiNER đa ngôn ngữ pretrained — dùng làm baseline zero-shot cho tiếng Việt
# (chưa có checkpoint GLiNER riêng cho tiếng Việt). Sau này thay bằng checkpoint
# fine-tuned mà không cần đổi API của module này.
DEFAULT_MODEL_NAME = "urchade/gliner_multi-v2.1"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50


class GLiNERModel(Protocol):
    """Interface tối thiểu mà một model GLiNER thật (hoặc mock trong test) cần có.

    Khớp đúng chữ ký ``predict_entities`` của package ``gliner``: text đầu vào,
    trả về list dict {start, end, text, label, score} với start/end là offset
    ký tự exclusive-end trên chính text đã truyền vào.
    """

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]: ...


def load_gliner_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = "cpu",
) -> GLiNERModel:
    """Tải GLiNER pretrained từ HuggingFace Hub.

    Cần mạng ở lần tải đầu tiên (checkpoint sẽ được cache local sau đó).
    Import ``gliner``/``torch`` được thực hiện trễ (bên trong hàm) để phần còn
    lại của module — và các module gọi ``predict_document`` với model giả lập
    trong test — không bắt buộc phải cài các thư viện nặng này.
    """
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_name)
    return model.to(device)


@dataclass(frozen=True)
class RawEntityPrediction:
    """Một dự đoán entity thô từ Layer 3, offset còn ở trên normalized_text.

    Đây CHƯA phải output cuối của Layer 3 — offset còn tính trên normalized_text,
    có thể trùng lặp hoặc chồng lấn nhau. Việc dedup/overlap-resolve/confidence
    filter/raw-offset-mapping thuộc về ``src.ner.postprocess`` (đúng như luồng
    "Postprocess NER" trong kiến trúc), không thực hiện ở đây.
    """

    document_id: str
    text: str
    entity_type: str
    normalized_start: int
    normalized_end: int
    confidence: float
    source: str = "gliner"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "type": self.entity_type,
            "normalized_position": [self.normalized_start, self.normalized_end],
            "confidence": self.confidence,
            "source": self.source,
        }


def iter_text_chunks(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    """Return overlapping word-based ``[start, end)`` character spans."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size.")

    words = list(re.finditer(r"\S+", text))
    if not words:
        return []

    chunks: list[tuple[int, int]] = []
    start_word = 0
    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        chunks.append((words[start_word].start(), words[end_word - 1].end()))
        if end_word == len(words):
            break
        start_word = end_word - chunk_overlap
    return chunks


def predict_document(
    document: MedicalDocument,
    model: GLiNERModel,
    *,
    labels: list[str] | None = None,
    threshold: float = 0.5,
    flat_ner: bool = True,
    multi_label: bool = False,
    normalize_kwargs: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[NormalizedDocument, list[RawEntityPrediction]]:
    """Chạy NER trên một ``MedicalDocument`` bằng model GLiNER đã load.

    Trả về cả ``NormalizedDocument`` (cần thiết để postprocess map offset về
    raw_text sau này) lẫn danh sách prediction thô. Model luôn chạy trên
    ``normalized_text`` — không module downstream nào được đọc raw_text trực
    tiếp, đúng nguyên tắc "sau Layer 2 mọi module chỉ làm việc với
    MedicalDocument/NormalizedDocument".
    """
    if labels is None:
        labels = DEFAULT_LABELS

    options = dict(normalize_kwargs or {})
    normalized = normalize_text(document.raw_text, **options)

    if not normalized.normalized_text.strip():
        return normalized, []

    predictions: list[RawEntityPrediction] = []
    text_length = len(normalized.normalized_text)
    chunks = iter_text_chunks(
        normalized.normalized_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    for chunk_start, chunk_end in chunks:
        chunk_text = normalized.normalized_text[chunk_start:chunk_end]
        raw_predictions = model.predict_entities(
            chunk_text,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )
        for raw in raw_predictions:
            local_start = int(raw["start"])
            local_end = int(raw["end"])
            if not 0 <= local_start < local_end <= len(chunk_text):
                # Bỏ span dị dạng, không để crash cả document.
                continue
            start = chunk_start + local_start
            end = chunk_start + local_end
            if end > text_length:
                continue

            predicted_label = str(
                raw["label"]
            )

            if predicted_label not in labels:
                continue

            prediction_text = (
                normalized.normalized_text[
                    start:end
                ]
            )

            predictions.append(
                RawEntityPrediction(
                    document_id=document.document_id,
                    text=prediction_text,
                    entity_type=predicted_label,
                    normalized_start=start,
                    normalized_end=end,
                    confidence=float(
                        raw["score"]
                    ),
                    source="gliner",
                )
            )

    return normalized, predictions


def predict_documents(
    documents: Iterable[MedicalDocument],
    model: GLiNERModel,
    *,
    labels: list[str] | None = None,
    threshold: float = 0.5,
    flat_ner: bool = True,
    multi_label: bool = False,
    normalize_kwargs: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, tuple[NormalizedDocument, list[RawEntityPrediction]]]:
    """Chạy ``predict_document`` cho nhiều document, map theo ``document_id``.

    Đây là vòng lặp tuần tự (gọi ``predict_entities`` từng văn bản một) — đơn
    giản và dễ test bằng mock. Nếu cần tăng tốc trên corpus lớn, có thể thay
    bằng ``model.inference(texts, labels, batch_size=...)`` để GLiNER tự batch
    nhiều văn bản trong một lần forward; API của hàm này không cần đổi khi đó.
    """
    return {
        document.document_id: predict_document(
            document,
            model,
            labels=labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            normalize_kwargs=normalize_kwargs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for document in documents
    }
