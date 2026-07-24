from __future__ import annotations

from src.ner.inference import RawEntityPrediction
from src.preprocessing import EntityAnnotation, NormalizedDocument


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def deduplicate_predictions(
    predictions: list[RawEntityPrediction],
) -> list[RawEntityPrediction]:
    """Gộp các prediction trùng hệt (cùng normalized span + cùng type).

    Khi trùng, giữ lại bản có confidence cao nhất.
    """
    best_by_key: dict[tuple[int, int, str], RawEntityPrediction] = {}
    for prediction in predictions:
        key = (
            prediction.normalized_start,
            prediction.normalized_end,
            prediction.entity_type,
        )
        current = best_by_key.get(key)
        if current is None or prediction.confidence > current.confidence:
            best_by_key[key] = prediction
    return list(best_by_key.values())


def resolve_overlaps(
    predictions: list[RawEntityPrediction],
) -> list[RawEntityPrediction]:
    """Giải quyết chồng lấn span theo kiểu greedy (flat NER).

    Ưu tiên confidence cao hơn; nếu bằng nhau thì ưu tiên span dài hơn (thường
    mang nhiều thông tin hơn, ví dụ "đau đầu dữ dội" tốt hơn "đau đầu"). Hai
    prediction chồng lấn nhau LUÔN xung đột dù khác type, vì một vùng text
    không thể vừa là TRIỆU_CHỨNG vừa là CHẨN_ĐOÁN cùng lúc trong cùng 1 span.
    """
    ordered = sorted(
        predictions,
        key=lambda p: (
            -p.confidence,
            -(p.normalized_end - p.normalized_start),
        ),
    )

    accepted: list[RawEntityPrediction] = []
    for candidate in ordered:
        conflicts = any(
            _spans_overlap(
                candidate.normalized_start,
                candidate.normalized_end,
                kept.normalized_start,
                kept.normalized_end,
            )
            for kept in accepted
        )
        if not conflicts:
            accepted.append(candidate)

    return accepted


def filter_by_confidence(
    predictions: list[RawEntityPrediction],
    min_confidence: float,
) -> list[RawEntityPrediction]:
    return [p for p in predictions if p.confidence >= min_confidence]


def map_to_raw_offsets(
    predictions: list[RawEntityPrediction],
    normalized: NormalizedDocument,
) -> list[EntityAnnotation]:
    """Map normalized offset về raw_text, trả kết quả dạng ``EntityAnnotation``.

    Dùng lại ``EntityAnnotation`` (thay vì tạo thêm một class ``EntityPrediction``
    riêng) để toàn bộ pipeline — NER, assertion, linking, validation, serializer —
    chỉ thao tác trên một schema thực thể duy nhất, đúng tinh thần thiết kế
    "một MedicalDocument, một EntityAnnotation cho mọi layer".
    """
    entities: list[EntityAnnotation] = []
    for prediction in predictions:
        raw_start, raw_end = normalized.normalized_span_to_raw(
            prediction.normalized_start, prediction.normalized_end
        )
        entities.append(
            EntityAnnotation(
                text=normalized.raw_text[
                    raw_start:raw_end
                ],
                entity_type=(
                    prediction.entity_type
                ),
                start=raw_start,
                end=raw_end,
                confidence=(
                    prediction.confidence
                ),
                source=prediction.source,
                metadata={
                    "normalized_position": [
                        prediction.normalized_start,
                        prediction.normalized_end,
                    ],
                },
            )
        )

    entities.sort(key=lambda entity: (entity.start, entity.end))
    return entities


def postprocess_predictions(
    predictions: list[RawEntityPrediction],
    normalized: NormalizedDocument,
    *,
    min_confidence: float = 0.3,
) -> list[EntityAnnotation]:
    """Pipeline hậu xử lý đầy đủ cho output của ``src.ner.inference``.

    Thứ tự cố định theo kiến trúc: remove duplicate -> overlap resolve ->
    confidence filter -> raw offset mapping.
    """
    step = deduplicate_predictions(predictions)
    step = resolve_overlaps(step)
    step = filter_by_confidence(step, min_confidence)
    return map_to_raw_offsets(step, normalized)
