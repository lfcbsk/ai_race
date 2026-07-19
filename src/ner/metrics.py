from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.preprocessing import EntityAnnotation

MatchMode = Literal["exact", "overlap"]


@dataclass
class TypeMetrics:
    """Số liệu TP/FP/FN cộng dồn cho một entity type (hoặc tổng micro)."""

    entity_type: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.entity_type,
            "support": self.true_positive + self.false_negative,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class NERMetricsReport:
    """Kết quả đánh giá NER: chi tiết theo type + tổng hợp micro/macro."""

    mode: MatchMode = "exact"
    per_type: dict[str, TypeMetrics] = field(default_factory=dict)

    def _metrics_for(self, entity_type: str) -> TypeMetrics:
        return self.per_type.setdefault(entity_type, TypeMetrics(entity_type))

    @property
    def micro(self) -> TypeMetrics:
        aggregate = TypeMetrics("micro")
        for metrics in self.per_type.values():
            aggregate.true_positive += metrics.true_positive
            aggregate.false_positive += metrics.false_positive
            aggregate.false_negative += metrics.false_negative
        return aggregate

    @property
    def macro_f1(self) -> float:
        scores = [metrics.f1 for metrics in self.per_type.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "per_type": {
                entity_type: metrics.to_dict()
                for entity_type, metrics in sorted(self.per_type.items())
            },
            "micro": self.micro.to_dict(),
            "macro_f1": round(self.macro_f1, 4),
        }


def _match_exact(gold: EntityAnnotation, pred: EntityAnnotation) -> bool:
    return (
        gold.start == pred.start
        and gold.end == pred.end
        and gold.entity_type == pred.entity_type
    )


def _match_overlap(gold: EntityAnnotation, pred: EntityAnnotation) -> bool:
    """Khớp lỏng: cùng type và span chồng lấn nhau ít nhất 1 ký tự.

    Hữu ích vì boundary NER tiếng Việt hay lệch 1-2 từ (ví dụ model đoán "đau
    đầu" trong khi gold là "đau đầu dữ dội") — exact match sẽ chấm 0 điểm dù
    model về cơ bản đã bắt đúng thực thể.
    """
    return gold.entity_type == pred.entity_type and gold.start < pred.end and pred.start < gold.end


def score_document(
    gold_entities: list[EntityAnnotation],
    pred_entities: list[EntityAnnotation],
    report: NERMetricsReport,
) -> None:
    """Chấm một document, cộng dồn TP/FP/FN vào ``report`` (in-place).

    Ghép cặp theo greedy — mỗi gold chỉ ghép với 1 pred chưa dùng, và ngược
    lại. Đây là xấp xỉ của bipartite matching tối ưu (không phải lúc nào cũng
    ra kết quả tối ưu tuyệt đối trong trường hợp nhiều span chồng chéo phức
    tạp), nhưng đủ tốt để theo dõi baseline và đơn giản để kiểm chứng.
    """
    match_fn = _match_exact if report.mode == "exact" else _match_overlap
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for gold_index, gold in enumerate(gold_entities):
        for pred_index, pred in enumerate(pred_entities):
            if pred_index in matched_pred:
                continue
            if match_fn(gold, pred):
                matched_gold.add(gold_index)
                matched_pred.add(pred_index)
                break

    for gold_index, gold in enumerate(gold_entities):
        metrics = report._metrics_for(gold.entity_type)
        if gold_index in matched_gold:
            metrics.true_positive += 1
        else:
            metrics.false_negative += 1

    for pred_index, pred in enumerate(pred_entities):
        if pred_index in matched_pred:
            continue
        report._metrics_for(pred.entity_type).false_positive += 1


def evaluate_ner(
    gold_by_document: dict[str, list[EntityAnnotation]],
    pred_by_document: dict[str, list[EntityAnnotation]],
    *,
    mode: MatchMode = "exact",
) -> NERMetricsReport:
    """Đánh giá NER trên nhiều document, khớp theo ``document_id``.

    Document có trong gold nhưng model không trả prediction nào vẫn được tính
    (toàn bộ entity của nó thành false negative).
    """
    report = NERMetricsReport(mode=mode)
    for document_id, gold_entities in gold_by_document.items():
        pred_entities = pred_by_document.get(document_id, [])
        score_document(gold_entities, pred_entities, report)
    return report
