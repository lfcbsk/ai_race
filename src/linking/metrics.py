from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .schemas import LinkingResult


@dataclass(frozen=True)
class LinkingMetrics:
    total: int
    top1_accuracy: float
    recall_at: dict[int, float]
    mean_reciprocal_rank: float
    selection_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "top1_accuracy": round(self.top1_accuracy, 6),
            "recall_at": {
                str(k): round(value, 6) for k, value in sorted(self.recall_at.items())
            },
            "mrr": round(self.mean_reciprocal_rank, 6),
            "selection_coverage": round(self.selection_coverage, 6),
        }


def evaluate_linking(
    results: Iterable[LinkingResult],
    gold_codes: dict[tuple[str, int], str],
    *,
    ks: tuple[int, ...] = (1, 5, 10, 20, 30),
) -> LinkingMetrics:
    result_by_key = {
        (result.document_id, result.entity_index): result for result in results
    }
    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    top1_correct = 0
    selected = 0
    for key, gold_code in gold_codes.items():
        result = result_by_key.get(key)
        if result is None:
            continue
        if result.selected_code is not None:
            selected += 1
        ranked_codes = [candidate.entry.code for candidate in result.candidates]
        if result.selected_code == gold_code:
            top1_correct += 1
        if gold_code in ranked_codes:
            rank = ranked_codes.index(gold_code) + 1
            reciprocal_rank_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1

    total = len(gold_codes)
    denominator = total or 1
    return LinkingMetrics(
        total=total,
        top1_accuracy=top1_correct / denominator,
        recall_at={k: hits[k] / denominator for k in ks},
        mean_reciprocal_rank=reciprocal_rank_sum / denominator,
        selection_coverage=selected / denominator,
    )
