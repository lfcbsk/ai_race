from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing import (
    EntityAnnotation,
)


@dataclass
class NERCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = (
            self.true_positive
            + self.false_positive
        )

        if denominator == 0:
            return 0.0

        return (
            self.true_positive
            / denominator
        )

    @property
    def recall(self) -> float:
        denominator = (
            self.true_positive
            + self.false_negative
        )

        if denominator == 0:
            return 0.0

        return (
            self.true_positive
            / denominator
        )

    @property
    def f1(self) -> float:
        denominator = (
            self.precision
            + self.recall
        )

        if denominator == 0:
            return 0.0

        return (
            2
            * self.precision
            * self.recall
            / denominator
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": (
                self.true_positive
            ),
            "false_positive": (
                self.false_positive
            ),
            "false_negative": (
                self.false_negative
            ),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def entity_key(
    entity: EntityAnnotation,
) -> tuple[int, int, str]:
    return (
        entity.start,
        entity.end,
        entity.entity_type,
    )


def evaluate_entities(
    gold_entities:
        list[EntityAnnotation],
    predicted_entities:
        list[EntityAnnotation],
) -> NERCounts:
    gold = {
        entity_key(entity)
        for entity in gold_entities
    }

    predicted = {
        entity_key(entity)
        for entity in predicted_entities
    }

    return NERCounts(
        true_positive=len(
            gold & predicted
        ),
        false_positive=len(
            predicted - gold
        ),
        false_negative=len(
            gold - predicted
        ),
    )