"""Chạy baseline NER end-to-end: Layer 1 (load) -> Layer 2 (normalize, tự
động trong predict_document) -> Layer 3 (NER inference + postprocess) ->
Evaluation Layer (metrics).

Đây CHƯA phải pipeline cuộc thi đầy đủ (chưa có assertion/linking/validation/
serializer) — mục tiêu duy nhất của file này hiện tại là đo baseline NER với
GLiNER pretrained, đúng như Giai đoạn 2 trong roadmap.

CHẠY:
    python -m src.main_pipeline --gold data/validated/validated_pass.jsonl

Cần mạng để tải checkpoint GLiNER lần đầu (từ HuggingFace Hub).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ner import (
    DEFAULT_LABELS,
    DEFAULT_MODEL_NAME,
    evaluate_ner,
    load_gliner_model,
    postprocess_predictions,
    predict_document,
)
from src.preprocessing import load_documents


def run_ner_baseline(
    gold_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    labels: list[str] | None = None,
    threshold: float = 0.5,
    min_confidence: float = 0.3,
) -> dict:
    """Đo baseline NER trên một tập gold (jsonl có sẵn field ``entities``).

    Trả về dict {"exact": NERMetricsReport.to_dict(), "overlap": ...}.
    """
    documents = load_documents(gold_path, strict=False)
    documents = [document for document in documents if document.entities]
    if not documents:
        raise ValueError(
            f"{gold_path} không có document nào có sẵn nhãn (entities) để làm gold."
        )

    model = load_gliner_model(model_name)

    gold_by_document = {document.document_id: document.entities for document in documents}
    pred_by_document: dict[str, list] = {}

    for document in documents:
        normalized, raw_predictions = predict_document(
            document,
            model,
            labels=labels or DEFAULT_LABELS,
            threshold=threshold,
        )
        pred_by_document[document.document_id] = postprocess_predictions(
            raw_predictions,
            normalized,
            min_confidence=min_confidence,
        )

    return {
        "num_documents": len(documents),
        "exact": evaluate_ner(gold_by_document, pred_by_document, mode="exact").to_dict(),
        "overlap": evaluate_ner(gold_by_document, pred_by_document, mode="overlap").to_dict(),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        default="data/validated/validated_pass.jsonl",
        help="File jsonl có sẵn field 'entities' để làm gold (mặc định: validated_pass.jsonl).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    args = parser.parse_args()

    report = run_ner_baseline(
        args.gold,
        model_name=args.model,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
