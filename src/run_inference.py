"""Run the local clinical pipeline on a directory of test-set TXT files.

Example (PowerShell):
    python -m src.run_inference --model models/medical-gliner --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.clinical_pipeline import ClinicalNLPPipeline, build_default_medical_linker
from src.linking import MedicalEntityLinker
from src.ner import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    GLiNERModel,
    load_gliner_model,
)
from src.preprocessing import load_documents


@dataclass(frozen=True)
class InferenceSummary:
    total: int
    succeeded: int
    failed: int
    output_path: str
    zip_path: str
    error_path: str


class _UnavailableGLiNER:
    """Model placeholder that lets the CLI still create a complete submission."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def predict_entities(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError(self.reason)


def _file_sort_key(path: Path) -> tuple[int, int | str, str]:
    """Sort 1.txt, 2.txt, 10.txt numerically, then other names alphabetically."""
    if path.stem.isdigit():
        return (0, int(path.stem), path.name.lower())
    return (1, path.stem.lower(), path.name.lower())


def discover_test_files(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Test input directory not found: {directory}")
    files = sorted(directory.glob("*.txt"), key=_file_sort_key)
    if not files:
        raise ValueError(f"No .txt test files found in: {directory}")
    return files


def validate_test_set(input_dir: str | Path) -> dict[str, Any]:
    """Read every TXT file without loading a model; useful before a long run."""
    files = discover_test_files(input_dir)
    total_characters = 0
    for path in files:
        document = load_documents(path, document_id=path.stem)[0]
        total_characters += len(document.raw_text)
    return {
        "input_dir": str(Path(input_dir)),
        "num_files": len(files),
        "total_characters": total_characters,
        "first_note_id": files[0].stem,
        "last_note_id": files[-1].stem,
    }


def package_predictions(
    predictions_path: str | Path,
    zip_path: str | Path,
) -> int:
    """Convert prediction JSONL records into ``output/<note_id>.json`` ZIP."""
    source = Path(predictions_path)
    destination = Path(zip_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Predictions JSONL path and ZIP path must be different.")

    records: list[dict[str, Any]] = []
    seen_note_ids: set[str] = set()
    with source.open(encoding="utf-8") as prediction_file:
        for line_number, line in enumerate(prediction_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            note_id = str(record.get("note_id", ""))
            if not note_id or "/" in note_id or "\\" in note_id:
                raise ValueError(f"Invalid note_id at JSONL line {line_number}: {note_id!r}")
            if note_id in seen_note_ids:
                raise ValueError(f"Duplicate note_id in predictions JSONL: {note_id!r}")
            seen_note_ids.add(note_id)
            records.append(record)

    records.sort(
        key=lambda record: _file_sort_key(Path(f"{record['note_id']}.json"))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    zip_tmp = destination.with_name(f"{destination.name}.tmp")
    with zipfile.ZipFile(
        zip_tmp, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for record in records:
            archive.writestr(
                f"output/{record['note_id']}.json",
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            )
    zip_tmp.replace(destination)
    return len(records)


def run_test_set(
    input_dir: str | Path,
    output_path: str | Path,
    model: GLiNERModel,
    *,
    entity_linker: MedicalEntityLinker | None = None,
    error_path: str | Path | None = None,
    zip_path: str | Path | None = None,
    threshold: float = 0.5,
    min_confidence: float = 0.3,
    include_text: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> InferenceSummary:
    """Run notes to JSONL, then create a competition-ready output ZIP.

    The intermediate file contains one JSON record per line. It is then split
    into ``output/<note_id>.json`` entries in the ZIP.

    A failure in one note is recorded separately and produces a valid empty
    prediction, so the archive always contains one JSON file per input TXT.
    """
    files = discover_test_files(input_dir)
    destination = Path(output_path)
    zip_destination = (
        Path(zip_path) if zip_path is not None else destination.with_name("output.zip")
    )
    errors_destination = (
        Path(error_path)
        if error_path is not None
        else destination.with_name("prediction_errors.jsonl")
    )
    if destination.resolve() == errors_destination.resolve():
        raise ValueError("Output path and error path must be different.")
    if zip_destination.resolve() in {
        destination.resolve(),
        errors_destination.resolve(),
    }:
        raise ValueError("JSONL, ZIP, and error paths must be different.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors_destination.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = destination.with_name(f"{destination.name}.tmp")
    errors_tmp = errors_destination.with_name(f"{errors_destination.name}.tmp")
    pipeline = ClinicalNLPPipeline(
        model,
        entity_linker=entity_linker,
        ner_threshold=threshold,
        min_ner_confidence=min_confidence,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    succeeded = 0
    failed = 0
    with output_tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file, errors_tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as error_file:
        for path in files:
            record: dict[str, Any] = {
                "note_id": path.stem,
                "entities": [],
            }
            try:
                document = load_documents(path, document_id=path.stem)[0]
                if include_text:
                    record["text"] = document.raw_text
                result = pipeline.process(document)
                record = result.competition_output(
                    strict=True,
                    include_text=include_text,
                )
                succeeded += 1
            except Exception as exc:  # keep other test notes runnable
                error = {
                    "note_id": path.stem,
                    "source_path": str(path),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                error_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                failed += 1
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    output_tmp.replace(destination)
    errors_tmp.replace(errors_destination)
    packaged = package_predictions(destination, zip_destination)
    if packaged != len(files):
        raise RuntimeError(f"Packaged {packaged} JSON files for {len(files)} inputs.")
    return InferenceSummary(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        output_path=str(destination),
        zip_path=str(zip_destination),
        error_path=str(errors_destination),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/test_set/input")
    parser.add_argument(
        "--output",
        default="data/test_set/predictions.jsonl",
        help="Intermediate JSONL containing one prediction per line.",
    )
    parser.add_argument(
        "--zip-output",
        help="Submission ZIP path (default: output.zip next to --output).",
    )
    parser.add_argument(
        "--errors", default="data/test_set/prediction_errors.jsonl"
    )
    parser.add_argument("--model", default="models/medical-gliner")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum whitespace-delimited words per GLiNER call.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlapping words between consecutive chunks.",
    )
    parser.add_argument("--icd", default="data/icd_mapping_final.json")
    parser.add_argument("--drug", default="data/drug_mapping_final.json")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow --model to be a Hugging Face model ID instead of a local path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count input files without loading GLiNER.",
    )
    return parser


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(validate_test_set(args.input), ensure_ascii=False, indent=2))
        return

    model_path = Path(args.model)
    model_error: str | None = None
    model: GLiNERModel
    if not args.allow_remote_model and not model_path.exists():
        model_error = (
            f"Local GLiNER checkpoint not found: {model_path}. "
            "Every output JSON will contain an empty entities list."
        )
        model = _UnavailableGLiNER(model_error)
    else:
        try:
            model = load_gliner_model(args.model, device=args.device)
        except Exception as exc:
            model_error = (
                f"Could not load GLiNER model {args.model!r}: "
                f"{type(exc).__name__}: {exc}. "
                "Every output JSON will contain an empty entities list."
            )
            model = _UnavailableGLiNER(model_error)

    if model_error is not None:
        print(f"WARNING: {model_error}", file=sys.stderr)
        linker = None
    else:
        linker = build_default_medical_linker(icd_path=args.icd, drug_path=args.drug)
    predictions_output = Path(args.output)
    zip_output = Path(args.zip_output) if args.zip_output else None
    if predictions_output.suffix.lower() == ".zip":
        if zip_output is not None:
            parser.error("Do not pass a ZIP path to both --output and --zip-output.")
        zip_output = predictions_output
        predictions_output = predictions_output.with_name("predictions.jsonl")

    summary = run_test_set(
        args.input,
        predictions_output,
        model,
        entity_linker=linker,
        error_path=args.errors,
        zip_path=zip_output,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
        include_text=args.include_text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
