from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.preprocessing import (
    load_synthetic_data,
    synthetic_to_documents,
)

from .datasets import (
    DatasetBuildResult,
    GLiNERSample,
    build_gliner_dataset,
)


@dataclass(frozen=True)
class NERTrainerConfig:
    model_path: str
    output_dir: str = "models/medical-gliner"
    validation_ratio: float = 0.1
    seed: int = 42
    max_steps: int = 2_000
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    others_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    save_steps: int = 200
    logging_steps: int = 20
    save_total_limit: int = 2
    device: str = "cpu"
    bf16: bool = False
    allow_remote_model: bool = False
    max_parameters: int = 9_000_000_000

    def validate(self) -> None:
        if not 0.0 <= self.validation_ratio < 1.0:
            raise ValueError("validation_ratio phải nằm trong [0, 1).")
        if self.max_steps <= 0:
            raise ValueError("max_steps phải lớn hơn 0.")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch size phải lớn hơn 0.")
        if self.learning_rate <= 0 or self.others_learning_rate <= 0:
            raise ValueError("Learning rate phải lớn hơn 0.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio phải nằm trong [0, 1).")
        if self.max_parameters <= 0:
            raise ValueError("max_parameters phải lớn hơn 0.")


@dataclass
class NERTrainingResult:
    output_dir: str
    train_samples: int
    eval_samples: int
    dataset_errors: int
    parameter_count: int | None
    trainer: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "dataset_errors": self.dataset_errors,
            "parameter_count": self.parameter_count,
        }


def validate_gliner_sample(sample: Any, sample_index: int = 0) -> GLiNERSample:
    if not isinstance(sample, dict):
        raise TypeError(f"Sample {sample_index} phải là object.")
    tokens = sample.get("tokenized_text")
    entities = sample.get("ner")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise TypeError(f"Sample {sample_index}: tokenized_text phải là list[str].")
    if not tokens:
        raise ValueError(f"Sample {sample_index}: tokenized_text rỗng.")
    if not isinstance(entities, list):
        raise TypeError(f"Sample {sample_index}: ner phải là list.")
    seen: set[tuple[int, int, str]] = set()
    normalized_entities: list[list[int | str]] = []
    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, (list, tuple)) or len(entity) != 3:
            raise ValueError(
                f"Sample {sample_index}, entity {entity_index}: phải có dạng [start, end, label]."
            )
        start, end, label = entity
        if not all(
            isinstance(offset, int) and not isinstance(offset, bool)
            for offset in (start, end)
        ):
            raise TypeError(
                f"Sample {sample_index}, entity {entity_index}: token offset phải là int."
            )
        if not 0 <= start <= end < len(tokens):
            raise ValueError(
                f"Sample {sample_index}, entity {entity_index}: token span ngoài văn bản."
            )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Sample {sample_index}, entity {entity_index}: label không hợp lệ."
            )
        key = (start, end, label)
        if key in seen:
            raise ValueError(
                f"Sample {sample_index}, entity {entity_index}: entity bị trùng {key}."
            )
        seen.add(key)
        normalized_entities.append([start, end, label])
    return {"tokenized_text": list(tokens), "ner": normalized_entities}


def validate_gliner_dataset(samples: Iterable[Any]) -> list[GLiNERSample]:
    return [
        validate_gliner_sample(sample, index)
        for index, sample in enumerate(samples)
    ]


def load_gliner_dataset(path: str | Path) -> list[GLiNERSample]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")
    if dataset_path.suffix.lower() == ".jsonl":
        payload: list[Any] = []
        with dataset_path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    payload.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL lỗi tại {dataset_path}, dòng {line_number}."
                    ) from exc
    elif dataset_path.suffix.lower() == ".json":
        with dataset_path.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise TypeError("GLiNER JSON dataset phải là một array.")
    else:
        raise ValueError("Dataset GLiNER chỉ hỗ trợ .json hoặc .jsonl.")
    return validate_gliner_dataset(payload)


def split_gliner_dataset(
    samples: Iterable[GLiNERSample],
    *,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[GLiNERSample], list[GLiNERSample]]:
    validated = validate_gliner_dataset(samples)
    if not validated:
        raise ValueError("Dataset GLiNER rỗng.")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio phải nằm trong [0, 1).")
    shuffled = list(validated)
    random.Random(seed).shuffle(shuffled)
    if validation_ratio == 0.0:
        return shuffled, []
    if len(shuffled) < 2:
        raise ValueError("Cần ít nhất 2 sample để tạo validation split.")
    eval_size = max(1, round(len(shuffled) * validation_ratio))
    eval_size = min(eval_size, len(shuffled) - 1)
    return shuffled[eval_size:], shuffled[:eval_size]


def prepare_synthetic_ner_dataset(
    contents_path: str | Path,
    labels_path: str | Path,
    *,
    strict: bool = True,
    error_log_path: str | Path | None = None,
    normalize_kwargs: dict[str, Any] | None = None,
) -> DatasetBuildResult:
    split = load_synthetic_data(contents_path, labels_path, strict=strict)
    documents = synthetic_to_documents(split)
    return build_gliner_dataset(
        documents,
        on_error="raise" if strict else "skip",
        error_log_path=error_log_path,
        normalize_kwargs=normalize_kwargs,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _default_model_factory(model_path: str) -> Any:
    from gliner import GLiNER

    return GLiNER.from_pretrained(model_path)


def _parameter_count(model: Any) -> int | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    return sum(parameter.numel() for parameter in parameters())


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def fine_tune_gliner(
    train_samples: Iterable[GLiNERSample],
    *,
    config: NERTrainerConfig,
    eval_samples: Iterable[GLiNERSample] | None = None,
    dataset_errors: int = 0,
    model_factory: Callable[[str], Any] | None = None,
) -> NERTrainingResult:
    config.validate()
    model_path = Path(config.model_path)
    if not config.allow_remote_model and not model_path.exists():
        raise FileNotFoundError(
            "Self-host mode yêu cầu model_path là checkpoint local. "
            "Dùng allow_remote_model=True nếu chủ động muốn tải từ Hub."
        )

    train_data = validate_gliner_dataset(train_samples)
    if not train_data:
        raise ValueError("Train dataset rỗng.")
    eval_data = (
        validate_gliner_dataset(eval_samples)
        if eval_samples is not None
        else []
    )
    _set_seed(config.seed)
    factory = model_factory or _default_model_factory
    model = factory(config.model_path)
    parameter_count = _parameter_count(model)
    if parameter_count is not None and parameter_count >= config.max_parameters:
        raise ValueError(
            f"Model có {parameter_count:,} parameters, vượt giới hạn "
            f"{config.max_parameters:,}."
        )
    if hasattr(model, "to"):
        model = model.to(config.device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train_dataset.json", train_data)
    _write_json(output_dir / "eval_dataset.json", eval_data)

    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data or None,
        output_dir=str(output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        others_lr=config.others_learning_rate,
        weight_decay=config.weight_decay,
        others_weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
    )
    if hasattr(trainer, "save_model"):
        trainer.save_model()

    result = NERTrainingResult(
        output_dir=str(output_dir),
        train_samples=len(train_data),
        eval_samples=len(eval_data),
        dataset_errors=dataset_errors,
        parameter_count=parameter_count,
        trainer=trainer,
    )
    _write_json(
        output_dir / "training_manifest.json",
        {"config": asdict(config), "result": result.to_dict()},
    )
    return result


def train_from_synthetic(
    contents_path: str | Path,
    labels_path: str | Path,
    *,
    config: NERTrainerConfig,
    error_log_path: str | Path | None = None,
    strict: bool = False,
) -> NERTrainingResult:
    prepared = prepare_synthetic_ner_dataset(
        contents_path,
        labels_path,
        strict=strict,
        error_log_path=error_log_path,
    )
    train_data, eval_data = split_gliner_dataset(
        prepared.samples,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    return fine_tune_gliner(
        train_data,
        eval_samples=eval_data,
        config=config,
        dataset_errors=len(prepared.errors),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GLiNER NER bằng checkpoint local."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train", help="GLiNER dataset .json/.jsonl")
    source.add_argument("--contents", help="Synthetic contents.jsonl")
    parser.add_argument("--labels", help="Synthetic labels.jsonl; bắt buộc với --contents")
    parser.add_argument("--model", required=True, help="Checkpoint GLiNER local")
    parser.add_argument("--output", default="models/medical-gliner")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = NERTrainerConfig(
        model_path=args.model,
        output_dir=args.output,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_steps=args.max_steps,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        bf16=args.bf16,
        allow_remote_model=args.allow_remote_model,
    )
    if args.contents:
        if not args.labels:
            parser.error("--labels là bắt buộc khi dùng --contents.")
        prepared = prepare_synthetic_ner_dataset(
            args.contents,
            args.labels,
            strict=False,
            error_log_path=(
                None
                if args.dry_run
                else Path(args.output) / "dataset_errors.jsonl"
            ),
        )
        samples = prepared.samples
        dataset_errors = len(prepared.errors)
    else:
        samples = load_gliner_dataset(args.train)
        dataset_errors = 0

    train_data, eval_data = split_gliner_dataset(
        samples,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "train_samples": len(train_data),
                    "eval_samples": len(eval_data),
                    "dataset_errors": dataset_errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result = fine_tune_gliner(
        train_data,
        eval_samples=eval_data,
        config=config,
        dataset_errors=dataset_errors,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
