from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ner.trainer import (
    NERTrainerConfig,
    fine_tune_gliner,
    split_gliner_dataset,
    validate_gliner_sample,
)


SAMPLES = [
    {"tokenized_text": ["Bệnh", "nhân", "sốt"], "ner": [[2, 2, "TRIỆU_CHỨNG"]]},
    {"tokenized_text": ["Có", "đau", "đầu"], "ner": [[1, 2, "TRIỆU_CHỨNG"]]},
    {"tokenized_text": ["Dùng", "aspirin"], "ner": [[1, 1, "THUỐC"]]},
    {"tokenized_text": ["Không", "ho"], "ner": [[1, 1, "TRIỆU_CHỨNG"]]},
]


class _Parameter:
    def numel(self):
        return 100


class _MockTrainer:
    def __init__(self):
        self.saved = False

    def save_model(self):
        self.saved = True


class _MockModel:
    def __init__(self):
        self.device = None
        self.kwargs = None
        self.trainer = _MockTrainer()

    def parameters(self):
        return [_Parameter()]

    def to(self, device):
        self.device = device
        return self

    def train_model(self, **kwargs):
        self.kwargs = kwargs
        return self.trainer


class NERTrainerTests(unittest.TestCase):
    def test_split_is_reproducible(self) -> None:
        train_a, eval_a = split_gliner_dataset(
            SAMPLES, validation_ratio=0.25, seed=7
        )
        train_b, eval_b = split_gliner_dataset(
            SAMPLES, validation_ratio=0.25, seed=7
        )
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual((len(train_a), len(eval_a)), (3, 1))

    def test_rejects_out_of_bounds_token_span(self) -> None:
        with self.assertRaisesRegex(ValueError, "ngoài văn bản"):
            validate_gliner_sample(
                {"tokenized_text": ["sốt"], "ner": [[0, 1, "X"]]}
            )

    def test_fine_tune_with_injected_model(self) -> None:
        model = _MockModel()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            output_dir = Path(temp_dir) / "model"
            config = NERTrainerConfig(
                model_path="mock-model",
                output_dir=str(output_dir),
                max_steps=5,
                train_batch_size=2,
                eval_batch_size=1,
                allow_remote_model=True,
            )
            result = fine_tune_gliner(
                SAMPLES[:3],
                eval_samples=SAMPLES[3:],
                config=config,
                model_factory=lambda _: model,
            )

            self.assertEqual(result.parameter_count, 100)
            self.assertEqual(model.kwargs["max_steps"], 5)
            self.assertTrue(model.trainer.saved)
            manifest = json.loads(
                (output_dir / "training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["result"]["train_samples"], 3)


if __name__ == "__main__":
    unittest.main()
