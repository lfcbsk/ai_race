from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.linking import MedicalEntityLinker
from src.run_inference import run_test_set, validate_test_set


class EmptyGLiNER:
    def predict_entities(self, text, labels, **kwargs):
        return []


class RunInferenceTests(unittest.TestCase):
    def test_validate_and_write_predictions_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            input_dir.mkdir()
            for name in ("10.txt", "2.txt", "1.txt"):
                (input_dir / name).write_text(
                    f"Medical note {name}", encoding="utf-8"
                )

            validation = validate_test_set(input_dir)
            self.assertEqual(validation["num_files"], 3)
            self.assertEqual(validation["first_note_id"], "1")
            self.assertEqual(validation["last_note_id"], "10")

            output_path = root / "output" / "predictions.jsonl"
            summary = run_test_set(
                input_dir,
                output_path,
                EmptyGLiNER(),
                entity_linker=MedicalEntityLinker(),
            )

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["note_id"] for record in records], ["1", "2", "10"])
            self.assertTrue(all(record["entities"] == [] for record in records))
            self.assertEqual(summary.succeeded, 3)
            self.assertEqual(summary.failed, 0)
            self.assertTrue(Path(summary.error_path).is_file())


if __name__ == "__main__":
    unittest.main()
