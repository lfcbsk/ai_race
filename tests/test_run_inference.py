from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.linking import MedicalEntityLinker
from src.run_inference import package_predictions, run_test_set, validate_test_set


class EmptyGLiNER:
    def predict_entities(self, text, labels, **kwargs):
        return []


class FailingGLiNER:
    def predict_entities(self, text, labels, **kwargs):
        raise RuntimeError("model failed")


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

            output_path = root / "predictions.jsonl"
            zip_path = root / "output.zip"
            summary = run_test_set(
                input_dir,
                output_path,
                EmptyGLiNER(),
                entity_linker=MedicalEntityLinker(),
                zip_path=zip_path,
            )

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["output/1.json", "output/2.json", "output/10.json"],
                )
                zipped_records = [
                    json.loads(archive.read(name).decode("utf-8"))
                    for name in archive.namelist()
                ]
            self.assertEqual([record["note_id"] for record in records], ["1", "2", "10"])
            self.assertEqual(zipped_records, records)
            self.assertTrue(all(record["entities"] == [] for record in records))
            self.assertEqual(summary.succeeded, 3)
            self.assertEqual(summary.failed, 0)
            self.assertTrue(Path(summary.error_path).is_file())

    def test_write_one_json_per_input_even_when_inference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            input_dir.mkdir()
            for note_id in range(1, 101):
                (input_dir / f"{note_id}.txt").write_text(
                    f"Medical note {note_id}", encoding="utf-8"
                )

            output_path = root / "predictions.jsonl"
            zip_path = root / "output.zip"
            summary = run_test_set(
                input_dir,
                output_path,
                FailingGLiNER(),
                entity_linker=MedicalEntityLinker(),
                zip_path=zip_path,
            )

            self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 100)
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(len(archive.namelist()), 100)
                self.assertEqual(archive.namelist()[0], "output/1.json")
                self.assertEqual(archive.namelist()[-1], "output/100.json")
                for note_id in range(1, 101):
                    record = json.loads(archive.read(f"output/{note_id}.json"))
                    self.assertEqual(record, {"note_id": str(note_id), "entities": []})

            self.assertEqual(summary.total, 100)
            self.assertEqual(summary.succeeded, 0)
            self.assertEqual(summary.failed, 100)

    def test_package_existing_predictions_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                '{"note_id":"2","entities":[]}\n'
                '{"note_id":"1","entities":[]}\n',
                encoding="utf-8",
            )

            zip_path = root / "output.zip"
            packaged = package_predictions(predictions_path, zip_path)

            self.assertEqual(packaged, 2)
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(
                    archive.namelist(), ["output/1.json", "output/2.json"]
                )


if __name__ == "__main__":
    unittest.main()
