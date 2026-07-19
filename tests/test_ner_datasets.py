from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ner.datasets import (
    NERDatasetError,
    build_gliner_dataset,
    convert_document_to_ner_sample,
)
from src.preprocessing import EntityAnnotation, MedicalDocument


class NERDatasetTests(unittest.TestCase):
    def make_document(
        self,
        text: str,
        entity_text: str,
        *,
        entity_type: str = "TRIỆU_CHỨNG",
    ) -> MedicalDocument:
        start = text.index(entity_text)
        return MedicalDocument(
            document_id="note-1",
            raw_text=text,
            entities=[
                EntityAnnotation(
                    text=entity_text,
                    entity_type=entity_type,
                    start=start,
                    end=start + len(entity_text),
                )
            ],
        )

    def test_maps_raw_span_to_gliner_token_span(self) -> None:
        document = self.make_document(
            "  Bệnh   nhân đau đầu . ", "đau đầu"
        )

        sample = convert_document_to_ner_sample(document)

        self.assertEqual(
            sample["tokenized_text"],
            ["Bệnh", "nhân", "đau", "đầu", "."],
        )
        self.assertEqual(sample["ner"], [[2, 3, "TRIỆU_CHỨNG"]])

    def test_rejects_raw_entity_text_mismatch(self) -> None:
        document = MedicalDocument(
            document_id="bad",
            raw_text="đau đầu",
            entities=[EntityAnnotation("sai", "X", 0, 3)],
        )

        with self.assertRaises(NERDatasetError) as context:
            convert_document_to_ner_sample(document)

        self.assertEqual(context.exception.error.code, "raw_text_mismatch")

    def test_rejects_entity_inside_token(self) -> None:
        document = self.make_document("foobar", "foo", entity_type="X")

        with self.assertRaises(NERDatasetError) as context:
            convert_document_to_ner_sample(document)

        self.assertEqual(context.exception.error.code, "token_alignment_error")

    def test_skips_and_logs_invalid_sample(self) -> None:
        good = self.make_document("bệnh nhân sốt", "sốt")
        bad = MedicalDocument(
            document_id="bad",
            raw_text="đau đầu",
            entities=[EntityAnnotation("sai", "X", 0, 3)],
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            log_path = Path(temp_dir) / "errors.jsonl"

            result = build_gliner_dataset(
                [good, bad],
                on_error="skip",
                error_log_path=log_path,
            )

            self.assertEqual(len(result.samples), 1)
            self.assertEqual(len(result.errors), 1)
            logged = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(logged[0]["document_id"], "bad")
            self.assertEqual(logged[0]["code"], "raw_text_mismatch")


if __name__ == "__main__":
    unittest.main()
