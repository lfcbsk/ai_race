from __future__ import annotations

import unittest

from src.clinical_pipeline import ClinicalNLPPipeline
from src.linking import MedicalEntityLinker, OntologyEntry
from src.ner import DrugRuleDetector
from src.preprocessing import MedicalDocument, normalize_text


class EmptyGLiNER:
    def predict_entities(self, text, labels, **kwargs):
        return []


class DrugRuleDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = DrugRuleDetector.from_entries(
            [
                OntologyEntry("1", "aspirin", ontology="RXNORM"),
                OntologyEntry("2", "metoprolol succinate", ontology="RXNORM"),
                OntologyEntry("3", "caffeine", ontology="RXNORM"),
            ]
        )

    def test_detects_drugs_in_medication_section_and_schedule_formats(self) -> None:
        text = (
            "Thuốc trước khi nhập viện\n"
            "- Aspirin 81mg SCDC\n"
            "- metoprolol succinate 100mg daily\n"
            "- Lasix 40mg S-C\n"
            "2. Bệnh sử hiện tại\n"
            "Một ngày uống nhiều caffeine từ cà phê\n"
        )
        normalized = normalize_text(text)

        predictions = self.detector.find_predictions(normalized, "note-1")

        self.assertEqual(
            [prediction.text for prediction in predictions],
            ["Aspirin", "metoprolol succinate", "Lasix"],
        )
        self.assertTrue(all(p.entity_type == "THUỐC" for p in predictions))

    def test_detects_drug_outside_section_when_dose_is_present(self) -> None:
        text = "Bệnh nhân được cho Aspirin 325mg tại khoa cấp cứu."
        normalized = normalize_text(text)

        predictions = self.detector.find_predictions(normalized, "note-1")

        self.assertEqual(len(predictions), 1)
        raw_start, raw_end = normalized.normalized_span_to_raw(
            predictions[0].normalized_start,
            predictions[0].normalized_end,
        )
        self.assertEqual(text[raw_start:raw_end], "Aspirin")

    def test_pipeline_merges_rule_prediction_with_empty_model(self) -> None:
        text = "Thuốc trước khi nhập viện: aspirin 81mg SCDC"
        pipeline = ClinicalNLPPipeline(
            EmptyGLiNER(),
            entity_linker=MedicalEntityLinker(),
            drug_rule_detector=self.detector,
        )

        result = pipeline.process(MedicalDocument("note-1", text))

        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].text, "aspirin")
        self.assertEqual(result.entities[0].entity_type, "THUỐC")


if __name__ == "__main__":
    unittest.main()
