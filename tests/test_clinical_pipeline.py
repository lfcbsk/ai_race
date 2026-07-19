from __future__ import annotations

import unittest

from src.clinical_pipeline import ClinicalNLPPipeline
from src.linking import HybridEntityLinker, MedicalEntityLinker, OntologyEntry
from src.preprocessing import MedicalDocument


class MockGLiNER:
    def predict_entities(self, text, labels, **kwargs):
        mention = "tăng huyết áp"
        start = text.index(mention)
        return [
            {
                "start": start,
                "end": start + len(mention),
                "text": mention,
                "label": "CHẨN_ĐOÁN",
                "score": 0.98,
            }
        ]


class ClinicalPipelineTests(unittest.TestCase):
    def test_end_to_end_local_pipeline(self) -> None:
        entry = OntologyEntry(
            "I10",
            "Tăng huyết áp vô căn",
            aliases=("tăng huyết áp",),
            ontology="ICD10",
        )
        pipeline = ClinicalNLPPipeline(
            MockGLiNER(),
            entity_linker=MedicalEntityLinker(
                icd_linker=HybridEntityLinker([entry])
            ),
        )
        document = MedicalDocument("note-1", "Tiền sử tăng huyết áp.")

        result = pipeline.process(document)

        self.assertTrue(result.validation.valid)
        self.assertEqual(result.entities[0].assertions, ["isHistorical"])
        self.assertEqual(result.entities[0].candidates, ["I10"])
        self.assertEqual(result.competition_output()["note_id"], "note-1")


if __name__ == "__main__":
    unittest.main()
