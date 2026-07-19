from __future__ import annotations

import unittest

from src.linking import (
    HybridEntityLinker,
    MedicalEntityLinker,
    OntologyEntry,
    evaluate_linking,
)
from src.preprocessing import EntityAnnotation, MedicalDocument


class LinkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.icd_entries = [
            OntologyEntry(
                "I10",
                "Tăng huyết áp vô căn",
                aliases=("cao huyết áp", "tăng huyết áp"),
                ontology="ICD10",
            ),
            OntologyEntry(
                "I11",
                "Bệnh tim do tăng huyết áp",
                aliases=("tăng huyết áp kèm bệnh tim",),
                ontology="ICD10",
            ),
            OntologyEntry(
                "J18",
                "Viêm phổi",
                aliases=("viêm phổi cộng đồng",),
                ontology="ICD10",
            ),
        ]

    def test_exact_alias_returns_immediately(self) -> None:
        text = "Bệnh nhân bị cao huyết áp."
        mention = "cao huyết áp"
        start = text.index(mention)
        result = HybridEntityLinker(self.icd_entries).link(
            MedicalDocument("x", text),
            EntityAnnotation(mention, "CHẨN_ĐOÁN", start, start + len(mention)),
            0,
        )
        self.assertEqual(result.selected_code, "I10")
        self.assertEqual(result.method, "exact_alias")
        self.assertEqual(result.confidence, 1.0)

    def test_bm25_and_reranker_retrieve_candidate(self) -> None:
        text = "Chẩn đoán viêm phổi mắc phải cộng đồng."
        mention = "viêm phổi mắc phải cộng đồng"
        start = text.index(mention)
        result = HybridEntityLinker(self.icd_entries).link(
            MedicalDocument("x", text),
            EntityAnnotation(mention, "CHẨN_ĐOÁN", start, start + len(mention)),
            0,
        )
        self.assertTrue(result.candidates)
        self.assertEqual(result.candidates[0].entry.code, "J18")

    def test_medical_router_only_links_diagnosis_and_drug(self) -> None:
        text = "Có tăng huyết áp và đau đầu."
        diagnosis_start = text.index("tăng huyết áp")
        symptom_start = text.index("đau đầu")
        entities = [
            EntityAnnotation("tăng huyết áp", "CHẨN_ĐOÁN", diagnosis_start, diagnosis_start + 14),
            EntityAnnotation("đau đầu", "TRIỆU_CHỨNG", symptom_start, symptom_start + 7),
        ]
        linked, results = MedicalEntityLinker(
            icd_linker=HybridEntityLinker(self.icd_entries)
        ).link_entities(MedicalDocument("x", text), entities)
        self.assertEqual(linked[0].candidates[0], "I10")
        self.assertEqual(linked[1].candidates, [])
        self.assertEqual(len(results), 1)

    def test_linking_metrics(self) -> None:
        text = "Có cao huyết áp."
        start = text.index("cao huyết áp")
        result = HybridEntityLinker(self.icd_entries).link(
            MedicalDocument("x", text),
            EntityAnnotation("cao huyết áp", "CHẨN_ĐOÁN", start, start + 12),
            0,
        )
        metrics = evaluate_linking([result], {("x", 0): "I10"})
        self.assertEqual(metrics.top1_accuracy, 1.0)
        self.assertEqual(metrics.recall_at[5], 1.0)


if __name__ == "__main__":
    unittest.main()
