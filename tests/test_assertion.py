from __future__ import annotations

import unittest

from src.assertion import AssertionDetector
from src.preprocessing import EntityAnnotation, MedicalDocument


def entity(text: str, mention: str, label: str = "TRIỆU_CHỨNG") -> EntityAnnotation:
    start = text.index(mention)
    return EntityAnnotation(mention, label, start, start + len(mention))


class AssertionTests(unittest.TestCase):
    def test_negation_stops_at_contrast_boundary(self) -> None:
        text = "Bệnh nhân không đau đầu, nhưng có chóng mặt."
        entities = [entity(text, "đau đầu"), entity(text, "chóng mặt")]
        output, _ = AssertionDetector().annotate(
            MedicalDocument("n1", text, entities=entities)
        )
        self.assertEqual(output[0].assertions, ["isNegated"])
        self.assertEqual(output[1].assertions, [])

    def test_historical_forward_and_backward_cues(self) -> None:
        text = "Tiền sử tăng huyết áp. Đau ngực cách đây 5 năm."
        entities = [
            entity(text, "tăng huyết áp", "CHẨN_ĐOÁN"),
            entity(text, "Đau ngực"),
        ]
        output, _ = AssertionDetector().annotate(
            MedicalDocument("n2", text, entities=entities)
        )
        self.assertIn("isHistorical", output[0].assertions)
        self.assertIn("isHistorical", output[1].assertions)

    def test_family_cue(self) -> None:
        text = "Mẹ bị đái tháo đường, bệnh nhân khỏe mạnh."
        output, _ = AssertionDetector().annotate(
            MedicalDocument(
                "n3",
                text,
                entities=[entity(text, "đái tháo đường", "CHẨN_ĐOÁN")],
            )
        )
        self.assertEqual(output[0].assertions, ["isFamily"])


if __name__ == "__main__":
    unittest.main()

