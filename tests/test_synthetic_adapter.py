from __future__ import annotations

import unittest

from src.preprocessing import SyntheticData, synthetic_to_documents


class SyntheticAdapterTests(unittest.TestCase):
    def test_separate_content_and_label_become_training_document(self) -> None:
        data = SyntheticData(
            contents=[{"note_id": "n1", "text": "Bệnh nhân sốt."}],
            labels=[
                {
                    "note_id": "n1",
                    "entities": [
                        {
                            "text": "sốt",
                            "type": "TRIỆU_CHỨNG",
                            "position": [10, 13],
                        }
                    ],
                }
            ],
        )
        documents = synthetic_to_documents(data)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].entities[0].text, "sốt")
        self.assertNotIn("entities", data.contents[0])
        self.assertNotIn("text", data.labels[0])


if __name__ == "__main__":
    unittest.main()
