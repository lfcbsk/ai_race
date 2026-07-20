from __future__ import annotations

import unittest

from src.ner import iter_text_chunks, postprocess_predictions, predict_document
from src.preprocessing import MedicalDocument


class RecordingGLiNER:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_entities(self, text, labels, **kwargs):
        self.calls.append(text)
        predictions = []
        start = 0
        while (match_start := text.find("disease", start)) >= 0:
            predictions.append(
                {
                    "start": match_start,
                    "end": match_start + len("disease"),
                    "text": "disease",
                    "label": "CHẨN_ĐOÁN",
                    "score": 0.9,
                }
            )
            start = match_start + len("disease")
        return predictions


class NERInferenceChunkTests(unittest.TestCase):
    def test_chunks_cover_long_text_with_overlap(self) -> None:
        text = "w0 w1 w2 w3 disease w5 w6 w7 w8 disease w10"

        chunks = iter_text_chunks(text, chunk_size=5, chunk_overlap=2)

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(text[start:end].split()) <= 5 for start, end in chunks))
        self.assertEqual(text[chunks[0][0] : chunks[0][1]], "w0 w1 w2 w3 disease")
        self.assertEqual(text[chunks[-1][0] : chunks[-1][1]], "w6 w7 w8 disease w10")

    def test_predict_document_restores_global_offsets_and_deduplicates(self) -> None:
        text = "w0 w1 w2 w3 disease w5 w6 w7 w8 disease w10"
        model = RecordingGLiNER()

        normalized, raw_predictions = predict_document(
            MedicalDocument("note-1", text),
            model,
            chunk_size=5,
            chunk_overlap=2,
        )
        entities = postprocess_predictions(raw_predictions, normalized)

        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(raw_predictions), 3)  # overlap predicts word 4 twice
        self.assertEqual(len(entities), 2)
        self.assertEqual(
            [entity.position for entity in entities],
            [
                [text.index("disease"), text.index("disease") + len("disease")],
                [text.rindex("disease"), text.rindex("disease") + len("disease")],
            ],
        )

    def test_rejects_invalid_chunk_configuration(self) -> None:
        with self.assertRaises(ValueError):
            iter_text_chunks("some text", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
