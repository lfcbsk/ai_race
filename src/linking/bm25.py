from __future__ import annotations

import math
from collections import Counter, defaultdict

from .schemas import OntologyEntry, RetrievalHit
from .text import tokenize


class BM25Index:
    def __init__(
        self,
        entries: list[OntologyEntry],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.entries = entries
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(entry.search_text) for entry in entries]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequency: dict[str, int] = defaultdict(int)
        for document in self.documents:
            for term in set(document):
                self.document_frequency[term] += 1
        self.average_length = (
            sum(map(len, self.documents)) / len(self.documents) if self.documents else 0.0
        )

    def search(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        query_terms = tokenize(query)
        if not query_terms or not self.entries:
            return []
        scores: list[tuple[int, float]] = []
        total_documents = len(self.entries)
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = len(self.documents[index])
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_frequency = math.log(
                    1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * document_length / max(self.average_length, 1.0)
                )
                score += inverse_frequency * frequency * (self.k1 + 1.0) / normalization
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalHit(
                code=self.entries[index].code,
                score=score,
                rank=rank,
                source="bm25",
            )
            for rank, (index, score) in enumerate(scores[:top_k], start=1)
        ]

