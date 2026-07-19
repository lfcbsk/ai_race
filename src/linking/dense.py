from __future__ import annotations

from typing import Protocol

from .schemas import OntologyEntry, RetrievalHit


class DenseRetriever(Protocol):
    def search(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]: ...


class SentenceTransformerDenseRetriever:
    """Optional self-hosted dense index backed by sentence-transformers."""

    def __init__(
        self,
        entries: list[OntologyEntry],
        model_path: str,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.entries = entries
        self.model = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=local_files_only,
        )
        self.embeddings = self.model.encode(
            [entry.search_text for entry in entries],
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

    def search(self, query: str, *, top_k: int = 20) -> list[RetrievalHit]:
        from sentence_transformers import util

        query_embedding = self.model.encode(
            query, normalize_embeddings=True, convert_to_tensor=True
        )
        hits = util.semantic_search(
            query_embedding, self.embeddings, top_k=min(top_k, len(self.entries))
        )[0]
        return [
            RetrievalHit(
                code=self.entries[int(hit["corpus_id"])].code,
                score=float(hit["score"]),
                rank=rank,
                source="dense",
            )
            for rank, hit in enumerate(hits, start=1)
        ]

