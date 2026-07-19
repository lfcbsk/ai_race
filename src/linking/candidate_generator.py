from __future__ import annotations

from .bm25 import BM25Index
from .dense import DenseRetriever
from .exact_match import ExactAliasIndex
from .fusion import reciprocal_rank_fusion
from .schemas import LinkingCandidate, OntologyEntry


class CandidateGenerator:
    def __init__(
        self,
        entries: list[OntologyEntry],
        *,
        dense: DenseRetriever | None = None,
    ) -> None:
        self.entries = entries
        self.entries_by_code = {entry.code: entry for entry in entries}
        self.exact = ExactAliasIndex(entries)
        self.bm25 = BM25Index(entries)
        self.dense = dense

    def exact_candidates(self, mention: str) -> list[LinkingCandidate]:
        return [
            LinkingCandidate(entry=entry, final_score=1.0)
            for entry in self.exact.search(mention)
        ]

    def generate(
        self,
        query: str,
        *,
        retrieval_top_k: int = 20,
        fused_top_k: int = 30,
    ) -> list[LinkingCandidate]:
        rankings = [self.bm25.search(query, top_k=retrieval_top_k)]
        if self.dense is not None:
            rankings.append(self.dense.search(query, top_k=retrieval_top_k))
        return reciprocal_rank_fusion(
            rankings,
            self.entries_by_code,
            top_k=fused_top_k,
        )

