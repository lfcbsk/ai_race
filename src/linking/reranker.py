from __future__ import annotations

from difflib import SequenceMatcher
from typing import Protocol

from .schemas import LinkingCandidate
from .text import normalize_mention, tokenize


class CandidateReranker(Protocol):
    def rerank(
        self, query: str, candidates: list[LinkingCandidate], *, top_k: int = 5
    ) -> list[LinkingCandidate]: ...


class LexicalReranker:
    """Dependency-free fallback used when a cross encoder is unavailable."""

    def rerank(
        self, query: str, candidates: list[LinkingCandidate], *, top_k: int = 5
    ) -> list[LinkingCandidate]:
        mention = query.splitlines()[0].split(":", 1)[-1].strip()
        normalized_query = normalize_mention(mention)
        query_tokens = set(tokenize(query))
        for candidate in candidates:
            alternatives = (candidate.entry.name, *candidate.entry.aliases)
            sequence_score = max(
                SequenceMatcher(None, normalized_query, normalize_mention(value)).ratio()
                for value in alternatives
            )
            candidate_tokens = set(tokenize(candidate.entry.search_text))
            union = query_tokens | candidate_tokens
            overlap = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            candidate.reranker_score = 0.75 * sequence_score + 0.25 * overlap
            candidate.final_score = candidate.reranker_score
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)[:top_k]


class CrossEncoderReranker:
    """Optional local BGE cross-encoder reranker."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_path,
            device=device,
            local_files_only=local_files_only,
        )

    def rerank(
        self, query: str, candidates: list[LinkingCandidate], *, top_k: int = 5
    ) -> list[LinkingCandidate]:
        scores = self.model.predict(
            [(query, candidate.entry.search_text) for candidate in candidates]
        )
        for candidate, score in zip(candidates, scores):
            candidate.reranker_score = float(score)
            candidate.final_score = float(score)
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)[:top_k]
