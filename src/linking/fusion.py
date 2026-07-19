from __future__ import annotations

from .schemas import LinkingCandidate, OntologyEntry, RetrievalHit


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalHit]],
    entries_by_code: dict[str, OntologyEntry],
    *,
    k: int = 60,
    top_k: int = 30,
) -> list[LinkingCandidate]:
    candidates: dict[str, LinkingCandidate] = {}
    for ranking in rankings:
        for hit in ranking:
            entry = entries_by_code.get(hit.code)
            if entry is None:
                continue
            candidate = candidates.setdefault(hit.code, LinkingCandidate(entry=entry))
            candidate.retrieval_scores[hit.source] = max(
                candidate.retrieval_scores.get(hit.source, float("-inf")), hit.score
            )
            candidate.fused_score += 1.0 / (k + hit.rank)
    ordered = sorted(candidates.values(), key=lambda item: item.fused_score, reverse=True)
    for candidate in ordered:
        candidate.final_score = candidate.fused_score
    return ordered[:top_k]

