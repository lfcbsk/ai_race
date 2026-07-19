from __future__ import annotations

import math
import re
from dataclasses import replace

from src.preprocessing import EntityAnnotation, MedicalDocument

from .candidate_generator import CandidateGenerator
from .disambiguation import CandidateDisambiguator
from .ontology import ensure_single_ontology
from .reranker import CandidateReranker, LexicalReranker
from .schemas import LinkingCandidate, LinkingResult, OntologyEntry


def _sentence_context(text: str, start: int, end: int) -> str:
    left = max((match.end() for match in re.finditer(r"[.!?;\n]", text[:start])), default=0)
    right_match = re.search(r"[.!?;\n]", text[end:])
    right = end + right_match.start() if right_match else len(text)
    return text[left:right].strip()


def _confidence(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


class HybridEntityLinker:
    def __init__(
        self,
        entries: list[OntologyEntry],
        *,
        candidate_generator: CandidateGenerator | None = None,
        reranker: CandidateReranker | None = None,
        disambiguator: CandidateDisambiguator | None = None,
        ambiguity_margin: float = 0.04,
        min_confidence: float = 0.35,
    ) -> None:
        self.ontology = ensure_single_ontology(entries)
        self.generator = candidate_generator or CandidateGenerator(entries)
        self.reranker = reranker or LexicalReranker()
        self.disambiguator = disambiguator
        self.ambiguity_margin = ambiguity_margin
        self.min_confidence = min_confidence

    def link(
        self,
        document: MedicalDocument,
        entity: EntityAnnotation,
        entity_index: int,
    ) -> LinkingResult:
        exact = self.generator.exact_candidates(entity.text)
        if len(exact) == 1:
            return LinkingResult(
                document_id=document.document_id,
                entity_index=entity_index,
                mention=entity.text,
                ontology=self.ontology,
                candidates=exact,
                selected_code=exact[0].entry.code,
                confidence=1.0,
                method="exact_alias",
            )

        context = _sentence_context(document.raw_text, entity.start, entity.end)
        query = f"{self.ontology} entity: {entity.text}\nContext: {context}"
        candidates = exact or self.generator.generate(query)
        candidates = self.reranker.rerank(query, candidates, top_k=5)
        if not candidates:
            return LinkingResult(
                document_id=document.document_id,
                entity_index=entity_index,
                mention=entity.text,
                ontology=self.ontology,
            )

        top_confidence = _confidence(candidates[0].final_score)
        second_confidence = (
            _confidence(candidates[1].final_score) if len(candidates) > 1 else 0.0
        )
        ambiguous = len(candidates) > 1 and abs(top_confidence - second_confidence) < self.ambiguity_margin
        selected_code = candidates[0].entry.code if top_confidence >= self.min_confidence else None
        method = "hybrid_reranker" if selected_code else "below_threshold"

        if ambiguous and self.disambiguator is not None:
            judged_code = self.disambiguator.choose(query, candidates)
            if judged_code in {candidate.entry.code for candidate in candidates}:
                selected_code = judged_code
                method = "local_disambiguator"
                ambiguous = False

        return LinkingResult(
            document_id=document.document_id,
            entity_index=entity_index,
            mention=entity.text,
            ontology=self.ontology,
            candidates=candidates,
            selected_code=selected_code,
            confidence=top_confidence,
            method=method,
            needs_review=ambiguous or selected_code is None,
        )


class MedicalEntityLinker:
    def __init__(
        self,
        *,
        icd_linker: HybridEntityLinker | None = None,
        drug_linker: HybridEntityLinker | None = None,
    ) -> None:
        self.icd_linker = icd_linker
        self.drug_linker = drug_linker

    def link_entities(
        self,
        document: MedicalDocument,
        entities: list[EntityAnnotation],
    ) -> tuple[list[EntityAnnotation], list[LinkingResult]]:
        output: list[EntityAnnotation] = []
        results: list[LinkingResult] = []
        for index, entity in enumerate(entities):
            linker = None
            if entity.entity_type == "CHẨN_ĐOÁN":
                linker = self.icd_linker
            elif entity.entity_type == "THUỐC":
                linker = self.drug_linker
            if linker is None:
                output.append(entity)
                continue
            result = linker.link(document, entity, index)
            results.append(result)
            candidate_codes = [candidate.entry.code for candidate in result.candidates]
            output.append(replace(entity, candidates=candidate_codes))
        return output, results

