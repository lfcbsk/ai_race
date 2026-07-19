from __future__ import annotations

import json
import re
from typing import Protocol

from .schemas import LinkingCandidate


class CandidateDisambiguator(Protocol):
    """Optional self-hosted final judge for close top candidates."""

    def choose(
        self, query: str, candidates: list[LinkingCandidate]
    ) -> str | None: ...


class NoOpDisambiguator:
    def choose(self, query: str, candidates: list[LinkingCandidate]) -> str | None:
        return None


class TransformersCandidateDisambiguator:
    """Optional local 3B-4B judge, invoked only for ambiguous top candidates."""

    def __init__(
        self,
        model_path: str,
        *,
        device: int | str = -1,
        local_files_only: bool = True,
        max_new_tokens: int = 48,
    ) -> None:
        from transformers import pipeline

        self._generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=device,
            model_kwargs={"local_files_only": local_files_only},
        )
        self.max_new_tokens = max_new_tokens

    def choose(self, query: str, candidates: list[LinkingCandidate]) -> str | None:
        options = "\n".join(
            f"- {candidate.entry.code}: {candidate.entry.name} | "
            f"{candidate.entry.description}"
            for candidate in candidates
        )
        prompt = (
            "Chọn đúng một mã ontology phù hợp nhất. Nếu không đủ thông tin, "
            "trả null. Chỉ trả JSON.\n"
            f"Query:\n{query}\nCandidates:\n{options}\n"
            'Schema: {"code": "string|null"}'
        )
        output = self._generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]
        match = re.search(r"\{.*?\}", output, flags=re.DOTALL)
        if not match:
            return None
        code = json.loads(match.group(0)).get("code")
        return str(code) if code is not None else None
