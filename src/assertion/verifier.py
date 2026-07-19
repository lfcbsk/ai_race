from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.preprocessing import EntityAnnotation

from .schemas import AssertionLabel, AssertionResult


class AssertionVerifier(Protocol):
    """Interface for a self-hosted verifier; no network API is required."""

    def verify(
        self,
        text: str,
        entity: EntityAnnotation,
        current: AssertionResult,
    ) -> tuple[list[AssertionLabel], float]: ...


class TransformersAssertionVerifier:
    """Optional local Transformers verifier for ambiguous assertion cases.

    ``model_path`` should point to a downloaded 3B-4B instruct checkpoint. By
    default only local files are accepted, keeping competition inference offline.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: int | str = -1,
        local_files_only: bool = True,
        max_new_tokens: int = 80,
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

    def verify(
        self,
        text: str,
        entity: EntityAnnotation,
        current: AssertionResult,
    ) -> tuple[list[AssertionLabel], float]:
        prompt = (
            "Bạn là bộ phân loại assertion y khoa. Trả JSON duy nhất.\n"
            f"Văn bản: {text}\n"
            f"Thực thể: {entity.text}\n"
            'Schema: {"isNegated": bool, "isHistorical": bool, '
            '"isFamily": bool, "confidence": number}'
        )
        output = self._generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]
        match = re.search(r"\{.*?\}", output, flags=re.DOTALL)
        if not match:
            return list(current.assertions), current.confidence
        payload: dict[str, Any] = json.loads(match.group(0))
        labels: list[AssertionLabel] = []
        for label in ("isNegated", "isHistorical", "isFamily"):
            if payload.get(label) is True:
                labels.append(label)  # type: ignore[arg-type]
        confidence = float(payload.get("confidence", current.confidence))
        return labels, max(0.0, min(1.0, confidence))

