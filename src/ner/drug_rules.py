from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.linking import OntologyEntry
from src.preprocessing import NormalizedDocument

from .inference import RawEntityPrediction


COMMON_DRUG_ALIASES = (
    "asa81",
    "augmentin",
    "cellcept",
    "combivent",
    "coumadin",
    "dilaudid",
    "gleevec",
    "laxis",
    "lasix",
    "prograf",
    "ranexa",
    "vicodin",
)


_DOSE_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml|iu|đv|đơn\s*vị)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_ADMINISTRATION_PATTERN = re.compile(
    r"\b(?:po|iv|im|sc|sl|bid|tid|qid|q\d+h|prn|daily|once|"
    r"tiêm|truyền|khí\s*dung|đặt\s*dưới\s*lưỡi)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_SCHEDULE_PATTERN = re.compile(
    r"\b(?:scdc|scđc|sáng|trưa|chiều|tối)\b|"
    r"(?<!\w)(?:s|tr|c|đ|t)\s*[-./]\s*(?:s|tr|c|đ|t)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_MEDICATION_ACTION_PATTERN = re.compile(
    r"\b(?:thuốc|điều\s*trị|được\s+(?:cho|kê)|bắt\s*đầu\s+dùng|"
    r"đang\s+dùng|đã\s+dùng|sử\s*dụng|nhận)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_MEDICATION_HEADER_PATTERN = re.compile(
    r"\b(?:thuốc\s+trước\s+khi\s+nhập\s+viện|danh\s+sách\s+thuốc|"
    r"thuốc\s+đang\s+dùng|toa\s+thuốc|đơn\s+thuốc)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_SECTION_END_PATTERN = re.compile(
    r"^\s*(?:\d+\.\s+|bệnh\s+sử|lý\s+do\s+(?:vào|nhập)\s+viện|"
    r"khám\s+(?:hiện\s+tại|tại\s+bệnh\s+viện)|đánh\s+giá\s+tại\s+bệnh\s+viện|"
    r"các\s+yếu\s+tố|triệu\s+chứng|diễn\s+biến|tình\s+trạng|sự\s+kiện)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_DOSAGE_IN_TERM_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml|iu|đv)\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _clean_drug_terms(entries: Iterable[OntologyEntry]) -> tuple[str, ...]:
    terms: set[str] = set()
    for entry in entries:
        for value in (entry.name, *entry.aliases):
            term = " ".join(value.strip().split())
            if not term or len(term) < 2 or len(term.split()) > 5:
                continue
            if _DOSAGE_IN_TERM_PATTERN.search(term) or any(
                character in term for character in "(),;:"
            ):
                continue
            terms.add(term)
    return tuple(sorted(terms, key=lambda value: (-len(value), value.casefold())))


def _medication_section_lines(text: str) -> set[int]:
    medication_lines: set[int] = set()
    in_medication_section = False
    for line_number, line in enumerate(text.splitlines()):
        if _MEDICATION_HEADER_PATTERN.search(line):
            in_medication_section = True
        elif in_medication_section and _SECTION_END_PATTERN.search(line):
            in_medication_section = False
        if in_medication_section:
            medication_lines.add(line_number)
    return medication_lines


@dataclass(frozen=True)
class DrugRuleDetector:
    terms: tuple[str, ...]
    confidence: float = 0.99

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[OntologyEntry],
        *,
        extra_terms: Iterable[str] = COMMON_DRUG_ALIASES,
    ) -> DrugRuleDetector:
        terms = set(_clean_drug_terms(entries))
        terms.update(" ".join(term.strip().split()) for term in extra_terms if term.strip())
        return cls(terms=tuple(sorted(terms, key=lambda value: (-len(value), value))))

    def find_predictions(
        self,
        normalized: NormalizedDocument,
        document_id: str,
    ) -> list[RawEntityPrediction]:
        if not self.terms:
            return []

        text = normalized.normalized_text
        medication_lines = _medication_section_lines(text)
        term_pattern = re.compile(
            r"(?<!\w)(?:"
            + "|".join(re.escape(term) for term in self.terms)
            + r")(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        )
        matches: list[RawEntityPrediction] = []
        current_offset = 0
        for line_number, line in enumerate(text.splitlines(keepends=True)):
            has_medication_context = line_number in medication_lines or any(
                pattern.search(line)
                for pattern in (
                    _DOSE_PATTERN,
                    _ADMINISTRATION_PATTERN,
                    _SCHEDULE_PATTERN,
                    _MEDICATION_ACTION_PATTERN,
                )
            )
            if has_medication_context:
                for match in term_pattern.finditer(line):
                    start = current_offset + match.start()
                    end = current_offset + match.end()
                    matches.append(
                        RawEntityPrediction(
                            document_id=document_id,
                            text=text[start:end],
                            entity_type="THUỐC",
                            normalized_start=start,
                            normalized_end=end,
                            confidence=self.confidence,
                        )
                    )
            current_offset += len(line)
        return matches
