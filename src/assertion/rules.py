from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .schemas import AssertionCue, AssertionLabel


@dataclass(frozen=True)
class CueRule:
    label: AssertionLabel
    pattern: str
    confidence: float
    direction: Literal["forward", "backward"] = "forward"


# Các cue được giữ tường minh để dễ audit và mở rộng bằng dữ liệu thật.
CUE_RULES: tuple[CueRule, ...] = (
    CueRule("isNegated", r"\bkhông\s+(?:có|ghi nhận|thấy|phát hiện|bị|mắc)?\s*", 0.97),
    CueRule("isNegated", r"\bchưa\s+(?:ghi nhận|thấy|phát hiện|từng|bị|mắc)?\s*", 0.94),
    CueRule("isNegated", r"\b(?:chẳng|chả)\s+(?:có|thấy|bị)?\s*", 0.92),
    CueRule("isNegated", r"\bphủ nhận\s+", 0.98),
    CueRule("isNegated", r"\bâm tính(?:\s+với)?\s+", 0.96),
    CueRule("isNegated", r"\bkhông còn\s+", 0.92),
    CueRule("isNegated", r"\bloại trừ\s+", 0.72),
    CueRule("isHistorical", r"\btiền sử(?:\s+(?:bị|mắc|có))?\s+", 0.98),
    CueRule("isHistorical", r"\btrước đây(?:\s+đã)?(?:\s+từng)?\s+", 0.96),
    CueRule("isHistorical", r"\bđã từng(?:\s+bị|\s+mắc|\s+có)?\s+", 0.96),
    CueRule("isHistorical", r"\btừng(?:\s+bị|\s+mắc|\s+có)\s+", 0.91),
    CueRule("isHistorical", r"\bcách đây\s+\d+(?:[,.]\d+)?\s*(?:ngày|tuần|tháng|năm)\b", 0.88, "backward"),
    CueRule("isHistorical", r"\bhồi\s+(?:nhỏ|bé|trước)\s+", 0.86),
    CueRule("isFamily", r"\btiền sử gia đình(?:\s+có)?\s+", 0.99),
    CueRule("isFamily", r"\bgia đình(?:\s+bệnh nhân)?(?:\s+có|\s+ghi nhận)?\s+", 0.96),
    CueRule("isFamily", r"\b(?:mẹ|má|bố|ba|cha|anh|chị|em|ông|bà|con)\s+(?:bị|mắc|có|từng bị|từng mắc)\s+", 0.94),
)


_COMPILED_RULES = tuple(
    (rule, re.compile(rule.pattern, flags=re.IGNORECASE | re.UNICODE))
    for rule in CUE_RULES
)


def find_assertion_cues(text: str) -> list[AssertionCue]:
    cues: list[AssertionCue] = []
    for rule, pattern in _COMPILED_RULES:
        for match in pattern.finditer(text):
            cues.append(
                AssertionCue(
                    label=rule.label,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=rule.confidence,
                    direction=rule.direction,
                )
            )
    cues.sort(key=lambda cue: (cue.start, -(cue.end - cue.start)))
    return cues
