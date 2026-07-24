from __future__ import annotations

ENTITY_TYPES: tuple[str, ...] = (
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
)

ENTITY_TYPE_SET = frozenset(ENTITY_TYPES)

ASSERTION_ENTITY_TYPES = frozenset(
    {
        "TRIỆU_CHỨNG",
        "CHẨN_ĐOÁN",
        "THUỐC",
    }
)

CANDIDATE_ENTITY_TYPES = frozenset(
    {
        "CHẨN_ĐOÁN",
        "THUỐC",
    }
)

NER_NORMALIZE_KWARGS: dict[str, object] = {
    "collapse_spaces": True,
    "normalize_punctuation": True,
    "preserve_line_breaks": True,
    "max_consecutive_newlines": 2,
    "strip_edges": True,
}