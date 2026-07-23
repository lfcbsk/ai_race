from __future__ import annotations
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any

try:
    from scripts.common import (
        CATALOG_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        clean_text,
        ensure_directories,
        normalize_for_matching,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    # Support direct execution: python scripts/prepare_sources.py
    from common import (
        CATALOG_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        clean_text,
        ensure_directories,
        normalize_for_matching,
        write_json,
        write_jsonl,
    )


# ============================================================
# INPUT PATHS
# ============================================================

VIMEDNER_PATH = RAW_DIR / "ViMedNER.txt"
RXNORM_DIR = RAW_DIR / "rxnorm_data"
ICD10_PATH = RAW_DIR / "DM ICD10-19_8_BYT.xlsx"


# ============================================================
# VIMEDNER
# ============================================================

DIRECT_LABEL_MAPPING = {
    "DISEASE": "CHẨN_ĐOÁN",
    "DIAGNOSIS": "CHẨN_ĐOÁN",
    "SYMPTOM": "TRIỆU_CHỨNG",
}

PROCEDURE_TERMS = {
    "phẫu thuật",
    "mổ",
    "nội soi",
    "sinh thiết",
    "đặt stent",
    "xạ trị",
    "hóa trị",
    "lọc máu",
    "thở máy",
    "đặt nội khí quản",
    "can thiệp",
    "chụp tử cung vòi trứng",
    "cholangiogram",
    "lấy mẫu bằng bàn chải",
    "nối mật tụy",
}


SURFACE_CORRECTIONS = {
    "o lắng": "lo lắng",
}

MANUAL_REJECT_SURFACES = {
    "bàn chân",
    "nâng hai cánh tay qua đầu",
    "õm vào trong",
    "xả màu vàng hay xanh",
    "rung rinh",
    "đau đầy",
    "không có tầm nhìn",
    "gây mất tầm nhìn",
    "có một khối ở bụng",
}

SYMPTOM_BAD_PREFIXES = (
    "có ",
    "có một ",
    "gây ",
    "được ",
    "nâng ",
    "thực hiện ",
)

EMBEDDED_ASSERTION_PATTERN = re.compile(
    r"\b(không|chưa|phủ nhận)\b",
    re.IGNORECASE,
)


def prepare_catalog_surface(
    raw_text: str,
    entity_type: str,
) -> tuple[str | None, str | None]:
    text = clean_text(raw_text)

    if not text:
        return None, "empty"

    text = text.rstrip(" \t\r\n,;:")
    normalized = normalize_for_matching(text)

    corrected = SURFACE_CORRECTIONS.get(
        normalized
    )
    if corrected:
        text = corrected
        normalized = normalize_for_matching(text)

    if len(text) < 3:
        return None, "too_short"

    if normalized in MANUAL_REJECT_SURFACES:
        return None, "manual_reject"

    if "\n" in text or "\r" in text:
        return None, "multiline_fragment"

    if entity_type == "TRIỆU_CHỨNG":
        if EMBEDDED_ASSERTION_PATTERN.search(text):
            return None, "embedded_assertion_cue"

        if normalized.startswith(
            SYMPTOM_BAD_PREFIXES
        ):
            return None, "contextual_prefix"

        if normalized.endswith(
            ("vào trong", "ra ngoài")
        ) and len(normalized.split()) <= 4:
            return None, "incomplete_directional_fragment"

    return text, None


def load_vimedner() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with VIMEDNER_PATH.open(
        encoding="utf-8"
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[WARN] ViMedNER JSON lỗi "
                    f"tại dòng {line_number}"
                )

    return records


def process_vimedner() -> dict[str, Any]:
    records = load_vimedner()

    target_mentions: list[dict[str, Any]] = []
    treatment_mentions: list[str] = []

    label_counts: Counter[str] = Counter()
    invalid_spans: list[dict[str, Any]] = []

    for document_index, record in enumerate(records):
        text = record.get("text", "")
        labels = record.get("label", [])

        for entity_index, label in enumerate(labels):
            if len(label) != 3:
                continue

            start, end, source_type = label
            label_counts[source_type] += 1

            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
            ):
                invalid_spans.append(
                    {
                        "document_index": (
                            document_index
                        ),
                        "label": label,
                    }
                )
                continue

            mention = clean_text(text[start:end])

            if not mention:
                continue

            if source_type == "TREATMENT":
                treatment_mentions.append(mention)
                continue

            target_type = (
                DIRECT_LABEL_MAPPING.get(
                    source_type
                )
            )

            if not target_type:
                continue

            target_mentions.append(
                {
                    "seed_id": (
                        f"vimedner_{document_index:06d}_"
                        f"{entity_index:03d}"
                    ),
                    "document_id": (
                        f"vimedner_{document_index:06d}"
                    ),
                    "document_text": text,
                    "text": mention,
                    "type": target_type,
                    "position": [start, end],
                    "source_type": source_type,
                    "source": "ViMedNER",
                }
            )

    write_jsonl(
        PROCESSED_DIR / "vimedner_entities.jsonl",
        target_mentions,
    )

    write_json(
        PROCESSED_DIR / "vimedner_audit.json",
        {
            "document_count": len(records),
            "target_entity_count": len(
                target_mentions
            ),
            "treatment_mention_count": len(
                treatment_mentions
            ),
            "label_counts": dict(label_counts),
            "invalid_span_count": len(
                invalid_spans
            ),
            "invalid_spans": invalid_spans[:100],
        },
    )

    return {
        "target_mentions": target_mentions,
        "treatment_mentions": sorted(
            set(treatment_mentions)
        ),
    }


# ============================================================
# RXNORM
# ============================================================

RXNORM_TTYS = (
    "IN",
    "PIN",
    "MIN",
    "SCDC",
    "SCD",
    "BN",
    "SBD",
    "DF",
)

STRENGTH_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>"
    r"mcg/ml|mg/ml|g/ml|mg|mcg|µg|g|ml|%"
    r")",
    re.I,
)

BRAND_PATTERN = re.compile(
    r"\[(?P<brand>[^\[\]]+)\]\s*$"
)

DOSE_FORM_PATTERNS = {
    "oral_tablet": re.compile(
        r"\boral tablet\b",
        re.I,
    ),
    "oral_capsule": re.compile(
        r"\boral capsule\b",
        re.I,
    ),
    "oral_solution": re.compile(
        r"\boral solution\b",
        re.I,
    ),
    "injectable_solution": re.compile(
        r"\binjectable solution\b",
        re.I,
    ),
    "topical_cream": re.compile(
        r"\btopical cream\b",
        re.I,
    ),
    "topical_ointment": re.compile(
        r"\btopical ointment\b",
        re.I,
    ),
}

DOSE_FORM_SURFACES = {
    "oral_tablet": [
        "viên",
        "viên nén",
        "tablet",
        "tab",
    ],
    "oral_capsule": [
        "viên nang",
        "capsule",
        "cap",
    ],
    "oral_solution": [
        "dung dịch uống",
        "oral solution",
    ],
    "topical_cream": [
        "kem bôi",
        "cream",
    ],
    "topical_ointment": [
        "thuốc mỡ",
        "ointment",
    ],
}

ROUTE_SURFACES = {
    "oral_tablet": [
        "po",
        "uống",
    ],
    "oral_capsule": [
        "po",
        "uống",
    ],
    "oral_solution": [
        "po",
        "uống",
    ],
    "topical_cream": [
        "bôi ngoài da",
    ],
    "topical_ointment": [
        "bôi ngoài da",
    ],
}

FREQUENCY_SURFACES = [
    "daily",
    "qd",
    "ngày 1 lần",
    "mỗi ngày",
    "mỗi sáng",
]


def extract_rxnorm_item(
    raw: dict[str, Any],
    fallback_tty: str,
) -> dict[str, Any] | None:
    name = clean_text(
        raw.get("name")
        or raw.get("str")
        or raw.get("concept_name")
    )

    rxcui = clean_text(
        raw.get("rxcui")
        or raw.get("RXCUI")
    )

    tty = clean_text(
        raw.get("tty")
        or raw.get("TTY")
        or fallback_tty
    )

    if not name or not rxcui or not tty:
        return None

    return {
        "concept_id": str(rxcui),
        "name": name,
        "tty": tty.upper(),
        "normalized_name": (
            normalize_for_matching(name)
        ),
    }


def load_rxnorm() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for tty in RXNORM_TTYS:
        path = (
            RXNORM_DIR
            / f"rxnorm_{tty}.json"
        )

        if not path.exists():
            print(
                f"[WARN] Thiếu RxNorm file: "
                f"{path.name}"
            )
            continue

        with path.open(encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, dict):
            payload = (
                payload.get("data")
                or payload.get("items")
                or payload.get("concepts")
                or []
            )

        for raw in payload:
            record = extract_rxnorm_item(
                raw,
                tty,
            )

            if record:
                records.append(record)

    unique = {
        (
            item["concept_id"],
            item["tty"],
            item["normalized_name"],
        ): item
        for item in records
    }

    return list(unique.values())


def parse_rxnorm_concept(
    concept: dict[str, Any],
) -> dict[str, Any]:
    name = concept["name"]
    brand_match = BRAND_PATTERN.search(name)
    brand = (
        brand_match.group("brand").strip()
        if brand_match
        else None
    )

    without_brand = BRAND_PATTERN.sub(
        "",
        name,
    ).strip()

    strength_matches = list(
        STRENGTH_PATTERN.finditer(
            without_brand
        )
    )

    complex_product = (
        len(strength_matches) > 1
        or " / " in without_brand
    )

    strength = None
    ingredient = without_brand

    if strength_matches:
        first = strength_matches[0]

        strength = {
            "value": (
                first.group("value")
                .replace(",", ".")
            ),
            "unit": (
                first.group("unit")
                .lower()
                .replace("µg", "mcg")
            ),
        }

        ingredient = without_brand[
            :first.start()
        ].strip(" ,;/:-")

    dose_form = None

    for form_name, pattern in (
        DOSE_FORM_PATTERNS.items()
    ):
        if pattern.search(without_brand):
            dose_form = form_name
            break

    return {
        **concept,
        "attributes": {
            "ingredient_text": ingredient,
            "strength": strength,
            "dose_form": dose_form,
            "brand": brand,
            "complex_product": complex_product,
        },
    }


def strength_variants(
    strength: dict[str, str] | None,
) -> list[str]:
    if not strength:
        return []

    value = strength["value"]
    unit = strength["unit"]

    return [
        f"{value} {unit}",
        f"{value}{unit}",
    ]


def build_drug_surfaces(
    concepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []

    def add_surface(
        concept: dict[str, Any],
        text: str,
        profile: str,
        *,
        reliability: str = "high",
        usable_for_linking: bool = True,
    ) -> None:
        text = clean_text(text)

        if not text:
            return

        surfaces.append(
            {
                "surface_id": (
                    f"RX_{concept['concept_id']}_"
                    f"{profile}_"
                    f"{len(surfaces):07d}"
                ),
                "text": text,
                "entity_type": "THUỐC",
                "candidates": [
                    concept["concept_id"]
                ],
                "concept_id": (
                    concept["concept_id"]
                ),
                "tty": concept["tty"],
                "profile": profile,
                "reliability": reliability,
                "usable_for_ner": True,
                "usable_for_linking": (
                    usable_for_linking
                ),
                "attributes": concept[
                    "attributes"
                ],
            }
        )

    for concept in concepts:
        tty = concept["tty"]
        attributes = concept["attributes"]

        ingredient = clean_text(
            attributes.get(
                "ingredient_text"
            )
        )
        strength = attributes.get("strength")
        dose_form = attributes.get("dose_form")
        brand = clean_text(
            attributes.get("brand")
        )
        complex_product = attributes.get(
            "complex_product",
            False,
        )

        if tty == "DF":
            continue

        if tty in {"IN", "PIN", "MIN"}:
            add_surface(
                concept,
                ingredient or concept["name"],
                "ingredient_only",
            )
            continue

        if tty == "BN":
            add_surface(
                concept,
                brand or concept["name"],
                "brand_only",
            )
            continue

        # Mọi clinical concept đều có canonical surface.
        add_surface(
            concept,
            concept["name"].lower(),
            "canonical",
        )

        # Không tự tạo simplified form cho product phức tạp.
        if complex_product:
            continue

        strength_forms = strength_variants(
            strength
        )

        if tty == "SCDC":
            for strength_text in strength_forms:
                if ingredient:
                    add_surface(
                        concept,
                        (
                            f"{ingredient} "
                            f"{strength_text}"
                        ),
                        "ingredient_strength",
                    )
            continue

        if tty not in {"SCD", "SBD"}:
            continue

        for strength_text in strength_forms:
            if ingredient:
                add_surface(
                    concept,
                    (
                        f"{ingredient} "
                        f"{strength_text}"
                    ),
                    "ingredient_strength",
                )

            if ingredient and dose_form:
                for form_text in (
                    DOSE_FORM_SURFACES.get(
                        dose_form,
                        [],
                    )
                ):
                    add_surface(
                        concept,
                        (
                            f"{ingredient} "
                            f"{strength_text} "
                            f"{form_text}"
                        ),
                        "ingredient_strength_form",
                    )

                for route in ROUTE_SURFACES.get(
                    dose_form,
                    [],
                ):
                    for frequency in (
                        FREQUENCY_SURFACES
                    ):
                        add_surface(
                            concept,
                            (
                                f"{ingredient} "
                                f"{strength_text} "
                                f"{route} "
                                f"{frequency}"
                            ),
                            "full_medication_order",
                        )

                if dose_form in {
                    "oral_tablet",
                    "oral_capsule",
                }:
                    add_surface(
                        concept,
                        (
                            f"{ingredient} "
                            f"{strength_text} "
                            f"1 viên/ngày"
                        ),
                        "vietnamese_order",
                    )

            if tty == "SBD" and brand:
                add_surface(
                    concept,
                    (
                        f"{brand} "
                        f"{strength_text}"
                    ),
                    "brand_strength",
                )

    unique = {
        (
            normalize_for_matching(
                record["text"]
            ),
            record["concept_id"],
        ): record
        for record in surfaces
    }

    return list(unique.values())


def build_rxnorm_families(
    concepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}

    role_by_tty = {
        "IN": "ingredients",
        "PIN": "ingredients",
        "MIN": "ingredients",
        "SCDC": "strength_components",
        "SCD": "clinical_drugs",
        "SBD": "clinical_drugs",
        "BN": "brands",
    }

    for concept in concepts:
        ingredient = clean_text(
            concept["attributes"].get(
                "ingredient_text"
            )
        )

        family_key = normalize_for_matching(
            ingredient or concept["name"]
        )

        family = families.setdefault(
            family_key,
            {
                "family_id": family_key,
                "ingredients": [],
                "strength_components": [],
                "clinical_drugs": [],
                "brands": [],
            },
        )

        role = role_by_tty.get(
            concept["tty"]
        )

        if role:
            family[role].append(
                {
                    "concept_id": (
                        concept["concept_id"]
                    ),
                    "name": concept["name"],
                    "tty": concept["tty"],
                }
            )

    return list(families.values())


# ============================================================
# ICD-10
# ============================================================

DISEASE_ALIASES = {
    "cao huyết áp": (
        "tăng huyết áp vô căn (nguyên phát)"
    ),
}


def load_icd10() -> list[dict[str, Any]]:
    import pandas as pd

    dataframe = pd.read_excel(
        ICD10_PATH,
        header=4,
    )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    required = {
        "Mã",
        "Tên bệnh",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"ICD-10 thiếu cột: {sorted(missing)}"
        )

    dataframe = dataframe[
        ["Mã", "Tên bệnh"]
    ].copy()

    dataframe = dataframe.dropna(
        subset=["Mã", "Tên bệnh"]
    )

    records: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        code = str(row["Mã"]).strip().upper()
        name = clean_text(row["Tên bệnh"])

        if not code or not name:
            continue

        records.append(
            {
                "concept_id": code,
                "name_vi": name,
                "normalized_name": (
                    normalize_for_matching(name)
                ),
            }
        )

    unique = {
        (
            record["concept_id"],
            record["normalized_name"],
        ): record
        for record in records
    }

    return list(unique.values())


def build_disease_surfaces(
    target_mentions: list[dict[str, Any]],
    icd_concepts: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    exact_index: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for concept in icd_concepts:
        exact_index[
            concept["normalized_name"]
        ].append(concept)

    surfaces: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    disease_texts = sorted(
        {
            mention["text"]
            for mention in target_mentions
            if mention["type"] == "CHẨN_ĐOÁN"
        }
    )

    for index, mention in enumerate(
        disease_texts
    ):
        normalized = normalize_for_matching(
            mention
        )

        canonical = DISEASE_ALIASES.get(
            normalized,
            normalized,
        )

        candidates = exact_index.get(
            normalize_for_matching(canonical),
            [],
        )

        if len(candidates) != 1:
            unresolved.append(
                {
                    "text": mention,
                    "reason": (
                        "no_unique_exact_icd_match"
                    ),
                }
            )
            continue

        concept = candidates[0]

        surfaces.append(
            {
                "surface_id": (
                    f"ICD_{concept['concept_id']}_"
                    f"{index:06d}"
                ),
                "text": mention,
                "entity_type": "CHẨN_ĐOÁN",
                "candidates": [
                    concept["concept_id"]
                ],
                "concept_id": (
                    concept["concept_id"]
                ),
                "profile": "vimedner_verified",
                "reliability": "high",
                "usable_for_ner": True,
                "usable_for_linking": True,
            }
        )

    return surfaces, unresolved


# ============================================================
# STATIC CATALOGS
# ============================================================

LAB_TESTS = [
    {
        "test_name": "WBC",
        "unit": "10^9/L",
        "normal_range": [4.0, 10.0],
        "decimals": 1,
    },
    {
        "test_name": "HGB",
        "unit": "g/L",
        "normal_range": [110, 165],
        "decimals": 0,
    },
    {
        "test_name": "PLT",
        "unit": "10^9/L",
        "normal_range": [150, 400],
        "decimals": 0,
    },
    {
        "test_name": "Glucose máu",
        "unit": "mmol/L",
        "normal_range": [3.9, 6.4],
        "decimals": 1,
    },
    {
        "test_name": "Creatinine máu",
        "unit": "µmol/L",
        "normal_range": [44, 106],
        "decimals": 0,
    },
]


ASSERTION_SCENARIOS = {
    "patient_current_positive": {
        "kind": "single_group",
        "weight": 20,
        "allowed_types": [
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
        ],
        "assertions": [],
        "min_entities": 1,
        "max_entities": 2,
        "instruction": (
            "Các entity là tình trạng hiện tại "
            "của chính bệnh nhân."
        ),
    },
    "patient_current_negated": {
        "kind": "single_group",
        "weight": 12,
        "allowed_types": [
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
        ],
        "assertions": ["isNegated"],
        "min_entities": 1,
        "max_entities": 2,
        "instruction": (
            "Phủ định rõ các entity bằng cue "
            "gắn trực tiếp với chúng."
        ),
    },
    "coordinated_negation": {
        "kind": "coordinated_negation",
        "weight": 15,
        "allowed_types": [
            "TRIỆU_CHỨNG"
        ],
        "assertions": ["isNegated"],
        "min_entities": 2,
        "max_entities": 5,
        "instruction": (
            "Dùng một cue phủ định chung có scope "
            "rõ cho toàn bộ danh sách entity."
        ),
    },
    "mixed_polarity": {
        "kind": "mixed_polarity",
        "weight": 12,
        "allowed_types": [
            "TRIỆU_CHỨNG"
        ],
        "min_entities": 2,
        "max_entities": 2,
        "instruction": (
            "Entity đầu tiên bị phủ định; entity "
            "thứ hai là triệu chứng hiện tại. "
            "Dùng liên từ tương phản rõ ràng."
        ),
    },
    "patient_historical": {
        "kind": "single_group",
        "weight": 15,
        "allowed_types": [
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
        ],
        "assertions": ["isHistorical"],
        "min_entities": 1,
        "max_entities": 2,
        "instruction": (
            "Mô tả các entity là tiền sử hoặc "
            "đã xảy ra trước đây."
        ),
    },
    "patient_historical_negated": {
        "kind": "single_group",
        "weight": 8,
        "allowed_types": [
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
        ],
        "assertions": [
            "isNegated",
            "isHistorical",
        ],
        "min_entities": 1,
        "max_entities": 2,
        "instruction": (
            "Phủ định rõ các tình trạng trong "
            "bối cảnh trước nhập viện hoặc quá khứ."
        ),
    },
    "family_current": {
        "kind": "single_group",
        "weight": 7,
        "allowed_types": [
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
        ],
        "assertions": ["isFamily"],
        "min_entities": 1,
        "max_entities": 2,
        "instruction": (
            "Các entity thuộc một người nhà, "
            "không phải bệnh nhân."
        ),
    },
    "family_historical": {
        "kind": "single_group",
        "weight": 9,
        "allowed_types": [
            "TRIỆU_CHỨNG",
            "CHẨN_ĐOÁN",
        ],
        "assertions": [
            "isFamily",
            "isHistorical",
        ],
        "min_entities": 1,
        "max_entities": 2,
        "instruction": (
            "Các entity thuộc tiền sử của "
            "một người nhà."
        ),
    },
    "home_medications": {
        "kind": "medication_group",
        "weight": 12,
        "allowed_types": ["THUỐC"],
        "assertions": ["isHistorical"],
        "min_entities": 1,
        "max_entities": 4,
        "instruction": (
            "Các thuốc được dùng tại nhà hoặc "
            "trước khi nhập viện."
        ),
    },
    "active_medications": {
        "kind": "medication_group",
        "weight": 8,
        "allowed_types": ["THUỐC"],
        "assertions": [],
        "min_entities": 1,
        "max_entities": 4,
        "instruction": (
            "Các thuốc đang được bệnh nhân "
            "sử dụng hiện tại."
        ),
    },
    "laboratory_results": {
        "kind": "laboratory_group",
        "weight": 10,
        "allowed_types": [
            "TÊN_XÉT_NGHIỆM",
            "KẾT_QUẢ_XÉT_NGHIỆM",
        ],
        "min_entities": 1,
        "max_entities": 3,
        "instruction": (
            "Viết kết quả xét nghiệm theo dạng "
            "tên xét nghiệm và giá trị kèm đơn vị."
        ),
    },
}


DOCUMENT_CONFIG = {
    "sections": {
        "medical_history": {
            "titles": [
                "Tiền sử bệnh nội khoa",
                "Tiền sử bệnh",
            ],
            "temporal_scope": "historical",
            "subject_scope": "patient_or_family",
            "render_style": "paragraph",
            "allowed_scenarios": [
                "patient_historical",
                "family_historical",
            ],
        },
        "history_present_illness": {
            "titles": [
                "Tiền sử bệnh hiện tại",
                "Bệnh sử",
            ],
            "temporal_scope": "mixed",
            "subject_scope": "patient",
            "render_style": "bullet_list",
            "allowed_scenarios": [
                "patient_current_positive",
                "patient_current_negated",
                "coordinated_negation",
                "mixed_polarity",
                "patient_historical",
            ],
        },
        "pre_admission_events": {
            "titles": [
                "Sự kiện trước khi nhập viện",
                "Diễn biến trước nhập viện",
            ],
            "temporal_scope": "historical",
            "subject_scope": "patient",
            "render_style": "bullet_list",
            "allowed_scenarios": [
                "patient_historical",
                "patient_historical_negated",
            ],
        },
        "hospital_assessment": {
            "titles": [
                "Đánh giá tại bệnh viện",
                "Đánh giá hiện tại",
            ],
            "temporal_scope": "current",
            "subject_scope": "patient",
            "render_style": "structured_or_paragraph",
            "allowed_scenarios": [
                "patient_current_positive",
                "patient_current_negated",
                "coordinated_negation",
                "mixed_polarity",
            ],
        },
        "home_medications": {
            "titles": [
                "Thuốc trước nhập viện",
                "Thuốc dùng tại nhà",
            ],
            "temporal_scope": "historical",
            "subject_scope": "patient",
            "render_style": "bullet_list",
            "allowed_scenarios": [
                "home_medications"
            ],
        },
        "active_medications": {
            "titles": [
                "Thuốc đang sử dụng",
                "Điều trị hiện tại",
            ],
            "temporal_scope": "current",
            "subject_scope": "patient",
            "render_style": "bullet_list",
            "allowed_scenarios": [
                "active_medications"
            ],
        },
        "laboratory": {
            "titles": [
                "Kết quả xét nghiệm",
                "Cận lâm sàng",
            ],
            "temporal_scope": "current",
            "subject_scope": "patient",
            "render_style": "structured_list",
            "allowed_scenarios": [
                "laboratory_results"
            ],
        },
    },
    "profiles": {
        "single_clinical_sentence": {
            "weight": 10,
            "required_sections": [
                "hospital_assessment"
            ],
            "optional_sections": [],
        },
        "medication_reconciliation": {
            "weight": 15,
            "required_sections": [
                "home_medications"
            ],
            "optional_sections": [
                "active_medications"
            ],
            "min_optional": 0,
            "max_optional": 1,
        },
        "outpatient_note": {
            "weight": 15,
            "required_sections": [
                "history_present_illness",
                "hospital_assessment",
            ],
            "optional_sections": [
                "active_medications",
                "laboratory",
            ],
            "min_optional": 0,
            "max_optional": 1,
        },
        "discharge_summary": {
            "weight": 20,
            "required_sections": [
                "medical_history",
                "hospital_assessment",
            ],
            "optional_sections": [
                "history_present_illness",
                "home_medications",
                "laboratory",
            ],
            "min_optional": 1,
            "max_optional": 2,
        },
        "longitudinal_hospital_note": {
            "weight": 30,
            "required_sections": [
                "medical_history",
                "history_present_illness",
                "hospital_assessment",
            ],
            "optional_sections": [
                "pre_admission_events",
                "home_medications",
                "active_medications",
                "laboratory",
            ],
            "min_optional": 1,
            "max_optional": 3,
        },
        "mixed_noisy_document": {
            "weight": 10,
            "required_sections": [
                "history_present_illness",
                "hospital_assessment",
            ],
            "optional_sections": [
                "medical_history",
                "pre_admission_events",
                "laboratory",
            ],
            "min_optional": 1,
            "max_optional": 2,
        },
    },
}


# ============================================================
# MAIN
# ============================================================

def run_prepare_data(*, seed: int = 42) -> None:
    ensure_directories()
    random.seed(seed)

    vimedner = process_vimedner()

    raw_rxnorm = load_rxnorm()
    rxnorm_concepts = [
        parse_rxnorm_concept(concept)
        for concept in raw_rxnorm
    ]

    rxnorm_families = build_rxnorm_families(
        rxnorm_concepts
    )

    drug_surfaces = build_drug_surfaces(
        rxnorm_concepts
    )

    # Index surface thuốc để xác minh TREATMENT từ ViMedNER.
    drug_surface_index: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for surface in drug_surfaces:
        drug_surface_index[
            normalize_for_matching(
                surface["text"]
            )
        ].append(surface)

    procedures: list[dict[str, Any]] = []
    unresolved_treatments: list[
        dict[str, Any]
    ] = []

    for index, treatment in enumerate(
        vimedner["treatment_mentions"]
    ):
        normalized = normalize_for_matching(
            treatment
        )

        if any(
            term in normalized
            for term in PROCEDURE_TERMS
        ):
            procedures.append(
                {
                    "negative_id": (
                        f"PROC_{index:06d}"
                    ),
                    "text": treatment,
                    "negative_type": "PROCEDURE",
                    "source": "ViMedNER",
                }
            )
            continue

        matched = drug_surface_index.get(
            normalized,
            [],
        )

        unique_candidates = {
            candidate
            for item in matched
            for candidate in item["candidates"]
        }

        if len(unique_candidates) == 1:
            concept_id = next(
                iter(unique_candidates)
            )

            drug_surfaces.append(
                {
                    "surface_id": (
                        f"VIMED_DRUG_{index:06d}"
                    ),
                    "text": treatment,
                    "entity_type": "THUỐC",
                    "candidates": [concept_id],
                    "concept_id": concept_id,
                    "tty": None,
                    "profile": (
                        "vimedner_verified"
                    ),
                    "reliability": "high",
                    "usable_for_ner": True,
                    "usable_for_linking": True,
                    "attributes": {},
                }
            )
        else:
            unresolved_treatments.append(
                {
                    "text": treatment,
                    "reason": (
                        "no_unique_exact_rxnorm_match"
                    ),
                }
            )

    icd_concepts = load_icd10()

    disease_surfaces, unresolved_diseases = (
        build_disease_surfaces(
            vimedner["target_mentions"],
            icd_concepts,
        )
    )

    drug_surface_names = {
        normalize_for_matching(surface["text"])
        for surface in drug_surfaces
    }

    filtered_disease_surfaces: list[
        dict[str, Any]
    ] = []
    rejected_disease_surfaces: list[
        dict[str, Any]
    ] = []

    for surface in disease_surfaces:
        cleaned, reason = prepare_catalog_surface(
            surface["text"],
            "CHẨN_ĐOÁN",
        )
        normalized = normalize_for_matching(
            cleaned or surface["text"]
        )

        if normalized in drug_surface_names:
            reason = "cross_type_drug_collision"

        if reason:
            rejected_disease_surfaces.append(
                {
                    **surface,
                    "rejection_reason": reason,
                }
            )
            continue

        filtered_disease_surfaces.append(
            {
                **surface,
                "text": cleaned,
            }
        )

    disease_surfaces = filtered_disease_surfaces

    symptom_texts = sorted(
        {
            mention["text"]
            for mention in vimedner["target_mentions"]
            if mention["type"] == "TRIỆU_CHỨNG"
        }
    )

    symptom_surfaces: list[dict[str, Any]] = []
    rejected_symptom_surfaces: list[
        dict[str, Any]
    ] = []

    for original_text in symptom_texts:
        cleaned, reason = prepare_catalog_surface(
            original_text,
            "TRIỆU_CHỨNG",
        )

        if reason:
            rejected_symptom_surfaces.append(
                {
                    "text": original_text,
                    "rejection_reason": reason,
                    "source": "ViMedNER",
                }
            )
            continue

        symptom_surfaces.append(
            {
                "surface_id": (
                    f"SYM_{len(symptom_surfaces):06d}"
                ),
                "text": cleaned,
                "original_text": original_text,
                "entity_type": "TRIỆU_CHỨNG",
                "candidates": [],
                "concept_id": None,
                "profile": "vimedner_filtered",
                "reliability": "medium",
                "usable_for_ner": True,
                "usable_for_linking": False,
            }
        )

    write_jsonl(
        PROCESSED_DIR / "rxnorm_concepts.jsonl",
        rxnorm_concepts,
    )
    write_jsonl(
        PROCESSED_DIR / "rxnorm_families.jsonl",
        rxnorm_families,
    )
    write_jsonl(
        PROCESSED_DIR / "icd10_concepts.jsonl",
        icd_concepts,
    )
    write_jsonl(
        PROCESSED_DIR
        / "unresolved_treatments.jsonl",
        unresolved_treatments,
    )
    write_jsonl(
        PROCESSED_DIR
        / "unresolved_diseases.jsonl",
        unresolved_diseases,
    )
    write_jsonl(
        PROCESSED_DIR
        / "rejected_disease_surfaces.jsonl",
        rejected_disease_surfaces,
    )
    write_jsonl(
        PROCESSED_DIR
        / "rejected_symptom_surfaces.jsonl",
        rejected_symptom_surfaces,
    )

    write_jsonl(
        CATALOG_DIR / "drug_surfaces.jsonl",
        drug_surfaces,
    )
    write_jsonl(
        CATALOG_DIR
        / "disease_surfaces.jsonl",
        disease_surfaces,
    )
    write_jsonl(
        CATALOG_DIR
        / "symptom_surfaces.jsonl",
        symptom_surfaces,
    )
    write_jsonl(
        CATALOG_DIR / "hard_negatives.jsonl",
        procedures,
    )
    write_jsonl(
        CATALOG_DIR / "lab_tests.jsonl",
        LAB_TESTS,
    )

    write_json(
        CATALOG_DIR
        / "assertion_scenarios.json",
        ASSERTION_SCENARIOS,
    )
    write_json(
        CATALOG_DIR
        / "document_profiles.json",
        DOCUMENT_CONFIG,
    )

    write_json(
        CATALOG_DIR / "generation_pool.json",
        {
            "version": "v3.0",
            "counts": {
                "rxnorm_concepts": len(
                    rxnorm_concepts
                ),
                "drug_surfaces": len(
                    drug_surfaces
                ),
                "disease_surfaces": len(
                    disease_surfaces
                ),
                "symptom_surfaces": len(
                    symptom_surfaces
                ),
                "rejected_symptom_surfaces": len(
                    rejected_symptom_surfaces
                ),
                "rejected_disease_surfaces": len(
                    rejected_disease_surfaces
                ),
                "hard_negatives": len(
                    procedures
                ),
                "lab_tests": len(LAB_TESTS),
            },
        },
    )

    print("\n=== PREPARE DATA COMPLETE ===")
    print(
        f"Drug surfaces: "
        f"{len(drug_surfaces)}"
    )
    print(
        f"Disease surfaces: "
        f"{len(disease_surfaces)}"
    )
    print(
        f"Symptom surfaces: "
        f"{len(symptom_surfaces)}"
    )
    print(
        f"Hard negatives: "
        f"{len(procedures)}"
    )
    print(
        f"Rejected symptom surfaces: "
        f"{len(rejected_symptom_surfaces)}"
    )
    print(
        f"Rejected disease surfaces: "
        f"{len(rejected_disease_surfaces)}"
    )


def main() -> None:
    run_prepare_data()


if __name__ == "__main__":
    main()