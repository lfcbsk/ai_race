"""
=====================================================================
PIPELINE HOÀN CHỈNH: seed (ViMedNER.txt) + ontology (RxNorm, ICD10)
-> synthetic note.txt + label.json theo đúng schema đề thi

SCHEMA ĐÍCH (từ ví dụ đề thi):
    CHẨN_ĐOÁN            (có thể multi-code ICD-10)
    TRIỆU_CHỨNG
    TÊN_XÉT_NGHIỆM         (tự xây, không có trong seed)
    KẾT_QUẢ_XÉT_NGHIỆM      (tự xây, không có trong seed)
    THUỐC                (dạng "Hoạt chất X.X MG/ML", có assertion)

CHẠY THEO THỨ TỰ:
    python build_synthetic_data.py survey      # bước 1, xem đủ label
    python build_synthetic_data.py build_pool  # bước 2, xây ontology pool
    python build_synthetic_data.py generate N  # bước 3, sinh N sample
=====================================================================
"""

import json
import re
import sys
import random
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from rapidfuzz import fuzz
from dotenv import load_dotenv

load_dotenv()
RAW = Path("raw")
WORK = Path("work")
WORK.mkdir(exist_ok=True, parents=True)


# =====================================================================
# 0. TIỆN ÍCH CHUNG (giữ nguyên logic bạn đã viết trong notebook)
# =====================================================================

def clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return None
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_diacritics(text):
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def normalize_medical_text(text):
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?()[]{}\"'")
    return text


# =====================================================================
# 1. LOAD + SURVEY VIMEDNER.TXT (seed DUY NHẤT)
# =====================================================================

def load_vimedner(path=RAW / "ViMedNER.txt"):
    
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return records


def survey_labels(records):
    """In đầy đủ tất cả entity type + vài mẫu, để xác nhận taxonomy
    thật trước khi map sang schema đích."""
    all_types = Counter()
    samples = {}
    for obj in records:
        text = obj["text"]
        for s, e, etype in obj.get("label", []):
            all_types[etype] += 1
            samples.setdefault(etype, [])
            if len(samples[etype]) < 8:
                samples[etype].append(text[s:e])

    print(f"Tổng số record: {len(records)}")
    print("\n=== Toàn bộ entity type trong ViMedNER.txt ===")
    for etype, count in all_types.most_common():
        print(f"\n{etype}: {count} lần")
        for s in samples[etype]:
            print(f"    - {s}")
    return all_types, samples


# Sau khi chạy survey_labels(), xác nhận/chỉnh lại mapping này.
# TREATMENT không map thẳng sang THUỐC vì có thể lẫn thủ thuật —
# xử lý riêng bằng filter_drug_like_treatments() bên dưới.
SEED_TO_TARGET_LABEL = {
    "DISEASE": "CHẨN_ĐOÁN",
    "DIAGNOSIS": "CHẨN_ĐOÁN",
    "SYMPTOM": "TRIỆU_CHỨNG",
    # "TREATMENT": xử lý riêng (có thể là thuốc HOẶC thủ thuật)
    # "CAUSE": không thuộc schema đích -> bỏ qua
}


def extract_mentions_by_target_type(records):
    """Trích toàn bộ mention text theo target type, cộng riêng 1 nhóm
    TREATMENT (chưa phân loại) để xử lý ở bước sau."""
    out = {"CHẨN_ĐOÁN": [], "TRIỆU_CHỨNG": [], "TREATMENT_RAW": []}
    for obj in records:
        text = obj["text"]
        for s, e, etype in obj.get("label", []):
            mention = clean_text(text[s:e])
            if not mention:
                continue
            if etype == "TREATMENT":
                out["TREATMENT_RAW"].append(mention)
            else:
                target = SEED_TO_TARGET_LABEL.get(etype)
                if target:
                    out[target].append(mention)
    return out


# =====================================================================
# 2. RXNORM: load đủ TTY + parse mention + mapping có kiểm soát
# =====================================================================

RXNORM_TTYS = ("IN", "PIN", "MIN", "SCD", "SCDC", "SBD", "BN")

BRAND_TO_INGREDIENT = {
    "panadol": ["acetaminophen"],
    "hapacol": ["acetaminophen"],
    "efferalgan": ["acetaminophen"],
    "augmentin": ["amoxicillin", "clavulanic acid"],
    "glucophage": ["metformin"],
    "ventolin": ["albuterol"],
    "no-spa": ["drotaverine"],
    "panadol extra": ["acetaminophen", "caffeine"],
    "biseptol": ["sulfamethoxazole", "trimethoprim"],
    "terpin codein": ["terpin hydrate", "codeine"],
}

ABBREVIATION_DICT = {
    "para": "acetaminophen",
    "pcm": "acetaminophen",
    "amox": "amoxicillin",
    "metro": "metronidazole",
    "genta": "gentamicin",
    "cefo": "cefoperazone",
    "vit c": "ascorbic acid",
    "vit b1": "thiamine",
}

SPELLING_VARIANTS = {
    "paraxetamol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "amoxycillin": "amoxicillin",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "lidocain": "lidocaine",
    "gentamycin": "gentamicin",
}

SALT_VARIANTS = {
    "metformin hcl": "metformin hydrochloride",
    "diclofenac natri": "diclofenac sodium",
    "salbutamol sulfat": "albuterol sulfate",
}

AMBIGUOUS_INGREDIENTS = {
    "insulin": ["insulin human", "insulin lispro", "insulin aspart", "insulin glargine"],
}

PROCEDURE_TERMS = {
    "phẫu thuật", "mổ", "nội soi", "xạ trị", "hóa trị", "lọc máu",
    "vật lý trị liệu", "thở máy", "đặt nội khí quản", "can thiệp",
}

ROUTE_PATTERNS = {
    "oral": re.compile(r"\b(?:po|uống|đường uống)\b", re.I),
    "intravenous": re.compile(r"\b(?:iv|tĩnh mạch|tiêm tĩnh mạch)\b", re.I),
    "intramuscular": re.compile(r"\b(?:im|tiêm bắp)\b", re.I),
    "subcutaneous": re.compile(r"\b(?:sc|sq|dưới da|tiêm dưới da)\b", re.I),
}

FREQUENCY_PATTERNS = {
    "daily": re.compile(r"\b(?:daily|qd|ngày 1 lần|mỗi ngày)\b", re.I),
    "twice_daily": re.compile(r"\b(?:bid|ngày 2 lần)\b", re.I),
    "three_times_daily": re.compile(r"\b(?:tid|ngày 3 lần)\b", re.I),
    "four_times_daily": re.compile(r"\b(?:qid|ngày 4 lần)\b", re.I),
    "q6h": re.compile(r"\bq6h\b", re.I),
    "q8h": re.compile(r"\bq8h\b", re.I),
}

STRENGTH_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>mcg/ml|mg/ml|g/l|mg|mcg|µg|g|ml|%)\b",
    re.I,
)

DOSE_FORM_PATTERNS = {
    "tablet": re.compile(r"\b(?:tablet|tab|viên nén|viên)\b", re.I),
    "capsule": re.compile(r"\b(?:capsule|cap|viên nang)\b", re.I),
    "solution": re.compile(r"\b(?:solution|dung dịch)\b", re.I),
    "injection": re.compile(r"\b(?:injection|tiêm|ống tiêm)\b", re.I),
    "syrup": re.compile(r"\b(?:syrup|siro)\b", re.I),
    "cream": re.compile(r"\b(?:cream|kem bôi)\b", re.I),
}


def _rxnorm_item(item, fallback_tty):
    name = clean_text(item.get("name") or item.get("str") or item.get("concept_name"))
    rxcui = clean_text(item.get("rxcui") or item.get("RXCUI"))
    tty = clean_text(item.get("tty") or item.get("TTY") or fallback_tty)
    if not name or not rxcui or not tty:
        return None
    return {
        "rxcui": str(rxcui),
        "name": name,
        "tty": tty.upper(),
        "normalized_name": normalize_medical_text(name),
    }


def load_all_rxnorm(path_dir=RAW / "rxnorm_data"):
    records, missing = [], []
    for tty in RXNORM_TTYS:
        path = path_dir / f"rxnorm_{tty}.json"
        if not path.exists():
            missing.append(tty)
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("items") or payload.get("concepts") or []
        for raw in payload:
            item = _rxnorm_item(raw, tty)
            if item:
                records.append(item)

    unique = {(r["rxcui"], r["tty"], r["normalized_name"]): r for r in records}
    records = list(unique.values())
    if missing:
        print("CẢNH BÁO thiếu file RxNorm TTY:", ", ".join(missing))
    print("RxNorm loaded:", Counter(r["tty"] for r in records))
    return records


def build_rxnorm_indexes(records):
    exact_all = defaultdict(list)
    exact_by_tty = defaultdict(lambda: defaultdict(list))
    by_tty = defaultdict(list)
    for rec in records:
        exact_all[rec["normalized_name"]].append(rec)
        exact_by_tty[rec["tty"]][rec["normalized_name"]].append(rec)
        by_tty[rec["tty"]].append(rec)
    return {"exact_all": exact_all, "exact_by_tty": exact_by_tty, "by_tty": by_tty}


def parse_medication_mention(raw_text):
    normalized = normalize_medical_text(raw_text)
    strength_match = STRENGTH_PATTERN.search(normalized)
    route = next((name for name, p in ROUTE_PATTERNS.items() if p.search(normalized)), None)
    frequency = next((name for name, p in FREQUENCY_PATTERNS.items() if p.search(normalized)), None)
    dose_form = next((name for name, p in DOSE_FORM_PATTERNS.items() if p.search(normalized)), None)

    removable = normalized
    for pattern in list(ROUTE_PATTERNS.values()) + list(FREQUENCY_PATTERNS.values()) + list(DOSE_FORM_PATTERNS.values()):
        removable = pattern.sub(" ", removable)
    removable = re.sub(r"\b(?:prn|khi cần|xl|xr|sr|cr|er)\b", " ", removable, flags=re.I)
    removable = STRENGTH_PATTERN.sub(" ", removable)
    drug_name = re.sub(r"\s+", " ", removable).strip(" ,;:-")

    return {
        "raw_text": raw_text,
        "normalized_text": normalized,
        "drug_name": drug_name,
        "strength": ({"value": strength_match.group("value").replace(",", "."),
                      "unit": strength_match.group("unit").lower()} if strength_match else None),
        "dose_form": dose_form,
        "route": route,
        "frequency": frequency,
        "prn": bool(re.search(r"\b(?:prn|khi cần)\b", normalized, re.I)),
    }


def infer_target_ttys(parsed, is_brand=False, ingredient_count=1):
    formulated = bool(parsed.get("strength") or parsed.get("dose_form"))
    if is_brand and formulated:
        return ["SBD", "BN", "SCD"]
    if is_brand:
        return ["BN", "SBD"]
    if ingredient_count > 1 and not formulated:
        return ["MIN", "IN"]
    if formulated:
        return ["SCD", "SCDC", "IN", "PIN"]
    return ["IN", "PIN", "MIN"]


def exact_rxnorm_candidates(name, ttys, indexes):
    normalized = normalize_medical_text(name)
    out = []
    for tty in ttys:
        out.extend(indexes["exact_by_tty"].get(tty, {}).get(normalized, []))
    return out



def fuzzy_rxnorm_candidates(name, ttys, indexes, threshold=0.90, top_k=5):
    q = normalize_medical_text(name)
    scored = []
    for tty in ttys:
        for rec in indexes["by_tty"].get(tty, []):
            score = fuzz.ratio(q, rec["normalized_name"]) / 100.0
            if score >= threshold:
                scored.append((rec, score))
    scored.sort(key=lambda x: (-x[1], RXNORM_TTYS.index(x[0]["tty"])))
    return [{**rec, "score": round(score, 4)} for rec, score in scored[:top_k]]


def _ingredient_concepts(names, indexes):
    concepts = []
    for name in names:
        candidates = exact_rxnorm_candidates(name, ["IN", "PIN"], indexes)
        if candidates:
            concepts.append({k: candidates[0][k] for k in ("rxcui", "name", "tty")})
        else:
            concepts.append({"rxcui": None, "name": name, "tty": None})
    return concepts


def resolve_drug_name(raw_name, indexes):
    parsed = parse_medication_mention(raw_name)
    base_name = parsed["drug_name"] or parsed["normalized_text"]
    normalized_base = normalize_medical_text(base_name)

    if any(term in normalized_base for term in PROCEDURE_TERMS):
        return {"original_text": raw_name, "entity_class": "PROCEDURE", "parsed": parsed,
                "status": "non_drug", "selected_concept": None, "candidates": []}

    if normalized_base in AMBIGUOUS_INGREDIENTS:
        return {"original_text": raw_name, "entity_class": "DRUG", "parsed": parsed,
                "status": "requires_context", "selected_concept": None,
                "candidates": AMBIGUOUS_INGREDIENTS[normalized_base], "confidence": None}

    alias_method = None
    is_brand = normalized_base in BRAND_TO_INGREDIENT
    ingredient_names = BRAND_TO_INGREDIENT.get(normalized_base, [])
    canonical_search = normalized_base

    if normalized_base in ABBREVIATION_DICT:
        canonical_search = ABBREVIATION_DICT[normalized_base]
        alias_method = "verified_abbreviation"
    elif normalized_base in SPELLING_VARIANTS:
        canonical_search = SPELLING_VARIANTS[normalized_base]
        alias_method = "verified_spelling_alias"
    elif normalized_base in SALT_VARIANTS:
        canonical_search = SALT_VARIANTS[normalized_base]
        alias_method = "verified_salt_alias"
    elif is_brand:
        alias_method = "verified_brand_alias"

    target_ttys = infer_target_ttys(parsed, is_brand=is_brand,
                                    ingredient_count=max(1, len(ingredient_names)))

    search_names = [normalized_base] if is_brand else [canonical_search]
    exact = []
    for search_name in search_names:
        exact.extend(exact_rxnorm_candidates(search_name, target_ttys, indexes))

    # Brand không có BN/SBD vẫn giữ ingredient links nhưng không giả mạo concept brand.
    selected = exact[0] if exact else None
    ingredient_concepts = _ingredient_concepts(
        ingredient_names or ([canonical_search] if not is_brand else []), indexes
    )

    if selected:
        return {
            "original_text": raw_name,
            "entity_class": "DRUG",
            "parsed": parsed,
            "selected_concept": {k: selected[k] for k in ("rxcui", "name", "tty")},
            "ingredient_concepts": ingredient_concepts,
            "mapping_method": alias_method or "exact_rxnorm",
            "confidence": 1.0,
            "status": "verified",
            "candidates": [],
        }

    # Với alias ingredient đã xác minh, có thể chọn exact IN/PIN dù mention có route/frequency.
    if alias_method and not is_brand:
        fallback = exact_rxnorm_candidates(canonical_search, ["IN", "PIN"], indexes)
        if fallback:
            selected = fallback[0]
            return {
                "original_text": raw_name, "entity_class": "DRUG", "parsed": parsed,
                "selected_concept": {k: selected[k] for k in ("rxcui", "name", "tty")},
                "ingredient_concepts": ingredient_concepts,
                "mapping_method": alias_method, "confidence": 1.0,
                "status": "verified", "candidates": [],
            }

    fuzzy_name = normalized_base if is_brand else canonical_search
    candidates = fuzzy_rxnorm_candidates(fuzzy_name, target_ttys, indexes)
    return {
        "original_text": raw_name,
        "entity_class": "DRUG" if candidates or is_brand or alias_method else "UNKNOWN_TREATMENT",
        "parsed": parsed,
        "selected_concept": None,
        "ingredient_concepts": ingredient_concepts,
        "mapping_method": "fuzzy_candidate_retrieval" if candidates else None,
        "confidence": candidates[0]["score"] if candidates else 0.0,
        "status": "review" if candidates or is_brand else "not_found",
        "candidates": candidates,
    }


def filter_drug_like_treatments(treatment_mentions, rxnorm_indexes):
    mappings = [resolve_drug_name(m, rxnorm_indexes) for m in sorted(set(treatment_mentions))]
    drug_like = [m["original_text"] for m in mappings
                 if m["entity_class"] == "DRUG" and m["status"] in {"verified", "review", "requires_context"}]
    excluded = [m for m in mappings if m["original_text"] not in drug_like]
    json.dump(mappings, open(WORK / "treatment_classification.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"TREATMENT: {len(drug_like)} drug-like; {len(excluded)} procedure/unknown.")
    return drug_like


def finalize_drug_mapping(drug_mentions, rxnorm_indexes):
    mappings = [resolve_drug_name(name, rxnorm_indexes)
                for name in sorted(set(m.strip() for m in drug_mentions if m.strip()))]
    verified = [m for m in mappings if m["status"] == "verified" and m.get("selected_concept")]
    review = [m for m in mappings if m["status"] in {"review", "requires_context"}]
    not_found = [m for m in mappings if m["status"] in {"not_found", "non_drug"}]
    payload = {"verified": verified, "review": review, "not_found": not_found}
    json.dump(payload, open(WORK / "drug_mapping_final.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Drug mapping: verified={len(verified)}, review={len(review)}, not_found={len(not_found)}")
    return verified


# =====================================================================
# 3. ICD-10: candidate retrieval + chỉ dùng mapping đã verified
# =====================================================================

DISEASE_ALIASES = {
    "cao huyết áp": "tăng huyết áp",
    "tăng huyết áp": "tăng huyết áp vô căn",
}


def load_icd10(path=RAW / "DM ICD10-19_8_BYT.xlsx"):
    import pandas as pd
    df = pd.read_excel(path, header=4)
    df.columns = df.columns.astype(str).str.strip()
    required = ["Mã", "Tên bệnh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột ICD-10: {missing}. Các cột hiện có: {list(df.columns)}")
    keep = ["Mã", "Tên bệnh"]
    group_col = next((c for c in ("Nhóm bệnh", "Tên nhóm", "Nhóm") if c in df.columns), None)
    if group_col:
        keep.append(group_col)
    df = df[keep].copy()
    rename = {"Mã": "code", "Tên bệnh": "name_vi"}
    if group_col:
        rename[group_col] = "group_vi"
    df = df.rename(columns=rename).dropna(subset=["code", "name_vi"])
    df["code"] = df["code"].astype(str).str.strip().str.upper()
    df["name_vi"] = df["name_vi"].astype(str).str.strip()
    if "group_vi" not in df:
        df["group_vi"] = None
    else:
        df["group_vi"] = df["group_vi"].where(df["group_vi"].notna(), None)
    df["normalized_name"] = df["name_vi"].map(normalize_medical_text)
    df = df.drop_duplicates(subset=["code", "normalized_name"])
    return df.to_dict("records")


from rapidfuzz import fuzz

def build_icd_indexes(records):
    exact = defaultdict(list)
    for rec in records:
        exact[rec["normalized_name"]].append(rec)
        # cache sẵn để fuzzy match không phải tính lại
        rec["_stripped"] = strip_diacritics(rec["normalized_name"])
        rec["_tokens"] = set(rec["_stripped"].split())
    return {"exact": exact, "records": records}


def fuzzy_match_icd(disease_mention, icd10_records, threshold=0.78, top_k=5):
    q = strip_diacritics(normalize_medical_text(disease_mention))
    q_tokens = set(q.split())
    scored = []
    for rec in icd10_records:
        candidate = rec["_stripped"]            # đã cache, không strip lại
        char_score = fuzz.ratio(q, candidate) / 100.0
        c_tokens = rec["_tokens"]                # đã cache, không split lại
        token_score = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
        score = 0.7 * char_score + 0.3 * token_score
        if score >= threshold:
            scored.append((rec, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def resolve_icd_mention(name, indexes):
    normalized = normalize_medical_text(name)
    canonical = DISEASE_ALIASES.get(normalized, normalized)
    alias_used = canonical != normalized
    exact = indexes["exact"].get(normalize_medical_text(canonical), [])
    if len(exact) == 1:
        rec = exact[0]
        return {
            "vn_mention": name,
            "selected_concept": {k: rec.get(k) for k in ("code", "name_vi", "group_vi")},
            "status": "verified",
            "mapping_method": "verified_alias" if alias_used else "exact_icd10",
            "confidence": 1.0,
            "candidates": [],
        }

    candidates = fuzzy_match_icd(canonical, indexes["records"])
    serialized = [
        {"code": rec["code"], "name_vi": rec["name_vi"], "group_vi": rec.get("group_vi"),
         "score": round(score, 4)} for rec, score in candidates
    ]
    # Chỉ auto-accept khi cực rõ: top1 >= .96 và cách top2 >= .08.
    if serialized:
        top1 = serialized[0]["score"]
        top2 = serialized[1]["score"] if len(serialized) > 1 else 0.0
        if top1 >= 0.96 and top1 - top2 >= 0.08:
            selected = {k: serialized[0].get(k) for k in ("code", "name_vi", "group_vi")}
            return {"vn_mention": name, "selected_concept": selected,
                    "status": "verified", "mapping_method": "high_precision_lexical",
                    "confidence": top1, "candidates": serialized}

    return {
        "vn_mention": name,
        "selected_concept": None,
        "status": "review" if serialized else "not_found",
        "mapping_method": "lexical_candidate_retrieval" if serialized else None,
        "confidence": serialized[0]["score"] if serialized else 0.0,
        "candidates": serialized,
    }


def finalize_icd_mapping(disease_mentions, icd10_indexes):
    mappings = [resolve_icd_mention(name, icd10_indexes)
                for name in sorted(set(m.strip() for m in disease_mentions if m.strip()))]
    verified = [m for m in mappings if m["status"] == "verified" and m.get("selected_concept")]
    review = [m for m in mappings if m["status"] == "review"]
    not_found = [m for m in mappings if m["status"] == "not_found"]
    json.dump({"verified": verified, "review": review, "not_found": not_found},
              open(WORK / "icd_mapping_final.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"ICD mapping: verified={len(verified)}, review={len(review)}, not_found={len(not_found)}")
    return verified


# =====================================================================
# 4. LAB TEST ONTOLOGY — tự xây (không có trong seed)
# =====================================================================

LAB_TEST_ONTOLOGY = [
    {"code": "WBC", "name_vi": "WBC", "unit": "10^9/L", "normal_range": [4.0, 10.0]},
    {"code": "NEUT%", "name_vi": "NEUT% (Tỷ lệ % bạch cầu trung tính)", "unit": "%", "normal_range": [40.0, 74.0]},
    {"code": "LYMPH%", "name_vi": "LYPH% (Tỷ lệ bạch cầu lympho)", "unit": "%", "normal_range": [19.0, 48.0]},
    {"code": "MONO%", "name_vi": "MONO% (Tỷ lệ % bạch cầu mono)", "unit": "%", "normal_range": [3.0, 9.0]},
    {"code": "RBC", "name_vi": "RBC (Số lượng hồng cầu)", "unit": "10^12/L", "normal_range": [3.8, 5.8]},
    {"code": "HGB", "name_vi": "HGB (Huyết sắc tố)", "unit": "g/L", "normal_range": [110, 165]},
    {"code": "HCT", "name_vi": "HCT (Thể tích khối hồng cầu)", "unit": "%", "normal_range": [34.0, 50.0]},
    {"code": "PLT", "name_vi": "PLT (Số lượng tiểu cầu)", "unit": "10^9/L", "normal_range": [150, 400]},
    {"code": "GLU", "name_vi": "Glucose máu", "unit": "mmol/L", "normal_range": [3.9, 6.4]},
    {"code": "CRE", "name_vi": "Creatinine máu", "unit": "umol/L", "normal_range": [44, 106]},
]


def sample_lab_result(lab_test):
    lo, hi = lab_test["normal_range"]
    if random.random() < 0.7:
        val = random.uniform(lo, hi)
    else:
        span = hi - lo
        val = random.choice([
            random.uniform(max(0, lo - span * 0.5), lo),
            random.uniform(hi, hi + span * 0.5),
        ])
    return f"{val:.2f}".replace(".", ",")


# =====================================================================
# 5. GỘP ONTOLOGY POOL — chỉ đưa mapping verified vào generator
# =====================================================================

def build_ontology_pool():
    records = load_vimedner()
    mentions = extract_mentions_by_target_type(records)

    rxnorm_records = load_all_rxnorm()
    rxnorm_indexes = build_rxnorm_indexes(rxnorm_records)
    drug_like = filter_drug_like_treatments(mentions["TREATMENT_RAW"], rxnorm_indexes)
    drugs_verified = finalize_drug_mapping(drug_like, rxnorm_indexes)

    icd10_records = load_icd10()
    icd10_indexes = build_icd_indexes(icd10_records)
    diseases_verified = finalize_icd_mapping(mentions["CHẨN_ĐOÁN"], icd10_indexes)

    symptoms = sorted(set(mentions["TRIỆU_CHỨNG"]))
    pool = {
        "drugs": drugs_verified,
        "diseases": diseases_verified,
        "symptoms": symptoms,
        "lab_tests": LAB_TEST_ONTOLOGY,
        "meta": {
            "rxnorm_ttys": list(RXNORM_TTYS),
            "only_verified_mappings": True,
            "lab_ranges_are_synthetic_templates": True,
            "not_for_clinical_use": True,
        },
    }
    with open(WORK / "ontology_pool_final.json", "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    print(f"\nOntology pool verified: drugs={len(pool['drugs'])}, "
          f"diseases={len(pool['diseases'])}, symptoms={len(pool['symptoms'])}, "
          f"lab_tests={len(pool['lab_tests'])}")
    print("Các mapping review nằm trong drug_mapping_final.json và icd_mapping_final.json; "
          "không được đưa vào nhãn synthetic cho tới khi xác minh.")
    return pool


# =====================================================================
# 6. CONSTRAINT GENERATOR + PROMPT
# =====================================================================
NEGATION_CUES = [
    "không có", "không xuất hiện", "không ghi nhận", "chưa từng",
    "đã ngừng", "không dùng", "không sử dụng", "phủ nhận", "không bị",
]
 
# Các cụm chứa từ "không" nhưng KHÔNG mang nghĩa phủ định thật —
# nếu prompt hướng dẫn model tránh, hoặc nếu cue thật trùng khớp với
# 1 trong các cụm này thì không tính là hợp lệ (false trigger).
NEGATION_FALSE_TRAPS = [
    "không loại trừ", "không thể loại trừ", "không rõ",
    "không chắc chắn", "không loại trừ khả năng",
]
 
FAMILY_CUES = [
    "bố bệnh nhân", "mẹ bệnh nhân", "anh trai", "chị gái", "em trai",
    "em gái", "ông của bệnh nhân", "bà của bệnh nhân", "người nhà",
    "gia đình", "họ hàng", "bố", "mẹ",
]
 
HISTORICAL_CUES = [
    "tiền sử", "trước đây", "đã từng", "trong quá khứ",
]
 
ASSERTION_CUES = {
    "isNegated": NEGATION_CUES,
    "isFamily": FAMILY_CUES,
    "isHistorical": HISTORICAL_CUES,
}
 
ASSERTION_HINT_VI = {
    "isNegated": "PHỦ ĐỊNH — dùng cụm như 'không có', 'không xuất hiện', 'phủ nhận'",
    "isFamily": "NGƯỜI NHÀ — gán cho người nhà bệnh nhân, dùng cụm như 'bố/mẹ bệnh nhân có...'",
    "isHistorical": "TIỀN SỬ — dùng cụm như 'tiền sử', 'trước đây', 'đã từng'",
}
 
 

 
# MULTI_CODE_DISEASE_PATTERNS = {
#     "dai thao duong": ["E11.9", "N18.9"],       # tiểu đường -> kèm bệnh thận mạn
#     "tang huyet ap": ["I10", "I11.9"],           # tăng huyết áp -> kèm bệnh tim do THA
#     "suy tim": ["I50.9", "I25.9"],               # suy tim -> kèm bệnh mạch vành
#     "viem phoi": ["J18.9", "J96.0"],             # viêm phổi -> kèm suy hô hấp cấp
#     "xo gan": ["K74.6", "K76.6"],                # xơ gan -> kèm tăng áp lực tĩnh mạch cửa
# }
 
 
# def find_multi_codes(vn_mention: str, primary_code: str) -> list[str]:
#     """Tìm mã ICD-10 phụ nếu bệnh chính khớp pattern đã biết.
#     Luôn trả về list có primary_code ở đầu."""
#     normalized = strip_diacritics(normalize_medical_text(vn_mention))
#     for pattern, extra_codes in MULTI_CODE_DISEASE_PATTERNS.items():
#         if pattern in normalized:
#             codes = [primary_code] + [c for c in extra_codes if c != primary_code]
#             return codes
#     return [primary_code]
 
 
 
def sample_assertions() -> list[str]:
    """Sinh ngẫu nhiên list assertion cho 1 concept.
    - ~55% không có assertion nào (list rỗng = hiện diện bình thường)
    - Các assertion còn lại độc lập với nhau (có thể kết hợp,
      ví dụ isFamily + isHistorical: 'gia đình có tiền sử...')
    """
    assertions = []
    if random.random() < 0.20:
        assertions.append("isNegated")
    if random.random() < 0.15:
        assertions.append("isFamily")
    if random.random() < 0.25:
        assertions.append("isHistorical")
    return assertions
 


def generate_constraint(pool):
    if not pool.get("diseases"):
        raise ValueError(
            "Ontology pool chưa có ICD-10 verified. Hãy review work/icd_mapping_final.json "
            "và bổ sung alias/verified mapping trước khi generate."
        )
    if not pool.get("symptoms"):
        raise ValueError("Không có triệu chứng trong ontology pool.")
 
    # --- TRIỆU_CHỨNG: mỗi symptom có assertions riêng ---
    n_symptoms = random.randint(2, 5)
    symptom_texts = random.sample(pool["symptoms"], k=min(n_symptoms, len(pool["symptoms"])))
    symptoms = [{"text": s, "assertions": sample_assertions()} for s in symptom_texts]

    # --- CHẨN_ĐOÁN: có assertions + có thể multi-code ICD-10 ---
    disease_mapping = random.choice(pool["diseases"])
    primary_code = str(disease_mapping["selected_concept"]["code"])
    disease = {"mapping": disease_mapping, "assertions": sample_assertions(), "candidates": [primary_code],}

    # --- THUỐC: có assertions ---
    # ~30% trường hợp: sinh kiểu "danh sách thuốc" dùng chung 1 assertion
    # cho cả nhóm (mô phỏng ví dụ thật: "Danh sách thuốc trước nhập viện"
    # -> mọi thuốc trong danh sách đều isHistorical, không cần cue riêng
    # lẻ từng thuốc). Các trường hợp còn lại: mỗi thuốc random độc lập.
    n_drugs = random.randint(0, 2)
    drug_entities = []
    is_group_mode = random.random() < 0.30 and n_drugs >= 2
    group_assertions = sample_assertions() if is_group_mode else None
 
    if pool.get("drugs"):
        for drug in random.sample(pool["drugs"], k=min(n_drugs, len(pool["drugs"]))):
            assertions = group_assertions if is_group_mode else sample_assertions()
            drug_entities.append({"drug": drug, "assertions": assertions})
 
    # --- Xét nghiệm: giữ nguyên logic cũ, không có assertion ---
    n_labs = random.randint(0, 3)
    labs = random.sample(pool["lab_tests"], k=min(n_labs, len(pool["lab_tests"])))
    lab_entities = [{"test": lt, "result": sample_lab_result(lt)} for lt in labs]
 
    return {
        "symptoms": symptoms,
        "disease": disease,
        "drugs": drug_entities,
        "labs": lab_entities,
    }
 


def drug_surface_text(drug_mapping):
    # Giữ đúng mention đã được map. Không thay brand bằng ingredient đầu tiên.
    return drug_mapping["original_text"]


 
def _assertion_instruction(assertions: list[str]) -> str:
    if not assertions:
        return "thể hiện BÌNH THƯỜNG (không phủ định, không phải người nhà, không phải tiền sử)"
    return "; ".join(ASSERTION_HINT_VI[a] for a in assertions)
 
 
def build_prompt(constraint):
    symptom_lines = [
        f"- Giữ nguyên chính xác chuỗi '{s['text']}'; {_assertion_instruction(s['assertions'])}."
        for s in constraint["symptoms"]
    ]
 
    disease_mention = constraint["disease"]["mapping"]["vn_mention"]
    disease_assertion_instr = _assertion_instruction(constraint["disease"]["assertions"])
    disease_codes = constraint["disease"]["candidates"]

 
    # Nếu mọi thuốc trong constraint dùng chung 1 assertion (giống nhau
    # hệt nhau), coi là "group mode" -> hướng dẫn model viết 1 câu mở
    # đầu chung (kiểu "Danh sách thuốc trước nhập viện:") thay vì lặp
    # lại hint riêng cho từng thuốc.
    drug_assertion_lists = [tuple(item["assertions"]) for item in constraint["drugs"]]
    is_group_mode = (
        len(drug_assertion_lists) >= 2
        and len(set(drug_assertion_lists)) == 1
        and drug_assertion_lists[0]  # không rỗng
    )

    drug_lines = []
    if is_group_mode:
        shared_instr = _assertion_instruction(list(drug_assertion_lists[0]))
        drug_lines.append(
            f"- QUAN TRỌNG: viết 1 câu MỞ ĐẦU chung cho cả danh sách thuốc dưới đây, "
            f"thể hiện rõ toàn bộ danh sách là: {shared_instr}. "
            f"(ví dụ: 'Danh sách thuốc trước nhập viện:' nếu là tiền sử). "
            f"Sau câu mở đầu, liệt kê các thuốc, KHÔNG cần lặp lại cue cho từng thuốc riêng lẻ."
        )
        for item in constraint["drugs"]:
            drug_name = item["drug"]["original_text"]
            concept = item["drug"]["selected_concept"]
            drug_lines.append(
                f"- Giữ nguyên chính xác chuỗi '{drug_name}'. "
                f"Concept nội bộ: {concept['name']} [{concept['tty']}], không cần viết ra văn bản."
            )
    else:
        for item in constraint["drugs"]:
            drug_name = item["drug"]["original_text"]
            concept = item["drug"]["selected_concept"]
            instr = _assertion_instruction(item["assertions"])
            drug_lines.append(
                f"- Giữ nguyên chính xác chuỗi '{drug_name}'; {instr}. "
                f"Concept nội bộ: {concept['name']} [{concept['tty']}], không cần viết concept nội bộ ra văn bản."
            )

    lab_lines = [
        f"- Viết đúng mã {lt['test']['code']}: {lt['result']} {lt['test']['unit']}"
        for lt in constraint["labs"]
    ]

    return f"""Bạn là bác sĩ viết tóm tắt bệnh án tiếng Việt theo phong cách hồ sơ khám bệnh.
                Viết MỘT đoạn văn liền mạch; thông tin nhân khẩu học phải hoàn toàn synthetic.
                
                Nội dung bắt buộc:
                1. Triệu chứng:
                {chr(10).join(symptom_lines)}
                2. Chẩn đoán: Giữ nguyên chính xác chuỗi '{disease_mention}'; {disease_assertion_instr}.
                3. Thuốc:
                {chr(10).join(drug_lines) if drug_lines else "- Không có thuốc bắt buộc trong ca này."}
                4. Xét nghiệm, viết cuối đoạn theo dạng MÃ:giá_trị đơn_vị;
                {chr(10).join(lab_lines) if lab_lines else "- Không có xét nghiệm bắt buộc trong ca này."}
                
                Quy tắc:
                - Không đổi, dịch, rút gọn hoặc sửa chính tả các chuỗi entity đã cho.
                - Với 'tiền sử' phải dùng cụm rõ như 'tiền sử', 'đã từng', 'trước đây'.
                - Với 'phủ định' phải dùng cụm phủ định gắn trực tiếp với khái niệm như 'không có', 'không xuất hiện'.
                - TRÁNH dùng các cụm mơ hồ như 'không loại trừ', 'không rõ', 'không chắc chắn' cho bất kỳ khái niệm nào — các cụm này KHÔNG phải phủ định thật và gây nhầm lẫn khi chấm điểm.
                - Với 'người nhà' phải gán rõ cho người nhà bệnh nhân (bố, mẹ, anh, chị...), không phải bản thân bệnh nhân.
                - Một khái niệm có thể vừa là tiền sử vừa là người nhà cùng lúc (ví dụ: 'mẹ bệnh nhân có tiền sử...').
                - Chỉ trả về đoạn văn, không markdown, không giải thích."""



# =====================================================================
# 7. RULE VALIDATOR
# =====================================================================

def find_all_occurrences(text, phrase):
    positions, start = [], 0
    while True:
        idx = text.find(phrase, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + max(1, len(phrase))
    return positions


def sentence_scope(text, start, end):
    left = max(text.rfind(".", 0, start), text.rfind(";", 0, start), text.rfind("\n", 0, start))
    right_candidates = [p for p in (text.find(".", end), text.find(";", end), text.find("\n", end)) if p != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right].lower()



def _document_scope(text: str, idx: int, window: int = 300) -> str:
    """Bối cảnh RỘNG HƠN sentence_scope — dùng cho trường hợp assertion
    được thiết lập bởi 1 câu/tiêu đề ở ĐẦU ĐOẠN áp dụng cho cả khối phía
    sau (ví dụ 'Danh sách thuốc trước nhập viện:' rồi liệt kê nhiều thuốc).
    Lấy `window` ký tự TRƯỚC vị trí idx (không lấy phía sau, vì assertion
    kiểu này luôn được thiết lập trước khi liệt kê)."""
    start = max(0, idx - window)
    return text[start:idx].lower()
 
 
def _check_assertions_in_scope(text: str, idx: int, end: int, assertions: list[str]) -> list[str]:
    """Trả về list lỗi nếu thiếu cue cho bất kỳ assertion nào trong list.
    Check theo 2 tầng:
    1. sentence_scope (câu chứa mention) — bắt cue cục bộ, sát nghĩa nhất.
    2. document_scope (300 ký tự trước đó) — bắt cue kiểu tiêu đề/mở đoạn
       áp dụng cho cả khối phía sau (ví dụ danh sách thuốc tiền sử).
    Chỉ cần 1 trong 2 tầng tìm thấy cue là coi như hợp lệ.
 
    Với isNegated: nếu cue tìm thấy thực ra là 1 phần của cụm "false trap"
    (VD: "không loại trừ" chứa "không" nhưng không mang nghĩa phủ định),
    thì KHÔNG tính là cue hợp lệ — tránh model lách qua validator bằng
    cách viết mơ hồ.
    """
    scope_local = sentence_scope(text, idx, end)
    scope_wide = _document_scope(text, idx)
 
    errors = []
    for assertion in assertions:
        cues = ASSERTION_CUES[assertion]
 
        def _valid_cue_found(scope: str) -> bool:
            for cue in cues:
                if cue not in scope:
                    continue
                if assertion == "isNegated":
                    # Kiểm tra cue không nằm trong 1 false trap
                    is_trapped = any(
                        trap in scope and cue in trap
                        for trap in NEGATION_FALSE_TRAPS
                    )
                    if is_trapped:
                        continue
                return True
            return False
 
        found_local = _valid_cue_found(scope_local)
        found_wide = _valid_cue_found(scope_wide)
        if not (found_local or found_wide):
            errors.append(f"thiếu cue hợp lệ cho assertion '{assertion}' (đã check cả câu và cả đoạn trước đó, loại trừ false trap)")
    return errors
 
 
def align_and_validate(generated_text, constraint):
    entities, errors, occupied = [], [], []
 
    # --- TRIỆU_CHỨNG ---
    for symptom in sorted(constraint["symptoms"], key=lambda s: len(s["text"]), reverse=True):
        text = symptom["text"]
        positions = find_all_occurrences(generated_text, text)
        if not positions:
            errors.append(f"Thiếu triệu chứng '{text}'")
            continue
        idx = next((p for p in positions if not any(p < e and p + len(text) > s for s, e in occupied)), positions[0])
        occupied.append((idx, idx + len(text)))
 
        assertion_errors = _check_assertions_in_scope(
            generated_text, idx, idx + len(text), symptom["assertions"]
        )
        errors.extend(f"'{text}': {e}" for e in assertion_errors)
 
        entities.append({
            "text": text,
            "position": [
                idx,
                idx + len(text),
            ],
            "type": "TRIỆU_CHỨNG",
            "assertions": symptom["assertions"],
        })
 
    # --- CHẨN_ĐOÁN ---
    disease_mapping = constraint["disease"]["mapping"]
    disease_assertions = constraint["disease"]["assertions"]
    disease_mention = disease_mapping["vn_mention"]
    idx = generated_text.find(disease_mention)
    selected_icd = disease_mapping.get("selected_concept")
 
    if idx == -1:
        errors.append(f"Thiếu chẩn đoán '{disease_mention}'")
    elif not selected_icd or disease_mapping.get("status") != "verified":
        errors.append(f"Chẩn đoán '{disease_mention}' chưa có ICD-10 verified")
    else:
        assertion_errors = _check_assertions_in_scope(
            generated_text, idx, idx + len(disease_mention), disease_assertions
        )
        errors.extend(f"'{disease_mention}': {e}" for e in assertion_errors)
 
        entities.append({
            "text": disease_mention,
            "position": [
                idx,
                idx + len(disease_mention),
            ],
            "type": "CHẨN_ĐOÁN",
            "assertions": disease_assertions,
            "candidates": [str(selected_icd["code"])],
        })
 
    # --- THUỐC ---
    for item in constraint["drugs"]:
        drug_mapping = item["drug"]
        drug_assertions = item["assertions"]
        drug_name = drug_mapping["original_text"]
        idx = generated_text.find(drug_name)
        selected = drug_mapping.get("selected_concept")
 
        if idx == -1:
            errors.append(f"Thiếu thuốc '{drug_name}'")
            continue
        if not selected or drug_mapping.get("status") != "verified":
            errors.append(f"Thuốc '{drug_name}' chưa có RxNorm verified")
            continue
 
        assertion_errors = _check_assertions_in_scope(
            generated_text, idx, idx + len(drug_name), drug_assertions
        )
        errors.extend(f"'{drug_name}': {e}" for e in assertion_errors)
 
        entities.append({
            "text": drug_name,
            "position": [
                idx,
                idx + len(drug_name),
            ],
            "type": "THUỐC",
            "assertions": drug_assertions,
            "candidates": [
                str(selected["rxcui"])
            ],
        })
 
    # --- XÉT NGHIỆM ---
    for lab in constraint["labs"]:
        code = str(lab["test"]["code"])
        result_value = str(lab["result"])
        unit = str(lab["test"]["unit"])

        # KẾT_QUẢ_XÉT_NGHIỆM phải gồm cả giá trị và đơn vị
        full_result = f"{result_value} {unit}"

        name_idx = generated_text.find(code)

        if name_idx == -1:
            errors.append(
                f"Thiếu tên xét nghiệm '{code}'"
            )
            continue

        result_idx = generated_text.find(
            full_result,
            name_idx + len(code),
        )

        if result_idx == -1:
            errors.append(
                f"Thiếu kết quả xét nghiệm đầy đủ "
                f"'{full_result}' của '{code}'"
            )
            continue

        entities.append({
            "text": code,
            "position": [
                name_idx,
                name_idx + len(code),
            ],
            "type": "TÊN_XÉT_NGHIỆM",
        })

        entities.append({
            "text": full_result,
            "position": [
                result_idx,
                result_idx + len(full_result),
            ],
            "type": "KẾT_QUẢ_XÉT_NGHIỆM",
        })
 
    entities.sort(key=lambda e: (e["position"][0], e["position"][1]))
    return entities, errors

VALID_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}

ASSERTION_TYPES = {
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
}

CANDIDATE_TYPES = {
    "CHẨN_ĐOÁN",
    "THUỐC",
}

VALID_ASSERTIONS = {
    "isNegated",
    "isFamily",
    "isHistorical",
}


def validate_output_schema(
    source_text: str,
    entities: list[dict],
) -> list[str]:
    errors = []

    for i, entity in enumerate(entities):
        prefix = f"Entity {i}"

        required = {"text", "position", "type"}
        missing = required - set(entity)

        if missing:
            errors.append(
                f"{prefix}: thiếu field {sorted(missing)}"
            )
            continue

        entity_type = entity["type"]

        if entity_type not in VALID_TYPES:
            errors.append(
                f"{prefix}: type không hợp lệ "
                f"{entity_type!r}"
            )

        position = entity["position"]

        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(
                isinstance(x, int)
                for x in position
            )
        ):
            errors.append(
                f"{prefix}: position phải là list "
                f"gồm 2 số nguyên"
            )
            continue

        start, end = position

        if not (
            0 <= start < end <= len(source_text)
        ):
            errors.append(
                f"{prefix}: position ngoài phạm vi "
                f"{position}"
            )
            continue

        actual_text = source_text[start:end]

        if actual_text != entity["text"]:
            errors.append(
                f"{prefix}: text-position mismatch: "
                f"entity={entity['text']!r}, "
                f"actual={actual_text!r}"
            )

        # Assertions chỉ dành cho 3 loại
        if entity_type in ASSERTION_TYPES:
            assertions = entity.get("assertions")

            if not isinstance(assertions, list):
                errors.append(
                    f"{prefix}: thiếu assertions dạng list"
                )
            else:
                invalid_assertions = (
                    set(assertions) - VALID_ASSERTIONS
                )

                if invalid_assertions:
                    errors.append(
                        f"{prefix}: assertion không hợp lệ "
                        f"{sorted(invalid_assertions)}"
                    )

                if len(assertions) > 3:
                    errors.append(
                        f"{prefix}: assertions vượt quá "
                        f"3 phần tử"
                    )
        elif "assertions" in entity:
            errors.append(
                f"{prefix}: type {entity_type} "
                f"không được có assertions"
            )

        # Candidates chỉ dành cho chẩn đoán và thuốc
        if entity_type in CANDIDATE_TYPES:
            candidates = entity.get("candidates")

            if not isinstance(candidates, list):
                errors.append(
                    f"{prefix}: thiếu candidates dạng list"
                )
            elif not all(
                isinstance(code, str)
                for code in candidates
            ):
                errors.append(
                    f"{prefix}: toàn bộ candidates "
                    f"phải là string"
                )
        elif "candidates" in entity:
            errors.append(
                f"{prefix}: type {entity_type} "
                f"không được có candidates"
            )

        allowed_fields = {
            "text",
            "position",
            "type",
        }

        if entity_type in ASSERTION_TYPES:
            allowed_fields.add("assertions")

        if entity_type in CANDIDATE_TYPES:
            allowed_fields.add("candidates")

        extra_fields = (
            set(entity.keys()) - allowed_fields
        )

        if extra_fields:
            errors.append(
                f"{prefix}: có field ngoài schema "
                f"{sorted(extra_fields)}"
            )

    return errors

# =====================================================================
# 8. GỌI LLM — THAY BẰNG API THẬT CỦA BẠN
# =====================================================================
import os
import time
import json
import requests
from typing import Any

LLM_MODELS = {
    "gemini_flash": {
        "provider": "gemini",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "api_key_env": "GEMINI_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enabled": False,
    },
    "groq_qwen": {
        "provider": "openai_compatible",
        "model": "qwen3-32b",   # cần verify lại trên console.groq.com/docs/models
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enabled": False,
    },
    "groq_llama_70b": {  # giữ nguyên, đang chạy tốt
        "provider": "openai_compatible",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enabled": False,
    },
    "groq_gpt_oss": {  # giữ nguyên, đang chạy tốt
        "provider": "openai_compatible",
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1500,
        "enabled": False,
    },
        "groq_llama_8b": {   # ← model mới, thay thế slot của groq_qwen
        "provider": "openai_compatible",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enabled": True,
    },
    "openrouter_free": {
        "provider": "openai_compatible",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1000,
        "enabled": False,
    },
    # =========================
    # OLLAMA LOCAL
    # =========================
    # "qwen_local": {
    #     "provider": "ollama",
    #     "model": "qwen3:4b",
    #     "base_url": "http://localhost:11434",
    #     "temperature": 0.7,
    #     "max_tokens": 500,
    #     "enabled": False,
    # },
}

SYSTEM_PROMPT = """
Bạn là một hệ thống tạo dữ liệu tổng hợp cho hồ sơ bệnh án điện tử, văn bản y khoa dạng tự do.

Nhiệm vụ của bạn là viết một văn bản lâm sàng tiếng Việt dựa trên
các ràng buộc được cung cấp.

Các ràng buộc có thể chứa nhiều khái niệm y khoa như chẩn đoán,
triệu chứng, thuốc, xét nghiệm, kết quả xét nghiệm và trạng thái
ngữ cảnh của thực thể.

Yêu cầu:
- Phải giữ nguyên chính xác tất cả chuỗi thực thể được cung cấp.
- Không bỏ sót thực thể.
- Không tự thêm thực thể y khoa mới.
- Không thay tên thuốc, bệnh hoặc xét nghiệm bằng từ đồng nghĩa.
- Phải thể hiện đúng trạng thái hiện tại, tiền sử hoặc phủ định.
- Không viết mã ICD-10 hoặc RxNorm.
- Không giải thích nhiệm vụ.
- Không dùng markdown.
- Chỉ trả về văn bản bệnh án.
""".strip()


def call_openai_compatible(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    api_key_name = config["api_key_env"]
    api_key = os.getenv(api_key_name)

    if not api_key:
        raise RuntimeError(
            f"Chưa thiết lập biến môi trường {api_key_name}"
        )

    url = (
        config["base_url"].rstrip("/")
        + "/chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if "openrouter.ai" in config["base_url"]:
        headers["HTTP-Referer"] = os.getenv(
            "OPENROUTER_SITE_URL",
            "http://localhost",
        )
        headers["X-Title"] = os.getenv(
            "OPENROUTER_APP_NAME",
            "medical-synthetic-data",
        )

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 1000),
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()

    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not text:
        raise ValueError(
            f"Model {config['model']} trả về nội dung rỗng"
        )

    return text


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    from google import genai
    from google.genai import types

    api_key_name = config["api_key_env"]
    api_key = os.getenv(api_key_name)

    if not api_key:
        raise RuntimeError(
            f"Chưa thiết lập biến môi trường {api_key_name}"
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=config["model"],
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get(
                "max_tokens",
                1000,
            ),
        ),
    )

    text = getattr(response, "text", None)

    if not text or not text.strip():
        raise ValueError(
            f"Model {config['model']} trả về nội dung rỗng"
        )

    return text.strip()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_key: str,
) -> str:
    if model_key not in LLM_MODELS:
        raise KeyError(
            f"Không tồn tại model_key: {model_key}"
        )

    config = LLM_MODELS[model_key]

    if not config.get("enabled", True):
        raise RuntimeError(
            f"Model {model_key} đang bị tắt"
        )

    provider = config["provider"]

    if provider == "gemini":
        return call_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )

    if provider == "openai_compatible":
        return call_openai_compatible(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )

    raise ValueError(
        f"Provider chưa được hỗ trợ: {provider}"
    )


# =====================================================================
# 9. ORCHESTRATOR
# =====================================================================

def generate_sample(
    note_id: str,
    constraint: dict[str, Any],
    model_key: str,
    max_retry: int = 3,
) -> dict[str, Any]:
    user_prompt = build_prompt(constraint)
    last_errors: list[str] = []

    for attempt in range(1, max_retry + 1):
        started_at = time.perf_counter()

        try:
            text = call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model_key=model_key,
            )

            latency_seconds = (
                time.perf_counter() - started_at
            )

            entities, validation_errors = (
                align_and_validate(
                    generated_text=text,
                    constraint=constraint,
                )
            )
            schema_errors = validate_output_schema(
                text,
                entities,
            )

            errors = [*validation_errors,*schema_errors,]

            if errors:
                last_errors = [
                    str(error)
                    for error in errors
                ]
                continue

            model_config = LLM_MODELS[model_key]

            return {
                "success": True,
                "note_id": note_id,
                "text": text,
                "entities": entities,
                "constraint": constraint,
                "meta": {
                    "source": "synthetic",
                    "pipeline_version": "v2.0",
                    "model_key": model_key,
                    "model_name": model_config["model"],
                    "provider": model_config["provider"],
                    "generation_attempt": attempt,
                    "latency_seconds": round(
                        latency_seconds,
                        3,
                    ),
                    "only_verified_ontology_labels": True,
                    "not_for_clinical_use": True,
                },
            }

        except Exception as exc:
            last_errors = [str(exc)]

            if attempt < max_retry:
                time.sleep(2 ** (attempt - 1))

    model_config = LLM_MODELS[model_key]

    return {
        "success": False,
        "note_id": note_id,
        "constraint": constraint,
        "error": {
            "model_key": model_key,
            "model_name": model_config["model"],
            "provider": model_config["provider"],
            "attempts": max_retry,
            "messages": last_errors,
        },
    }


from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any
import json


def run_generation(
    n: int,
    pool: dict[str, Any],
    max_retry: int = 3,
    model_keys: list[str] | None = None,
) -> None:
    """
    Sinh dữ liệu synthetic và ghi nối tiếp vào các file:

    - contents.jsonl: văn bản đầu vào
    - labels.jsonl: nhãn entity
    - generation_failures.jsonl: mẫu lỗi
    - benchmark_history.jsonl: lịch sử từng lần chạy

    Mỗi lần chạy có một run_id riêng để tránh trùng note_id.
    contents.jsonl và labels.jsonl liên kết bằng note_id.
    """
    if n <= 0:
        raise ValueError("n phải lớn hơn 0.")

    output_dir = WORK / "generated"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contents_path = output_dir / "contents.jsonl"
    labels_path = output_dir / "labels.jsonl"
    failures_path = (
        output_dir / "generation_failures.jsonl"
    )

    # JSONL để có thể append từng lần benchmark
    benchmark_path = (
        output_dir / "benchmark_history.jsonl"
    )

    # Mỗi lần chạy có ID riêng, tránh note_id bị trùng
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    if model_keys is None:
        model_keys = [
            model_key
            for model_key, config
            in LLM_MODELS.items()
            if config.get("enabled", True)
        ]

    if not model_keys:
        raise ValueError(
            "Không có model nào được bật "
            "trong LLM_MODELS."
        )

    unknown_models = [
        model_key
        for model_key in model_keys
        if model_key not in LLM_MODELS
    ]

    if unknown_models:
        raise KeyError(
            f"Không tìm thấy các model: "
            f"{unknown_models}"
        )

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
            "failed": 0,
            "latencies": [],
        }
    )

    total_success = 0
    total_failed = 0

    # Dùng append mode "a" thay vì write mode "w"
    with (
        contents_path.open(
            "a",
            encoding="utf-8",
        ) as contents_file,
        labels_path.open(
            "a",
            encoding="utf-8",
        ) as labels_file,
        failures_path.open(
            "a",
            encoding="utf-8",
        ) as failures_file,
    ):
        for constraint_index in range(n):
            constraint_id = (
                f"{run_id}_"
                f"constraint_{constraint_index:05d}"
            )

            # Một constraint chung cho các model
            # để benchmark công bằng
            constraint = generate_constraint(pool)

            for model_key in model_keys:
                note_id = (
                    f"{constraint_id}_{model_key}"
                )

                result = generate_sample(
                    note_id=note_id,
                    constraint=constraint,
                    model_key=model_key,
                    max_retry=max_retry,
                )

                stats[model_key]["total"] += 1

                if result["success"]:
                    content_record = {
                        "note_id": result["note_id"],
                        "text": result["text"],
                    }

                    label_record = {
                        "note_id": result["note_id"],
                        "entities": result["entities"],
                    }

                    contents_file.write(
                        json.dumps(
                            content_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    labels_file.write(
                        json.dumps(
                            label_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # Đẩy dữ liệu xuống ổ đĩa ngay
                    contents_file.flush()
                    labels_file.flush()

                    stats[model_key]["success"] += 1

                    stats[model_key][
                        "latencies"
                    ].append(
                        result["meta"][
                            "latency_seconds"
                        ]
                    )

                    total_success += 1

                else:
                    failure_record = {
                        **result,
                        "run_id": run_id,
                    }

                    failures_file.write(
                        json.dumps(
                            failure_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    failures_file.flush()

                    stats[model_key]["failed"] += 1
                    total_failed += 1

                status = (
                    "OK"
                    if result["success"]
                    else "FAILED"
                )

                print(
                    f"[{constraint_index + 1}/{n}] "
                    f"{model_key}: {status}"
                )

    benchmark_results: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "requested_constraints": n,
        "models": model_keys,
        "maximum_possible_samples": (
            n * len(model_keys)
        ),
        "successful_samples": total_success,
        "failed_samples": total_failed,
        "output_files": {
            "contents": str(contents_path),
            "labels": str(labels_path),
            "failures": str(failures_path),
        },
        "model_results": {},
    }

    for model_key, model_stats in stats.items():
        latencies = model_stats["latencies"]

        average_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else None
        )

        success_rate = (
            model_stats["success"]
            / model_stats["total"]
            if model_stats["total"]
            else 0.0
        )

        benchmark_results[
            "model_results"
        ][model_key] = {
            "model_name": (
                LLM_MODELS[model_key]["model"]
            ),
            "provider": (
                LLM_MODELS[model_key]["provider"]
            ),
            "total": model_stats["total"],
            "success": model_stats["success"],
            "failed": model_stats["failed"],
            "success_rate": round(
                success_rate,
                4,
            ),
            "average_latency_seconds": (
                round(average_latency, 3)
                if average_latency is not None
                else None
            ),
        }

    # Benchmark cũng append từng dòng thay vì ghi đè
    with benchmark_path.open(
        "a",
        encoding="utf-8",
    ) as benchmark_file:
        benchmark_file.write(
            json.dumps(
                benchmark_results,
                ensure_ascii=False,
            )
            + "\n"
        )

    print("\n=== HOÀN TẤT GENERATION ===")
    print(f"Run ID: {run_id}")
    print(f"Số constraint: {n}")
    print(f"Số model: {len(model_keys)}")
    print(
        "Số sample tối đa: "
        f"{n * len(model_keys)}"
    )
    print(f"Sample thành công: {total_success}")
    print(f"Sample thất bại: {total_failed}")
    print(f"Content: {contents_path}")
    print(f"Labels: {labels_path}")
    print(f"Failures: {failures_path}")
    print(f"Benchmark: {benchmark_path}")

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "survey"

    if cmd == "survey":
        records = load_vimedner()
        survey_labels(records)

    elif cmd == "build_pool":
        build_ontology_pool()

    elif cmd == "generate":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        pool = json.load(open(WORK / "ontology_pool_final.json", encoding="utf-8"))
        run_generation(n, pool)

    else:
        print("Dùng: python build_synthetic_data.py [survey|build_pool|generate N]")