"""
=====================================================================
VALIDATION GATE — dùng model y khoa riêng để double-check synthetic
note sau khi build_synthetic_data.py generate đã chạy xong.

Đọc work/generated/contents.jsonl + labels.jsonl, gửi từng note cho
1 model chấm y khoa qua API OpenAI-compatible để kiểm tra:
    1. Tính hợp lý lâm sàng (triệu chứng/chẩn đoán/thuốc có phi lý không)
    2. Assertion (isHistorical/isNegated/isFamily) có đúng nghĩa thật không
    3. Không có claim y khoa bị bịa thêm ngoài constraint

CHẠY:
    python validate_medical.py run          # chạy full validation
    python validate_medical.py run --limit 50   # test thử 50 note đầu

OUTPUT:
    work/validated/validated_pass.jsonl      # note pass, dùng để train
    work/validated/validated_flagged.jsonl   # note bị flag kèm lý do
    work/validated/validation_errors.jsonl   # lỗi API/kỹ thuật, không kết luận note sai
    work/validated/validation_report.json    # thống kê tổng
=====================================================================
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

WORK = Path("work")
GENERATED_DIR = WORK / "generated"
VALIDATED_DIR = WORK / "validated"
VALIDATED_DIR.mkdir(exist_ok=True, parents=True)


# =====================================================================
# 1. CẤU HÌNH MODEL Y KHOA
# =====================================================================
# Mặc định dùng Groq. Có thể đổi provider/model qua .env miễn là API hỗ trợ
# endpoint OpenAI-compatible /chat/completions.

MEDICAL_MODEL_CONFIG = {
    "provider": "openai_compatible",
    "model": os.getenv(
        "VALIDATOR_MODEL",
        "openai/gpt-oss-120b",
    ),
    "base_url": os.getenv(
        "VALIDATOR_BASE_URL",
        "https://api.groq.com/openai/v1",
    ),
    "api_key_env": "VALIDATOR_API_KEY",
    "temperature": 0.0,
    "max_tokens": 500,
    "timeout": 60,
}

ALLOWED_ENTITY_TYPES = {
    "CHẨN_ĐOÁN",
    "TRIỆU_CHỨNG",
    "THUỐC",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
}

ALLOWED_ASSERTIONS = {
    "isHistorical",
    "isNegated",
    "isFamily",
}

BAD_OUTPUT_PATTERNS = [
    r"```",
    r"^\s*dưới đây là",
    r"^\s*đoạn bệnh án",
    r"^\s*theo yêu cầu",
    r"^\s*nội dung bắt buộc",
    r"^\s*constraint",
]


def rule_based_validate(
    note_text: str,
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    hard_errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(note_text, str):
        return {
            "passed": False,
            "hard_errors": ["Text không phải chuỗi"],
            "warnings": [],
        }

    note_text = note_text.strip()

    if not note_text:
        return {
            "passed": False,
            "hard_errors": ["Văn bản rỗng"],
            "warnings": [],
        }

    if len(note_text) < 30:
        warnings.append("Văn bản rất ngắn")

    if not entities:
        hard_errors.append("Không có entity")

    for pattern in BAD_OUTPUT_PATTERNS:
        if re.search(pattern, note_text, re.I):
            hard_errors.append(
                f"Output chứa dấu hiệu model giải thích hoặc copy prompt: {pattern}"
            )

    seen_entities: set[tuple] = set()
    occupied: list[tuple[int, int, str]] = []

    for index, entity in enumerate(entities):
        prefix = f"Entity #{index}"

        entity_type = entity.get("type")
        entity_text = entity.get("text")
        position = entity.get("position")

        if entity_type not in ALLOWED_ENTITY_TYPES:
            hard_errors.append(
                f"{prefix}: type không hợp lệ: {entity_type}"
            )

        if not isinstance(entity_text, str) or not entity_text:
            hard_errors.append(
                f"{prefix}: text entity rỗng"
            )

        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(isinstance(value, int) for value in position)
        ):
            hard_errors.append(
                f"{prefix}: position phải là [start, end] integer"
            )
            continue

        start, end = position

        if start < 0 or end <= start or end > len(note_text):
            hard_errors.append(
                f"{prefix}: span [{start}, {end}] không hợp lệ"
            )
            continue

        actual_text = note_text[start:end]

        if actual_text != entity_text:
            hard_errors.append(
                f"{prefix}: span mismatch, "
                f"label='{entity_text}', actual='{actual_text}'"
            )

        entity_key = (
            entity_type,
            start,
            end,
            entity_text,
        )

        if entity_key in seen_entities:
            hard_errors.append(
                f"{prefix}: entity bị trùng hoàn toàn"
            )
        else:
            seen_entities.add(entity_key)

        for old_start, old_end, old_type in occupied:
            is_overlap = (
                start < old_end
                and end > old_start
            )

            if is_overlap:
                warnings.append(
                    f"{prefix}: overlap với {old_type} "
                    f"[{old_start}, {old_end}]"
                )

        occupied.append(
            (start, end, entity_type)
        )

        assertions = entity.get("assertions", [])
        if not isinstance(assertions, list):
            hard_errors.append(
                f"{prefix}: assertions phải là list"
            )
        else:
            unknown_assertions = set(assertions) - ALLOWED_ASSERTIONS
            if unknown_assertions:
                hard_errors.append(
                    f"{prefix}: assertion không hợp lệ: "
                    f"{sorted(unknown_assertions)}"
                )
            if len(assertions) != len(set(assertions)):
                hard_errors.append(
                    f"{prefix}: assertion bị trùng"
                )

        if entity_type in {"CHẨN_ĐOÁN", "THUỐC"}:
            candidates = entity.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                hard_errors.append(
                    f"{prefix}: thiếu candidates chuẩn hóa"
                )

    return {
        "passed": len(hard_errors) == 0,
        "hard_errors": hard_errors,
        "warnings": warnings,
    }

VALIDATION_SYSTEM_PROMPT = """
Bạn là hệ thống kiểm tra dữ liệu bệnh án tổng hợp dùng để huấn luyện
mô hình NLP y tế.

Các kiểm tra cấu trúc, span và mã ontology đã được thực hiện bằng
rule-based validator. Bạn chỉ đánh giá các vấn đề cần hiểu ngữ nghĩa.

Tiêu chí:

1. assertion_correct:
Ngữ cảnh của từng entity có thực sự thể hiện đúng các assertion
isHistorical, isNegated, isFamily hay không. Danh sách assertion rỗng
nghĩa là entity hiện tại, không phủ định và thuộc về bệnh nhân.

2. internally_consistent:
Đoạn bệnh án có mâu thuẫn nội tại nghiêm trọng hay không, ví dụ cùng
một thuốc vừa được mô tả là đang sử dụng vừa được mô tả là không sử
dụng trong cùng ngữ cảnh.

3. clinically_plausible:
Tổ hợp thông tin có hoàn toàn phi lý về lâm sàng hay không. Chỉ đánh
false đối với lỗi rõ ràng; không đánh false chỉ vì thiếu thông tin.

4. suspicious_unlabeled_claims:
Văn bản có chứa thêm chẩn đoán, triệu chứng, thuốc hoặc xét nghiệm rõ
ràng nhưng không xuất hiện trong danh sách entity hay không. Đây chỉ
là cảnh báo vì bạn không được cung cấp constraint gốc.

Chỉ trả về một JSON object hợp lệ:

{
  "assertion_correct": true,
  "internally_consistent": true,
  "clinically_plausible": true,
  "suspicious_unlabeled_claims": false,
  "confidence": 0.0,
  "issues": []
}
""".strip()


def build_validation_prompt(note_text: str, entities: list[dict]) -> str:
    entity_lines = []
    for ent in entities:
        assertions = ent.get("assertions", [])
        assertion_text = ", ".join(assertions) if assertions else "present"
        candidate_text = ", ".join(ent.get("candidates", [])) or "-"
        entity_lines.append(
            f"- {ent.get('type')}: '{ent.get('text')}' | "
            f"assertions: {assertion_text} | candidates: {candidate_text}"
        )

    entity_block = "\n".join(entity_lines) if entity_lines else "(không có)"

    return f"""Đoạn bệnh án cần kiểm tra:
---
{note_text}
---

Danh sách entity đã được gắn nhãn:
{entity_block}

Hãy đánh giá theo 4 tiêu chí đã nêu và trả lời đúng format JSON."""


VALIDATION_JSON_SCHEMA = {
    "name": "medical_validation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "assertion_correct": {"type": "boolean"},
            "internally_consistent": {"type": "boolean"},
            "clinically_plausible": {"type": "boolean"},
            "suspicious_unlabeled_claims": {"type": "boolean"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "assertion_correct",
            "internally_consistent",
            "clinically_plausible",
            "suspicious_unlabeled_claims",
            "confidence",
            "issues",
        ],
        "additionalProperties": False,
    },
}


def response_format_for_model(model: str) -> dict[str, Any]:
    """Dùng strict schema khi model Groq hỗ trợ, còn lại ép JSON object."""
    if model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        return {
            "type": "json_schema",
            "json_schema": VALIDATION_JSON_SCHEMA,
        }
    return {"type": "json_object"}


# =====================================================================
# 2. GỌI MODEL Y KHOA (OpenAI-compatible, dùng chung cho Ollama/API)
# =====================================================================

def call_medical_model(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    api_key_env = config["api_key_env"]
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise RuntimeError(
            f"Chưa thiết lập biến môi trường "
            f"{api_key_env}"
        )

    url = (
        config["base_url"].rstrip("/")
        + "/chat/completions"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

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
        "temperature": config.get(
            "temperature",
            0.0,
        ),
        "max_tokens": config.get(
            "max_tokens",
            500,
        ),
        "response_format": response_format_for_model(config["model"]),
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=config.get("timeout", 60),
    )

    if not response.ok:
        raise RuntimeError(
            f"API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not text:
        raise ValueError(
            "Medical model trả về nội dung rỗng"
        )

    return text


def parse_validation_json(raw_text: str) -> dict[str, Any]:
    """Model đôi khi bọc markdown fence hoặc thêm text thừa quanh JSON.
    Cố gắng trích JSON object đầu tiên tìm được."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Không tìm thấy JSON hợp lệ trong output: {raw_text[:200]}")
    return json.loads(match.group(0))


def validate_verdict(verdict: Any) -> dict[str, Any]:
    if not isinstance(verdict, dict):
        raise ValueError("Verdict không phải JSON object")

    boolean_keys = {
        "assertion_correct",
        "internally_consistent",
        "clinically_plausible",
        "suspicious_unlabeled_claims",
    }
    required_keys = boolean_keys | {"confidence", "issues"}
    missing_keys = required_keys - set(verdict)
    if missing_keys:
        raise ValueError(f"Verdict thiếu key: {sorted(missing_keys)}")

    invalid_boolean_keys = [
        key for key in boolean_keys
        if not isinstance(verdict[key], bool)
    ]
    if invalid_boolean_keys:
        raise ValueError(
            f"Verdict có field không phải boolean: {invalid_boolean_keys}"
        )

    confidence = verdict["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence phải là số trong khoảng [0, 1]")

    issues = verdict["issues"]
    if not isinstance(issues, list) or not all(
        isinstance(issue, str) for issue in issues
    ):
        raise ValueError("issues phải là list string")

    return verdict


# =====================================================================
# 3. VALIDATE 1 NOTE
# =====================================================================

def validate_note(
    note_id: str,
    note_text: str,
    entities: list[dict[str, Any]],
    max_retry: int = 2,
    use_api_judge: bool = True,
) -> dict[str, Any]:
    rule_result = rule_based_validate(
        note_text=note_text,
        entities=entities,
    )

    # Lỗi chắc chắn: loại ngay, không gọi API
    if not rule_result["passed"]:
        return {
            "note_id": note_id,
            "passed": False,
            "status": "REJECT",
            "rule_validation": rule_result,
            "semantic_validation": None,
            "reason": "hard_rule_failed",
        }

    # Chế độ chỉ kiểm tra bằng rule
    if not use_api_judge:
        return {
            "note_id": note_id,
            "passed": True,
            "status": "PASS",
            "rule_validation": rule_result,
            "semantic_validation": None,
            "reason": "rule_only_pass",
        }

    prompt = build_validation_prompt(
        note_text,
        entities,
    )

    last_error = None

    for attempt in range(1, max_retry + 1):
        try:
            raw = call_medical_model(
                VALIDATION_SYSTEM_PROMPT,
                prompt,
                MEDICAL_MODEL_CONFIG,
            )

            verdict = validate_verdict(parse_validation_json(raw))
            confidence = float(verdict["confidence"])

            serious_semantic_error = (
                not bool(verdict["assertion_correct"])
                or not bool(
                    verdict["internally_consistent"]
                )
                or not bool(
                    verdict["clinically_plausible"]
                )
            )

            # Chỉ reject khi model đủ tự tin
            if (
                serious_semantic_error
                and confidence >= 0.85
            ):
                status = "REJECT"
                passed = False

            elif (
                bool(
                    verdict[
                        "suspicious_unlabeled_claims"
                    ]
                )
                or confidence < 0.70
                or bool(rule_result["warnings"])
            ):
                status = "FLAG_REVIEW"
                passed = False

            else:
                status = "PASS"
                passed = True

            return {
                "note_id": note_id,
                "passed": passed,
                "status": status,
                "rule_validation": rule_result,
                "semantic_validation": verdict,
                "attempt": attempt,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < max_retry:
                time.sleep(
                    2 ** (attempt - 1)
                )

    # API timeout/lỗi JSON không chứng minh dữ liệu sai
    return {
        "note_id": note_id,
        "passed": False,
        "status": "ERROR",
        "rule_validation": rule_result,
        "semantic_validation": None,
        "error": last_error,
        "reason": "api_judge_failed",
        "attempt": max_retry,
    }

# =====================================================================
# 4. ORCHESTRATOR
# =====================================================================

def load_generated_data() -> dict[str, dict[str, Any]]:
    """Merge contents.jsonl + labels.jsonl theo note_id."""
    contents_path = GENERATED_DIR / "contents.jsonl"
    labels_path = GENERATED_DIR / "labels.jsonl"

    if not contents_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {contents_path} hoặc {labels_path}. "
            "Hãy chạy build_synthetic_data.py generate trước."
        )

    contents = {}
    with contents_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            contents[obj["note_id"]] = obj["text"]

    merged = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            note_id = obj["note_id"]
            if note_id in contents:
                merged[note_id] = {
                    "text": contents[note_id],
                    "entities": obj["entities"],
                    "meta": obj.get("meta", {}),
                }

    return merged


def run_validation(limit: int | None = None) -> None:
    data = load_generated_data()
    note_ids = list(data.keys())
    if limit:
        note_ids = note_ids[:limit]

    pass_path = VALIDATED_DIR / "validated_pass.jsonl"
    flagged_path = VALIDATED_DIR / "validated_flagged.jsonl"
    errors_path = VALIDATED_DIR / "validation_errors.jsonl"
    report_path = VALIDATED_DIR / "validation_report.json"

    total_pass = 0
    total_flagged = 0
    total_errors = 0
    status_counter: dict[str, int] = {}
    issue_counter: dict[str, int] = {}

    with (
        pass_path.open("w", encoding="utf-8") as pass_file,
        flagged_path.open("w", encoding="utf-8") as flagged_file,
        errors_path.open("w", encoding="utf-8") as errors_file,
    ):
        for i, note_id in enumerate(note_ids, 1):
            item = data[note_id]
            result = validate_note(note_id, item["text"], item["entities"])

            output_record = {
                "note_id": note_id,
                "text": item["text"],
                "entities": item["entities"],
                "meta": item["meta"],
                "validation_status": result["status"],
                "rule_validation": result["rule_validation"],
                "semantic_validation": result.get("semantic_validation"),
                "error": result.get("error"),
                "reason": result.get("reason"),
            }

            status_counter[result["status"]] = (
                status_counter.get(result["status"], 0) + 1
            )

            if result["status"] == "PASS":
                pass_file.write(
                    json.dumps(output_record, ensure_ascii=False)
                    + "\n"
                )

                total_pass += 1

            elif result["status"] == "ERROR":
                errors_file.write(
                    json.dumps(output_record, ensure_ascii=False)
                    + "\n"
                )
                total_errors += 1

            else:
                flagged_file.write(
                    json.dumps(output_record, ensure_ascii=False)
                    + "\n"
                )

                total_flagged += 1

            semantic_validation = result.get(
                "semantic_validation"
            )

            if semantic_validation:
                for issue in semantic_validation.get(
                    "issues",
                    [],
                ):
                    issue_counter[issue] = (
                        issue_counter.get(issue, 0) + 1
                    )

            print(
                f"[{i}/{len(note_ids)}] "
                f"{note_id}: {result['status']}"
            )

    report = {
        "total_checked": len(note_ids),
        "total_pass": total_pass,
        "total_flagged": total_flagged,
        "total_errors": total_errors,
        "pass_rate": round(total_pass / len(note_ids), 4) if note_ids else 0.0,
        "status_counts": status_counter,
        "medical_model": MEDICAL_MODEL_CONFIG["model"],
        "top_issues": sorted(issue_counter.items(), key=lambda x: -x[1])[:20],
        "output_files": {
            "pass": str(pass_path),
            "flagged": str(flagged_path),
            "errors": str(errors_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== HOÀN TẤT VALIDATION ===")
    print(f"Tổng kiểm tra: {len(note_ids)}")
    print(f"Pass: {total_pass} ({report['pass_rate']*100:.1f}%)")
    print(f"Flagged: {total_flagged}")
    print(f"Lỗi kỹ thuật/API: {total_errors}")
    print(f"Pass file: {pass_path}")
    print(f"Flagged file: {flagged_path}")
    print(f"Report: {report_path}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["run"])
    parser.add_argument("--limit", type=int, default=None, help="Chỉ test N note đầu tiên")
    args = parser.parse_args()

    if args.cmd == "run":
        run_validation(limit=args.limit)
