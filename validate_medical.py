"""
=====================================================================
VALIDATION GATE — dùng model y khoa riêng để double-check synthetic
note sau khi build_synthetic_data.py generate đã chạy xong.

Đọc work/generated/contents.jsonl + labels.jsonl, gửi từng note cho
1 model y khoa (mặc định: Ollama local) để kiểm tra:
    1. Tính hợp lý lâm sàng (triệu chứng/chẩn đoán/thuốc có phi lý không)
    2. Assertion (present/isHistorical/absent) có đúng nghĩa thật không
    3. Không có claim y khoa bị bịa thêm ngoài constraint

CHẠY:
    python validate_medical.py run          # chạy full validation
    python validate_medical.py run --limit 50   # test thử 50 note đầu

OUTPUT:
    work/validated/validated_pass.jsonl      # note pass, dùng để train
    work/validated/validated_flagged.jsonl   # note bị flag kèm lý do
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
# Mặc định dùng Ollama local (miễn phí, chạy offline). Đổi "model" theo
# tên bạn đã pull, ví dụ: meditron, medllama2, alibayram/medgemma...
#
# Nếu muốn dùng model qua API (Groq/OpenRouter) thay vì local, đổi
# "base_url" + "api_key_env" tương ứng — code call_medical_model dùng
# chung format OpenAI-compatible nên không cần sửa logic gọi.

MEDICAL_MODEL_CONFIG = {
    "provider": "ollama_openai_compatible",
    "model": os.getenv("MEDICAL_MODEL", "meditron"),
    "base_url": os.getenv("MEDICAL_MODEL_BASE_URL", "http://localhost:11434/v1"),
    "api_key_env": "OLLAMA_FAKE_KEY",  # ollama không check key, chỉ cần biến tồn tại
    "temperature": 0.0,  # cần độ ổn định cao cho việc chấm điểm, không cần sáng tạo
    "max_tokens": 800,
    "timeout": 120,
}

VALIDATION_SYSTEM_PROMPT = """
Bạn là một bác sĩ chuyên khoa, nhiệm vụ là kiểm tra tính hợp lý lâm
sàng của một đoạn bệnh án tổng hợp (synthetic) dùng để huấn luyện mô
hình NER y tế.

Bạn KHÔNG đánh giá văn phong, KHÔNG đánh giá chính tả. Bạn chỉ đánh
giá 3 tiêu chí lâm sàng dưới đây, dựa trên đoạn văn và danh sách
entity kèm theo.

Tiêu chí:
1. clinically_plausible: Tổ hợp triệu chứng + chẩn đoán + thuốc có
   hợp lý về mặt y khoa không? (ví dụ: thuốc dùng sai chỉ định hoàn
   toàn không liên quan tới chẩn đoán là KHÔNG hợp lý)
2. assertion_correct: Với mỗi thuốc có assertion (present/isHistorical
   /absent), ngữ cảnh trong câu có thực sự thể hiện đúng trạng thái đó
   không? (không chỉ dựa vào từ khóa, mà dựa vào NGHĨA thật của câu)
3. no_fabricated_claims: Đoạn văn có bịa thêm thông tin y khoa nghiêm
   trọng nào ngoài các entity đã cho không? (thêm 1-2 câu mô tả bối
   cảnh khám bệnh là bình thường, KHÔNG tính là bịa đặt)

CHỈ trả lời bằng JSON đúng format sau, không thêm text nào khác,
không dùng markdown code fence:

{
  "clinically_plausible": true hoặc false,
  "assertion_correct": true hoặc false,
  "no_fabricated_claims": true hoặc false,
  "issues": ["mô tả ngắn gọn từng vấn đề nếu có, để trống nếu không có vấn đề"]
}
""".strip()


def build_validation_prompt(note_text: str, entities: list[dict]) -> str:
    entity_lines = []
    for ent in entities:
        if ent["type"] == "THUỐC":
            entity_lines.append(
                f"- THUỐC: '{ent['text']}' | assertion: {ent.get('assertion')}"
            )
        elif ent["type"] == "CHẨN_ĐOÁN":
            entity_lines.append(f"- CHẨN_ĐOÁN: '{ent['text']}'")
        elif ent["type"] == "TRIỆU_CHỨNG":
            entity_lines.append(f"- TRIỆU_CHỨNG: '{ent['text']}'")

    entity_block = "\n".join(entity_lines) if entity_lines else "(không có)"

    return f"""Đoạn bệnh án cần kiểm tra:
---
{note_text}
---

Danh sách entity đã được gắn nhãn:
{entity_block}

Hãy đánh giá theo 3 tiêu chí đã nêu và trả lời đúng format JSON."""


# =====================================================================
# 2. GỌI MODEL Y KHOA (OpenAI-compatible, dùng chung cho Ollama/API)
# =====================================================================

def call_medical_model(system_prompt: str, user_prompt: str, config: dict[str, Any]) -> str:
    api_key = os.getenv(config["api_key_env"], "not_needed")

    url = config["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not_needed":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 800),
    }

    response = requests.post(url, headers=headers, json=payload, timeout=config.get("timeout", 120))
    response.raise_for_status()
    data = response.json()

    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not text:
        raise ValueError("Medical model trả về nội dung rỗng")
    return text


def parse_validation_json(raw_text: str) -> dict[str, Any]:
    """Model đôi khi bọc markdown fence hoặc thêm text thừa quanh JSON.
    Cố gắng trích JSON object đầu tiên tìm được."""
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Không tìm thấy JSON hợp lệ trong output: {raw_text[:200]}")
    return json.loads(match.group(0))


# =====================================================================
# 3. VALIDATE 1 NOTE
# =====================================================================

def validate_note(note_id: str, note_text: str, entities: list[dict], max_retry: int = 2) -> dict[str, Any]:
    prompt = build_validation_prompt(note_text, entities)
    last_error = None

    for attempt in range(1, max_retry + 1):
        try:
            raw = call_medical_model(VALIDATION_SYSTEM_PROMPT, prompt, MEDICAL_MODEL_CONFIG)
            verdict = parse_validation_json(raw)

            required_keys = {"clinically_plausible", "assertion_correct", "no_fabricated_claims"}
            if not required_keys.issubset(verdict.keys()):
                raise ValueError(f"Thiếu key bắt buộc trong verdict: {verdict}")

            passed = (
                bool(verdict["clinically_plausible"])
                and bool(verdict["assertion_correct"])
                and bool(verdict["no_fabricated_claims"])
            )

            return {
                "note_id": note_id,
                "passed": passed,
                "verdict": verdict,
                "attempt": attempt,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retry:
                time.sleep(1)

    # Hết retry mà vẫn lỗi -> flag để xem thủ công, không tự động pass
    return {
        "note_id": note_id,
        "passed": False,
        "verdict": None,
        "error": last_error,
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
    report_path = VALIDATED_DIR / "validation_report.json"

    total_pass = 0
    total_flagged = 0
    issue_counter: dict[str, int] = {}

    with (
        pass_path.open("w", encoding="utf-8") as pass_file,
        flagged_path.open("w", encoding="utf-8") as flagged_file,
    ):
        for i, note_id in enumerate(note_ids, 1):
            item = data[note_id]
            result = validate_note(note_id, item["text"], item["entities"])

            if result["passed"]:
                pass_file.write(
                    json.dumps(
                        {
                            "note_id": note_id,
                            "text": item["text"],
                            "entities": item["entities"],
                            "meta": item["meta"],
                            "medical_validation": result["verdict"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_pass += 1
            else:
                flagged_file.write(
                    json.dumps(
                        {
                            "note_id": note_id,
                            "text": item["text"],
                            "entities": item["entities"],
                            "meta": item["meta"],
                            "medical_validation": result.get("verdict"),
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_flagged += 1

                if result.get("verdict"):
                    for issue in result["verdict"].get("issues", []):
                        issue_counter[issue] = issue_counter.get(issue, 0) + 1

            status = "PASS" if result["passed"] else "FLAGGED"
            print(f"[{i}/{len(note_ids)}] {note_id}: {status}")

    report = {
        "total_checked": len(note_ids),
        "total_pass": total_pass,
        "total_flagged": total_flagged,
        "pass_rate": round(total_pass / len(note_ids), 4) if note_ids else 0.0,
        "medical_model": MEDICAL_MODEL_CONFIG["model"],
        "top_issues": sorted(issue_counter.items(), key=lambda x: -x[1])[:20],
        "output_files": {
            "pass": str(pass_path),
            "flagged": str(flagged_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== HOÀN TẤT VALIDATION ===")
    print(f"Tổng kiểm tra: {len(note_ids)}")
    print(f"Pass: {total_pass} ({report['pass_rate']*100:.1f}%)")
    print(f"Flagged: {total_flagged}")
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