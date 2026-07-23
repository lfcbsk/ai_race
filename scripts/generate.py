from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from typing import Any
import requests

try:
    from scripts.common import (
        BlockSpec,
        CATALOG_DIR,
        CaseSpec,
        EntitySpec,
        GENERATED_DIR,
        NegativeSpec,
        SectionSpec,
        append_jsonl,
        ensure_directories,
        get_qwen_config,
        read_json,
        read_jsonl,
        validate_case_spec,
        validate_markers,
    )
except ModuleNotFoundError:
    # Support direct execution: python scripts/generate.py
    from common import (
        BlockSpec,
        CATALOG_DIR,
        CaseSpec,
        EntitySpec,
        GENERATED_DIR,
        NegativeSpec,
        SectionSpec,
        append_jsonl,
        ensure_directories,
        get_qwen_config,
        read_json,
        read_jsonl,
        validate_case_spec,
        validate_markers,
    )


QWEN_SYSTEM_PROMPT = """
Bạn là bộ dựng văn bản cho dữ liệu NLP y khoa tổng hợp.

Đầu vào là SECTION_SPEC chứa:
- target entity có marker [[E...]]...[[/E...]];
- hard-negative phrase có marker [[N...]]...[[/N...]];
- scenario lâm sàng;
- kiểu trình bày của section.

Bạn chỉ được viết ngữ cảnh lâm sàng xung quanh các chuỗi
đã được cung cấp.

QUY TẮC TUYỆT ĐỐI:

1. Giữ nguyên chính xác mọi nội dung nằm trong marker.
2. Mỗi marker phải xuất hiện đúng một lần.
3. Không sửa, dịch, rút gọn hoặc thay chính tả entity.
4. Không thêm tên thuốc, bệnh, triệu chứng hoặc xét nghiệm
   mục tiêu ngoài SECTION_SPEC.
5. Không viết mã ICD-10 hoặc RxNorm.
6. Hard-negative phrase phải xuất hiện nhưng không được đổi
   thành target entity.
7. Cue phủ định phải có scope rõ ràng.
8. Nếu các entity có assertion khác nhau, phải tách clause
   hoặc dùng liên từ tương phản rõ.
9. Local cue rõ ràng phải phản ánh đúng scenario.
10. Được phép dùng bullet, fragment hoặc paragraph theo style.
11. Chỉ trả về nội dung của section, không tự viết tiêu đề.
12. Không giải thích, không trả JSON, không dùng markdown fence.
""".strip()


def load_catalogs() -> dict[str, Any]:
    return {
        "drugs": read_jsonl(
            CATALOG_DIR
            / "drug_surfaces.jsonl"
        ),
        "diseases": read_jsonl(
            CATALOG_DIR
            / "disease_surfaces.jsonl"
        ),
        "symptoms": read_jsonl(
            CATALOG_DIR
            / "symptom_surfaces.jsonl"
        ),
        "negatives": read_jsonl(
            CATALOG_DIR
            / "hard_negatives.jsonl"
        ),
        "labs": read_jsonl(
            CATALOG_DIR
            / "lab_tests.jsonl"
        ),
        "scenarios": read_json(
            CATALOG_DIR
            / "assertion_scenarios.json"
        ),
        "document_config": read_json(
            CATALOG_DIR
            / "document_profiles.json"
        ),
    }


def weighted_choice(
    records: dict[str, dict[str, Any]],
) -> str:
    keys = list(records)
    weights = [
        records[key].get("weight", 1)
        for key in keys
    ]

    return random.choices(
        keys,
        weights=weights,
        k=1,
    )[0]


def sample_profile(
    document_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    profiles = document_config["profiles"]
    profile_id = weighted_choice(profiles)

    return profile_id, profiles[profile_id]


def select_section_types(
    profile: dict[str, Any],
) -> list[str]:
    selected = list(
        profile["required_sections"]
    )

    optional = list(
        profile.get("optional_sections", [])
    )

    if optional:
        minimum = profile.get(
            "min_optional",
            0,
        )
        maximum = min(
            profile.get(
                "max_optional",
                len(optional),
            ),
            len(optional),
        )

        count = random.randint(
            minimum,
            maximum,
        )

        selected.extend(
            random.sample(optional, count)
        )

    return selected


def sample_surface(
    catalog: list[dict[str, Any]],
    *,
    require_linking: bool = False,
) -> dict[str, Any]:
    eligible = [
        item
        for item in catalog
        if (
            not require_linking
            or item.get(
                "usable_for_linking",
                False,
            )
        )
    ]

    if not eligible:
        raise ValueError(
            "Không có surface phù hợp."
        )

    return random.choice(eligible)


def build_entity(
    entity_id: str,
    surface: dict[str, Any],
    assertions: list[str],
) -> EntitySpec:
    return EntitySpec(
        entity_id=entity_id,
        surface_text=surface["text"],
        entity_type=surface["entity_type"],
        assertions=list(assertions),
        candidates=list(
            surface.get("candidates", [])
        ),
        surface_id=surface.get("surface_id"),
        concept_id=surface.get(
            "concept_id"
        ),
    )


def sample_lab_value(
    lab: dict[str, Any],
) -> str:
    low, high = lab["normal_range"]
    decimals = int(lab.get("decimals", 1))

    if random.random() < 0.70:
        value = random.uniform(low, high)
    else:
        span = high - low

        if random.random() < 0.5:
            value = random.uniform(
                max(0, low - span * 0.5),
                low,
            )
        else:
            value = random.uniform(
                high,
                high + span * 0.5,
            )

    if decimals == 0:
        return str(round(value))

    return f"{value:.{decimals}f}".replace(
        ".",
        ",",
    )


def build_block(
    block_id: str,
    scenario_id: str,
    scenario: dict[str, Any],
    catalogs: dict[str, Any],
    counters: dict[str, int],
) -> BlockSpec:
    kind = scenario["kind"]

    entity_count = random.randint(
        scenario.get("min_entities", 1),
        scenario.get("max_entities", 1),
    )

    entities: list[EntitySpec] = []

    def next_entity_id() -> str:
        entity_id = (
            f"E{counters['entity']}"
        )
        counters["entity"] += 1
        return entity_id

    if kind == "mixed_polarity":
        first = sample_surface(
            catalogs["symptoms"]
        )
        second = sample_surface(
            catalogs["symptoms"]
        )

        entities.append(
            build_entity(
                next_entity_id(),
                first,
                ["isNegated"],
            )
        )

        entities.append(
            build_entity(
                next_entity_id(),
                second,
                [],
            )
        )

    elif kind == "medication_group":
        selected: list[
            dict[str, Any]
        ] = []

        attempts = 0

        while (
            len(selected) < entity_count
            and attempts < 50
        ):
            attempts += 1

            candidate = sample_surface(
                catalogs["drugs"],
                require_linking=True,
            )

            key = (
                candidate["concept_id"],
                candidate["text"],
            )

            if any(
                (
                    item["concept_id"],
                    item["text"],
                )
                == key
                for item in selected
            ):
                continue

            selected.append(candidate)

        for surface in selected:
            entities.append(
                build_entity(
                    next_entity_id(),
                    surface,
                    scenario["assertions"],
                )
            )

    elif kind == "laboratory_group":
        selected_labs = random.sample(
            catalogs["labs"],
            k=min(
                entity_count,
                len(catalogs["labs"]),
            ),
        )

        for lab in selected_labs:
            test_surface = {
                "text": lab["test_name"],
                "entity_type": (
                    "TÊN_XÉT_NGHIỆM"
                ),
                "candidates": [],
                "surface_id": None,
                "concept_id": None,
            }

            result_value = sample_lab_value(
                lab
            )

            result_surface = {
                "text": (
                    f"{result_value} "
                    f"{lab['unit']}"
                ),
                "entity_type": (
                    "KẾT_QUẢ_XÉT_NGHIỆM"
                ),
                "candidates": [],
                "surface_id": None,
                "concept_id": None,
            }

            entities.append(
                build_entity(
                    next_entity_id(),
                    test_surface,
                    [],
                )
            )

            entities.append(
                build_entity(
                    next_entity_id(),
                    result_surface,
                    [],
                )
            )

    else:
        allowed_types = scenario[
            "allowed_types"
        ]

        for _ in range(entity_count):
            entity_type = random.choice(
                allowed_types
            )

            if entity_type == "TRIỆU_CHỨNG":
                surface = sample_surface(
                    catalogs["symptoms"]
                )
            elif entity_type == "CHẨN_ĐOÁN":
                surface = sample_surface(
                    catalogs["diseases"],
                    require_linking=True,
                )
            elif entity_type == "THUỐC":
                surface = sample_surface(
                    catalogs["drugs"],
                    require_linking=True,
                )
            else:
                raise ValueError(
                    f"Unsupported type: "
                    f"{entity_type}"
                )

            entities.append(
                build_entity(
                    next_entity_id(),
                    surface,
                    scenario.get(
                        "assertions",
                        [],
                    ),
                )
            )

    hard_negatives: list[NegativeSpec] = []

    if (
        catalogs["negatives"]
        and random.random() < 0.30
        and kind != "medication_group"
    ):
        negative = random.choice(
            catalogs["negatives"]
        )

        negative_id = (
            f"N{counters['negative']}"
        )
        counters["negative"] += 1

        hard_negatives.append(
            NegativeSpec(
                negative_id=negative_id,
                surface_text=negative["text"],
                negative_type=negative[
                    "negative_type"
                ],
            )
        )

    return BlockSpec(
        block_id=block_id,
        scenario_id=scenario_id,
        entities=entities,
        hard_negatives=hard_negatives,
        instructions=[
            scenario["instruction"]
        ],
    )


def sample_case_spec(
    case_id: str,
    catalogs: dict[str, Any],
) -> CaseSpec:
    profile_id, profile = sample_profile(
        catalogs["document_config"]
    )

    section_types = select_section_types(
        profile
    )

    section_library = catalogs[
        "document_config"
    ]["sections"]

    counters = {
        "entity": 0,
        "negative": 0,
    }

    sections: list[SectionSpec] = []

    for section_index, section_type in enumerate(
        section_types
    ):
        section_config = section_library[
            section_type
        ]

        scenario_id = random.choice(
            section_config[
                "allowed_scenarios"
            ]
        )

        scenario = catalogs[
            "scenarios"
        ][scenario_id]

        block = build_block(
            block_id=(
                f"B{section_index}"
            ),
            scenario_id=scenario_id,
            scenario=scenario,
            catalogs=catalogs,
            counters=counters,
        )

        sections.append(
            SectionSpec(
                section_id=(
                    f"S{section_index}"
                ),
                section_type=section_type,
                title=random.choice(
                    section_config["titles"]
                ),
                temporal_scope=(
                    section_config[
                        "temporal_scope"
                    ]
                ),
                subject_scope=(
                    section_config[
                        "subject_scope"
                    ]
                ),
                render_style=(
                    section_config[
                        "render_style"
                    ]
                ),
                blocks=[block],
            )
        )

    case_spec = CaseSpec(
        case_id=case_id,
        document_profile=profile_id,
        structure_style="section_aware",
        noise_profile=(
            "light_noise"
            if profile_id
            == "mixed_noisy_document"
            else "clean"
        ),
        sections=sections,
        metadata={
            "pipeline_version": "v3.0"
        },
    )

    errors = validate_case_spec(case_spec)

    if errors:
        raise ValueError(
            f"CaseSpec không hợp lệ: {errors}"
        )

    return case_spec


def build_section_prompt(
    case_spec: CaseSpec,
    section: SectionSpec,
) -> str:
    blocks_payload: list[
        dict[str, Any]
    ] = []

    for block in section.blocks:
        blocks_payload.append(
            {
                "block_id": block.block_id,
                "scenario_id": (
                    block.scenario_id
                ),
                "target_entities": [
                    {
                        "id": entity.entity_id,
                        "type": (
                            entity.entity_type
                        ),
                        "marked_text": (
                            entity.marked_text
                        ),
                    }
                    for entity in block.entities
                ],
                "hard_negatives": [
                    {
                        "id": (
                            negative.negative_id
                        ),
                        "type": (
                            negative.negative_type
                        ),
                        "marked_text": (
                            negative.marked_text
                        ),
                    }
                    for negative in (
                        block.hard_negatives
                    )
                ],
                "instructions": (
                    block.instructions
                ),
            }
        )

    payload = {
        "document_profile": (
            case_spec.document_profile
        ),
        "noise_profile": (
            case_spec.noise_profile
        ),
        "section": {
            "section_id": section.section_id,
            "title": section.title,
            "temporal_scope": (
                section.temporal_scope
            ),
            "subject_scope": (
                section.subject_scope
            ),
            "render_style": (
                section.render_style
            ),
            "blocks": blocks_payload,
        },
    }

    return (
        "SECTION_SPEC:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nViết nội dung bên dưới tiêu đề "
        + f"{section.title!r}. "
        + "Không viết lại tiêu đề."
    )


def call_qwen(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
) -> str:
    config = get_qwen_config()

    response = requests.post(
        f"{config.base_url}/chat/completions",
        headers={
            "Authorization": (
                f"Bearer {config.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        },
        json={
            "model": config.model,
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
            "temperature": temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        },
        timeout=config.timeout_seconds,
    )

    response.raise_for_status()

    payload = response.json()

    output = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not output:
        raise ValueError(
            "Qwen trả về output rỗng."
        )

    return output


def build_repair_prompt(
    original_prompt: str,
    previous_output: str,
    errors: list[str],
) -> str:
    return f"""
PROMPT GỐC:
{original_prompt}

OUTPUT CHƯA HỢP LỆ:
{previous_output}

LỖI:
{json.dumps(errors, ensure_ascii=False, indent=2)}

Sửa output để đáp ứng prompt gốc.

- Giữ nguyên mọi marker.
- Không sửa nội dung trong marker.
- Không thêm marker mới.
- Chỉ trả về section đã sửa.
""".strip()


def render_section(
    case_spec: CaseSpec,
    section: SectionSpec,
    *,
    max_retry: int = 3,
) -> dict[str, Any]:
    prompt = build_section_prompt(
        case_spec,
        section,
    )

    mini_case = CaseSpec(
        case_id=case_spec.case_id,
        document_profile=(
            case_spec.document_profile
        ),
        structure_style=(
            case_spec.structure_style
        ),
        noise_profile=(
            case_spec.noise_profile
        ),
        sections=[section],
    )

    previous_output = ""
    errors: list[str] = []

    for attempt in range(1, max_retry + 1):
        if attempt == 1:
            current_prompt = prompt
            temperature = 0.45
        else:
            current_prompt = (
                build_repair_prompt(
                    prompt,
                    previous_output,
                    errors,
                )
            )
            temperature = 0.15

        output = call_qwen(
            QWEN_SYSTEM_PROMPT,
            current_prompt,
            temperature=temperature,
        )

        marked_section = (
            f"{section.title}\n"
            f"{output.strip()}"
        )

        errors = validate_markers(
            marked_section,
            mini_case,
        )

        if not errors:
            return {
                "section_id": (
                    section.section_id
                ),
                "section_type": (
                    section.section_type
                ),
                "title": section.title,
                "marked_text": (
                    marked_section
                ),
                "attempt": attempt,
            }

        previous_output = output

    raise ValueError(
        f"Render section thất bại: {errors}"
    )


def clear_output_files() -> None:
    for filename in (
        "case_specs.jsonl",
        "marked_notes.jsonl",
        "generation_failures.jsonl",
    ):
        path = GENERATED_DIR / filename

        if path.exists():
            path.unlink()


def run_generation(
    *,
    num_samples: int = 50,
    seed: int = 42,
    resume: bool = False,
) -> None:
    ensure_directories()
    random.seed(seed)

    if not resume:
        clear_output_files()

    catalogs = load_catalogs()

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    success = 0
    failed = 0

    for index in range(num_samples):
        case_id = (
            f"{run_id}_{index:06d}"
        )

        started_at = time.perf_counter()

        try:
            case_spec = sample_case_spec(
                case_id,
                catalogs,
            )

            rendered_sections = [
                render_section(
                    case_spec,
                    section,
                )
                for section in (
                    case_spec.sections
                )
            ]

            marked_document = "\n\n".join(
                section["marked_text"]
                for section in rendered_sections
            )

            marker_errors = validate_markers(
                marked_document,
                case_spec,
            )

            if marker_errors:
                raise ValueError(
                    marker_errors
                )

            append_jsonl(
                GENERATED_DIR
                / "case_specs.jsonl",
                case_spec.to_dict(),
            )

            append_jsonl(
                GENERATED_DIR
                / "marked_notes.jsonl",
                {
                    "case_id": case_id,
                    "marked_text": (
                        marked_document
                    ),
                    "sections": (
                        rendered_sections
                    ),
                    "meta": {
                        "model": (
                            get_qwen_config().model
                        ),
                        "latency_seconds": round(
                            time.perf_counter()
                            - started_at,
                            3,
                        ),
                    },
                },
            )

            success += 1

        except Exception as exc:
            append_jsonl(
                GENERATED_DIR
                / "generation_failures.jsonl",
                {
                    "case_id": case_id,
                    "error": str(exc),
                },
            )

            failed += 1

        print(
            f"[{index + 1}/{num_samples}] "
            f"success={success}, failed={failed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    run_generation(
        num_samples=args.num_samples,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
