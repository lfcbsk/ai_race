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
        expand_placeholders,
        get_qwen_config,
        normalize_for_matching,
        placeholder_for,
        read_json,
        read_jsonl,
        validate_case_spec,
        validate_placeholder_output,
        validate_rendered_sample,
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
        expand_placeholders,
        get_qwen_config,
        normalize_for_matching,
        placeholder_for,
        read_json,
        read_jsonl,
        validate_case_spec,
        validate_placeholder_output,
        validate_rendered_sample,
    )


QWEN_SYSTEM_PROMPT = """
Bạn là bộ dựng văn bản bệnh án tiếng Việt cho dữ liệu NLP y khoa tổng hợp.

Đầu vào cung cấp các placeholder như <<E0>>, <<E1>> và <<N0>>.
Bạn chỉ viết ngữ cảnh lâm sàng xung quanh placeholder; Python sẽ tự chèn
surface text sau khi bạn trả lời.

QUY TẮC TUYỆT ĐỐI:

1. Mỗi placeholder được cung cấp phải xuất hiện đúng một lần.
2. Chỉ dùng placeholder; không tự chép lại surface_hint.
3. Không tạo marker dạng [[...]] và không tạo placeholder mới.
4. Không thêm tên thuốc, bệnh, triệu chứng hay xét nghiệm mục tiêu khác.
5. Không viết mã ICD-10 hoặc RxNorm.
6. Cue phủ định phải nằm cùng scope với placeholder bị phủ định.
7. Placeholder dương tính không được nằm dưới scope phủ định.
8. Ngữ cảnh gia đình và tiền sử phải có cue rõ ràng.
9. Không nhắc các thuật ngữ nội bộ như entity, assertion, scenario,
   hard-negative, SECTION_SPEC hoặc marker.
10. Chỉ trả về nội dung section; không viết lại tiêu đề.
11. Không giải thích, không trả JSON, không dùng markdown fence.
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
    excluded_texts: set[str] | None = None,
) -> dict[str, Any]:
    excluded_texts = excluded_texts or set()

    eligible = [
        item
        for item in catalog
        if (
            (
                not require_linking
                or item.get(
                    "usable_for_linking",
                    False,
                )
            )
            and normalize_for_matching(
                item.get("text", "")
            )
            not in excluded_texts
        )
    ]

    if not eligible:
        raise ValueError(
            "Không có surface phù hợp sau lọc trùng."
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
    used_surfaces: set[str],
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
            catalogs["symptoms"],
            excluded_texts=used_surfaces,
        )
        used_surfaces.add(
            normalize_for_matching(first["text"])
        )

        second = sample_surface(
            catalogs["symptoms"],
            excluded_texts=used_surfaces,
        )
        used_surfaces.add(
            normalize_for_matching(second["text"])
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
                excluded_texts=used_surfaces,
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
            used_surfaces.add(
                normalize_for_matching(
                    candidate["text"]
                )
            )

        for surface in selected:
            used_surfaces.add(
                normalize_for_matching(
                    surface["text"]
                )
            )

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
                    catalogs["symptoms"],
                    excluded_texts=used_surfaces,
                )
            elif entity_type == "CHẨN_ĐOÁN":
                surface = sample_surface(
                    catalogs["diseases"],
                    require_linking=True,
                    excluded_texts=used_surfaces,
                )
            elif entity_type == "THUỐC":
                surface = sample_surface(
                    catalogs["drugs"],
                    require_linking=True,
                    excluded_texts=used_surfaces,
                )
            else:
                raise ValueError(
                    f"Unsupported type: "
                    f"{entity_type}"
                )

            used_surfaces.add(
                normalize_for_matching(
                    surface["text"]
                )
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
        and kind in {
            "single_group",
            "coordinated_negation",
            "mixed_polarity",
        }
    ):
        eligible_negatives = [
            item
            for item in catalogs["negatives"]
            if normalize_for_matching(
                item.get("text", "")
            )
            not in used_surfaces
        ]

        if eligible_negatives:
            negative = random.choice(
                eligible_negatives
            )
            used_surfaces.add(
                normalize_for_matching(
                    negative["text"]
                )
            )
        else:
            negative = None

        negative_id = (
            f"N{counters['negative']}"
        )
        if negative is not None:
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
    used_surfaces: set[str] = set()

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
            used_surfaces=used_surfaces,
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
    blocks_payload: list[dict[str, Any]] = []

    for block in section.blocks:
        block_payload: dict[str, Any] = {
            "scenario": block.scenario_id,
            "targets": [
                {
                    "placeholder": placeholder_for(
                        entity.entity_id
                    ),
                    "surface_hint": entity.surface_text,
                    "type": entity.entity_type,
                    "assertions": entity.assertions,
                }
                for entity in block.entities
            ],
        }

        if block.hard_negatives:
            block_payload["non_target_phrases"] = [
                {
                    "placeholder": placeholder_for(
                        negative.negative_id
                    ),
                    "surface_hint": (
                        negative.surface_text
                    ),
                    "type": negative.negative_type,
                }
                for negative in block.hard_negatives
            ]

        blocks_payload.append(block_payload)

    payload = {
        "document_profile": case_spec.document_profile,
        "noise_profile": case_spec.noise_profile,
        "section_style": section.render_style,
        "blocks": blocks_payload,
    }

    return (
        "Dữ liệu điều khiển:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nViết nội dung bệnh án bên dưới tiêu đề "
        + f"{section.title!r}. Không viết lại tiêu đề. "
        + "Chỉ xuất placeholder, tuyệt đối không chép surface_hint."
    )


def _join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return ", ".join(tokens[:-1]) + f" và {tokens[-1]}"


def _render_block_template(block: BlockSpec) -> str:
    entity_tokens = [
        placeholder_for(entity.entity_id)
        for entity in block.entities
    ]
    negative_tokens = [
        placeholder_for(negative.negative_id)
        for negative in block.hard_negatives
    ]
    joined = _join_tokens(entity_tokens)
    scenario = block.scenario_id

    if scenario == "patient_current_positive":
        text = f"Bệnh nhân hiện ghi nhận {joined}."
    elif scenario in {
        "patient_current_negated",
        "coordinated_negation",
    }:
        text = f"Bệnh nhân không ghi nhận {joined}."
    elif scenario == "mixed_polarity":
        text = (
            f"Bệnh nhân không ghi nhận {entity_tokens[0]}, "
            f"nhưng hiện có {entity_tokens[1]}."
        )
    elif scenario == "patient_historical":
        text = f"Trước đây bệnh nhân từng ghi nhận {joined}."
    elif scenario == "patient_historical_negated":
        text = (
            f"Trước nhập viện, bệnh nhân không ghi nhận "
            f"{joined}."
        )
    elif scenario == "family_current":
        text = f"Người nhà bệnh nhân hiện có {joined}."
    elif scenario == "family_historical":
        text = (
            f"Trong tiền sử gia đình, người nhà bệnh nhân "
            f"từng ghi nhận {joined}."
        )
    elif scenario == "home_medications":
        text = "\n".join(
            f"- {token}"
            for token in entity_tokens
        )
    elif scenario == "active_medications":
        text = "\n".join(
            f"- {token}"
            for token in entity_tokens
        )
    elif scenario == "laboratory_results":
        lines: list[str] = []
        for index in range(0, len(entity_tokens), 2):
            pair = entity_tokens[index:index + 2]
            if len(pair) == 2:
                lines.append(f"- {pair[0]}: {pair[1]}")
        text = "\n".join(lines)
    else:
        raise ValueError(
            f"Chưa có template cho scenario {scenario!r}"
        )

    if negative_tokens:
        text += (
            "\nNgoài ra, bệnh nhân đã được thực hiện "
            + _join_tokens(negative_tokens)
            + "."
        )

    return text


def render_section_template(
    section: SectionSpec,
) -> str:
    return "\n".join(
        _render_block_template(block)
        for block in section.blocks
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
YÊU CẦU GỐC:
{original_prompt}

DRAFT CHƯA HỢP LỆ:
{previous_output}

LỖI CẦN SỬA:
{json.dumps(errors, ensure_ascii=False, indent=2)}

Chỉ trả về section đã sửa.
- Giữ mỗi placeholder đúng một lần.
- Không chép surface_hint.
- Không tạo placeholder mới hoặc marker [[...]].
- Không nhắc thuật ngữ nội bộ của pipeline.
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
        document_profile=case_spec.document_profile,
        structure_style=case_spec.structure_style,
        noise_profile=case_spec.noise_profile,
        sections=[section],
    )

    deterministic_scenarios = {
        "patient_current_negated",
        "coordinated_negation",
        "mixed_polarity",
        "patient_historical_negated",
        "home_medications",
        "active_medications",
        "laboratory_results",
    }

    use_template = all(
        block.scenario_id in deterministic_scenarios
        for block in section.blocks
    )

    config = get_qwen_config()
    previous_output = ""
    errors: list[str] = []

    attempts = 1 if use_template else max_retry

    for attempt in range(1, attempts + 1):
        if use_template:
            draft = render_section_template(section)
        else:
            current_prompt = (
                prompt
                if attempt == 1
                else build_repair_prompt(
                    prompt,
                    previous_output,
                    errors,
                )
            )
            temperature = (
                config.temperature
                if attempt == 1
                else config.repair_temperature
            )
            draft = call_qwen(
                QWEN_SYSTEM_PROMPT,
                current_prompt,
                temperature=temperature,
            )

        errors = validate_placeholder_output(
            draft,
            mini_case,
        )

        if not errors:
            marked_section = (
                f"{section.title}\n"
                f"{expand_placeholders(draft, mini_case).strip()}"
            )
            errors = validate_rendered_sample(
                marked_section,
                mini_case,
            )

            if not errors:
                return {
                    "section_id": section.section_id,
                    "section_type": section.section_type,
                    "title": section.title,
                    "marked_text": marked_section,
                    "attempt": attempt,
                    "render_mode": (
                        "template"
                        if use_template
                        else "qwen"
                    ),
                }

        previous_output = draft

    # Fallback deterministic để không mất case vì lỗi format của LLM.
    fallback_draft = render_section_template(section)
    fallback_errors = validate_placeholder_output(
        fallback_draft,
        mini_case,
    )

    if not fallback_errors:
        marked_section = (
            f"{section.title}\n"
            f"{expand_placeholders(fallback_draft, mini_case).strip()}"
        )
        fallback_errors = validate_rendered_sample(
            marked_section,
            mini_case,
        )

    if not fallback_errors:
        return {
            "section_id": section.section_id,
            "section_type": section.section_type,
            "title": section.title,
            "marked_text": marked_section,
            "attempt": attempts + 1,
            "render_mode": "template_fallback",
        }

    raise ValueError(
        f"Render section thất bại: {errors}; "
        f"fallback: {fallback_errors}"
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

            document_errors = validate_rendered_sample(
                marked_document,
                case_spec,
            )

            if document_errors:
                raise ValueError(
                    document_errors
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
                    "error_type": type(exc).__name__,
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