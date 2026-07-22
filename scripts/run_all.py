from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# Cho phép chạy trực tiếp:
# python scripts/synthetic/run_all.py
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from scripts.generate import run_generation
from scripts.prepare_sources import run_prepare_data
from scripts.validate_medical import run_validate_export


@dataclass
class StageResult:
    name: str
    success: bool
    duration_seconds: float
    error: str | None = None


def run_stage(
    name: str,
    function: Callable[[], None],
) -> StageResult:
    print("\n" + "=" * 70)
    print(f"STAGE: {name}")
    print("=" * 70)

    started_at = time.perf_counter()

    try:
        function()

        duration = (
            time.perf_counter()
            - started_at
        )

        print(
            f"\n[OK] {name} hoàn tất "
            f"trong {duration:.2f} giây."
        )

        return StageResult(
            name=name,
            success=True,
            duration_seconds=duration,
        )

    except Exception as exc:
        duration = (
            time.perf_counter()
            - started_at
        )

        print(
            f"\n[FAILED] {name} thất bại "
            f"sau {duration:.2f} giây."
        )
        print(f"Lỗi: {exc}")

        traceback.print_exc()

        return StageResult(
            name=name,
            success=False,
            duration_seconds=duration,
            error=str(exc),
        )


def print_summary(
    results: list[StageResult],
) -> None:
    print("\n" + "=" * 70)
    print("TỔNG KẾT PIPELINE")
    print("=" * 70)

    for result in results:
        status = (
            "SUCCESS"
            if result.success
            else "FAILED"
        )

        print(
            f"{result.name:<30} "
            f"{status:<10} "
            f"{result.duration_seconds:>8.2f}s"
        )

        if result.error:
            print(
                f"  └─ Error: {result.error}"
            )

    successful = sum(
        result.success
        for result in results
    )

    print("-" * 70)
    print(
        f"Hoàn thành: {successful}/"
        f"{len(results)} stage"
    )


def run_all(
    *,
    num_samples: int,
    seed: int,
    resume: bool,
    skip_prepare: bool,
    skip_generate: bool,
    skip_validate: bool,
    continue_on_error: bool,
) -> bool:
    if num_samples <= 0:
        raise ValueError(
            "num_samples phải lớn hơn 0."
        )

    results: list[StageResult] = []

    if not skip_prepare:
        result = run_stage(
            "1. Prepare data",
            lambda: run_prepare_data(
                seed=seed,
            ),
        )

        results.append(result)

        if (
            not result.success
            and not continue_on_error
        ):
            print_summary(results)
            return False

    if not skip_generate:
        result = run_stage(
            "2. Generate with Qwen",
            lambda: run_generation(
                num_samples=num_samples,
                seed=seed,
                resume=resume,
            ),
        )

        results.append(result)

        if (
            not result.success
            and not continue_on_error
        ):
            print_summary(results)
            return False

    if not skip_validate:
        result = run_stage(
            "3. Validate and export",
            run_validate_export,
        )

        results.append(result)

        if (
            not result.success
            and not continue_on_error
        ):
            print_summary(results)
            return False

    print_summary(results)

    return all(
        result.success
        for result in results
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chạy toàn bộ synthetic data pipeline: "
            "prepare → Qwen generation → validate/export."
        )
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help=(
            "Số CaseSpec cần sinh bằng Qwen. "
            "Mặc định: 50."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Mặc định: 42.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append vào output generation hiện có. "
            "Nếu không bật, output generation cũ "
            "sẽ được xóa trước khi sinh."
        ),
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help=(
            "Bỏ qua bước xử lý nguồn và "
            "build catalogs."
        ),
    )

    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Bỏ qua bước gọi Qwen.",
    )

    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help=(
            "Bỏ qua bước validate và export."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Tiếp tục stage sau kể cả khi "
            "stage trước thất bại."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline_started_at = (
        time.perf_counter()
    )

    success = run_all(
        num_samples=args.num_samples,
        seed=args.seed,
        resume=args.resume,
        skip_prepare=args.skip_prepare,
        skip_generate=args.skip_generate,
        skip_validate=args.skip_validate,
        continue_on_error=(
            args.continue_on_error
        ),
    )

    total_duration = (
        time.perf_counter()
        - pipeline_started_at
    )

    print(
        f"\nTổng thời gian: "
        f"{total_duration:.2f} giây."
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()