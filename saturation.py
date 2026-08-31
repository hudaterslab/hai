#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
지정한 동영상을 3초마다 분석하여 화면을 3×3 격자로 나누고,
각 격자의 HSV 채도(S 채널) 중앙값과 그레이스케일 명도 중앙값을
CSV와 콘솔에 기록합니다.

필수 패키지:
    pip install opencv-python numpy

사용 방법:
    1. 아래 INPUT_VIDEO에 동영상 경로를 입력합니다.
    2. python grid_saturation_logger_no_args.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 사용자 설정
# ============================================================

# 분석할 동영상 경로
# Windows 경로는 문자열 앞에 r을 붙이는 것을 권장합니다.
INPUT_VIDEO = r"C:/Users/hudaterss/Desktop/code/issue/this.mp4"

# 분석 간격(초)
INTERVAL_SEC = 1.0

# 출력 CSV 경로
# None이면 입력 동영상과 같은 폴더에
# "원본파일명_grid_saturation.csv"로 자동 생성됩니다.
OUTPUT_CSV = None

# 저채도 픽셀로 계산할 채도 기준값(S 채널: 0~255)
LOW_SAT_THRESHOLD = 15

# 격자 크기
GRID_ROWS = 3
GRID_COLS = 3


def seconds_to_hms(seconds: float) -> str:
    """초를 HH:MM:SS.mmm 문자열로 변환합니다."""
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def iter_grid_bounds(height: int, width: int):
    """
    3×3 격자의 번호와 좌표를 행 우선 순서로 반환합니다.

    grid_1 | grid_2 | grid_3
    grid_4 | grid_5 | grid_6
    grid_7 | grid_8 | grid_9
    """
    for row in range(GRID_ROWS):
        y1 = round(row * height / GRID_ROWS)
        y2 = round((row + 1) * height / GRID_ROWS)

        for col in range(GRID_COLS):
            x1 = round(col * width / GRID_COLS)
            x2 = round((col + 1) * width / GRID_COLS)

            grid_number = row * GRID_COLS + col + 1
            yield grid_number, x1, y1, x2, y2


def analyze_frame(frame: np.ndarray) -> dict[str, object]:
    """
    프레임을 3×3 격자로 나누어 다음 값을 계산합니다.

    - 채도: HSV의 S 채널 중앙값
    - 명도: multi_event.py와 동일하게 BGR 영상을 GRAY로 변환한 뒤 중앙값
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # multi_event.py의 _gray_plain()과 동일한 변환 방식
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    grid_sat_medians: list[float] = []
    grid_brightness_medians: list[float] = []

    for _, x1, y1, x2, y2 in iter_grid_bounds(height, width):
        sat_cell = saturation[y1:y2, x1:x2]
        gray_cell = gray[y1:y2, x1:x2]

        if sat_cell.size == 0 or gray_cell.size == 0:
            grid_sat_medians.append(float("nan"))
            grid_brightness_medians.append(float("nan"))
        else:
            grid_sat_medians.append(float(np.median(sat_cell)))
            grid_brightness_medians.append(float(np.median(gray_cell)))

    return {
        "grid_sat_medians": grid_sat_medians,
        "grid_brightness_medians": grid_brightness_medians,
        "overall_sat_median": float(np.median(saturation)),
        "overall_sat_mean": float(np.mean(saturation)),
        "overall_brightness_median": float(np.median(gray)),
        "overall_brightness_mean": float(np.mean(gray)),
        "low_sat_ratio": float(np.mean(saturation <= LOW_SAT_THRESHOLD)),
    }


def format_grid_console(grid_values: list[float]) -> str:
    """9개 격자값을 콘솔용 한 줄 문자열로 만듭니다."""
    rows = []

    for row in range(GRID_ROWS):
        start = row * GRID_COLS
        values = grid_values[start:start + GRID_COLS]
        rows.append(" | ".join(f"{value:.1f}" for value in values))

    return " / ".join(rows)


def format_grid_csv_cell(grid_values: list[float]) -> str:
    """
    9개 격자값을 CSV 한 셀 안에 3×3 형태로 저장합니다.

    예:
    1.0 | 2.0 | 3.0
    4.0 | 5.0 | 6.0
    7.0 | 8.0 | 9.0
    """
    rows = []

    for row in range(GRID_ROWS):
        start = row * GRID_COLS
        values = grid_values[start:start + GRID_COLS]

        formatted_values = [
            "x" if math.isnan(value) else f"{value:.1f}"
            for value in values
        ]

        rows.append(" | ".join(formatted_values))

    return "\n".join(rows)


def resolve_paths() -> tuple[Path, Path]:
    """입력 동영상과 출력 CSV 경로를 확인합니다."""
    input_path = Path(INPUT_VIDEO).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"입력 동영상을 찾을 수 없습니다.\n"
            f"INPUT_VIDEO 경로를 확인하세요:\n{input_path}"
        )

    if not math.isfinite(INTERVAL_SEC) or INTERVAL_SEC <= 0:
        raise ValueError("INTERVAL_SEC은 0보다 큰 숫자여야 합니다.")

    if OUTPUT_CSV is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_grid_saturation.csv"
        )
    else:
        output_path = Path(OUTPUT_CSV).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return input_path, output_path


def main() -> int:
    try:
        input_path, output_path = resolve_paths()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {input_path}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    print(f"입력 영상 : {input_path}")
    print(f"출력 CSV : {output_path}")
    print(f"분석 간격 : {INTERVAL_SEC:.3f}초")
    print(f"영상 FPS  : {fps:.3f}" if fps > 0 else "영상 FPS  : 확인 불가")

    if duration_sec > 0:
        print(f"영상 길이 : {seconds_to_hms(duration_sec)}")

    print()
    print("격자 번호:")
    print("1 | 2 | 3")
    print("4 | 5 | 6")
    print("7 | 8 | 9")
    print()

    fieldnames = [
        "sample_index",
        "timestamp_sec",
        "timestamp_hms",
        "frame_index",
        "grid_sat_median_3x3",
        "grid_brightness_median_3x3",
        "overall_sat_median",
        "overall_sat_mean",
        "overall_brightness_median",
        "overall_brightness_mean",
        f"low_sat_ratio_s_le_{LOW_SAT_THRESHOLD}",
    ]

    sample_index = 0
    next_sample_sec = 0.0
    last_processed_time = -1.0

    try:
        # utf-8-sig를 사용해 Windows Excel에서도 한글이 깨지지 않게 합니다.
        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                timestamp_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

                # 일부 코덱에서 POS_MSEC가 0으로 고정되거나 역행할 경우
                # FPS 기반 시간으로 보완합니다.
                if fps > 0:
                    fps_time = frame_index / fps

                    if timestamp_sec <= last_processed_time:
                        timestamp_sec = fps_time
                elif timestamp_sec <= last_processed_time:
                    continue

                # 다음 분석 시점에 도달하지 않았으면 현재 프레임은 건너뜁니다.
                if timestamp_sec + 1e-6 < next_sample_sec:
                    continue

                stats = analyze_frame(frame)
                grid_sat_medians = stats["grid_sat_medians"]
                grid_brightness_medians = stats["grid_brightness_medians"]

                row = {
                    "sample_index": sample_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp_hms": seconds_to_hms(timestamp_sec),
                    "frame_index": frame_index,
                    "overall_sat_median": round(
                        float(stats["overall_sat_median"]), 3
                    ),
                    "overall_sat_mean": round(
                        float(stats["overall_sat_mean"]), 3
                    ),
                    "overall_brightness_median": round(
                        float(stats["overall_brightness_median"]), 3
                    ),
                    "overall_brightness_mean": round(
                        float(stats["overall_brightness_mean"]), 3
                    ),
                    f"low_sat_ratio_s_le_{LOW_SAT_THRESHOLD}": round(
                        float(stats["low_sat_ratio"]), 6
                    ),
                }

                # 채도와 명도 각각 9개 값을 CSV 한 셀 안에 3×3으로 저장
                row["grid_sat_median_3x3"] = format_grid_csv_cell(
                    grid_sat_medians
                )
                row["grid_brightness_median_3x3"] = format_grid_csv_cell(
                    grid_brightness_medians
                )

                writer.writerow(row)
                csv_file.flush()

                print(
                    f"[{sample_index:04d}] "
                    f"{seconds_to_hms(timestamp_sec)} | "
                    f"채도={format_grid_console(grid_sat_medians)} | "
                    f"명도={format_grid_console(grid_brightness_medians)} | "
                    f"전체 채도 median={stats['overall_sat_median']:.1f} | "
                    f"전체 명도 median={stats['overall_brightness_median']:.1f} | "
                    f"저채도비율={stats['low_sat_ratio']:.3f}"
                )

                sample_index += 1
                last_processed_time = timestamp_sec

                # 영상 프레임 시간이 정확히 3초에 맞지 않아도
                # 다음 목표 시각은 0, 3, 6, 9초 단위로 유지합니다.
                while next_sample_sec <= timestamp_sec + 1e-6:
                    next_sample_sec += INTERVAL_SEC

    except KeyboardInterrupt:
        print("\n사용자 중단으로 분석을 종료합니다.")
    except OSError as exc:
        print(f"[오류] CSV 파일을 쓸 수 없습니다: {exc}", file=sys.stderr)
        return 1
    finally:
        cap.release()

    print()
    print(f"완료: {sample_index}개 시점 기록")
    print(f"CSV 저장 위치: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())