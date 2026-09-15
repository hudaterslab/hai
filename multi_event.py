import os
os.environ["GST_VAAPI_DISPLAY"] = "drm"
os.environ["GST_VAAPI_DRM_DEVICE"] = "/dev/dri/renderD128"
os.environ["LIBVA_DRIVER_NAME"] = "iHD"
os.environ["GST_VAAPI_ALL_DRIVERS"] = "1"
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vah264dec:MAX,vah265dec:MAX"
import sys
import gc
import json
import csv
import shutil
import subprocess
import cv2
import math
import numpy as np
import time
import datetime
import traceback
import threading
import queue
import logging
import psutil
import atexit
from collections import deque, defaultdict
import concurrent.futures
import re
import requests
import pytz
from fractions import Fraction
from urllib.parse import urlsplit, unquote
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
import argparse
# ----------------- light tower --------------------#
# import struct
# import sys
# import usb.core
# from usb.core import Device
# --------------------------------------------------#
warnings = requests.packages.urllib3.exceptions.InsecureRequestWarning
requests.packages.urllib3.disable_warnings(warnings)

API_SEND_STATE = {"consecutive_failures": 0, "last_failure_at": None}
API_SEND_STATE_LOCK = threading.Lock()
EVENT_AUDIT_LOCK = threading.Lock()

# ==========================================
# [1] 시스템 기본 설정 및 상수
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_COMMON_FILE = os.path.join(PROJECT_ROOT, "system_config.json")
CONFIG_CAMERAS_FILE = os.path.join(PROJECT_ROOT, "cameras.json")
CAMERA_LIST_FILE = os.path.join(PROJECT_ROOT, "cameras.csv")
EVENT_ROOT_DIR = os.path.join(PROJECT_ROOT, "CCTV_EVENT_ALERT")

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WATCHDOG_TIMEOUT = 30.0

# PCTC 단일 MAIN 모델 클래스 정의
PCTC_CLASS_NAMES = (
    "helmet",
    "head",
    "person",
    "car",
    "Truck",
    "Y/T",
    "Chassis",
    "Container",
    "Corn",
    "Y/C",
    "Spreader",
)

ID_PCTC_HELMET = 0
ID_PCTC_HEAD = 1
ID_PCTC_PERSON = 2
ID_PCTC_CAR = 3
ID_PCTC_TRUCK = 4
ID_PCTC_YT = 5
ID_PCTC_CHASSIS = 6
ID_PCTC_CONTAINER = 7
ID_PCTC_CORN = 8
ID_PCTC_YC = 9
ID_PCTC_SPREADER = 10

PCTC_MAIN_IGNORED_CLASS_IDS = {ID_PCTC_HELMET, ID_PCTC_HEAD}
PCTC_MAIN_TRACK_CLASS_IDS = {
    ID_PCTC_PERSON,
    ID_PCTC_CAR,
    ID_PCTC_TRUCK,
    ID_PCTC_YT,
    ID_PCTC_CHASSIS,
    ID_PCTC_CONTAINER,
    ID_PCTC_CORN,
    ID_PCTC_YC,
    ID_PCTC_SPREADER,
}

# 전용 HELMET 모델 클래스
ID_H_HELMET = 0
ID_H_HEAD = 1
ID_H_PERSON = 2

INTRUSION_CLASS_IDS = set(PCTC_MAIN_TRACK_CLASS_IDS)
PARKING_CLASS_IDS = {
    ID_PCTC_CAR,
    ID_PCTC_TRUCK,
    ID_PCTC_YT,
    ID_PCTC_CHASSIS,
    ID_PCTC_CONTAINER,
    ID_PCTC_YC,
}
ABNORMAL_DRIVE_CLASS_IDS = {
    ID_PCTC_YT,
    ID_PCTC_CAR,
    ID_PCTC_CHASSIS,
}
DANGER_APPROACH_CLASS_IDS = {
    ID_PCTC_PERSON,
    ID_PCTC_CAR,
    ID_PCTC_TRUCK,
    ID_PCTC_YT,
    ID_PCTC_CHASSIS,
}
PLATE_PRIVACY_CLASS_IDS = set(PARKING_CLASS_IDS)

SAFETY_EVENT_NAMES = (
    "intrusion",
    "illegal_parking",
    "no_helmet",
    "yard_crossing",
    "abnormal_drive",
    "walkway_out",
    "spreader_danger_zone",
    "container_collision_risk",
)

PORT_EVENT_NAMES = (
    "yard_crossing",
    "abnormal_drive",
    "walkway_out",
    "spreader_danger_zone",
    "container_collision_risk",
)

ROI_POLYGON_EVENT_NAMES = {
    "intrusion",
    "illegal_parking",
    "no_helmet",
    "yard_crossing",
    "walkway_out",
}

SUPPORTED_CAMERA_EVENTS = set(SAFETY_EVENT_NAMES) | {
    "roi_change",
    "roi_change_apply",
}

EVENT_NAME_ALIASES = {
    "wrong_way_uturn": "abnormal_drive",
    "walkway_departure": "walkway_out",
}

EVENT_CONFIG_NAMES = frozenset(SAFETY_EVENT_NAMES)
CONFIG_SCHEMA_VERSION = 4

MODEL_SECTION_ALLOWED_KEYS = {
    "models": frozenset(("MAIN", "FACE", "HELMET", "PLATE")),
    "model_confidences": frozenset(("MAIN", "PERSON", "HELMET_PERSON", "FACE", "HELMET", "PLATE")),
    "model_output_formats": frozenset(("MAIN", "FACE", "HELMET", "PLATE")),
    "model_engine_pool_sizes": frozenset(("MAIN", "FACE", "HELMET", "PLATE")),
    "model_input_shapes": frozenset(("MAIN", "FACE", "HELMET", "PLATE")),  # <--- 이 줄 추가
}

SYSTEM_CONFIG_ALLOWED_KEYS = frozenset((
    "config_schema_version",
    "terminal_id",
    "system_performance",
    "logging",
    "event_config",
    "models",
    "model_confidences",
    "model_output_formats",
    "model_engine_pool_sizes",
    "model_input_shapes",
    "inference_runtime",
    "video_decode",
    "BATCH_SIZE",
    "REC_FPS",
    "LOOP_FPS",
    "PERF_LOG_INTERVAL_SEC",
    "REC_PRE_SEC",
    "REC_POST_SEC",
    "VIDEO_EVENT_MARK_SEC",
    "VIDEO_EVENT_BORDER_THICKNESS",
    "EVENT_FRAME_SAVE_DELAY_SEC",
    "EVENT_FRAME_SAVE_MAX_COUNT",
    "OUTPUT_RETENTION_DAYS",
    "OUTPUT_CLEANUP_INTERVAL_SEC",
    "ROI_SETUP_REQUIRED_API_ENABLED",
    "INTERACTIVE_INPUT_GUARD_SEC",
    "VISUAL_ALARM_DURATION",
    "roi_align_learning",
    "verbose_logs",
))

def pctc_class_name(class_id):
    try:
        class_id = int(class_id)
    except Exception:
        return "unknown"
    if 0 <= class_id < len(PCTC_CLASS_NAMES):
        return PCTC_CLASS_NAMES[class_id]
    return f"class_{class_id}"


def get_visible_pctc_class_ids(events):
    event_class_ids = {
        "intrusion": INTRUSION_CLASS_IDS,
        "illegal_parking": PARKING_CLASS_IDS,
        "no_helmet": {ID_PCTC_PERSON},
        "yard_crossing": {ID_PCTC_PERSON},
        "walkway_out": {ID_PCTC_PERSON},
        "abnormal_drive": ABNORMAL_DRIVE_CLASS_IDS,
        "spreader_danger_zone": {
            ID_PCTC_PERSON,
            ID_PCTC_CAR,
            ID_PCTC_TRUCK,
            ID_PCTC_YT,
            ID_PCTC_CHASSIS,
            ID_PCTC_CONTAINER,
            ID_PCTC_SPREADER,
        },
        "container_collision_risk": {
        ID_PCTC_CONTAINER,
        ID_PCTC_SPREADER,
    },
    }
    visible = set()
    for event_name in events or []:
        visible.update(event_class_ids.get(event_name, set()))
    return visible & PCTC_MAIN_TRACK_CLASS_IDS


def get_display_pctc_class_ids(events):
    """Return classes rendered and inferred for the current terminal view.

    Event detectors still apply their own class filters.  Showing every valid PCTC
    terminal class by default prevents a valid model detection from disappearing
    merely because the selected event menu did not reference that class.
    """
    cfg = globals().get("SYS_CFG", {})
    runtime_cfg = cfg.get("inference_runtime", {}) if isinstance(cfg, dict) else {}
    if bool(runtime_cfg.get("display_all_pctc_objects", True)):
        return set(PCTC_MAIN_TRACK_CLASS_IDS)
    return get_visible_pctc_class_ids(events)

DEBUG_MODE = False

# ------------------------------------------------------------
# ROI 보정(Aligner) 튜닝 파라미터
# ------------------------------------------------------------
ALIGN_INTERVAL_SEC = 300.0                 # 화각변경 검사 주기(초)
ROI_CHANGE_EVENT = "roi_change"            # 이 이벤트가 지정된 카메라만 화각변경 '감지+알림'
ROI_CHANGE_APPLY_EVENT = "roi_change_apply"  # 감지 + 측정된 평행이동만큼 ROI를 '자동 보정'까지 하는 카메라
GRID_APPLY_MAX_SHIFT_PX = 150.0            # 자동 보정 허용 이동 상한(px). 초과 시 보정 안 하고 알림만(=사람 재설정 필요)
GRID_APPLY_SHIFT_SIGN = 1.0                # ROI 보정 방향 부호. 보정이 반대로 되면 -1.0으로 (phaseCorrelate 부호 실측 후 조정)
ANCHOR_STARTUP_DELAY_SEC = 10.0            # RTSP 연결 직후 무효 프레임 회피용 안정화 대기
ANCHOR_RETRY_INTERVAL_SEC = 30.0           # 앵커 등록 실패 시 재시도 간격

# suspect/confirm 상태머신 (ROIAlignLearningStore.record_check)
ROI_DRIFT_CONFIRM_COUNT = 3                # 이동 확정에 필요한 연속 횟수
GRID_DISTURBED_CONFIRM_COUNT = 3           # 전 칸 이동이지만 방향이 흩어진 큰 변화 알림에 필요한 연속 횟수
GRID_ABNORMAL_CONFIRM_COUNT = 3            # suspect/disturbed를 합산한 카메라별 연속 이상 횟수

ANCHOR_BASE = "base"
ANCHOR_UPDATED = "updated"
ROI_ALIGN_CSV_LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "roi_align", "roi_align_decisions.csv")
ROI_ALIGN_LEARNING_DEFAULTS = {
    "confirm_count_required": ROI_DRIFT_CONFIRM_COUNT,
    "disturbed_confirm_count_required": GRID_DISTURBED_CONFIRM_COUNT,
    "abnormal_count_required": GRID_ABNORMAL_CONFIRM_COUNT,
}
# test주석
# ============================================================
# 전체 화면 3×3 격자 기반 화각 변경(틀어짐) 감지
#   - 전체 프레임을 3×3로 나눠 각 칸의 평행이동 벡터를 phaseCorrelate로 측정.
#   - 측정 성공한 칸이 모두 GRID_SHAKE_THRESHOLD_PX(10px)를 초과해 움직였고,
#     그중 같은 방향인 칸이 round(n_moving × GRID_QUORUM_FRACTION) 이상이면
#     카메라 틀어짐으로 본다.
#       * 객체 이동: 일부 칸만 움직임 → 같은 방향 칸 수 부족 → 틀어짐 아님(사물=차/사람/택배)
#       * 조명 변화(주/야·IR): 밝기만 변하고 벡터(평행이동)는 없음 → 틀어짐 아님
#   - 이벤트(cameras.json events에 "roi_change") 지정 카메라만 동작.
# ============================================================
GRID_ROWS = 3
GRID_COLS = 3
GRID_SHAKE_THRESHOLD_PX = 7.5       # 칸의 이동량이 이 값을 초과하면 '움직인 칸'(px)
GRID_CELL_MIN_STD = 10.0             # 칸 픽셀 표준편차가 이 미만이면 텍스처 없음 → 측정 제외
# 적응형 정족수: 카메라마다 쓸 수 있는(텍스처 있는) 칸 수가 다르므로(멀티터미널 다양한 장면),
#   고정값 대신 그 프레임의 텍스처 칸 수(n_textured)에 비례해 정족수를 정한다.
#   quorum = max(GRID_QUORUM_FLOOR, round(n_textured × GRID_QUORUM_FRACTION))
#   예) 9칸 → 5, 하늘3칸이라 6칸 → 4, 5칸 → 3. (측정칸이 정족수 미만이면 판단 보류=알람 안 함)
GRID_QUORUM_FRACTION = 0.45           # 텍스처 칸 중 이 비율이 측정돼야 판단 가능
GRID_QUORUM_FLOOR = 3                # 정족수 하한(최소 이만큼은 측정돼야 판단)
GRID_DIRECTION_COS_MIN = 0.4         # 움직인 칸 벡터와 대표(median) 방향의 코사인 유사도가 이 이상이면 '같은 방향'(0.6≈±53°)

# --- homography 기반 ROI 자동 보정(1순위) 파라미터 ----------------------------
# confirm 시 앵커(틀어지기 전)↔현재 프레임을 ORB 특징점 매칭 + RANSAC homography로 정합해
# ROI 점들을 변환한다. 렌즈 왜곡으로 지역별 이동량이 다른 경우(실측: 중앙 99px vs 구석 76px)
# 전역 평행이동보다 ROI 위치에서 정확하다. 게이트를 하나라도 통과 못 하면 평행이동 보정으로 폴백.
GRID_HOMOGRAPHY_MAX_FEATURES = 1500     # ORB 특징점 수 상한
GRID_HOMOGRAPHY_MIN_INLIERS = 15        # RANSAC 인라이어 최소 수(이 미만이면 매칭 신뢰 불가)
GRID_HOMOGRAPHY_RANSAC_REPROJ_PX = 5.0  # RANSAC 재투영 오차 임계(px)
GRID_HOMOGRAPHY_SHIFT_TOL_PX = 40.0     # H의 화면중심 이동량과 격자 median 측정값의 허용 차(교차검증)
# 스케일 게이트: 렌즈 왜곡이 있으면 최적 H가 스케일 성분을 갖는 게 정상(실측 sv=1.16에서
# 상한 1.15로 아깝게 탈락했던 이력 있음 → 0.75~1.35로 완화. 오매칭 방어는 인라이어 수 +
# 중심이동 교차검증 + ROI 점 변위 상한이 담당).
GRID_HOMOGRAPHY_SCALE_MIN = 0.75        # 허용 스케일 하한
GRID_HOMOGRAPHY_SCALE_MAX = 1.35        # 허용 스케일 상한
GRID_HOMOGRAPHY_PERSPECTIVE_MAX = 1e-3  # 원근 성분(H[2,0], H[2,1]) 상한(ROI 찌그러짐 방어)
# ROI 지역 잔차 정밀 보정: H는 전 화면 최적 근사라 ROI 지점에는 몇 px 잔차가 남을 수 있다.
# H로 워핑한 앵커(=보정이 완벽할 때의 현재 화면 예측)와 실제 현재 프레임을 ROI 중심 패치에서
# phaseCorrelate로 1회 비교해 잔차를 측정하고 ROI에 추가 반영한다.
GRID_APPLY_REFINE_PATCH_PX = 192        # 잔차 측정 패치 한 변 크기(px)
GRID_APPLY_REFINE_MAX_PX = 15.0         # 측정된 잔차가 이보다 크면 이상 측정으로 보고 무시

def _format_grid_cell_diag(c):
    """격자 칸 1개가 '얼마나 움직였는지'(px)만 적는다(CSV/로그 공용).
      측정칸          → "12.3"  (그 칸의 평행이동량 px)
      측정 불가(x)    → "x"     (텍스처 없음 std<GRID_CELL_MIN_STD, 또는 phaseCorrelate 실패)
    """
    if c.get("m"):
        return f"{c['shift']:.1f}"
    return "x"

def _format_grid_cell_std(c):
    """격자 칸 1개의 std(텍스처) 값. GRID_CELL_MIN_STD 이상이면 측정칸이 된다(어느 칸이 통과했는지 확인용)."""
    return f"{float(c.get('std', 0.0)):.1f}"

def deep_merge_dict(base, override):
    """딕셔너리를 깊은 병합(Deep Merge)하는 유틸리티 함수"""
    import copy
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

def _migrate_event_config(raw_event_config):
    """Normalize aliases and keep only terminal safety-event configuration."""
    normalized = {}
    for raw_name, raw_value in dict(raw_event_config or {}).items():
        event_name = EVENT_NAME_ALIASES.get(str(raw_name), str(raw_name))
        if event_name not in EVENT_CONFIG_NAMES:
            continue
        if event_name not in normalized:
            normalized[event_name] = raw_value
        elif isinstance(raw_value, dict) and isinstance(normalized.get(event_name), dict):
            normalized[event_name] = deep_merge_dict(raw_value, normalized[event_name])

    no_helmet = dict(normalized.get("no_helmet") or {})
    if "trigger_total_sec" not in no_helmet and "trigger_sec" in no_helmet:
        no_helmet["trigger_total_sec"] = no_helmet.get("trigger_sec")
    no_helmet.pop("trigger_sec", None)
    if no_helmet:
        normalized["no_helmet"] = no_helmet

    return normalized


def _schema_version(value):
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _migrate_system_config(raw_config):
    """Apply an allow-list schema so retired model/event keys cannot survive."""
    raw = dict(raw_config or {})
    source_schema = _schema_version(raw.get("config_schema_version"))
    migrated = {
        key: value
        for key, value in raw.items()
        if key in SYSTEM_CONFIG_ALLOWED_KEYS
    }
    migrated["event_config"] = _migrate_event_config(migrated.get("event_config"))

    for section_name, allowed_keys in MODEL_SECTION_ALLOWED_KEYS.items():
        section = dict(migrated.get(section_name) or {})
        migrated[section_name] = {
            key: value
            for key, value in section.items()
            if key in allowed_keys
        }

    # Schema 2 removes stale single-purpose defaults and lets the runtime inspect
    # the actual tensor metadata/output before choosing a decoder.
    if source_schema < 2:
        models = dict(migrated.get("models") or {})
        if not str(models.get("MAIN") or "").strip():
            models["MAIN"] = "pctc_v1.dxnn"
        migrated["models"] = models

        output_formats = dict(migrated.get("model_output_formats") or {})
        output_formats["MAIN"] = "auto"
        migrated["model_output_formats"] = output_formats

        confidences = dict(migrated.get("model_confidences") or {})
        try:
            if abs(float(confidences.get("MAIN", 0.6)) - 0.6) < 1e-9:
                confidences["MAIN"] = 0.35
        except Exception:
            confidences["MAIN"] = 0.35
        try:
            if abs(float(confidences.get("PERSON", 0.5)) - 0.5) < 1e-9:
                confidences["PERSON"] = 0.30
        except Exception:
            confidences["PERSON"] = 0.30
        migrated["model_confidences"] = confidences

    # Schema 3 enables the HELMET model's person class as a secondary person detector.
    # It is decoded at a lower person threshold, while helmet/head retain HELMET's own threshold.
    if source_schema < 3:
        confidences = dict(migrated.get("model_confidences") or {})
        confidences.setdefault("HELMET_PERSON", confidences.get("PERSON", 0.30))
        migrated["model_confidences"] = confidences

    migrated["config_schema_version"] = CONFIG_SCHEMA_VERSION
    return migrated

def load_system_config():
    default_config = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "terminal_id": "99999",
        "system_performance": {
            "target_fps": 10.0,
            "dynamic_cpu_adjust_enabled": False,
        },
        "logging": {
            "dir": "./logs",
            "level": "INFO",
            "file_level": "INFO",
            "console_level": "INFO",
            "debug_file_level": "DEBUG",
            "retention_days": 14,
            "event_audit_enabled": True,
            "disk_free_warn_gb": 5.0,
        },
        "event_config": {
            "intrusion": {
                "enabled": False,
                "cooldown_sec": 600,
                "proximity_ratio": 0.0,
                "blur_face": True,
                "blur_plate": True,
            },
            "illegal_parking": {
                "enabled": False,
                "cooldown_sec": 600,
                "trigger_sec": 5.0,
                "move_threshold_ratio": 0.1,
                "blur_plate": True,
            },
            "no_helmet": {
                "enabled": False,
                "cooldown_sec": 60,
                "blur_face": True,
                "blur_plate": True,
                "min_streak_sec": 2.0,
                "trigger_total_sec": 3.0,
                "max_gap_sec": 1.5,
                "window_sec": 30.0,
                "ignore_top_ratio": 0.2,
            },
            "yard_crossing": {
                "enabled": False,
                "cooldown_sec": 60,
                "trigger_sec": 3.0,
                "blur_face": True,
                "blur_plate": False,
            },
            "abnormal_drive": {
                "enabled": False,
                "cooldown_sec": 60,
                "min_displacement_ratio": 0.35,
                "opposite_cos_threshold": -0.35,
                "uturn_min_forward_ratio": 0.45,
                "uturn_return_ratio": 0.30,
                "uturn_reverse_cos_threshold": -0.20,
                "blur_face": False,
                "blur_plate": True,
            },
            "walkway_out": {
                "enabled": False,
                "cooldown_sec": 60,
                "outside_grace_sec": 1,
                "boundary_margin_ratio": 0.05,
                "blur_face": True,
                "blur_plate": False,
            },
            "spreader_danger_zone": {
                "enabled": False,
                "cooldown_sec": 30,
                "spreader_container_link_ratio": 1.25,
                "danger_distance_ratio": 1.20,
                "min_danger_distance_px": 30.0,
                "trigger_hold_sec": 0.5,
                "blur_face": True,
                "blur_plate": True,
            },
            "container_collision_risk": {
                "enabled": False,
                "cooldown_sec": 30,

                # 스프레더와 들고 있는 컨테이너 결합 판단
                "pair_center_x_ratio": 0.30,
                "pair_vertical_gap_ratio": 0.35,
                "pair_confirm_frames": 3,

                # 컨테이너 횡이동 판단
                "motion_window_sec": 0.4,
                "horizontal_move_ratio": 0.04,
                "horizontal_dominance": 1.5,

                # 진행방향 충돌 예측
                "lookahead_sec": 1.0,
                "max_lookahead_ratio": 2.0,

                # 적층 컨테이너 상단과 확보해야 하는 높이
                "min_clearance_ratio": 0.10,
                "min_clearance_px": 10.0,

                # 순간 BBox 흔들림 방지
                "trigger_hold_sec": 0.5,

                "blur_face": True,
                "blur_plate": True,
            },
        },
        "models": {
            "MAIN": "pctc_v1.dxnn",
            "FACE": "yolov8m-face_ppu.dxnn",
            "HELMET": "helmet_260622.dxnn",
            "PLATE": "license_plate_detector_v2.dxnn",
        },
        "model_confidences": {
            "MAIN": 0.35,
            "FACE": 0.35,
            "HELMET": 0.85,
            "PERSON": 0.30,
            "HELMET_PERSON": 0.30,
            "PLATE": 0.1,
        },
        "model_output_formats": {
            "MAIN": "auto",
            "FACE": "auto",
            "HELMET": "auto",
            "PLATE": "yolo",
        },
        "model_engine_pool_sizes": {
            "MAIN": 2,
            "FACE": 1,
            "HELMET": 1,
            "PLATE": 1,
        },
        "model_input_shapes": {
            "MAIN": [640, 384],
            "FACE": [640, 640],
            "HELMET": [640, 640],
            "PLATE": [640, 640]
        },
        "inference_runtime": {
            "display_all_pctc_objects": True,
            "max_detection_area_ratio": 0.95,
            "log_first_output_signature": True,
            "empty_detection_log_interval_sec": 10.0,
            "filter_diagnostic_interval_sec": 10.0,
            "helmet_person_assist_enabled": True,
            "helmet_person_class_id": ID_H_PERSON,
            "person_merge_iou_threshold": 0.4,
        },
        "video_decode": {
            "backend": "gstreamer",
            "hw_acceleration": "auto",
            "hw_device": "/dev/dri/renderD128",
            "vaapi_driver": "iHD",
            "fallback_to_cpu": True,
            "fps_limit": 10.0,
            "gstreamer_latency_ms": 50,
            "gstreamer_protocols": "tcp",
            "gstreamer_tcp_timeout_us": 3000000,
            "gstreamer_drop_on_latency": True,
            "gstreamer_log_drain_sleep_sec": 0.002,
            "gstreamer_loop_sleep_sec": 0.001,
            "log_interval_sec": 10.0,
            "verbose_logs": False,
        },
        "BATCH_SIZE": 9,
        "REC_FPS": 3,
        "LOOP_FPS": 10.0,
        "PERF_LOG_INTERVAL_SEC": 10.0,
        "REC_PRE_SEC": 10,
        "REC_POST_SEC": 10,
        "VIDEO_EVENT_MARK_SEC": 2.0,
        "VIDEO_EVENT_BORDER_THICKNESS": 18,
        "EVENT_FRAME_SAVE_DELAY_SEC": 10.0,
        "EVENT_FRAME_SAVE_MAX_COUNT": 0,
        "OUTPUT_RETENTION_DAYS": 14,
        "OUTPUT_CLEANUP_INTERVAL_SEC": 86400,
        "ROI_SETUP_REQUIRED_API_ENABLED": False,
        "INTERACTIVE_INPUT_GUARD_SEC": 0.35,
        "VISUAL_ALARM_DURATION": 5.0,
    }

    if not os.path.exists(CONFIG_COMMON_FILE):
        try:
            with open(CONFIG_COMMON_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as exc:
            print(f"[Warning] 기본 설정 파일 저장 실패: {exc}")
        return default_config

    try:
        with open(CONFIG_COMMON_FILE, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        migrated_config = _migrate_system_config(loaded_config)
        merged_config = deep_merge_dict(default_config, migrated_config)
        merged_config = _migrate_system_config(merged_config)

        try:
            with open(CONFIG_COMMON_FILE, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, indent=4, ensure_ascii=False)
        except Exception as exc:
            print(f"[Warning] 설정 파일 마이그레이션 쓰기 실패: {exc}")
        return merged_config
    except Exception as exc:
        print(f"[Warning] 설정 파일 로드 실패. 기본값을 사용합니다: {exc}")
        return default_config

SYS_CFG = load_system_config()
BATCH_SIZE = SYS_CFG.get("BATCH_SIZE", 9)
IMAGE_SAVER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def get_model_input_shape(model_key, default=(640, 640)):
    shapes = SYS_CFG.get("model_input_shapes", {})
    val = shapes.get(model_key)
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (int(val[0]), int(val[1]))  # (Width, Height)
    return default

def resolve_model_path(model_path):
    """설정 파일의 모델 경로가 상대 경로면 프로젝트 폴더 기준 절대 경로로 바꿉니다."""
    if not model_path:
        return model_path
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(PROJECT_ROOT, model_path)

def get_model_output_format(model_key):
    formats = SYS_CFG.get("model_output_formats", {})
    return str(formats.get(model_key, "auto")).strip().lower()

def get_main_model_output_format(model_path=None):
    return get_model_output_format("MAIN")

def get_model_engine_pool_size(model_key, default=1):
    sizes = SYS_CFG.get("model_engine_pool_sizes", {})
    try:
        return max(1, int(sizes.get(model_key, default)))
    except Exception:
        return max(1, int(default))

def detection_array(rows):
    """추론 결과 리스트를 트래커/로그가 기대하는 Nx6 numpy 배열 형태로 맞춥니다."""
    if rows is None or len(rows) == 0:
        return np.empty((0, 6))
    return np.array(rows, dtype=float)

def split_unified_event_detections(
    raw_dets,
    main_conf,
    person_conf,
    max_area_threshold,
    class_ids=None,
):
    selected_class_ids = set(class_ids or PCTC_MAIN_TRACK_CLASS_IDS)
    selected_class_ids &= PCTC_MAIN_TRACK_CLASS_IDS
    filtered = []

    if raw_dets is None:
        return detection_array(filtered)

    for det in raw_dets:
        if len(det) < 6:
            continue
        obj_w = max(0.0, float(det[2]) - float(det[0]))
        obj_h = max(0.0, float(det[3]) - float(det[1]))
        if (obj_w * obj_h) > float(max_area_threshold):
            continue

        class_id = int(det[5])
        if class_id in PCTC_MAIN_IGNORED_CLASS_IDS or class_id not in selected_class_ids:
            continue

        confidence = float(det[4])
        threshold = person_conf if class_id == ID_PCTC_PERSON else main_conf
        if confidence >= threshold:
            filtered.append(det)

    return detection_array(filtered)

def split_helmet_model_detections(
    raw_dets,
    helmet_conf,
    person_conf,
    max_area_threshold,
    person_class_id=ID_H_PERSON,
):
    """Split the dedicated HELMET model into helmet/head tracks and person-assist detections.

    HELMET model classes are 0=helmet, 1=head, 2=person.  The person result is remapped
    to the shared PCTC person class ID before it reaches the main tracker.
    """
    safety_rows = []
    person_rows = []
    if raw_dets is None:
        return detection_array(safety_rows), detection_array(person_rows)

    for det in raw_dets:
        if len(det) < 6:
            continue
        obj_w = max(0.0, float(det[2]) - float(det[0]))
        obj_h = max(0.0, float(det[3]) - float(det[1]))
        if (obj_w * obj_h) > float(max_area_threshold):
            continue

        class_id = int(det[5])
        confidence = float(det[4])
        if class_id in (ID_H_HELMET, ID_H_HEAD):
            if confidence >= helmet_conf:
                safety_rows.append(det)
        elif class_id == int(person_class_id) and confidence >= person_conf:
            person_det = list(map(float, det[:6]))
            person_det[5] = float(ID_PCTC_PERSON)
            person_rows.append(person_det)

    return detection_array(safety_rows), detection_array(person_rows)


def _fuse_person_detection(main_det, helmet_det):
    """Fuse two matched person detections into one shared person box.

    Coordinates use the union box so a close-up partial detection from one model does not
    discard body extent supplied by the other model. Confidence uses the stronger score.
    """
    return [
        min(float(main_det[0]), float(helmet_det[0])),
        min(float(main_det[1]), float(helmet_det[1])),
        max(float(main_det[2]), float(helmet_det[2])),
        max(float(main_det[3]), float(helmet_det[3])),
        max(float(main_det[4]), float(helmet_det[4])),
        float(ID_PCTC_PERSON),
    ]


def merge_person_detections(main_dets, helmet_person_dets, iou_threshold=0.35):
    """One-to-one IoU fusion for PCTC-person and HELMET-person detections.

    Non-person PCTC objects pass through untouched. Person pairs are greedily matched by
    descending IoU, fused once, and unmatched detections from either model are preserved.
    The returned array therefore contains at most one bbox for each matched physical person.
    """
    main_rows = [list(map(float, det[:6])) for det in (main_dets if main_dets is not None else []) if len(det) >= 6]
    helmet_rows = [list(map(float, det[:6])) for det in (helmet_person_dets if helmet_person_dets is not None else []) if len(det) >= 6]

    non_person = [det for det in main_rows if int(det[5]) != ID_PCTC_PERSON]
    main_persons = [det for det in main_rows if int(det[5]) == ID_PCTC_PERSON]
    helmet_persons = [det for det in helmet_rows if int(det[5]) == ID_PCTC_PERSON]

    threshold = min(1.0, max(0.0, float(iou_threshold)))
    candidates = []
    for main_index, main_det in enumerate(main_persons):
        for helmet_index, helmet_det in enumerate(helmet_persons):
            iou = float(calculate_iou(main_det[:4], helmet_det[:4]))
            if iou >= threshold:
                candidates.append((iou, main_index, helmet_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_main = set()
    matched_helmet = set()
    fused_persons = []
    for _iou, main_index, helmet_index in candidates:
        if main_index in matched_main or helmet_index in matched_helmet:
            continue
        matched_main.add(main_index)
        matched_helmet.add(helmet_index)
        fused_persons.append(_fuse_person_detection(main_persons[main_index], helmet_persons[helmet_index]))

    fused_persons.extend(det for index, det in enumerate(main_persons) if index not in matched_main)
    fused_persons.extend(det for index, det in enumerate(helmet_persons) if index not in matched_helmet)
    return detection_array(non_person + fused_persons), {
        "main_person": len(main_persons),
        "helmet_person": len(helmet_persons),
        "fused_pairs": len(matched_main),
        "unmatched_main": len(main_persons) - len(matched_main),
        "unmatched_helmet": len(helmet_persons) - len(matched_helmet),
    }


def draw_abnormal_drive_zones(image, zones, thickness=2):
    if image is None:
        return image
    for index, zone in enumerate(zones or [], start=1):
        polygon = zone.get("roi_poly", []) if isinstance(zone, dict) else []
        direction = zone.get("direction_points", []) if isinstance(zone, dict) else []
        if len(polygon) >= 3:
            poly = np.array(polygon, dtype=np.int32)
            cv2.polylines(image, [poly], True, (255, 0, 255), thickness)
            label_point = tuple(map(int, polygon[0]))
            cv2.putText(
                image,
                f"ABNORMAL ROI #{index}",
                (label_point[0], max(18, label_point[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        if len(direction) == 2:
            start_pt = tuple(map(int, direction[0]))
            end_pt = tuple(map(int, direction[1]))
            cv2.arrowedLine(image, start_pt, end_pt, (255, 0, 255), thickness, tipLength=0.2)
            cv2.putText(
                image,
                f"ALLOWED #{index}",
                (start_pt[0], max(18, start_pt[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return image


def create_roi_snapshot(cam, frame):
    """현재 카메라 프레임에 공용 ROI, abnormal-drive ROI와 이벤트명을 그립니다."""
    if frame is None:
        return None
    image = frame.copy()

    if cam.roi_poly and len(cam.roi_poly) > 2:
        cv2.polylines(image, [np.array(cam.roi_poly, np.int32)], True, (0, 255, 255), 2)
    draw_abnormal_drive_zones(image, getattr(cam, "abnormal_drive_zones", []), thickness=2)

    y_pos = 30
    for event_name in cam.events:
        if event_name in (ROI_CHANGE_EVENT, ROI_CHANGE_APPLY_EVENT):
            continue
        display_name = EVENT_REGISTRY[event_name].gui_name if event_name in EVENT_REGISTRY else event_name.upper()
        cv2.putText(image, f"Event: {display_name}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        y_pos += 30

    return image

def _send_roi_snapshot_task( cam_id, terminal_id, img, roi_info_str, w, h, is_req_roi_setup=False, send_type="hourly"):
    """관제 서버로 ROI 스냅샷을 백그라운드에서 전송합니다."""
    url = "1https://tmlsafety.hudaters.net/receiver/api/v1/cctv/roi/img"
    send_label = {
        "roi_check_5min": "5분 ROI 화각검사",
        "roi_refresh": "ROI 설정반영 스냅샷",
        "hourly": "1시간 정기 ROI 스냅샷",
    }.get(str(send_type), f"ROI 스냅샷({send_type})")
    try:
        # 이미지를 메모리 상에서 JPEG 바이너리로 인코딩
        _, img_encoded = cv2.imencode('.jpg', img)

        data = {
            "terminalId": str(terminal_id),
            "cctvId": int(cam_id),
            "imageWidth": int(w),
            "imageHeight": int(h),
            "cctvServerId": "1",
            "isReqRoiSetup": bool(is_req_roi_setup),
            "roiInfo": roi_info_str
        }

        files = {
            "image": (f"snapshot_cam{cam_id}.jpg", img_encoded.tobytes(), "image/jpeg")
        }

        resp = requests.post(url, data=data, files=files, verify=False, timeout=15)

        if resp.status_code == 200:
            logger.info(
                f" [ROI Snapshot][{send_label}] CAM:{cam_id} 전송 성공 "
                f"isReqRoiSetup={bool(is_req_roi_setup)}"
            )
        else:
            logger.error(
                f" [ROI Snapshot][{send_label}] CAM:{cam_id} API 에러 "
                f"({resp.status_code}): {resp.text}"
            )
    except Exception as e:
        logger.error(f" [ROI Snapshot][{send_label}] CAM:{cam_id} 전송 실패: {e}")

# ==========================================
# [2] 로깅 시스템 초기화
# ==========================================
LOG_DIR = SYS_CFG.get("logging", {}).get("dir", "./logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

def _parse_log_level(value, default=logging.INFO):
    return getattr(logging, str(value or "").upper(), default)

_log_cfg = SYS_CFG.get("logging", {})
_file_log_level = _parse_log_level(_log_cfg.get("file_level", _log_cfg.get("level", "INFO")), logging.INFO)
_console_log_level = _parse_log_level(_log_cfg.get("console_level", "INFO"), logging.INFO)
_logger_floor_level = min(_file_log_level, _console_log_level)

logger = logging.getLogger("CCTV_SYSTEM")
logger.setLevel(_logger_floor_level)
formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

log_filename = datetime.datetime.now().strftime("cctv_%Y%m%d.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

log_retention_days = max(1, int(SYS_CFG.get("logging", {}).get("retention_days", 14)))
file_handler = TimedRotatingFileHandler(log_filepath, when="H", interval=1, backupCount=24 * log_retention_days, encoding='utf-8')
file_handler.setLevel(_file_log_level)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(_console_log_level)
stream_handler.setFormatter(formatter)

# 비동기 로깅을 위한 큐(Queue) 설정
log_queue = queue.Queue(-1)
queue_handler = QueueHandler(log_queue)
queue_handler.setLevel(_logger_floor_level)
logger.addHandler(queue_handler)

LOG_LISTENER = QueueListener(log_queue, file_handler, stream_handler, respect_handler_level=True)
LOG_LISTENER.start()

def graceful_shutdown():
    """시스템 종료 시 스레드 풀과 로거를 안전하게 정리합니다."""
    logger.info("[SYSTEM] Waiting for background I/O tasks to finish.")
    try:
        IMAGE_SAVER_POOL.shutdown(wait=True)
    except Exception:
        pass
    if LOG_LISTENER is not None:
        try:
            LOG_LISTENER.stop()
        except Exception:
            pass

atexit.register(graceful_shutdown)

def cleanup_old_files(root_dir, retention_days, label):
    """지정된 폴더에서 보관 기간이 지난 파일을 삭제하고 빈 폴더를 정리합니다."""
    # 산출물은 계속 쌓이면 디스크를 채우기 때문에 14일 같은 보관 기간을 둡니다.
    # 삭제 기준은 파일의 마지막 수정 시간입니다. 즉, retention_days보다 오래된 파일만 지웁니다.
    # root_dir 바깥은 건드리지 않고, os.walk로 root_dir 내부만 순회합니다.
    if retention_days <= 0:
        logger.info(f"[Retention] {label} 보관 정리가 비활성화되어 있습니다.")
        return

    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        logger.debug(f"[Retention] {label} 폴더가 아직 없습니다: {root_dir}")
        return

    cutoff_ts = time.time() - (float(retention_days) * 86400.0)
    removed_files = 0
    removed_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                if os.path.getmtime(file_path) < cutoff_ts:
                    os.remove(file_path)
                    removed_files += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"[Retention] 오래된 {label} 파일 삭제 실패: {file_path} | {e}")

        if dirpath == root_dir:
            continue

        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
                removed_dirs += 1
        except OSError:
            pass
        except Exception as e:
            logger.debug(f"[Retention] 빈 {label} 폴더 정리 실패: {dirpath} | {e}")

    if removed_files or removed_dirs:
        logger.info(f"[Retention] {label} {retention_days}일 보관 정리 완료: 파일 {removed_files}개, 빈 폴더 {removed_dirs}개 삭제")

def run_output_retention_cleanup(retention_days):
    # 이벤트 산출물: 원본 이미지, 원본 영상, infer JSONL, BBox JSON 등이 모두 포함됩니다.
    cleanup_old_files(EVENT_ROOT_DIR, retention_days, "이벤트 산출물")
    # 실행 로그도 같은 보관 정책으로 한 번 더 정리합니다.
    # TimedRotatingFileHandler가 시간 단위 회전을 맡고, 이 함수가 오래된 날짜 파일을 보조 정리합니다.
    cleanup_old_files(LOG_DIR, retention_days, "실행 로그")

# ==========================================
# [3] 딥엑스 NPU 엔진 및 환경변수 설정
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"

# [수정] sys.exit(1) 강제 종료를 제거하고 상태 플래그(HAS_DX_ENGINE) 도입
HAS_DX_ENGINE = False
try:
    from dx_engine import InferenceEngine, InferenceOption
    HAS_DX_ENGINE = True
except ImportError:
    logger.warning(" [환경 알림] dx_engine 모듈을 찾을 수 없습니다. 서버(GPU/CPU) 환경으로 간주합니다.")

# ==========================================
# [4] 공통 유틸리티 함수
# ==========================================
def sanitize_camera_url(url: str) -> str:
    """RTSP URL에서 공백 및 불순물을 제거합니다."""
    if not url: return ""
    try:
        clean_url = url.encode('ascii', 'ignore').decode('ascii')
        return re.sub(r'\s+', '', clean_url.strip())
    except Exception:
        return re.sub(r'\s+', '', str(url).strip())

def extract_ip(rtsp_url: str) -> str:
    """RTSP URL에서 식별용 고유 ID(IP+Channel)를 추출합니다."""
    try:
        clean_url = sanitize_camera_url(rtsp_url)
        if "://" not in clean_url:
            clean_url = f"rtsp://{clean_url}"

        parsed = urlsplit(clean_url)
        host = parsed.netloc.rsplit("@", 1)[-1].strip("[]").split(":")[0].split(".")[-1]

        # Path와 Query(채널 정보 등)를 포함하여 고유한 키 생성
        path = re.sub(r'[^a-zA-Z0-9]', '_', parsed.path)
        query = re.sub(r'[^a-zA-Z0-9]', '_', parsed.query)

        uid = f"{host}{path}_{query}".strip('_')
        return uid if uid else "unknown_cam"
    except Exception as e:
        logger.warning(f"고유 식별자 추출 실패: {e}")
        return "unknown_cam"

def load_rtsp_list_from_csv(csv_path):
    """CSV 파일에서 카메라 RTSP URL 목록을 로드합니다."""
    if not os.path.exists(csv_path):
        logger.error(f"카메라 목록 CSV를 찾을 수 없습니다: {csv_path}")
        return []

    rtsp_list = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                first_col = line.split(',')[0].strip()
                if first_col.lower() in ['url', 'rtsp', 'rtsp_url', 'camera_url']:
                    continue
                url = sanitize_camera_url(first_col)
                if url:
                    rtsp_list.append(url)
    except Exception as e:
        logger.error(f"카메라 리스트 로드 중 예외 발생: {e}")
        pass

    unique_list = []
    for u in rtsp_list:
        if u not in unique_list:
            unique_list.append(u)

    logger.info(f"카메라 CSV 로드 완료: {len(unique_list)}대")
    return unique_list

def calculate_iou(box1, box2):
    """두 BBox 간의 IoU(Intersection over Union)를 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area)

def get_foot_point(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int(y1 + (y2 - y1) * (2/3)))

def get_check_point(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int(y2))

def get_center_point(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def to_json_safe(value):
    """numpy 값, tuple, deque 등을 JSON 저장 가능한 기본 타입으로 바꿉니다."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value

def int_box(box):
    return [int(round(float(v))) for v in box]

def int_point(pt):
    return [int(round(float(pt[0]))), int(round(float(pt[1])))]

def now_kst():
    return datetime.datetime.now(pytz.timezone('Asia/Seoul'))

def safe_id_part(value):
    text = str(value if value is not None else "-")
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")
    return text or "-"

def make_event_id(cam_id, ip, event_name, tid, fid, ts=None):
    ts = ts or now_kst()
    stamp = ts.strftime('%Y%m%dT%H%M%S%f')[:-3]
    return (
        f"{stamp}_cam{safe_id_part(cam_id)}_{safe_id_part(ip)}_"
        f"{safe_id_part(event_name)}_tid{safe_id_part(tid)}_fid{safe_id_part(fid)}"
    )

def append_event_audit_record(record, stage="event", status="ok", extra=None):
    if not SYS_CFG.get("logging", {}).get("event_audit_enabled", True):
        return
    try:
        audit_record = dict(record or {})
        if extra:
            audit_record.update(extra)
        audit_record.setdefault("ts", now_kst().isoformat())
        audit_record["audit_stage"] = stage
        audit_record["artifact_status"] = status
        ts_text = str(audit_record.get("ts", ""))
        day = ts_text[:10].replace("-", "") if len(ts_text) >= 10 else now_kst().strftime('%Y%m%d')
        audit_path = os.path.join(EVENT_ROOT_DIR, "logs", f"event_{day}.jsonl")
        with EVENT_AUDIT_LOCK:
            _write_jsonl_records(audit_path, [audit_record])
    except Exception as e:
        logger.error(f"[EVENT AUDIT] write failed | event_id={(record or {}).get('event_id', '-')} | {e}")

def log_disk_health(paths, threshold_gb=None):
    try:
        threshold_gb = float(
            threshold_gb
            if threshold_gb is not None
            else SYS_CFG.get("logging", {}).get("disk_free_warn_gb", 5.0)
        )
    except Exception:
        threshold_gb = 5.0

    seen_roots = set()
    for label, path in paths:
        target = os.path.abspath(path or PROJECT_ROOT)
        while not os.path.exists(target):
            parent = os.path.dirname(target)
            if parent == target:
                target = PROJECT_ROOT
                break
            target = parent
        if target in seen_roots:
            continue
        seen_roots.add(target)
        try:
            usage = shutil.disk_usage(target)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            
            # [최적화] 디스크가 위험 수준일 때만 알림을 울리고 평시 도배 로그는 삭제함
            if free_gb < threshold_gb:
                logger.warning(
                    f"⚠️ [DISK HEALTH] label={label} path={target} free_gb={free_gb:.2f} "
                    f"total_gb={total_gb:.2f} threshold_gb={threshold_gb:.2f}"
                )
        except Exception as e:
            logger.warning(f"[DISK HEALTH] check failed | label={label} path={path} | {e}")

def record_api_send_state(ok, event_id="-"):
    with API_SEND_STATE_LOCK:
        failures = int(API_SEND_STATE.get("consecutive_failures", 0) or 0)
        if ok:
            if failures > 0:
                logger.info(f"[API SEND RECOVERED] event_id={event_id or '-'} previous_failures={failures}")
            API_SEND_STATE["consecutive_failures"] = 0
            API_SEND_STATE["last_failure_at"] = None
            return

        failures += 1
        API_SEND_STATE["consecutive_failures"] = failures
        API_SEND_STATE["last_failure_at"] = now_kst().isoformat()
        if failures in (1, 3, 10) or failures % 10 == 0:
            logger.warning(f"[API SEND DEGRADED] event_id={event_id or '-'} consecutive_failures={failures}")

def get_git_metadata():
    meta = {"commit": "-", "branch": "-"}
    try:
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        ).strip()
        meta["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        ).strip()
    except Exception:
        pass
    return meta

def ccw(p1, p2, p3):
    """세 점의 방향성을 판별합니다. (선분 교차 알고리즘용)"""
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val > 0: return 1
    elif val < 0: return -1
    return 0

def normalize_roi_points(points, width, height):
    if not points or width <= 0 or height <= 0:
        return []
    return [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in points]

def denormalize_roi_points(points, width, height):
    if not points or width <= 0 or height <= 0:
        return []
    return [[int(round(float(x) * width)), int(round(float(y) * height))] for x, y in points]

def _sanitize_norm_points(value, min_points=0, exact_points=None):
    if not isinstance(value, list):
        return []
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            return []
        if not (math.isfinite(x) and math.isfinite(y)):
            return []
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return []
        points.append([round(x, 6), round(y, 6)])
    if exact_points is not None and len(points) != int(exact_points):
        return []
    if len(points) < int(min_points):
        return []
    return points


def sanitize_abnormal_drive_zones(value):
    if not isinstance(value, list):
        return []
    valid_zones = []
    for zone in value:
        if not isinstance(zone, dict):
            continue
        polygon = _sanitize_norm_points(zone.get("roi_poly_norm"), min_points=3)
        direction = _sanitize_norm_points(zone.get("direction_points_norm"), exact_points=2)
        if len(polygon) < 3 or len(direction) != 2 or direction[0] == direction[1]:
            continue
        valid_zones.append({
            "roi_poly_norm": polygon,
            "direction_points_norm": direction,
        })
    return valid_zones


def sanitize_camera_config(raw_conf):
    conf = dict(raw_conf or {})
    raw_events = conf.get("events") if isinstance(conf.get("events"), list) else []
    events = []
    for raw_name in raw_events:
        event_name = EVENT_NAME_ALIASES.get(str(raw_name), str(raw_name))
        if event_name not in SUPPORTED_CAMERA_EVENTS or event_name in events:
            continue
        events.append(event_name)

    roi_poly_norm = _sanitize_norm_points(conf.get("roi_poly_norm"), min_points=0)
    zones = sanitize_abnormal_drive_zones(conf.get("abnormal_drive_zones_norm"))

    legacy_direction = []
    for legacy_key in ("wrong_way_direction_norm", "abnormal_drive_direction_norm"):
        if not legacy_direction:
            legacy_direction = _sanitize_norm_points(conf.get(legacy_key), exact_points=2)
        conf.pop(legacy_key, None)

    if (
        "abnormal_drive" in events
        and not zones
        and len(roi_poly_norm) >= 3
        and len(legacy_direction) == 2
        and legacy_direction[0] != legacy_direction[1]
    ):
        zones = [{
            "roi_poly_norm": roi_poly_norm,
            "direction_points_norm": legacy_direction,
        }]

    if not any(event_name in ROI_POLYGON_EVENT_NAMES for event_name in events):
        roi_poly_norm = []

    conf["events"] = events
    conf["roi_poly_norm"] = roi_poly_norm
    conf["roi_lines_norm"] = []
    conf["abnormal_drive_zones_norm"] = zones
    return conf


def sanitize_camera_configs(raw_configs):
    if not isinstance(raw_configs, dict):
        return {}
    return {
        str(camera_key): sanitize_camera_config(camera_conf)
        for camera_key, camera_conf in raw_configs.items()
        if isinstance(camera_conf, dict)
    }

def create_mosaic_image(images, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT):
    """여러 카메라의 영상을 하나의 모자이크 화면으로 합성합니다."""
    if not images:
        return None

    count = len(images)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))

    cell_w = screen_w // cols
    cell_h = screen_h // rows

    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h

        if img is None:
            cell_img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(cell_img, "No Signal", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cell_img = cv2.resize(img, (cell_w, cell_h))

        mosaic[y:y+cell_h, x:x+cell_w] = cell_img
        cv2.rectangle(mosaic, (x, y), (x+cell_w, y+cell_h), (100, 100, 100), 1)

    return mosaic

# ==========================================
# [5] API 통신 및 이미지 저장 (NAS 연동 제외)
# ==========================================
def send_event_image_to_receiver(image_path, event_name, terminal_id, cctv_id, bboxes, img_width=None, img_height=None, event_id=None):
    """수신 서버(Receiver API)로 이벤트 이미지를 POST 전송합니다."""
    event_id = event_id or "-"
    api_audit = {
        "event_id": event_id,
        "event_name": event_name,
        "terminal_id": str(terminal_id),
        "cctv_id": int(cctv_id),
        "image_path": image_path,
        "bbox_count": len(bboxes or [])
    }
    if(terminal_id == "99999"):
        logger.debug(f"[API SKIP] event_id={event_id} reason=default_terminal image={image_path}")
        append_event_audit_record(api_audit, stage="api_send", status="skipped", extra={"reason": "default_terminal"})
        logger.debug(f"[API 스킵] 기본 단말 ID(99999) 사용 중: {image_path}")
        return

    url = "1https://tmlsafety.hudaters.net/receiver/api/v1/cctv/img"
    event_type_mapping = {
        "no_helmet": 2,
        "illegal_parking": 4,
        "intrusion": 5,
    }

    if event_name not in event_type_mapping:
        logger.debug(f"[API SKIP] event_id={event_id} reason=unknown_event event={event_name}")
        append_event_audit_record(api_audit, stage="api_send", status="skipped", extra={"reason": "unknown_event"})
        logger.debug(f"[API 스킵] 정의되지 않은 이벤트 타입: {event_name}")
        return

    api_event_type = event_type_mapping[event_name]
    collected_at = now_kst().strftime('%Y-%m-%dT%H:%M:%S')

    # [검증 완료] bboxes 배열(리스트 내 딕셔너리)을 JSON 문자열로 안전하게 직렬화
    detected_objects_json = json.dumps(bboxes) if bboxes else "[]"

    data = {
        "collectedAt": collected_at,
        "eventType": api_event_type,
        "terminalId": str(terminal_id),
        "cctvId": int(cctv_id),
        "detectedObjects": detected_objects_json
    }

    if img_width: data["imageWidth"] = int(img_width)
    if img_height: data["imageHeight"] = int(img_height)

    if not os.path.exists(image_path):
        logger.error(f"[API ERROR] event_id={event_id} image_missing={image_path}")
        record_api_send_state(False, event_id)
        append_event_audit_record(api_audit, stage="api_send", status="failed", extra={"reason": "image_missing"})
        logger.error(f"[API 에러] 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        started_at = time.monotonic()
        with open(image_path, 'rb') as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, data=data, files=files, verify=False, timeout=10)
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            bbox_count = len(bboxes or [])

            if response.status_code == 200:
                logger.info(
                    f"[API SEND OK] event_id={event_id} terminal={terminal_id} cam={cctv_id} "
                    f"event={event_name} status={response.status_code} elapsed_ms={elapsed_ms} "
                    f"bbox_count={bbox_count} image={os.path.basename(image_path)}"
                )
                record_api_send_state(True, event_id)
                append_event_audit_record(
                    api_audit,
                    stage="api_send",
                    status="ok",
                    extra={"status_code": response.status_code, "elapsed_ms": elapsed_ms}
                )
                logger.info(f" [API 전송 성공] 단말:{terminal_id} | CAM:{cctv_id} | 이벤트:{event_name}")
            else:
                logger.error(f" [API 전송 실패] 상태코드: {response.status_code} | 메시지: {response.text}")
            if response.status_code != 200:
                logger.error(
                    f"[API SEND FAIL] event_id={event_id} terminal={terminal_id} cam={cctv_id} "
                    f"event={event_name} status={response.status_code} elapsed_ms={elapsed_ms} "
                    f"bbox_count={bbox_count} body={response.text}"
                )
                record_api_send_state(False, event_id)
                append_event_audit_record(
                    api_audit,
                    stage="api_send",
                    status="failed",
                    extra={"status_code": response.status_code, "elapsed_ms": elapsed_ms}
                )
    except requests.exceptions.RequestException as e:
        logger.error(f"[API NETWORK ERROR] event_id={event_id} image={image_path} | {e}")
        record_api_send_state(False, event_id)
        append_event_audit_record(api_audit, stage="api_send", status="failed", extra={"reason": "network_error"})
        logger.error(f" [API 네트워크 예외 발생]: {e}")
    except Exception as e:
        logger.error(f"[API UNEXPECTED ERROR] event_id={event_id} image={image_path} | {e}\n{traceback.format_exc()}")
        record_api_send_state(False, event_id)
        append_event_audit_record(api_audit, stage="api_send", status="failed", extra={"reason": "unexpected_error"})
        logger.error(f" [API 기타 예외 발생]: {e}\n{traceback.format_exc()}")

def _draw_event_api_image(frame, event_type, bbox, tid, objects_meta=None):
    """관제 API용 이미지에 이벤트 객체를 표시합니다."""
    api_img = frame.copy()
    draw_items = objects_meta or [{"label": event_type, "box": bbox, "tid": tid}]
    drawn = False

    for obj in draw_items:
        try:
            x1, y1, x2, y2 = int_box(obj.get("box", bbox))
        except Exception:
            continue
        class_name = str(obj.get("class_name") or obj.get("label") or event_type)
        cv2.rectangle(api_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            api_img,
            f"{event_type}: {class_name}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        drawn = True

    if not drawn:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(api_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(api_img, str(event_type), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    return api_img

def _write_image_file(path, image, label="image", event_id="-"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ok = cv2.imwrite(path, image)
        if not ok:
            logger.error(f"[EVIDENCE SAVE FAIL] event_id={event_id or '-'} label={label} path={path} reason=cv2.imwrite_false")
            return False
        logger.debug(f"[EVIDENCE SAVE OK] event_id={event_id or '-'} label={label} path={path}")
        return True
    except Exception as e:
        logger.error(f"[EVIDENCE SAVE FAIL] event_id={event_id or '-'} label={label} path={path} | {e}")
        return False

def _save_and_send_task(img, img_path, api_img, api_img_path, api_params):
    event_id = api_params.get('event_id', '-')
    audit_base = {
        "event_id": event_id,
        "ts": api_params.get('event_ts', now_kst().isoformat()),
        "event_name": api_params.get('event_name', '-'),
        "terminal_id": api_params.get('terminal_id', '-'),
        "cctv_id": api_params.get('cctv_id', '-'),
        "ip": api_params.get('ip', '-'),
        "image_path": img_path,
        "api_img_path": api_img_path
    }
    """비동기 스레드에서 파일 쓰기 및 API 전송을 처리합니다."""
    try:
        if not _write_image_file(img_path, img, label="event_image", event_id=event_id):
            append_event_audit_record(audit_base, stage="event_image_saved", status="failed", extra={"failed_path": img_path})
            return
        append_event_audit_record(audit_base, stage="event_image_saved", status="ok", extra={"saved_path": img_path})
    except Exception as e:
        logger.error(f"[이미지 저장 실패] 경로: {img_path} | 예외: {e}")
        return

    try:
        if not _write_image_file(api_img_path, api_img, label="api_image", event_id=event_id):
            append_event_audit_record(audit_base, stage="api_image_saved", status="failed", extra={"failed_path": api_img_path})
            return
        append_event_audit_record(audit_base, stage="api_image_saved", status="ok", extra={"saved_path": api_img_path})
    except Exception as e:
        logger.error(f"[API image save failed] path: {api_img_path} | error: {e}")
        return

    try:
        send_event_image_to_receiver(
            image_path=api_img_path,
            event_name=api_params['event_name'],
            terminal_id=api_params['terminal_id'],
            cctv_id=api_params['cctv_id'],
            bboxes=api_params['bboxes'],
            img_width=api_params['img_width'],
            img_height=api_params['img_height'],
            event_id=event_id
        )
    except Exception as e:
        logger.error(f"[Task 내부 API 호출 에러] {e}")

def _write_jsonl_records(path, records):
    # JSONL은 "한 줄에 기록 하나"인 로그 파일입니다.
    # 영상 편집 프로그램이 없어도 메모장으로 열어서 특정 프레임의 탐지 결과를 줄 단위로 확인할 수 있습니다.
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(to_json_safe(record), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[InferLog] JSONL write failed: {path} | {e}")

def save_event_image_with_mark(frame, ip, event_type, bbox, tid, terminal_id="99999", cctv_id=1, objects_meta=None, trajectories=None, event_id=None, event_ts=None):
    """원본 프레임 이미지를 로컬에 저장하고 탐지 메타데이터를 API 큐에 등록합니다."""
    if IMAGE_SAVER_POOL._work_queue.qsize() > 50:
        logger.warning("이미지 저장 큐가 포화 상태입니다. 저장을 스킵합니다.")
        return

    try:
        img = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        now = datetime.datetime.now()
        event_ts = event_ts or now_kst().isoformat()
        event_id = event_id or make_event_id(cctv_id, ip, event_type, tid, "unknown", now_kst())

        dpath = os.path.join(EVENT_ROOT_DIR, "events", ip, "images", str(event_type))
        api_dpath = os.path.join(EVENT_ROOT_DIR, "events", ip, "images_api", str(event_type))
        os.makedirs(dpath, exist_ok=True)
        os.makedirs(api_dpath, exist_ok=True)

        fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{ip}_{event_type}_{tid}.jpg"
        img_path = os.path.join(dpath, fname)
        api_img_path = os.path.join(api_dpath, fname)
        evidence_paths = {
            "event_id": event_id,
            "ts": event_ts,
            "image_path": img_path,
            "api_img_path": api_img_path,
            "image_basename": fname
        }
        
        api_img = _draw_event_api_image(img, event_type, [x1, y1, x2, y2], tid, objects_meta)

        h, w = frame.shape[:2]

        if objects_meta:
            ai_detected_bboxes = []
            for o in objects_meta:

                item = {
                    "box": [int(b) for b in o['box']],
                    "label": str(o.get("class_name") or o.get("label") or event_type),
                    "class_name": str(o.get("class_name") or pctc_class_name(o.get("class_id", -1))),
                    "class_id": int(o.get("class_id", -1)),
                    "score": round(float(o.get('score', 0.95)), 2)
                }
                if o.get('tid') is not None:
                    item["tid"] = int(o.get('tid'))
                ai_detected_bboxes.append(item)
        else:
            ai_detected_bboxes = [
                {
                    "box": [x1, y1, x2, y2],
                    "label": str(event_type),
                    "score": 0.95,
                    "tid": int(tid)
                }
            ]

        api_params = {
            'ip': ip,
            'event_id': event_id,
            'event_ts': event_ts,
            'event_name': event_type,
            'terminal_id': str(terminal_id),
            'cctv_id': int(cctv_id),
            'bboxes': ai_detected_bboxes,
            'img_width': w,
            'img_height': h
        }

        IMAGE_SAVER_POOL.submit(_save_and_send_task, img, img_path, api_img, api_img_path, api_params)
        append_event_audit_record(
            {
                "event_id": event_id,
                "ts": event_ts,
                "event_name": event_type,
                "terminal_id": str(terminal_id),
                "cctv_id": int(cctv_id),
                "ip": ip,
                "tid": int(tid),
                "bbox": [x1, y1, x2, y2],
                "image_path": img_path,
                "api_img_path": api_img_path,
                "image_basename": fname
            },
            stage="evidence_queued",
            status="queued"
        )
        logger.info(
            f"[EVIDENCE QUEUED] event_id={event_id} cam={cctv_id} event={event_type} "
            f"tid={tid} image={img_path} api_image={api_img_path}"
        )
        return evidence_paths

    except Exception as e:
        logger.error(f"[EventLogic Error] 이미지 마킹 중 예외 발생: {e}")
        return None

# ==========================================
# [6] DeepX NPU 모델 추론 (YOLOv8 버그 픽스 반영)
# ==========================================
class YoLoDeepX:
    """DEEPX inference wrapper with runtime output-format detection.

    The runtime metadata is preferred, but every first output is inspected again so a
    stale system_config.json value cannot force a PPU tensor through a raw-YOLO parser
    (or the reverse).  PPU BBOX records follow dxrt::DeviceBoundingBox_t (32 bytes).
    """

    PPU_BBOX_STRIDE = 32
    PPU_FACE_STRIDE = 64

    def __init__(
        self,
        engine_path,
        output_format="auto",
        pool_size=1,
        model_key=None,
        class_count=None,
        ppu_box_format="auto",
        output_hint=None,
        input_shape=None,
    ):
        if not HAS_DX_ENGINE:
            raise RuntimeError("dx_engine is not installed; YoLoDeepX can only run on a DeepX/NPU runtime.")

        self.engine_path = engine_path
        self.input_shape_override = input_shape
        self.model_key = str(model_key or os.path.basename(str(engine_path or "")) or "MODEL")
        self.class_count = int(class_count) if class_count is not None else None
        self.requested_output_format = str(output_format or "auto").strip().lower()
        self.configured_output_format = self._normalize_configured_output_format(output_format)
        self.output_hint = self._normalize_configured_output_format(output_hint)
        self.output_format = self.configured_output_format or self.output_hint or "auto"
        self.ppu_box_format = self._normalize_ppu_box_format(ppu_box_format)
        self.pool_size = max(1, int(pool_size or 1))
        self.engine_pool = queue.Queue(maxsize=self.pool_size)
        self.engines_ref = []

        self.input_height = 640
        self.input_width = 640
        self.input_layout = "hwc"
        self.input_has_batch = False
        self.input_dtype = np.uint8
        self.input_dtype_name = "uint8"
        self.input_metadata_detail = "metadata unavailable"

        self.output_metadata_format = None
        self.output_metadata_detail = "metadata unavailable"
        self.output_metadata_text = ""
        self.runtime_output_format = None
        self._first_output_logged = False
        self._last_empty_log_at = 0.0
        self._last_format_change = None

        runtime_cfg = SYS_CFG.get("inference_runtime", {}) if isinstance(SYS_CFG, dict) else {}
        self.log_first_output_signature = bool(runtime_cfg.get("log_first_output_signature", True))
        try:
            self.empty_detection_log_interval_sec = max(
                1.0,
                float(runtime_cfg.get("empty_detection_log_interval_sec", 10.0)),
            )
        except Exception:
            self.empty_detection_log_interval_sec = 10.0

        try:
            io = InferenceOption()
            for _ in range(self.pool_size):
                engine = InferenceEngine(self.engine_path, io)
                self.engine_pool.put(engine)
                self.engines_ref.append(engine)

            first_engine = self.engines_ref[0]
            self._load_input_shape(first_engine)
            self.output_format = self._resolve_output_format(output_format, first_engine)
            logger.info(
                f"[DeepX][{self.model_key}] model loaded: {os.path.basename(self.engine_path)} "
                f"(output={self.output_format}, requested={self.requested_output_format}, "
                f"hint={self.output_hint or '-'}, pool={self.pool_size}, "
                f"input={self.input_width}x{self.input_height}, "
                f"layout={self.input_layout}, dtype={self.input_dtype_name})"
            )
            logger.info(
                f"[DeepX][{self.model_key}] input metadata: {self.input_metadata_detail} | "
                f"output metadata: {self.output_metadata_detail}"
            )
        except Exception as exc:
            self.release()
            logger.error(f"[DeepX Load Fail][{self.model_key}] engine init failed ({engine_path}): {exc}")
            raise

    def __del__(self):
        self.release()

    def release(self):
        while hasattr(self, "engine_pool") and not self.engine_pool.empty():
            try:
                self.engine_pool.get_nowait()
            except Exception:
                break
        for engine in getattr(self, "engines_ref", []):
            try:
                del engine
            except Exception:
                pass
        self.engines_ref = []

    def _normalize_configured_output_format(self, output_format):
        fmt = str(output_format or "auto").strip().lower()
        if fmt in ("", "auto", "detect"):
            return None
        if fmt in ("ppu", "bbox", "deepx_ppu", "yolov8_ppu"):
            return "ppu"
        if fmt in ("ppu_face", "face_ppu", "face"):
            return "ppu_face"
        if fmt in ("yolo_xyxy", "xyxy"):
            return "yolo_xyxy"
        if fmt in ("yolo_tlwh", "tlwh"):
            return "yolo_tlwh"
        if fmt in ("end2end", "yolo_end2end", "nms", "xyxy_score_class"):
            return "yolo_end2end"
        if fmt in ("yolo", "yolov8", "raw", "standard", "raw_yolo"):
            return "yolo"
        logger.warning(
            f"[DeepX][{getattr(self, 'model_key', 'MODEL')}] unknown output format "
            f"'{output_format}'; runtime detection will be used."
        )
        return None

    @staticmethod
    def _normalize_ppu_box_format(value):
        fmt = str(value or "auto").strip().lower()
        if fmt in ("corner", "xyxy", "x1y1x2y2"):
            return "corner"
        if fmt in ("center", "xywh", "cxcywh"):
            return "center"
        return "auto"

    def _resolve_output_format(self, output_format, engine=None):
        configured = self._normalize_configured_output_format(output_format)
        detected, detail = self._detect_output_format_from_engine(engine)
        self.output_metadata_format = detected
        self.output_metadata_detail = detail or "metadata unavailable"

        if detected:
            if configured and configured != detected and not ({configured, detected} <= {"ppu", "ppu_face"}):
                logger.warning(
                    f"[DeepX][{self.model_key}] configured output={configured} disagrees with "
                    f"runtime metadata={detected}; metadata wins ({self.output_metadata_detail})."
                )
            return detected
        if configured:
            logger.info(
                f"[DeepX][{self.model_key}] output metadata was inconclusive; "
                f"using configured format={configured}."
            )
            return configured
        if self.output_hint:
            logger.info(
                f"[DeepX][{self.model_key}] output metadata was inconclusive; "
                f"using non-binding hint={self.output_hint} until the first output is inspected."
            )
            return self.output_hint

        fallback = self._detect_output_format_from_filename()
        logger.warning(
            f"[DeepX][{self.model_key}] output metadata was inconclusive "
            f"({self.output_metadata_detail}); initial fallback={fallback}. "
            "The first inference output will be inspected again."
        )
        return fallback

    def _detect_output_format_from_filename(self):
        model_name = os.path.basename(str(self.engine_path or "")).lower()
        if ("_ppu" in model_name or "-ppu" in model_name) and "face" in model_name:
            return "ppu_face"
        if "_ppu" in model_name or "-ppu" in model_name:
            return "ppu"
        return "yolo"

    def _detect_output_format_from_engine(self, engine):
        if engine is None:
            return None, "engine unavailable"

        info_reader = None
        for method_name in ("get_output_tensors_info", "get_outputs_info", "get_output_tensor_info"):
            if hasattr(engine, method_name):
                info_reader = getattr(engine, method_name)
                break
        if info_reader is None:
            return None, "output tensor metadata API unavailable"

        try:
            output_info = info_reader()
        except Exception as exc:
            return None, f"output tensor metadata read failed: {exc}"

        entries = self._tensor_info_entries(output_info)
        if not entries:
            return None, "empty output tensor metadata"

        metadata_text = self._safe_json_text(entries).lower()
        self.output_metadata_text = metadata_text
        shapes = [shape for shape in (self._tensor_shape(entry) for entry in entries) if shape]
        dtype_values = [self._tensor_dtype(entry) for entry in entries]
        dtypes = [str(value).lower() for value in dtype_values if str(value)]
        detail = f"shapes={shapes or '-'} dtypes={dtypes or '-'}"

        numeric_types = set()
        for value in dtype_values:
            try:
                numeric_types.add(int(value))
            except Exception:
                continue

        if any("face" in dtype for dtype in dtypes) or 11 in numeric_types:
            return "ppu_face", detail
        if any("bbox" in dtype for dtype in dtypes) or 10 in numeric_types:
            return "ppu", detail
        if re.search(r"(?:^|[^a-z])face(?:[^a-z]|$)", metadata_text) and "type" in metadata_text:
            return "ppu_face", detail
        if "bbox" in metadata_text or "postprocess" in metadata_text or "ppu" in metadata_text:
            return "ppu", detail
        if self._metadata_looks_raw_yolo(shapes, dtypes):
            return "yolo", detail
        if self._metadata_looks_ppu(shapes, dtypes):
            return "ppu", detail
        return None, detail

    @staticmethod
    def _tensor_info_entries(value):
        if value is None:
            return []
        if isinstance(value, dict):
            for key in ("outputs", "output", "tensors", "tensor_info"):
                nested = value.get(key)
                if isinstance(nested, (list, tuple)):
                    return list(nested)
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _safe_json_text(value):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    @staticmethod
    def _tensor_shape(entry):
        shape = None
        if isinstance(entry, dict):
            for key in ("shape", "dims", "dimension", "tensor_shape"):
                if key in entry:
                    shape = entry.get(key)
                    break
        else:
            for key in ("shape", "dims", "dimension", "tensor_shape"):
                if hasattr(entry, key):
                    shape = getattr(entry, key)
                    break
        if shape is None:
            return []
        try:
            return [int(x) for x in list(shape)]
        except Exception:
            return [int(x) for x in re.findall(r"-?\d+", str(shape))]

    @staticmethod
    def _tensor_dtype(entry):
        if isinstance(entry, dict):
            for key in ("dtype", "data_type", "type", "format"):
                if key in entry and entry.get(key) is not None:
                    return entry.get(key)
        else:
            for key in ("dtype", "data_type", "type", "format"):
                if hasattr(entry, key):
                    return getattr(entry, key)
        return ""

    @staticmethod
    def _shape_without_ones(shape):
        return [int(x) for x in shape if int(x) > 1]

    def _metadata_looks_raw_yolo(self, shapes, dtypes):
        for shape in shapes:
            dims = self._shape_without_ones(shape)
            if len(dims) < 2:
                continue
            if max(dims) >= 1000 and any(5 <= dim <= 512 for dim in dims):
                return True
        return False

    def _metadata_looks_ppu(self, shapes, dtypes):
        has_byte_output = any(
            dtype in ("uint8", "byte", "bytes", "void", "v32")
            or "uint8" in dtype
            or "bbox" in dtype
            for dtype in dtypes
        )
        if has_byte_output and any(self._shape_looks_ppu_rows(shape) for shape in shapes):
            return True
        return any(self._shape_looks_ppu_rows(shape) for shape in shapes) and not self._metadata_looks_raw_yolo(shapes, dtypes)

    @staticmethod
    def _shape_looks_ppu_rows(shape):
        dims = [int(x) for x in shape if int(x) > 1]
        if not dims:
            return False
        if len(dims) == 1:
            return dims[0] <= 4096 or (dims[0] % 32 == 0 and dims[0] <= 131072)
        return dims[-1] in (6, 7, 8, 32, 64) and max(dims) < 4096

    def _load_input_shape(self, engine):
        try:
            input_info = engine.get_input_tensors_info()
            entries = self._tensor_info_entries(input_info)
            if not entries:
                raise ValueError("empty input tensor metadata")
            entry = entries[0]
            shape = self._tensor_shape(entry)
            dtype_text = str(self._tensor_dtype(entry) or "uint8").lower()
            self.input_metadata_detail = f"shape={shape or '-'} dtype={dtype_text or '-'}"
        except Exception as exc:
            logger.warning(
                f"[DeepX][{self.model_key}] input metadata read failed; using HWC uint8 640x640: {exc}"
            )
            return

        if "float" in dtype_text or dtype_text in ("1", "fp32", "f32"):
            self.input_dtype = np.float32
            self.input_dtype_name = "float32"
        else:
            self.input_dtype = np.uint8
            self.input_dtype_name = "uint8"

        if len(shape) == 4:
            self.input_has_batch = True
            if shape[-1] in (1, 3, 4):
                self.input_layout = "nhwc"
                self.input_height, self.input_width = int(shape[1]), int(shape[2])
            elif shape[1] in (1, 3, 4):
                self.input_layout = "nchw"
                self.input_height, self.input_width = int(shape[2]), int(shape[3])
        elif len(shape) == 3:
            self.input_has_batch = False
            if shape[-1] in (1, 3, 4):
                self.input_layout = "hwc"
                self.input_height, self.input_width = int(shape[0]), int(shape[1])
            elif shape[0] in (1, 3, 4):
                self.input_layout = "chw"
                self.input_height, self.input_width = int(shape[1]), int(shape[2])

        if self.input_height <= 0 or self.input_width <= 0:
            self.input_height = 640
            self.input_width = 640
        if hasattr(self, 'input_shape_override') and self.input_shape_override:
            self.input_width, self.input_height = self.input_shape_override
            self.input_metadata_detail += f" (Overridden to {self.input_width}x{self.input_height})"
            
    def letter_box(self, img, new_shape=None):
        if new_shape is None:
            new_shape = (self.input_height, self.input_width)
        h, w = img.shape[:2]
        scale = min(float(new_shape[0]) / max(1, h), float(new_shape[1]) / max(1, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((int(new_shape[0]), int(new_shape[1]), 3), 114, dtype=np.uint8)
        dw = (int(new_shape[1]) - nw) // 2
        dh = (int(new_shape[0]) - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        return canvas, scale, (dw, dh)

    def _prepare_input_tensor(self, npu_input):
        # DX-RT's Python API receives one sample per list item.  A leading batch
        # dimension from model metadata must therefore not be added here.
        input_tensor = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)
        if self.input_layout in ("nchw", "chw"):
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
        if self.input_dtype == np.float32:
            input_tensor = input_tensor.astype(np.float32) / 255.0
        else:
            input_tensor = input_tensor.astype(np.uint8, copy=False)
        return np.ascontiguousarray(input_tensor)

    @staticmethod
    def _normalize_outputs(output_tensor):
        if output_tensor is None:
            return []
        if isinstance(output_tensor, (bytes, bytearray, memoryview, np.ndarray)):
            return [output_tensor]
        try:
            outputs = list(output_tensor)
        except TypeError:
            return [output_tensor]
        if len(outputs) == 1 and isinstance(outputs[0], (list, tuple)):
            return list(outputs[0])
        return outputs

    def _output_tensor_signature(self, output_tensor):
        parts = []
        for index, tensor in enumerate(self._normalize_outputs(output_tensor)[:8]):
            try:
                arr = np.asarray(tensor)
                fields = tuple(arr.dtype.names or ())
                field_text = f",fields={fields}" if fields else ""
                parts.append(
                    f"#{index}:shape={tuple(arr.shape)},dtype={arr.dtype},bytes={arr.nbytes}{field_text}"
                )
            except Exception as exc:
                try:
                    size = len(tensor)
                except Exception:
                    size = "?"
                parts.append(f"#{index}:type={type(tensor).__name__},len={size},error={exc}")
        return " | ".join(parts) if parts else "no output tensors"

    @staticmethod
    def _squeezed_2d_array(tensor):
        arr = np.asarray(tensor)
        if arr.size == 0:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        return arr

    def _looks_like_raw_yolo_array(self, tensor):
        try:
            arr = np.asarray(tensor)
            if arr.dtype.kind not in "fc":
                return False
            dims = [int(x) for x in arr.shape if int(x) > 1]
            return len(dims) >= 2 and max(dims) >= 1000 and any(5 <= dim <= 512 for dim in dims)
        except Exception:
            return False

    def _looks_like_end2end_array(self, tensor):
        try:
            arr = self._squeezed_2d_array(tensor)
            if arr is None or arr.dtype.kind not in "fc" or arr.shape[1] not in (6, 7):
                return False
            sample = np.asarray(arr[: min(64, len(arr)), :6], dtype=np.float64)
            if sample.size == 0 or not np.all(np.isfinite(sample)):
                return False
            scores = sample[:, 4]
            labels = sample[:, 5]
            return (
                np.all((scores >= -1e-6) & (scores <= 1.01))
                and np.mean(np.abs(labels - np.round(labels)) < 1e-4) >= 0.9
            )
        except Exception:
            return False

    def _to_uint8_buffer(self, tensor):
        if isinstance(tensor, (bytes, bytearray, memoryview)):
            return np.frombuffer(tensor, dtype=np.uint8).copy()
        arr = np.asarray(tensor)
        if arr.size == 0:
            return np.empty((0,), dtype=np.uint8)
        if arr.dtype.kind == "O":
            raise TypeError("object-dtype output cannot be interpreted as a packed PPU buffer")
        return np.ascontiguousarray(arr).view(np.uint8).ravel().copy()

    def _packed_ppu_summary(self, output_tensor, stride=None, conf_thres=0.0):
        outputs = self._normalize_outputs(output_tensor)
        if not outputs:
            return {"plausible": False, "reason": "no output tensors"}
        try:
            flat = self._to_uint8_buffer(outputs[0])
        except Exception as exc:
            return {"plausible": False, "reason": str(exc)}

        candidate_strides = [int(stride)] if stride else []
        for value in (self.PPU_BBOX_STRIDE, self.PPU_FACE_STRIDE):
            if value not in candidate_strides:
                candidate_strides.append(value)

        for record_stride in candidate_strides:
            if record_stride <= 0 or flat.size == 0 or flat.size % record_stride != 0:
                continue
            records = flat.reshape(flat.size // record_stride, record_stride)
            if len(records) > 4096 or record_stride < 24:
                continue
            coords = np.ascontiguousarray(records[:, :16]).view("<f4").reshape(-1, 4)
            scores = np.ascontiguousarray(records[:, 20:24]).view("<f4").reshape(-1)
            if record_stride == self.PPU_BBOX_STRIDE:
                labels = np.ascontiguousarray(records[:, 24:28]).view("<u4").reshape(-1)
            else:
                labels = np.zeros(len(scores), dtype=np.uint32)
            finite = np.isfinite(scores) & np.all(np.isfinite(coords), axis=1)
            score_range_ok = finite & (scores >= -1e-6) & (scores <= 1.01)
            nonnegative_size = finite & (coords[:, 2] >= 0.0) & (coords[:, 3] >= 0.0)
            valid_fraction = float(np.mean(score_range_ok)) if len(scores) else 0.0
            plausible = bool(
                len(scores)
                and valid_fraction >= 0.95
                and float(np.mean(nonnegative_size)) >= 0.90
            )
            positive = score_range_ok & (scores > 0.0)
            label_sample = sorted({int(x) for x in labels[positive][:32]})[:12]
            return {
                "plausible": plausible,
                "stride": record_stride,
                "records": int(len(scores)),
                "max_score": float(np.max(scores[finite])) if np.any(finite) else None,
                "min_score": float(np.min(scores[finite])) if np.any(finite) else None,
                "above_threshold": int(np.sum(score_range_ok & (scores >= float(conf_thres)))),
                "positive_scores": int(np.sum(positive)),
                "labels": label_sample,
                "valid_fraction": valid_fraction,
            }
        return {
            "plausible": False,
            "reason": f"buffer_bytes={flat.size} is not compatible with 32/64-byte PPU records",
        }

    def _detect_output_format_from_outputs(self, output_tensor):
        outputs = self._normalize_outputs(output_tensor)
        if not outputs:
            return None, "no output tensors"
        first = outputs[0]
        try:
            arr = np.asarray(first)
        except Exception as exc:
            return None, f"cannot convert first output to ndarray: {exc}"

        field_names = {str(name).lower() for name in (arr.dtype.names or ())}
        if {"x", "y", "w", "h", "score"}.issubset(field_names):
            return ("ppu" if "label" in field_names else "ppu_face"), "structured PPU dtype"
        if arr.dtype.kind == "V" and arr.dtype.itemsize in (self.PPU_BBOX_STRIDE, self.PPU_FACE_STRIDE):
            return (
                "ppu_face" if arr.dtype.itemsize == self.PPU_FACE_STRIDE else "ppu",
                f"void record size={arr.dtype.itemsize}",
            )

        # A clear raw-YOLO tensor must win over the fact that its byte size may
        # coincidentally be divisible by 32.
        if self._looks_like_raw_yolo_array(first):
            return "yolo", f"raw YOLO shape={tuple(arr.shape)}"
        if self._looks_like_end2end_array(first):
            if self.output_metadata_format in ("ppu", "ppu_face"):
                return self.output_metadata_format, "PPU metadata with decoded row tensor"
            if self.configured_output_format == "yolo_end2end":
                return "yolo_end2end", "explicit end-to-end configuration"
            if self.configured_output_format in ("ppu", "ppu_face"):
                return self.configured_output_format, "explicit PPU configuration with decoded row tensor"
            if self.output_hint in ("ppu", "ppu_face"):
                return self.output_hint, "PPU hint with decoded row tensor"
            return "yolo_end2end", f"end-to-end rows shape={tuple(np.squeeze(arr).shape)}"

        if arr.dtype == np.uint8 and arr.ndim and arr.shape[-1] == self.PPU_FACE_STRIDE:
            return "ppu_face", f"uint8 packed records shape={tuple(arr.shape)}"
        if arr.dtype == np.uint8 and arr.ndim and arr.shape[-1] == self.PPU_BBOX_STRIDE:
            return "ppu", f"uint8 packed records shape={tuple(arr.shape)}"

        if self.output_metadata_format in ("ppu", "ppu_face"):
            return self.output_metadata_format, f"metadata={self.output_metadata_format}"

        face_summary = self._packed_ppu_summary(output_tensor, stride=self.PPU_FACE_STRIDE)
        if face_summary.get("plausible") and self.output_format == "ppu_face":
            return "ppu_face", f"packed PPU probe={face_summary}"
        bbox_summary = self._packed_ppu_summary(output_tensor, stride=self.PPU_BBOX_STRIDE)
        if bbox_summary.get("plausible"):
            return "ppu", f"packed PPU probe={bbox_summary}"

        if self.configured_output_format:
            return self.configured_output_format, "configured fallback"
        if self.output_hint:
            return self.output_hint, "non-binding output hint fallback"
        return None, f"unrecognized output signature: {self._output_tensor_signature(output_tensor)}"

    def _select_runtime_output_format(self, output_tensor):
        detected, detail = self._detect_output_format_from_outputs(output_tensor)
        selected = detected or self.runtime_output_format or self.output_format or "yolo"
        if selected == "auto":
            selected = "yolo"
        if selected != self.runtime_output_format:
            change_key = (self.runtime_output_format, selected, detail)
            if change_key != self._last_format_change:
                logger.info(
                    f"[DeepX][{self.model_key}] runtime output selected={selected} "
                    f"(previous={self.runtime_output_format or self.output_format}, reason={detail})"
                )
                self._last_format_change = change_key
            self.runtime_output_format = selected
        return selected, detail

    @staticmethod
    def _safe_nms(boxes_xywh, scores, class_ids, conf_thres, iou_thres):
        if len(boxes_xywh) == 0:
            return np.empty((0,), dtype=np.int64)
        boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        offsets = class_ids.astype(np.float32) * 7680.0
        shifted = boxes_xywh.copy()
        shifted[:, 0] += offsets
        shifted[:, 1] += offsets
        indices = cv2.dnn.NMSBoxes(
            shifted.tolist(),
            scores.tolist(),
            float(conf_thres),
            float(iou_thres),
        )
        if indices is None or len(indices) == 0:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(indices).reshape(-1).astype(np.int64)

    def _normalize_raw_prediction(self, output_tensor):
        outputs = self._normalize_outputs(output_tensor)
        if not outputs:
            return None
        pred = np.asarray(outputs[0])
        if pred.size == 0:
            return None
        if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
            pred = pred.transpose((0, 2, 1))
        pred = np.squeeze(pred)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if pred.ndim != 2:
            raise ValueError(f"unexpected prediction rank/shape: {pred.shape}")
        return np.asarray(pred, dtype=np.float32)

    def _scale_normalized_xywh(self, boxes_xywh):
        boxes = np.asarray(boxes_xywh, dtype=np.float32).copy()
        if boxes.size and np.nanmax(np.abs(boxes)) <= 2.0:
            boxes[:, [0, 2]] *= float(self.input_width)
            boxes[:, [1, 3]] *= float(self.input_height)
        return boxes

    def _scale_normalized_xyxy(self, boxes_xyxy):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        if boxes.size and np.nanmax(np.abs(boxes)) <= 2.0:
            boxes[:, [0, 2]] *= float(self.input_width)
            boxes[:, [1, 3]] *= float(self.input_height)
        return boxes

    def postprocess(self, output_tensor, conf_thres=0.40, iou_thres=0.4):
        """Raw YOLOv8 center-xywh output: [4 box columns + class scores]."""
        try:
            pred = self._normalize_raw_prediction(output_tensor)
            if pred is None or pred.shape[1] <= 4:
                return []
            class_scores = pred[:, 4:]
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
            valid = np.isfinite(scores) & np.all(np.isfinite(pred[:, :4]), axis=1)
            valid &= scores >= float(conf_thres)
            if not np.any(valid):
                return []
            boxes = self._scale_normalized_xywh(pred[valid, :4])
            scores = scores[valid]
            class_ids = class_ids[valid]
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 0] -= boxes_xywh[:, 2] * 0.5
            boxes_xywh[:, 1] -= boxes_xywh[:, 3] * 0.5
            keep = self._safe_nms(boxes_xywh, scores, class_ids, conf_thres, iou_thres)
            return [
                [
                    [
                        float(boxes_xywh[i, 0]),
                        float(boxes_xywh[i, 1]),
                        float(boxes_xywh[i, 0] + boxes_xywh[i, 2]),
                        float(boxes_xywh[i, 1] + boxes_xywh[i, 3]),
                    ],
                    float(scores[i]),
                    int(class_ids[i]),
                ]
                for i in keep
            ]
        except Exception as exc:
            logger.error(f"NPU Postprocess Error ({os.path.basename(self.engine_path)}): {exc}")
            return []

    def postprocess_xyxy(self, output_tensor, conf_thres=0.40, iou_thres=0.4):
        """Raw XYXY output with class-score columns after the first four values."""
        try:
            pred = self._normalize_raw_prediction(output_tensor)
            if pred is None or pred.shape[1] <= 4:
                return []
            class_scores = pred[:, 4:]
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
            valid = np.isfinite(scores) & np.all(np.isfinite(pred[:, :4]), axis=1)
            valid &= scores >= float(conf_thres)
            if not np.any(valid):
                return []
            boxes_xyxy = self._scale_normalized_xyxy(pred[valid, :4])
            scores = scores[valid]
            class_ids = class_ids[valid]
            boxes_xywh = np.column_stack(
                (
                    boxes_xyxy[:, 0],
                    boxes_xyxy[:, 1],
                    boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                    boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
                )
            )
            keep = self._safe_nms(boxes_xywh, scores, class_ids, conf_thres, iou_thres)
            return [
                [boxes_xyxy[i].astype(float).tolist(), float(scores[i]), int(class_ids[i])]
                for i in keep
            ]
        except Exception as exc:
            logger.error(f"NPU XYXY Postprocess Error ({os.path.basename(self.engine_path)}): {exc}")
            return []

    def postprocess_tlwh(self, output_tensor, conf_thres=0.40, iou_thres=0.4):
        """Top-left XYWH output with class-score columns after the first four values."""
        try:
            pred = self._normalize_raw_prediction(output_tensor)
            if pred is None or pred.shape[1] <= 4:
                return []
            class_scores = pred[:, 4:]
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
            valid = np.isfinite(scores) & np.all(np.isfinite(pred[:, :4]), axis=1)
            valid &= scores >= float(conf_thres)
            if not np.any(valid):
                return []
            boxes_xywh = self._scale_normalized_xywh(pred[valid, :4])
            scores = scores[valid]
            class_ids = class_ids[valid]
            keep = self._safe_nms(boxes_xywh, scores, class_ids, conf_thres, iou_thres)
            return [
                [
                    [
                        float(boxes_xywh[i, 0]),
                        float(boxes_xywh[i, 1]),
                        float(boxes_xywh[i, 0] + boxes_xywh[i, 2]),
                        float(boxes_xywh[i, 1] + boxes_xywh[i, 3]),
                    ],
                    float(scores[i]),
                    int(class_ids[i]),
                ]
                for i in keep
            ]
        except Exception as exc:
            logger.error(f"NPU TLWH Postprocess Error ({os.path.basename(self.engine_path)}): {exc}")
            return []

    def postprocess_end2end(self, output_tensor, conf_thres=0.40, iou_thres=0.4):
        """End-to-end NMS rows: [x1, y1, x2, y2, score, class_id]."""
        try:
            outputs = self._normalize_outputs(output_tensor)
            if not outputs:
                return []
            rows = self._squeezed_2d_array(outputs[0])
            if rows is None or rows.shape[1] < 6:
                return []
            rows = np.asarray(rows[:, :6], dtype=np.float32)
            boxes_xyxy = self._scale_normalized_xyxy(rows[:, :4])
            scores = rows[:, 4]
            class_ids = np.rint(rows[:, 5]).astype(np.int64)
            valid = np.isfinite(scores) & np.all(np.isfinite(boxes_xyxy), axis=1)
            valid &= scores >= float(conf_thres)
            valid &= boxes_xyxy[:, 2] > boxes_xyxy[:, 0]
            valid &= boxes_xyxy[:, 3] > boxes_xyxy[:, 1]
            if not np.any(valid):
                return []
            boxes_xyxy = boxes_xyxy[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]
            boxes_xywh = np.column_stack(
                (
                    boxes_xyxy[:, 0],
                    boxes_xyxy[:, 1],
                    boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                    boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
                )
            )
            keep = self._safe_nms(boxes_xywh, scores, class_ids, conf_thres, iou_thres)
            return [
                [boxes_xyxy[i].astype(float).tolist(), float(scores[i]), int(class_ids[i])]
                for i in keep
            ]
        except Exception as exc:
            logger.error(f"NPU end-to-end Postprocess Error ({os.path.basename(self.engine_path)}): {exc}")
            return []

    def _extract_structured_ppu_rows(self, arr):
        names = tuple(arr.dtype.names or ())
        if not names:
            return None
        lower_to_name = {str(name).lower(): name for name in names}
        required = ("x", "y", "w", "h", "score")
        if not all(name in lower_to_name for name in required):
            return None
        flat = arr.reshape(-1)
        boxes = np.column_stack(
            [np.asarray(flat[lower_to_name[name]], dtype=np.float32) for name in ("x", "y", "w", "h")]
        )
        scores = np.asarray(flat[lower_to_name["score"]], dtype=np.float32)
        label_name = lower_to_name.get("label") or lower_to_name.get("class_id") or lower_to_name.get("class")
        labels = (
            np.asarray(flat[label_name], dtype=np.int64)
            if label_name is not None
            else np.zeros(len(scores), dtype=np.int64)
        )
        return boxes, scores, labels

    def _extract_ppu_rows(self, output_tensor, ppu_format="ppu"):
        outputs = self._normalize_outputs(output_tensor)
        if not outputs:
            return None
        first = outputs[0]
        arr = np.asarray(first)
        if arr.size == 0:
            return None

        structured = self._extract_structured_ppu_rows(arr)
        if structured is not None:
            return structured

        direct = self._squeezed_2d_array(first)
        if direct is not None and direct.dtype.kind in "fc" and direct.shape[1] == 6:
            rows = np.asarray(direct, dtype=np.float32)
            labels = np.rint(rows[:, 5]).astype(np.int64)
            return rows[:, :4], rows[:, 4], labels

        flat = self._to_uint8_buffer(first)
        preferred_stride = self.PPU_FACE_STRIDE if ppu_format == "ppu_face" else self.PPU_BBOX_STRIDE
        strides = [preferred_stride]
        for stride in (self.PPU_BBOX_STRIDE, self.PPU_FACE_STRIDE):
            if stride not in strides:
                strides.append(stride)

        selected_stride = None
        for stride in strides:
            if flat.size and flat.size % stride == 0 and flat.size // stride <= 4096:
                if stride == preferred_stride:
                    selected_stride = stride
                    break
                if selected_stride is None:
                    selected_stride = stride
        if selected_stride is None:
            raise ValueError(f"PPU buffer length {flat.size} is not a valid 32/64-byte record array")

        records = flat.reshape(flat.size // selected_stride, selected_stride)
        boxes = np.ascontiguousarray(records[:, :16]).view("<f4").reshape(-1, 4)
        scores = np.ascontiguousarray(records[:, 20:24]).view("<f4").reshape(-1)
        if selected_stride == self.PPU_BBOX_STRIDE:
            labels = np.ascontiguousarray(records[:, 24:28]).view("<u4").reshape(-1).astype(np.int64)
        else:
            labels = np.zeros(len(scores), dtype=np.int64)
        return boxes, scores, labels

    def _resolved_ppu_box_format(self):
        if self.ppu_box_format in ("center", "corner"):
            return self.ppu_box_format
        model_name = os.path.basename(str(self.engine_path or "")).lower()
        if "yolov10" in model_name:
            return "corner"
        return "center"

    def postprocess_ppu(self, output_tensor, conf_thres=0.40, iou_thres=0.4, ppu_format="ppu"):
        try:
            extracted = self._extract_ppu_rows(output_tensor, ppu_format=ppu_format)
            if extracted is None:
                return []
            boxes_raw, scores, labels = extracted
            boxes_raw = np.asarray(boxes_raw, dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            labels = np.asarray(labels, dtype=np.int64).reshape(-1)
            if not (len(boxes_raw) == len(scores) == len(labels)):
                raise ValueError("PPU box/score/label lengths disagree")

            valid = np.all(np.isfinite(boxes_raw), axis=1) & np.isfinite(scores)
            valid &= scores >= float(conf_thres)
            valid &= boxes_raw[:, 2] > 0.0
            valid &= boxes_raw[:, 3] > 0.0
            valid &= labels >= 0
            if self.class_count is not None:
                valid &= labels < int(self.class_count)
            if not np.any(valid):
                return []

            boxes_raw = boxes_raw[valid]
            scores = scores[valid]
            labels = labels[valid]
            boxes_raw = self._scale_normalized_xywh(boxes_raw)

            if self._resolved_ppu_box_format() == "corner":
                boxes_xyxy = boxes_raw.copy()
                boxes_xywh = np.column_stack(
                    (
                        boxes_xyxy[:, 0],
                        boxes_xyxy[:, 1],
                        boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                        boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
                    )
                )
            else:
                boxes_xywh = boxes_raw.copy()
                boxes_xywh[:, 0] -= boxes_xywh[:, 2] * 0.5
                boxes_xywh[:, 1] -= boxes_xywh[:, 3] * 0.5
                boxes_xyxy = np.column_stack(
                    (
                        boxes_xywh[:, 0],
                        boxes_xywh[:, 1],
                        boxes_xywh[:, 0] + boxes_xywh[:, 2],
                        boxes_xywh[:, 1] + boxes_xywh[:, 3],
                    )
                )

            positive_size = (boxes_xywh[:, 2] > 0.0) & (boxes_xywh[:, 3] > 0.0)
            if not np.any(positive_size):
                return []
            boxes_xywh = boxes_xywh[positive_size]
            boxes_xyxy = boxes_xyxy[positive_size]
            scores = scores[positive_size]
            labels = labels[positive_size]
            keep = self._safe_nms(boxes_xywh, scores, labels, conf_thres, iou_thres)
            return [
                [boxes_xyxy[i].astype(float).tolist(), float(scores[i]), int(labels[i])]
                for i in keep
            ]
        except Exception as exc:
            logger.error(
                f"NPU PPU Postprocess Error [{self.model_key}] "
                f"({os.path.basename(self.engine_path)}): {exc}"
            )
            return []

    def _postprocess_by_format(self, output_format, output_tensor, conf_thres):
        if output_format == "ppu_face":
            return self.postprocess_ppu(output_tensor, conf_thres=conf_thres, ppu_format="ppu_face")
        if output_format == "ppu":
            return self.postprocess_ppu(output_tensor, conf_thres=conf_thres, ppu_format="ppu")
        if output_format == "yolo_xyxy":
            return self.postprocess_xyxy(output_tensor, conf_thres=conf_thres)
        if output_format == "yolo_tlwh":
            return self.postprocess_tlwh(output_tensor, conf_thres=conf_thres)
        if output_format == "yolo_end2end":
            return self.postprocess_end2end(output_tensor, conf_thres=conf_thres)
        return self.postprocess(output_tensor, conf_thres=conf_thres)

    def _log_empty_detection(self, output_tensor, selected_format, conf_thres, reason):
        now = time.monotonic()
        if now - self._last_empty_log_at < self.empty_detection_log_interval_sec:
            return
        self._last_empty_log_at = now
        signature = self._output_tensor_signature(output_tensor)
        detail = ""
        if selected_format in ("ppu", "ppu_face"):
            stride = self.PPU_FACE_STRIDE if selected_format == "ppu_face" else self.PPU_BBOX_STRIDE
            summary = self._packed_ppu_summary(output_tensor, stride=stride, conf_thres=conf_thres)
            detail = f" ppu={summary}"
        logger.warning(
            f"[DeepX][{self.model_key}] zero detections after decode "
            f"format={selected_format} threshold={float(conf_thres):.3f} "
            f"reason={reason}; output={signature}{detail}"
        )

    def infer(self, img, conf_override=None):
        if img is None:
            return np.empty((0, 6), dtype=float)

        h_orig, w_orig = img.shape[:2]
        npu_input, scale, offset = self.letter_box(img)
        input_tensor = self._prepare_input_tensor(npu_input)
        engine = self.engine_pool.get()

        try:
            output_tensor = engine.run([input_tensor])
            threshold = float(conf_override if conf_override is not None else 0.40)
            threshold = min(1.0, max(0.0, threshold))
            selected_format, reason = self._select_runtime_output_format(output_tensor)

            if self.log_first_output_signature and not self._first_output_logged:
                self._first_output_logged = True
                logger.info(
                    f"[DeepX][{self.model_key}] first inference output: "
                    f"format={selected_format}, reason={reason}, "
                    f"signature={self._output_tensor_signature(output_tensor)}"
                )

            raw_detections = self._postprocess_by_format(selected_format, output_tensor, threshold)
            if not raw_detections:
                self._log_empty_detection(output_tensor, selected_format, threshold, reason)
                return np.empty((0, 6), dtype=float)

            dw, dh = offset
            rows = []
            for box, score, class_id in raw_detections:
                if len(box) < 4:
                    continue
                x1 = float(np.clip((float(box[0]) - dw) / scale, 0, w_orig))
                y1 = float(np.clip((float(box[1]) - dh) / scale, 0, h_orig))
                x2 = float(np.clip((float(box[2]) - dw) / scale, 0, w_orig))
                y2 = float(np.clip((float(box[3]) - dh) / scale, 0, h_orig))
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2, float(score))):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                rows.append([x1, y1, x2, y2, float(score), int(class_id)])
            return detection_array(rows)
        except Exception as exc:
            logger.error(
                f"NPU Inference Error [{self.model_key}] "
                f"({os.path.basename(self.engine_path)}): {exc}\n{traceback.format_exc()}"
            )
            return np.empty((0, 6), dtype=float)
        finally:
            self.engine_pool.put(engine)
# ==========================================
# [7] 객체 트래커 및 영상 녹화기
# ==========================================
# [1] 시스템 기본 설정 및 상수 영역 하단에 추가
DEBUG_MODE = False
class SimpleTracker:
    def __init__(self, max_lost=30, history_len=60): # 15FPS 기준 약 4초 분량의 궤적 저장
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost
        self.history_len = history_len

    def update(self, detections):
        used_dets = set()

        for tid, trk in self.tracks.items():
            best_iou = 0
            best_idx = -1

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                if int(det[5]) != trk['cls']:
                    continue

                iou = calculate_iou(trk['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > 0.2:
                # 중심점 좌표 계산 및 히스토리에 누적
                cx = int((detections[best_idx][0] + detections[best_idx][2]) / 2)
                cy = int((detections[best_idx][1] + detections[best_idx][3]) / 2)

                self.tracks[tid].update({
                    'bbox': detections[best_idx][:4],
                    'lost': 0,
                    'conf': detections[best_idx][4]
                })
                self.tracks[tid]['history'].append((cx, cy))
                used_dets.add(best_idx)
            else:
                self.tracks[tid]['lost'] += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.max_lost}

        res_tracks = []
        for i, det in enumerate(detections):
            if i not in used_dets:
                cx = int((det[0] + det[2]) / 2)
                cy = int((det[1] + det[3]) / 2)
                self.tracks[self.next_id] = {
                    'bbox': det[:4], 'lost': 0, 'cls': int(det[5]), 'conf': det[4],
                    'history': deque([(cx, cy)], maxlen=self.history_len) # 신규 객체 궤적 초기화
                }
                self.next_id += 1

        for tid, trk in self.tracks.items():
            if trk['lost'] <= 15:
                res_tracks.append([*trk['bbox'], tid, trk.get('conf', 1.0), trk['cls']])

        return np.array(res_tracks)

def draw_video_event_markers(frame, frame_timestamp, markers, mark_sec=None, border_thickness=None):
    if frame is None:
        return frame
    try:
        frame_timestamp = float(frame_timestamp)
    except Exception:
        return frame

    mark_sec = float(SYS_CFG.get("VIDEO_EVENT_MARK_SEC", 2.0) if mark_sec is None else mark_sec)
    mark_sec = max(0.0, mark_sec)
    thickness = int(SYS_CFG.get("VIDEO_EVENT_BORDER_THICKNESS", 18) if border_thickness is None else border_thickness)
    thickness = max(1, thickness)

    active = []
    for marker in markers or []:
        try:
            trigger_ts = float(marker.get("ts"))
        except Exception:
            continue
        if trigger_ts <= frame_timestamp <= trigger_ts + mark_sec:
            active.append(marker)
    if not active:
        return frame

    marked = frame.copy()
    height, width = marked.shape[:2]
    inset = max(1, thickness // 2)
    cv2.rectangle(marked, (inset, inset), (max(inset, width - inset - 1), max(inset, height - inset - 1)), (0, 0, 255), thickness)

    event_names = []
    for marker in active:
        name = str(marker.get("event_name") or "unknown")
        if name not in event_names:
            event_names.append(name)
    first_ts = min(float(marker.get("ts")) for marker in active)
    trigger_dt = datetime.datetime.fromtimestamp(first_ts, pytz.timezone("Asia/Seoul"))
    time_text = trigger_dt.strftime("%H:%M:%S.%f")[:-3]
    label = f"EVENT TRIGGER {time_text} {','.join(event_names)}"
    cv2.putText(marked, label, (max(10, thickness), max(35, thickness + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return marked


class VideoRecorder:
    def __init__(self, ip, cam_id=None):
        self.ip = ip
        self.cam_id = cam_id if cam_id is not None else "-"
        self.pre_sec = max(1.0, float(SYS_CFG.get("REC_PRE_SEC", 10.0)))
        self.max_buffer_len = max(1, int(15 * self.pre_sec))
        self.buffer = deque(maxlen=self.max_buffer_len)
        self.write_queue = queue.Queue()

        self.recording = False
        self.record_end_time = 0.0
        self.current_event = "unknown"
        self.current_meta = None
        self.current_record_started_at = None
        self.recorded_fps = 3.0
        self.event_markers = []
        self.marker_lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def update(self, frame, infer_meta=None, timestamp=None):
        if frame is None:
            return
        frame_timestamp = float(timestamp if timestamp is not None else time.time())
        self.buffer.append((frame.copy(), infer_meta, frame_timestamp))

        if self.recording:
            if time.time() > self.record_end_time:
                self.recording = False
                self.write_queue.put(None)
                logger.info(f"[녹화종료] {self.ip} - {self.current_event}")
            else:
                self.write_queue.put((frame.copy(), infer_meta, frame_timestamp))

    def trigger(self, event_name, objects_meta=None, event_meta=None, current_fps=3.0):
        trigger_ts = time.time()
        post_sec = float(SYS_CFG.get("REC_POST_SEC", 10.0))
        pre_sec = float(SYS_CFG.get("REC_PRE_SEC", 10.0))
        event_id = str((event_meta or {}).get("event_id", ""))
        marker = {
            "ts": trigger_ts,
            "event_name": str(event_name),
            "event_id": event_id,
        }

        if self.recording:
            self.record_end_time = max(self.record_end_time, trigger_ts + post_sec)
            with self.marker_lock:
                self.event_markers.append(marker)
            return

        logger.info(f"[녹화시작] {self.ip} - {event_name} (FPS: {current_fps:.1f})")
        self.recording = True
        self.record_end_time = trigger_ts + post_sec
        self.current_event = str(event_name)
        self.current_meta = dict(event_meta or {})
        self.current_record_started_at = trigger_ts
        self.recorded_fps = max(1.0, float(current_fps))
        with self.marker_lock:
            self.event_markers = [marker]

        target_start_time = trigger_ts - pre_sec
        for buffered_item in list(self.buffer):
            if buffered_item[2] >= target_start_time:
                self.write_queue.put(buffered_item)

    def _writer_loop(self):
        writer = None
        infer_log_file = None
        video_frame_index = 0

        while self.running:
            try:
                item = self.write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                if writer:
                    writer.release()
                if infer_log_file:
                    infer_log_file.close()
                writer = None
                infer_log_file = None
                video_frame_index = 0
                continue

            frame, infer_meta, frame_timestamp = item

            if writer is None:
                dpath = os.path.join(EVENT_ROOT_DIR, "events", self.ip, "videos", self.current_event)
                os.makedirs(dpath, exist_ok=True)

                time_str = datetime.datetime.fromtimestamp(self.current_record_started_at).strftime("%Y%m%d_%H%M%S")
                fname = f"{time_str}_{self.ip}_{self.current_event}.mp4"
                video_path = os.path.join(dpath, fname)
                infer_log_path = os.path.join(dpath, f"{time_str}_{self.ip}_{self.current_event}.infer.jsonl")
                meta_path = os.path.join(dpath, f"{time_str}_{self.ip}_{self.current_event}.meta.json")

                if isinstance(self.current_meta, dict):
                    self.current_meta.update({
                        "video_path": video_path,
                        "infer_log_path": infer_log_path,
                        "recorded_fps": self.recorded_fps,
                    })
                    try:
                        with open(meta_path, "w", encoding="utf-8") as meta_file:
                            json.dump(to_json_safe(self.current_meta), meta_file, indent=4, ensure_ascii=False)
                    except Exception as exc:
                        logger.error(f"메타데이터 저장 실패: {exc}")

                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, self.recorded_fps, (width, height))
                try:
                    infer_log_file = open(infer_log_path, "w", encoding="utf-8")
                except Exception:
                    infer_log_file = None

            if writer:
                if infer_log_file and infer_meta is not None:
                    try:
                        log_record = dict(infer_meta)
                        log_record["video_frame_index"] = video_frame_index
                        infer_log_file.write(json.dumps(to_json_safe(log_record), ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                with self.marker_lock:
                    markers = [dict(marker) for marker in self.event_markers]
                output_frame = draw_video_event_markers(frame, frame_timestamp, markers)
                writer.write(output_frame)
                video_frame_index += 1

# ==========================================
# [8] PCTC 이벤트 감지 로직
# ==========================================
def _track_object(track, class_id=None, label=None):
    class_id = int(track[6] if class_id is None else class_id)
    class_name = pctc_class_name(class_id)
    return {
        "label": str(label or class_name),
        "class_name": class_name,
        "box": [int(round(float(v))) for v in track[:4]],
        "score": float(track[5]),
        "tid": int(track[4]),
        "class_id": class_id,
    }


def _bbox_scale(box):
    return max(1.0, float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _bbox_gap(box_a, box_b):
    dx = max(float(box_a[0]) - float(box_b[2]), float(box_b[0]) - float(box_a[2]), 0.0)
    dy = max(float(box_a[1]) - float(box_b[3]), float(box_b[1]) - float(box_a[3]), 0.0)
    return math.hypot(dx, dy)


def _point_to_bbox_distance(point, box):
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, box[:4])
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _unit_vector(start, end):
    vx = float(end[0]) - float(start[0])
    vy = float(end[1]) - float(start[1])
    length = math.hypot(vx, vy)
    if length <= 1e-9:
        return None
    return (vx / length, vy / length)


class BaseEventDetector:
    gui_name = "BASE"
    
    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        self.config = dict(config or {})
        self.roi_poly = np.array(roi_poly, dtype=np.int32) if roi_poly and len(roi_poly) >= 3 else np.empty((0, 2), dtype=np.int32)
        self.roi_lines = roi_lines or []
        self.abnormal_drive_zones = list(abnormal_drive_zones or [])
        self.fps = SYS_CFG.get("REC_FPS", 3)

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        return []


class IntrusionDetector(BaseEventDetector):
    gui_name = "INTRUSION"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.proximity_ratio = max(0.0, float(self.config.get("proximity_ratio", 0.0)))

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        if self.roi_poly.size == 0:
            return []
        privacy_tracks = kwargs.get("privacy_tracks", [])
        triggered = []

        for track in tracks:
            tid = int(track[4])
            class_id = int(track_map.get(tid, -1))
            if class_id not in INTRUSION_CLASS_IDS:
                continue
            anchor = get_foot_point(*track[:4]) if class_id == ID_PCTC_PERSON else get_check_point(*track[:4])
            signed_distance = float(cv2.pointPolygonTest(self.roi_poly, anchor, True))
            proximity_px = _bbox_scale(track[:4]) * self.proximity_ratio
            inside_roi = signed_distance >= 0.0
            if signed_distance < -proximity_px:
                continue

            triggered.append({
                "tid": tid,
                "bbox": track[:4],
                "frame": frame.copy() if frame is not None else None,
                "fid": fid,
                "privacy_tracks": privacy_tracks,
                "privacy_fid": fid,
                "objects": [_track_object(track, class_id)],
                "decision_trace": {
                    "detector": "IntrusionDetector",
                    "reason": "inside_or_near_intrusion_roi",
                    "class_id": class_id,
                    "class_name": pctc_class_name(class_id),
                    "roi_anchor": int_point(anchor),
                    "signed_distance_px": round(signed_distance, 3),
                    "proximity_ratio": round(self.proximity_ratio, 4),
                    "proximity_px": round(proximity_px, 3),
                    "inside_roi": inside_roi,
                },
            })
        return triggered


class ParkingDetector(BaseEventDetector):
    gui_name = "PARKING"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.states = {}
        self.trigger_sec = max(0.0, float(self.config.get("trigger_sec", 5.0)))
        self.move_threshold_ratio = max(0.0, float(self.config.get("move_threshold_ratio", 0.1)))

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        if self.roi_poly.size == 0:
            self.states.clear()
            return []

        current_time = time.time()
        current_ids = set()
        privacy_tracks = kwargs.get("privacy_tracks", [])
        triggered = []

        for track in tracks:
            tid = int(track[4])
            class_id = int(track_map.get(tid, -1))
            if class_id not in PARKING_CLASS_IDS:
                continue
            anchor = get_check_point(*track[:4])
            if cv2.pointPolygonTest(self.roi_poly, anchor, False) < 0:
                continue

            current_ids.add(tid)
            center = get_center_point(*track[:4])
            object_size = _bbox_scale(track[:4])
            movement_threshold = object_size * self.move_threshold_ratio
            state = self.states.get(tid)

            if state is None or get_distance(center, state["anchor_center"]) >= max(1e-6, movement_threshold):
                self.states[tid] = {
                    "start_time": current_time,
                    "anchor_center": center,
                    "triggered": False,
                }
                state = self.states[tid]

            stationary_sec = current_time - float(state["start_time"])
            if not state["triggered"] and stationary_sec >= self.trigger_sec:
                triggered.append({
                    "tid": tid,
                    "bbox": track[:4],
                    "frame": frame.copy() if frame is not None else None,
                    "fid": fid,
                    "privacy_tracks": privacy_tracks,
                    "privacy_fid": fid,
                    "objects": [_track_object(track, class_id)],
                    "decision_trace": {
                        "detector": "ParkingDetector",
                        "reason": "stationary_duration_exceeded",
                        "class_id": class_id,
                        "class_name": pctc_class_name(class_id),
                        "roi_anchor": int_point(anchor),
                        "anchor_center": int_point(state["anchor_center"]),
                        "current_center": int_point(center),
                        "duration_sec": round(stationary_sec, 3),
                        "trigger_sec": round(self.trigger_sec, 3),
                        "move_threshold_ratio": round(self.move_threshold_ratio, 4),
                        "move_threshold_px": round(movement_threshold, 3),
                    },
                })
                state["triggered"] = True

        for tid in list(self.states):
            if tid not in current_ids:
                self.states.pop(tid, None)
        return triggered


class HelmetDetector(BaseEventDetector):
    gui_name = "NO-HELMET"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.sessions = []
        self.min_streak_sec = float(self.config.get("min_streak_sec", 2.0))
        self.trigger_total_sec = float(self.config.get("trigger_total_sec", 3.0))
        self.max_gap_sec = float(self.config.get("max_gap_sec", 1.5))
        self.window_sec = float(self.config.get("window_sec", 30.0))
        self.ignore_top_ratio = float(self.config.get("ignore_top_ratio", 0.2))
        self.red_helmet_tids = set()

    def _get_roi_crop(self, frame, box):
        if frame is None:
            return None
        image_height, image_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width, x2), min(image_height, y2)
        box_height = y2 - y1
        if box_height <= 0 or x2 - x1 <= 0:
            return None
        roi = frame[y1:y1 + int(box_height * 0.5), x1:x2]
        return roi.copy() if roi.size else None

    def _is_red_helmet_median(self, roi_buffer):
        if not roi_buffer:
            return False
        h_means, s_means, r_means = [], [], []
        for roi in roi_buffer:
            if roi is None or roi.size == 0:
                continue
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            r_means.append(np.mean(rgb_roi[:, :, 0]))
            h_means.append(np.mean(hsv_roi[:, :, 0]))
            s_means.append(np.mean(hsv_roi[:, :, 1]))
        if not h_means:
            return False
        return (
            10 <= np.median(h_means) <= 40
            and np.median(s_means) >= 60
            and np.median(r_means) >= 100
        )

    @staticmethod
    def _intersection_over_head_area(head_box, person_box):
        inter_width = max(0.0, min(head_box[2], person_box[2]) - max(head_box[0], person_box[0]))
        inter_height = max(0.0, min(head_box[3], person_box[3]) - max(head_box[1], person_box[1]))
        head_area = max(1.0, (head_box[2] - head_box[0]) * (head_box[3] - head_box[1]))
        return (inter_width * inter_height) / head_area

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        helmet_tracks = kwargs.get("helmet_tracks", [])
        privacy_tracks = kwargs.get("privacy_tracks", [])
        current_time = time.time()
        unhelmeted_heads = [track for track in helmet_tracks if int(track[6]) == ID_H_HEAD]
        current_matches = []
        ignore_y_threshold = frame.shape[0] * self.ignore_top_ratio if frame is not None else 0.0

        for person in tracks:
            person_tid = int(person[4])
            if track_map.get(person_tid) != ID_PCTC_PERSON or person_tid in self.red_helmet_tids:
                continue
            px1, py1, px2, py2 = person[:4]
            if py1 <= ignore_y_threshold:
                continue
            foot_point = get_foot_point(*person[:4])
            if self.roi_poly.size and cv2.pointPolygonTest(self.roi_poly, foot_point, False) < 0:
                continue

            person_height = max(1.0, py2 - py1)
            person_width = max(1.0, px2 - px1)
            best_ioa = 0.0
            best_head = None
            for head in unhelmeted_heads:
                hx1, hy1, hx2, hy2 = head[:4]
                head_center_x = (hx1 + hx2) / 2.0
                head_center_y = (hy1 + hy2) / 2.0
                if head_center_y > py1 + person_height * 0.4:
                    continue
                margin = person_width * 0.15
                if not (px1 - margin <= head_center_x <= px2 + margin):
                    continue
                ioa = self._intersection_over_head_area(head[:4], person[:4])
                if ioa > best_ioa:
                    best_ioa = ioa
                    best_head = head

            if best_head is None or best_ioa < 0.5:
                continue
            head_center = get_center_point(*best_head[:4])
            if self.roi_poly.size and cv2.pointPolygonTest(self.roi_poly, head_center, False) < 0:
                continue

            current_matches.append({
                "tid": person_tid,
                "head_bbox": best_head[:4],
                "person_bbox": person[:4],
                "objects": [
                    _track_object(person, ID_PCTC_PERSON),
                    {
                        "label": "head",
                        "class_name": "head",
                        "box": [int(round(float(v))) for v in best_head[:4]],
                        "score": float(best_head[5]),
                        "tid": int(best_head[4]),
                        "class_id": ID_H_HEAD,
                    },
                ],
                "decision_context": {
                    "person_tid": person_tid,
                    "head_tid": int(best_head[4]),
                    "head_score": float(best_head[5]),
                    "ioa_with_person": round(best_ioa, 4),
                    "min_ioa": 0.5,
                    "head_center": int_point(head_center),
                    "person_foot_point": int_point(foot_point),
                    "ignore_top_ratio": round(self.ignore_top_ratio, 4),
                },
            })

        for match in current_matches:
            session = next((item for item in self.sessions if item["last_tid"] == match["tid"] or calculate_iou(match["person_bbox"], item["last_person_bbox"]) > 0.3), None)
            roi_crop = self._get_roi_crop(frame, match["head_bbox"])
            if session is None:
                roi_buffer = deque(maxlen=5)
                if roi_crop is not None:
                    roi_buffer.append(roi_crop)
                self.sessions.append({
                    "start_time": current_time,
                    "last_seen_time": current_time,
                    "streaks": [{"start_time": current_time, "end_time": current_time}],
                    "last_tid": match["tid"],
                    "last_person_bbox": match["person_bbox"],
                    "bbox": match["head_bbox"],
                    "fid": fid,
                    "triggered": False,
                    "roi_buffer": roi_buffer,
                    "objects": match["objects"],
                    "decision_context": match["decision_context"],
                })
                continue

            gap_sec = current_time - session["last_seen_time"]
            if gap_sec <= self.max_gap_sec:
                session["streaks"][-1]["end_time"] = current_time
            else:
                session["streaks"].append({"start_time": current_time, "end_time": current_time})
            session.update({
                "last_seen_time": current_time,
                "last_tid": match["tid"],
                "last_person_bbox": match["person_bbox"],
                "bbox": match["head_bbox"],
                "fid": fid,
                "objects": match["objects"],
                "decision_context": match["decision_context"],
            })
            if roi_crop is not None:
                session["roi_buffer"].append(roi_crop)

        triggered = []
        active_sessions = []
        for session in self.sessions:
            if session["last_tid"] in self.red_helmet_tids:
                continue
            if current_time - session["start_time"] > self.window_sec:
                continue
            valid_durations = []
            total_valid_sec = 0.0
            for streak in session["streaks"]:
                duration = streak["end_time"] - streak["start_time"]
                if duration >= self.min_streak_sec:
                    total_valid_sec += duration
                    valid_durations.append(round(duration, 3))

            if not session["triggered"] and total_valid_sec >= self.trigger_total_sec:
                if self._is_red_helmet_median(session["roi_buffer"]):
                    self.red_helmet_tids.add(session["last_tid"])
                else:
                    triggered.append({
                        "tid": session["last_tid"],
                        "bbox": session["bbox"],
                        "frame": frame.copy() if frame is not None else None,
                        "fid": session["fid"],
                        "privacy_tracks": privacy_tracks,
                        "privacy_fid": fid,
                        "objects": session["objects"],
                        "decision_trace": {
                            "detector": "HelmetDetector",
                            "reason": "no_helmet_duration_exceeded",
                            "total_valid_sec": round(total_valid_sec, 3),
                            "trigger_total_sec": round(self.trigger_total_sec, 3),
                            "min_streak_sec": round(self.min_streak_sec, 3),
                            "max_gap_sec": round(self.max_gap_sec, 3),
                            "valid_streaks_sec": valid_durations,
                            "red_helmet_veto": False,
                            **session["decision_context"],
                        },
                    })
                session["triggered"] = True
            active_sessions.append(session)

        self.sessions = active_sessions
        return triggered


class YardCrossingDetector(BaseEventDetector):
    gui_name = "YARD-CROSS"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.trigger_sec = max(0.0, float(self.config.get("trigger_sec", 3.0)))
        self.states = {}

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        if self.roi_poly.size == 0:
            self.states.clear()
            return []
        current_time = time.time()
        inside_ids = set()
        privacy_tracks = kwargs.get("privacy_tracks", [])
        triggered = []

        for track in tracks:
            tid = int(track[4])
            if track_map.get(tid) != ID_PCTC_PERSON:
                continue
            foot_point = get_foot_point(*track[:4])
            if cv2.pointPolygonTest(self.roi_poly, foot_point, False) < 0:
                continue
            inside_ids.add(tid)
            state = self.states.setdefault(tid, {"entered_at": current_time, "triggered": False})
            dwell_sec = current_time - state["entered_at"]
            if not state["triggered"] and dwell_sec >= self.trigger_sec:
                triggered.append({
                    "tid": tid,
                    "bbox": track[:4],
                    "frame": frame.copy() if frame is not None else None,
                    "fid": fid,
                    "privacy_tracks": privacy_tracks,
                    "privacy_fid": fid,
                    "objects": [_track_object(track, ID_PCTC_PERSON)],
                    "decision_trace": {
                        "detector": "YardCrossingDetector",
                        "reason": "person_dwell_in_yard_crossing_roi",
                        "class_id": ID_PCTC_PERSON,
                        "class_name": pctc_class_name(ID_PCTC_PERSON),
                        "foot_point": int_point(foot_point),
                        "dwell_sec": round(dwell_sec, 3),
                        "trigger_sec": round(self.trigger_sec, 3),
                    },
                })
                state["triggered"] = True

        for tid in list(self.states):
            if tid not in inside_ids:
                self.states.pop(tid, None)
        return triggered


class AbnormalDriveDetector(BaseEventDetector):
    gui_name = "ABNORMAL-DRIVE"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.min_displacement_ratio = max(0.0, float(self.config.get("min_displacement_ratio", 0.35)))
        self.opposite_cos_threshold = float(self.config.get("opposite_cos_threshold", -0.35))
        self.uturn_min_forward_ratio = max(0.0, float(self.config.get("uturn_min_forward_ratio", 0.45)))
        self.uturn_return_ratio = max(0.0, float(self.config.get("uturn_return_ratio", 0.30)))
        self.uturn_reverse_cos_threshold = float(self.config.get("uturn_reverse_cos_threshold", -0.20))
        self.states = {}

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        current_time = time.time()
        active_keys = set()
        privacy_tracks = kwargs.get("privacy_tracks", [])
        triggered = []

        prepared_zones = []
        for zone_index, zone in enumerate(self.abnormal_drive_zones or []):
            polygon = zone.get("roi_poly", [])
            direction = zone.get("direction_points", [])
            if len(polygon) < 3 or len(direction) != 2:
                continue
            allowed_unit = _unit_vector(direction[0], direction[1])
            if allowed_unit is None:
                continue
            prepared_zones.append((zone_index, np.array(polygon, dtype=np.int32), polygon, direction, allowed_unit))

        for track in tracks:
            tid = int(track[4])
            class_id = int(track_map.get(tid, -1))
            if class_id not in ABNORMAL_DRIVE_CLASS_IDS:
                continue
            current_center = get_check_point(*track[:4])
            object_size = _bbox_scale(track[:4])
            min_displacement_px = object_size * self.min_displacement_ratio

            for zone_index, polygon_array, polygon, direction, allowed_unit in prepared_zones:
                if cv2.pointPolygonTest(polygon_array, current_center, False) < 0:
                    continue
                key = (zone_index, tid)
                active_keys.add(key)
                state = self.states.get(key)
                if state is None:
                    self.states[key] = {
                        "entry_center": current_center,
                        "last_center": current_center,
                        "max_forward_projection_px": 0.0,
                        "triggered": False,
                        "entered_at": current_time,
                    }
                    continue

                entry_center = state["entry_center"]
                travel_vector = (
                    float(current_center[0]) - float(entry_center[0]),
                    float(current_center[1]) - float(entry_center[1]),
                )
                travel_distance = math.hypot(*travel_vector)
                forward_projection = travel_vector[0] * allowed_unit[0] + travel_vector[1] * allowed_unit[1]
                state["max_forward_projection_px"] = max(state["max_forward_projection_px"], forward_projection)
                max_forward_projection = state["max_forward_projection_px"]

                travel_cos = None
                if travel_distance > 1e-9:
                    travel_cos = forward_projection / travel_distance

                last_center = state["last_center"]
                local_vector = (
                    float(current_center[0]) - float(last_center[0]),
                    float(current_center[1]) - float(last_center[1]),
                )
                local_distance = math.hypot(*local_vector)
                local_cos = None
                if local_distance > 1e-9:
                    local_cos = (local_vector[0] * allowed_unit[0] + local_vector[1] * allowed_unit[1]) / local_distance

                reason = None
                uturn_forward_px = object_size * self.uturn_min_forward_ratio
                uturn_return_px = object_size * self.uturn_return_ratio
                if (
                    not state["triggered"]
                    and max_forward_projection >= uturn_forward_px
                    and max_forward_projection - forward_projection >= uturn_return_px
                    and local_cos is not None
                    and local_cos <= self.uturn_reverse_cos_threshold
                ):
                    reason = "u_turn_after_forward_travel"
                elif (
                    not state["triggered"]
                    and travel_distance >= min_displacement_px
                    and travel_cos is not None
                    and travel_cos <= self.opposite_cos_threshold
                ):
                    reason = "opposite_direction_travel"

                if reason:
                    triggered.append({
                        "tid": tid,
                        "bbox": track[:4],
                        "frame": frame.copy() if frame is not None else None,
                        "fid": fid,
                        "privacy_tracks": privacy_tracks,
                        "privacy_fid": fid,
                        "objects": [_track_object(track, class_id)],
                        "decision_trace": {
                            "detector": "AbnormalDriveDetector",
                            "reason": reason,
                            "zone_index": zone_index + 1,
                            "class_id": class_id,
                            "class_name": pctc_class_name(class_id),
                            "roi_anchor": int_point(current_center),
                            "roi_polygon": [int_point(point) for point in polygon],
                            "allowed_direction_points": [int_point(point) for point in direction],
                            "allowed_direction_unit": [round(allowed_unit[0], 6), round(allowed_unit[1], 6)],
                            "entry_center": int_point(entry_center),
                            "current_center": int_point(current_center),
                            "travel_distance_px": round(travel_distance, 3),
                            "forward_projection_px": round(forward_projection, 3),
                            "max_forward_projection_px": round(max_forward_projection, 3),
                            "travel_cos": None if travel_cos is None else round(travel_cos, 6),
                            "local_cos": None if local_cos is None else round(local_cos, 6),
                            "min_displacement_px": round(min_displacement_px, 3),
                        },
                    })
                    state["triggered"] = True

                state["last_center"] = current_center

        for key in list(self.states):
            if key not in active_keys:
                self.states.pop(key, None)
        return triggered


class WalkwayOutDetector(BaseEventDetector):
    gui_name = "WALKWAY-OUT"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.outside_grace_sec = max(0.0, float(self.config.get("outside_grace_sec", 1.0)))
        self.boundary_margin_ratio = max(0.0, float(self.config.get("boundary_margin_ratio", 0.05)))
        self.states = {}

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        if self.roi_poly.size == 0:
            self.states.clear()
            return []
        current_time = time.time()
        visible_person_ids = set()
        privacy_tracks = kwargs.get("privacy_tracks", [])
        triggered = []

        for track in tracks:
            tid = int(track[4])
            if track_map.get(tid) != ID_PCTC_PERSON:
                continue
            visible_person_ids.add(tid)
            foot_point = get_foot_point(*track[:4])
            signed_distance = float(cv2.pointPolygonTest(self.roi_poly, foot_point, True))
            boundary_margin_px = _bbox_scale(track[:4]) * self.boundary_margin_ratio

            if signed_distance >= -boundary_margin_px:
                self.states.pop(tid, None)
                continue

            state = self.states.setdefault(tid, {"outside_since": current_time, "triggered": False})
            outside_sec = current_time - state["outside_since"]
            if not state["triggered"] and outside_sec >= self.outside_grace_sec:
                triggered.append({
                    "tid": tid,
                    "bbox": track[:4],
                    "frame": frame.copy() if frame is not None else None,
                    "fid": fid,
                    "privacy_tracks": privacy_tracks,
                    "privacy_fid": fid,
                    "objects": [_track_object(track, ID_PCTC_PERSON)],
                    "decision_trace": {
                        "detector": "WalkwayOutDetector",
                        "reason": "person_outside_walkway_roi",
                        "foot_point": int_point(foot_point),
                        "signed_distance_px": round(signed_distance, 3),
                        "boundary_margin_px": round(boundary_margin_px, 3),
                        "outside_sec": round(outside_sec, 3),
                        "outside_grace_sec": round(self.outside_grace_sec, 3),
                    },
                })
                state["triggered"] = True

        for tid in list(self.states):
            if tid not in visible_person_ids:
                self.states.pop(tid, None)
        return triggered


class SpreaderDangerZoneDetector(BaseEventDetector):
    gui_name = "SPREADER-DANGER"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)
        self.link_ratio = max(0.0, float(self.config.get("spreader_container_link_ratio", 1.25)))
        self.danger_distance_ratio = max(0.0, float(self.config.get("danger_distance_ratio", 1.20)))
        self.min_danger_distance_px = max(0.0, float(self.config.get("min_danger_distance_px", 30.0)))
        self.trigger_hold_sec = max(0.0, float(self.config.get("trigger_hold_sec", 0.5)))
        self.states = {}

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        current_time = time.time()
        privacy_tracks = kwargs.get("privacy_tracks", [])
        spreaders = [track for track in tracks if track_map.get(int(track[4])) == ID_PCTC_SPREADER]
        containers = [track for track in tracks if track_map.get(int(track[4])) == ID_PCTC_CONTAINER]
        targets = [track for track in tracks if track_map.get(int(track[4])) in DANGER_APPROACH_CLASS_IDS]
        active_keys = set()
        triggered = []

        if not spreaders or not containers:
            self.states.clear()
            return []

        operations = []
        for spreader in spreaders:
            nearest_container = min(containers, key=lambda item: _bbox_gap(spreader[:4], item[:4]))
            gap = _bbox_gap(spreader[:4], nearest_container[:4])
            spreader_height = max(1.0, float(spreader[3]) - float(spreader[1]))
            container_height = max(1.0, float(nearest_container[3]) - float(nearest_container[1]))
            reference_height = max(spreader_height, container_height)
            link_distance = max(10.0, reference_height * self.link_ratio)
            if gap > link_distance:
                continue
            operation_box = [
                min(float(spreader[0]), float(nearest_container[0])),
                min(float(spreader[1]), float(nearest_container[1])),
                max(float(spreader[2]), float(nearest_container[2])),
                max(float(spreader[3]), float(nearest_container[3])),
            ]
            danger_distance = max(self.min_danger_distance_px, reference_height * self.danger_distance_ratio)
            operations.append((spreader, nearest_container, operation_box, gap, link_distance, danger_distance))

        for spreader, container, operation_box, gap, link_distance, danger_distance in operations:
            spreader_tid = int(spreader[4])
            container_tid = int(container[4])
            for target in targets:
                target_tid = int(target[4])
                target_class_id = int(track_map.get(target_tid, -1))
                anchor = get_foot_point(*target[:4]) if target_class_id == ID_PCTC_PERSON else get_check_point(*target[:4])
                distance_to_operation = _point_to_bbox_distance(anchor, operation_box)
                if distance_to_operation > danger_distance:
                    continue

                key = (spreader_tid, container_tid, target_tid)
                active_keys.add(key)
                state = self.states.setdefault(key, {"near_since": current_time, "triggered": False})
                hold_sec = current_time - state["near_since"]
                if not state["triggered"] and hold_sec >= self.trigger_hold_sec:
                    triggered.append({
                        "tid": target_tid,
                        "bbox": target[:4],
                        "frame": frame.copy() if frame is not None else None,
                        "fid": fid,
                        "privacy_tracks": privacy_tracks,
                        "privacy_fid": fid,
                        "objects": [
                            _track_object(spreader, ID_PCTC_SPREADER),
                            _track_object(container, ID_PCTC_CONTAINER),
                            _track_object(target, target_class_id),
                        ],
                        "decision_trace": {
                            "detector": "SpreaderDangerZoneDetector",
                            "reason": "target_near_active_spreader_container_operation",
                            "spreader_track_id": spreader_tid,
                            "container_track_id": container_tid,
                            "target_track_id": target_tid,
                            "target_class_id": target_class_id,
                            "target_class_name": pctc_class_name(target_class_id),
                            "target_anchor": int_point(anchor),
                            "operation_box": int_box(operation_box),
                            "spreader_container_gap_px": round(gap, 3),
                            "link_distance_px": round(link_distance, 3),
                            "distance_to_operation_px": round(distance_to_operation, 3),
                            "danger_distance_px": round(danger_distance, 3),
                            "hold_sec": round(hold_sec, 3),
                            "trigger_hold_sec": round(self.trigger_hold_sec, 3),
                        },
                    })
                    state["triggered"] = True

        for key in list(self.states):
            if key not in active_keys:
                self.states.pop(key, None)
        return triggered

class ContainerCollisionRiskDetector(BaseEventDetector):
    gui_name = "CONTAINER-COLLISION"

    def __init__(self, config, roi_poly=None, roi_lines=None, abnormal_drive_zones=None):
        super().__init__(config, roi_poly, roi_lines, abnormal_drive_zones)

        # Spreader <-> Moving Container 결합
        self.pair_center_x_ratio = max(
            0.0, float(self.config.get("pair_center_x_ratio", 0.30))
        )
        self.pair_vertical_gap_ratio = max(
            0.0, float(self.config.get("pair_vertical_gap_ratio", 0.35))
        )
        self.pair_confirm_frames = max(
            1, int(self.config.get("pair_confirm_frames", 3))
        )

        # 횡이동 판단
        self.motion_window_sec = max(
            0.1, float(self.config.get("motion_window_sec", 0.4))
        )
        self.horizontal_move_ratio = max(
            0.0, float(self.config.get("horizontal_move_ratio", 0.04))
        )
        self.horizontal_dominance = max(
            1.0, float(self.config.get("horizontal_dominance", 1.5))
        )

        # 충돌 예측
        self.lookahead_sec = max(
            0.1, float(self.config.get("lookahead_sec", 1.0))
        )
        self.max_lookahead_ratio = max(
            0.5, float(self.config.get("max_lookahead_ratio", 2.0))
        )

        # 안전 높이
        self.min_clearance_ratio = max(
            0.0, float(self.config.get("min_clearance_ratio", 0.10))
        )
        self.min_clearance_px = max(
            0.0, float(self.config.get("min_clearance_px", 10.0))
        )

        self.trigger_hold_sec = max(
            0.0, float(self.config.get("trigger_hold_sec", 0.5))
        )

        self.motion_history = defaultdict(lambda: deque(maxlen=30))
        self.pair_votes = defaultdict(int)
        self.states = {}


    @staticmethod
    def _center(box):
        return (
            (float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0,
        )


    def _find_carried_container(self, spreader, containers):
        sx1, sy1, sx2, sy2 = map(float, spreader[:4])
        spreader_cx = (sx1 + sx2) / 2.0

        best_container = None
        best_score = float("inf")

        for container in containers:
            cx1, cy1, cx2, cy2 = map(float, container[:4])

            container_width = max(1.0, cx2 - cx1)
            container_height = max(1.0, cy2 - cy1)
            container_cx = (cx1 + cx2) / 2.0

            # X축 중심이 어느 정도 일치해야 함
            center_diff = abs(spreader_cx - container_cx)

            if center_diff > container_width * self.pair_center_x_ratio:
                continue

            # Spreader 하단과 Container 상단의 간격
            vertical_gap = cy1 - sy2

            # 컨테이너가 스프레더보다 지나치게 위에 있으면 제외
            if vertical_gap < -(container_height * 0.50):
                continue

            if vertical_gap > container_height * self.pair_vertical_gap_ratio:
                continue

            score = (
                center_diff / container_width
                + abs(vertical_gap) / container_height
            )

            if score < best_score:
                best_score = score
                best_container = container

        return best_container


    def _get_motion(self, tid, box, current_time):
        cx, cy = self._center(box)

        history = self.motion_history[tid]
        history.append((current_time, cx, cy))

        if len(history) < 2:
            return None

        reference = None

        # 약 motion_window_sec 전의 가장 가까운 좌표 사용
        for item in reversed(history):
            if current_time - item[0] >= self.motion_window_sec:
                reference = item
                break

        if reference is None:
            return None

        dt = max(0.001, current_time - reference[0])

        dx = cx - reference[1]
        dy = cy - reference[2]

        vx = dx / dt
        vy = dy / dt

        return dx, dy, vx, vy


    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        current_time = time.time()
        privacy_tracks = kwargs.get("privacy_tracks", [])

        spreaders = [
            track for track in tracks
            if track_map.get(int(track[4])) == ID_PCTC_SPREADER
        ]

        containers = [
            track for track in tracks
            if track_map.get(int(track[4])) == ID_PCTC_CONTAINER
        ]

        triggered = []

        if not spreaders or len(containers) < 2:
            self.states.clear()
            return triggered

        active_pair_keys = set()
        active_risk_keys = set()

        visible_container_ids = {
            int(container[4])
            for container in containers
        }

        for spreader in spreaders:

            spreader_tid = int(spreader[4])

            carried = self._find_carried_container(
                spreader,
                containers
            )

            if carried is None:
                continue

            carried_tid = int(carried[4])

            pair_key = (
                spreader_tid,
                carried_tid
            )

            active_pair_keys.add(pair_key)
            self.pair_votes[pair_key] += 1

            # 움직임 히스토리는 결합 확정 전부터 누적
            motion = self._get_motion(
                carried_tid,
                carried,
                current_time
            )

            if self.pair_votes[pair_key] < self.pair_confirm_frames:
                continue

            if motion is None:
                continue

            dx, dy, vx, vy = motion

            cx1, cy1, cx2, cy2 = map(
                float,
                carried[:4]
            )

            moving_width = max(
                1.0,
                cx2 - cx1
            )

            moving_height = max(
                1.0,
                cy2 - cy1
            )

            moving_cx = (
                cx1 + cx2
            ) / 2.0

            # ------------------------------
            # 횡이동 여부 판단
            # ------------------------------

            if abs(dx) < moving_width * self.horizontal_move_ratio:
                continue

            # X 이동량이 Y 이동량보다 충분히 커야 함
            if abs(dx) < abs(dy) * self.horizontal_dominance:
                continue

            direction = (
                1 if dx > 0
                else -1
            )

            # ------------------------------
            # 약 1초 후 X 위치 예측
            # ------------------------------

            predicted_shift = vx * self.lookahead_sec

            max_shift = (
                moving_width
                * self.max_lookahead_ratio
            )

            predicted_shift = max(
                -max_shift,
                min(max_shift, predicted_shift)
            )

            predicted_x1 = (
                cx1 + predicted_shift
            )

            predicted_x2 = (
                cx2 + predicted_shift
            )

            # 현재부터 예상 위치까지 swept 영역
            swept_x1 = min(
                cx1,
                predicted_x1
            )

            swept_x2 = max(
                cx2,
                predicted_x2
            )

            # 이동 중 컨테이너의 하단 Y
            moving_bottom_y = cy2

            required_clearance = max(
                self.min_clearance_px,
                moving_height
                * self.min_clearance_ratio
            )

            best_obstacle = None
            best_clearance = float("inf")

            # ------------------------------
            # 이동방향 앞의 적층 Container 검색
            # ------------------------------

            for obstacle in containers:

                obstacle_tid = int(
                    obstacle[4]
                )

                if obstacle_tid == carried_tid:
                    continue

                ox1, oy1, ox2, oy2 = map(
                    float,
                    obstacle[:4]
                )

                obstacle_cx = (
                    ox1 + ox2
                ) / 2.0

                # 진행방향 반대쪽 Container 제외
                if direction > 0:
                    if obstacle_cx <= moving_cx:
                        continue
                else:
                    if obstacle_cx >= moving_cx:
                        continue

                # 예상 이동 X 경로와 겹치는지
                horizontal_overlap = max(
                    0.0,
                    min(swept_x2, ox2)
                    - max(swept_x1, ox1)
                )

                if horizontal_overlap <= 0:
                    continue

                # 적층 컨테이너의 상단
                obstacle_top_y = oy1

                # 화면 좌표는 위로 갈수록 Y값이 작음
                clearance = (
                    obstacle_top_y
                    - moving_bottom_y
                )

                if clearance < best_clearance:
                    best_clearance = clearance
                    best_obstacle = obstacle


            if best_obstacle is None:
                continue

            # --------------------------------
            # 최종 위험 판단
            #
            # 안전:
            # moving_bottom_y가
            # obstacle_top_y보다 충분히 작아야 함
            #
            # 위험:
            # clearance가 요구값보다 작음
            # --------------------------------

            if best_clearance >= required_clearance:
                continue

            obstacle_tid = int(
                best_obstacle[4]
            )

            risk_key = (
                spreader_tid,
                carried_tid,
                obstacle_tid
            )

            active_risk_keys.add(
                risk_key
            )

            state = self.states.setdefault(
                risk_key,
                {
                    "risk_since": current_time,
                    "triggered": False,
                }
            )

            hold_sec = (
                current_time
                - state["risk_since"]
            )

            # 순간적인 BBox 흔들림은 무시
            if (
                not state["triggered"]
                and hold_sec >= self.trigger_hold_sec
            ):

                triggered.append({
                    "tid": carried_tid,

                    # 대표 객체는 이동중 Container
                    "bbox": carried[:4],

                    "frame": (
                        frame.copy()
                        if frame is not None
                        else None
                    ),

                    "fid": fid,

                    "privacy_tracks":
                        privacy_tracks,

                    "privacy_fid":
                        fid,

                    "objects": [
                        _track_object(
                            spreader,
                            ID_PCTC_SPREADER
                        ),

                        _track_object(
                            carried,
                            ID_PCTC_CONTAINER,
                            label="moving_container"
                        ),

                        _track_object(
                            best_obstacle,
                            ID_PCTC_CONTAINER,
                            label="obstacle_container"
                        ),
                    ],

                    "decision_trace": {
                        "detector":
                            "ContainerCollisionRiskDetector",

                        "reason":
                            "horizontal_container_move_with_insufficient_clearance",

                        "spreader_track_id":
                            spreader_tid,

                        "moving_container_track_id":
                            carried_tid,

                        "obstacle_container_track_id":
                            obstacle_tid,

                        "direction":
                            "right"
                            if direction > 0
                            else "left",

                        "dx_px":
                            round(float(dx), 3),

                        "dy_px":
                            round(float(dy), 3),

                        "vx_px_sec":
                            round(float(vx), 3),

                        "vy_px_sec":
                            round(float(vy), 3),

                        "moving_bottom_y":
                            round(
                                float(moving_bottom_y),
                                3
                            ),

                        "obstacle_top_y":
                            round(
                                float(best_obstacle[1]),
                                3
                            ),

                        "clearance_px":
                            round(
                                float(best_clearance),
                                3
                            ),

                        "required_clearance_px":
                            round(
                                float(required_clearance),
                                3
                            ),

                        "predicted_shift_px":
                            round(
                                float(predicted_shift),
                                3
                            ),

                        "risk_hold_sec":
                            round(
                                float(hold_sec),
                                3
                            ),
                    },
                })

                state["triggered"] = True


        # Pair가 끊어지면 confirm count 초기화
        for key in list(self.pair_votes):
            if key not in active_pair_keys:
                self.pair_votes.pop(key, None)

        # 위험 상태가 해제되면 다시 이벤트 발생 가능하게 초기화
        for key in list(self.states):
            if key not in active_risk_keys:
                self.states.pop(key, None)

        # 사라진 Container 히스토리 정리
        for tid in list(self.motion_history):
            if tid not in visible_container_ids:
                self.motion_history.pop(
                    tid,
                    None
                )

        return triggered
    
EVENT_REGISTRY = {
    "intrusion": IntrusionDetector,
    "illegal_parking": ParkingDetector,
    "no_helmet": HelmetDetector,
    "yard_crossing": YardCrossingDetector,
    "abnormal_drive": AbnormalDriveDetector,
    "walkway_out": WalkwayOutDetector,
    "spreader_danger_zone": SpreaderDangerZoneDetector,
    "container_collision_risk": ContainerCollisionRiskDetector,
}
# [9] 터미널 마법사 및 설정 UI
# ==========================================
def _flush_terminal_input():
    """원격 터미널에서 다음 질문으로 넘어온 잔여 키 입력을 비웁니다."""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return

        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception as e:
        logger.debug(f"터미널 입력 버퍼 정리 실패: {e}")

def _flush_cv2_key_buffer(duration_sec=0.10):
    """OpenCV 창에 남은 Enter/ESC 키 이벤트가 다음 단계로 넘어가지 않게 비웁니다."""
    end_time = time.time() + max(0.0, float(duration_sec))
    while time.time() < end_time:
        try:
            cv2.waitKey(1)
        except Exception:
            break
        time.sleep(0.01)

def guard_interactive_input(delay_sec=None, flush_cv=True, flush_terminal=True):
    # RDP/VNC/SSH 환경에서는 키 입력이 늦게 도착해 다음 input/ROI 창에 들어가는 경우가 있습니다.
    # 짧게 기다린 뒤 OpenCV 키 큐와 터미널 입력 큐를 비워 연속 Enter 오입력을 줄입니다.
    guard_sec = SYS_CFG.get("INTERACTIVE_INPUT_GUARD_SEC", 0.35) if delay_sec is None else delay_sec
    guard_sec = max(0.0, float(guard_sec))
    if guard_sec > 0:
        time.sleep(guard_sec)
    if flush_cv:
        _flush_cv2_key_buffer(min(0.15, guard_sec if guard_sec > 0 else 0.10))
    if flush_terminal:
        _flush_terminal_input()

def guarded_input(prompt, delay_sec=None):
    guard_interactive_input(delay_sec=delay_sec)
    return input(prompt)

def capture_snapshot(url):
    """설정 마법사용 스냅샷 캡처"""
    try:
        cap = cv2.VideoCapture(sanitize_camera_url(url), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except Exception as e:
        logger.error(f"스냅샷 캡처 실패: {e}")
        return None

def get_roi_points_scaled(frame, title, mode="poly"):
    """마우스 클릭을 통해 ROI(관심 영역)를 획득합니다."""
    pts = []
    orig_h, orig_w = frame.shape[:2]
    scale = 960 / orig_w
    disp_h = int(orig_h * scale)
    disp_frame = cv2.resize(frame, (960, disp_h))

    guard_interactive_input()

    cv2.namedWindow(title)
    def mouse_cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            if mode == "line" and len(pts) >= 2:
                return
            pts.append([int(x / scale), int(y / scale)])

    cv2.setMouseCallback(title, mouse_cb)

    # [UX 개선] 전체 화면 적용 방법(Skip)을 명시적으로 안내
    if mode == "poly":
        logger.info(f"'{title}' 설정 - 화면을 클릭하여 점을 찍으십시오. (전체 화면 적용 시 그냥 Enter 또는 ESC)")
    else:
        logger.info(f"'{title}' 설정 - 화면을 클릭하여 선분을 그리십시오. (Enter: 완료, ESC: 취소)")

    while True:
        temp = disp_frame.copy()
        dp = [[int(p[0] * scale), int(p[1] * scale)] for p in pts]

        if mode == "line":
            if len(dp) == 1:
                cv2.circle(temp, tuple(dp[0]), 5, (0, 0, 255), -1)
            elif len(dp) == 2:
                cv2.line(temp, tuple(dp[0]), tuple(dp[1]), (0, 0, 255), 2)
        else:
            if len(dp) > 0:
                cv2.polylines(temp, [np.array(dp, np.int32)], True, (0, 255, 0), 2)

        cv2.imshow(title, temp)
        k = cv2.waitKey(1)
        if k == 13: # Enter
            break
        if k == 27: # ESC
            pts = []
            break
        if mode == "line" and len(pts) == 2:
            cv2.waitKey(500)
            break

    cv2.destroyWindow(title)
    guard_interactive_input()
    return normalize_roi_points(pts, orig_w, orig_h)

WIZARD_SAFETY_EVENT_CHOICES = (
    ("1", "침입 감지"),
    ("2", "불법 주정차"),
    ("3", "안전모 미착용"),
    ("4", "야적장 사이 횡단 금지"),
    ("5", "역주행/유턴 금지"),
    ("6", "보행로 이탈 감지"),
    ("7", "위험구역 진입 감지"),
    ("8", "컨테이너 충돌 위험"),
)
WIZARD_SAFETY_SELECTION_MAP = {
    "1": "intrusion",
    "2": "illegal_parking",
    "3": "no_helmet",
    "4": "yard_crossing",
    "5": "abnormal_drive",
    "6": "walkway_out",
    "7": "spreader_danger_zone",
    "8": "container_collision_risk",
}
WIZARD_CAMERA_OPTION_CHOICES = (
    ("0", "사용 안 함"),
    ("1", "카메라 화각변경 감지"),
    ("2", "카메라 화각변경 감지 + ROI 자동보정"),
)
WIZARD_CAMERA_OPTION_MAP = {
    "0": None,
    "1": ROI_CHANGE_EVENT,
    "2": ROI_CHANGE_APPLY_EVENT,
}


def _print_wizard_event_menu():
    print("=== 안전 이벤트 선택 (1~8, 복수 선택 가능) ===", flush=True)
    for number, label in WIZARD_SAFETY_EVENT_CHOICES:
        print(f"  {number}. {label}", flush=True)
    


def _parse_wizard_event_selection(value):
    text = str(value or "").strip()
    if not text:
        return []
    tokens = [token for token in re.split(r"[,\s]+", text) if token]
    invalid = [token for token in tokens if token not in WIZARD_SAFETY_SELECTION_MAP]
    if invalid:
        raise ValueError(f"허용 번호는 1~8입니다. 잘못된 입력: {', '.join(invalid)}")
    events = []
    for token in tokens:
        event_name = WIZARD_SAFETY_SELECTION_MAP[token]
        if event_name not in events:
            events.append(event_name)
    return events


def _print_wizard_camera_option_menu():
    print("=== 카메라 화각변경 옵션 (안전 이벤트와 별도) ===", flush=True)
    for number, label in WIZARD_CAMERA_OPTION_CHOICES:
        print(f"  {number}. {label}", flush=True)


def _parse_wizard_camera_option(value):
    token = str(value or "").strip()
    if token not in WIZARD_CAMERA_OPTION_MAP:
        raise ValueError("허용 번호는 0, 1, 2입니다.")
    return WIZARD_CAMERA_OPTION_MAP[token]


def _prompt_wizard_safety_events(ip):
    while True:
        _print_wizard_event_menu()
        try:
            return _parse_wizard_event_selection(guarded_input(f"[{ip}] 안전 이벤트 선택 (쉼표 또는 공백, 미선택은 Enter): "))
        except ValueError as exc:
            print(str(exc), flush=True)


def _prompt_wizard_camera_option(ip):
    while True:
        _print_wizard_camera_option_menu()
        try:
            return _parse_wizard_camera_option(guarded_input(f"[{ip}] 화각변경 옵션 선택 (0/1/2): "))
        except ValueError as exc:
            print(str(exc), flush=True)


def run_wizard_batch_mode(rtsp_list, existing_configs=None):
    logger.info("=== 설정 마법사 시작 ===")
    configs = sanitize_camera_configs(existing_configs or {})

    for batch_start in range(0, len(rtsp_list), BATCH_SIZE):
        batch = rtsp_list[batch_start:batch_start + BATCH_SIZE]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            frames = list(executor.map(capture_snapshot, batch))

        display = []
        for frame in frames:
            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Conn Fail", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                display.append(blank)
            else:
                display.append(frame)

        mosaic = create_mosaic_image(display)
        columns = max(1, math.ceil(math.sqrt(len(display))))
        rows = max(1, math.ceil(len(display) / columns))
        cell_width = SCREEN_WIDTH // columns
        cell_height = SCREEN_HEIGHT // rows
        for local_index in range(len(display)):
            row, column = divmod(local_index, columns)
            x, y = column * cell_width, row * cell_height
            cv2.rectangle(mosaic, (x, y), (x + 50, y + 50), (255, 255, 255), -1)
            camera_number = batch_start + local_index + 1
            cv2.putText(mosaic, str(camera_number), (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        cv2.imshow("Select Cameras", mosaic)
        cv2.waitKey(1)
        example_ids = f"{batch_start + 1},{batch_start + 2}" if len(batch) > 1 else f"{batch_start + 1}"
        selection = guarded_input(f">> [Batch {batch_start // BATCH_SIZE + 1}] 설정할 카메라 번호 (예: {example_ids} / 건너뛰기: 엔터): ").strip()
        if not selection:
            continue

        try:
            camera_numbers = [int(token) for token in re.split(r"[,\s]+", selection) if token]
        except ValueError as exc:
            logger.error(f"카메라 번호 입력 오류: {exc}")
            continue

        for camera_number in camera_numbers:
            local_index = camera_number - batch_start - 1
            if not (0 <= local_index < len(batch)) or frames[local_index] is None:
                continue

            url = batch[local_index]
            ip = extract_ip(url)
            safety_events = _prompt_wizard_safety_events(ip)
            camera_option = _prompt_wizard_camera_option(ip)
            events = list(safety_events)
            if camera_option:
                events.append(camera_option)

            roi_poly_norm = []
            if any(event_name in ROI_POLYGON_EVENT_NAMES for event_name in safety_events):
                roi_poly_norm = get_roi_points_scaled(frames[local_index], f"Polygon - CAM: {ip}")

            abnormal_drive_zones_norm = []
            if "abnormal_drive" in safety_events:
                zone_index = 1
                while True:
                    polygon = get_roi_points_scaled(frames[local_index], f"abnormal_drive ROI #{zone_index} - CAM: {ip}")
                    direction = get_roi_points_scaled(frames[local_index], f"Allowed START to END #{zone_index} - CAM: {ip}", mode="line")
                    if len(polygon) >= 3 and len(direction) == 2 and direction[0] != direction[1]:
                        abnormal_drive_zones_norm.append({
                            "roi_poly_norm": polygon,
                            "direction_points_norm": direction,
                        })
                    else:
                        print(f"경고: abnormal_drive ROI #{zone_index}는 polygon 3점 이상과 방향점 2개가 필요하여 추가하지 않았습니다.", flush=True)
                    answer = guarded_input("Add another abnormal_drive ROI? (y/n): ").strip().lower()
                    if answer != "y":
                        break
                    zone_index += 1

            configs[ip] = sanitize_camera_config({
                "url": url,
                "events": events,
                "roi_poly_norm": roi_poly_norm,
                "roi_lines_norm": [],
                "abnormal_drive_zones_norm": abnormal_drive_zones_norm,
            })

    cv2.destroyWindow("Select Cameras")
    return configs

# ==========================================
# [10] 카메라 제어 (FrameReader / Camera)
# ==========================================

class ROIAlignLearningStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {"cameras": {}}
        self.roi_setup_reported = self._load_reported_from_csv()

    def _load_reported_from_csv(self, path=ROI_ALIGN_CSV_LOG_FILE):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # 신/구 컬럼명 모두 허용
                    requested = str(row.get("healthcheck", row.get("healthcheck_requested", ""))).strip().lower()
                    if requested in ("true", "1", "yes", "y"):
                        return True
        except Exception as e:
            logger.warning(f"[ROI DRIFT] CSV state load failed: {e}")
        return False

    # 3×3 격자 전용 CSV 스키마(decision은 normal/suspect/confirm/disturbed 4종).
    #   decision        : normal(이동 없음) / suspect(이동 감지, 누적 중) / confirm(연속 N회 도달 → API)
    #                     / disturbed(전 칸 이동 + 방향 불일치: 큰 회전/줌/장면 전환, 연속 3회 도달 → 확정)
    #   suspect_count   : 연속 suspect 횟수(normal이 나오면 0으로 리셋). confirm_count_required(기본 3) 도달 시 confirm
    #   disturbed_count : 연속 disturbed 횟수. disturbed_confirm_count_required(기본 3) 도달 시 확정
    #   abnormal_count  : 카메라별 연속 suspect/disturbed 합산 횟수. normal일 때만 0으로 초기화
    #   cells_measurable: std 게이트 통과(측정 가능)한 칸 수. cells_moving == cells_measurable 이면 '전부 움직임'(①)
    #   cells_moving    : >GRID_SHAKE_THRESHOLD_PX(10px) 로 움직인 칸 수
    #   cells_consistent: 움직인 칸 중 같은 방향인 칸 수
    #   consistent_quorum: 같은 방향 정족수 = round(cells_moving × GRID_QUORUM_FRACTION). cells_consistent >= 이 값(②)
    #     → ①(전부 움직임) & ②(방향 정족수 충족) 둘 다면 그 검사가 '틀어짐(moved)' = suspect 후보
    #   grid_cells      : 칸별 이동량(9칸 '|' 구분). 측정칸=이동 px, 제외칸="x"(텍스처 없음/측정 실패)
    #   grid_cells_std  : 칸별 std(텍스처, 9칸 '|'). >= GRID_CELL_MIN_STD(10) 이면 측정칸 → 어느 칸이 통과했는지 확인
    #   frame_std       : 전체 프레임 표준편차(텍스처/대비)
    #   anchor_refreshed: 이번 검사에서 앵커를 갱신했는지(True/False)
    #   healthcheck     : ROI 재설정 필요(pending) 상태. confirm/disturbed 확정부터 관제센터가
    #                     ROI를 내려줄(update_config) 때까지 계속 True. 발사 순간은 reason이 채워진 행
    def append_csv_log(self, row, path=ROI_ALIGN_CSV_LOG_FILE):
        fieldnames = [
            "timestamp", "camera_key", "decision",
            "suspect_count", "disturbed_count", "abnormal_count", "cells_measurable", "cells_moving", "cells_consistent", "consistent_quorum",
            "grid_cells", "grid_cells_std", "frame_std",
            "anchor_refreshed", "healthcheck", "reason",
        ]

        def write_one_csv(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            exists = os.path.exists(target_path) and os.path.getsize(target_path) > 0
            if exists:
                # 스키마(컬럼) 변경 시 기존 로그를 백업으로 밀어내고 새 헤더로 시작(컬럼 어긋남 방지)
                try:
                    with open(target_path, "r", newline="", encoding="utf-8") as f:
                        header_line = f.readline()
                    current_header = next(csv.reader([header_line])) if header_line else []
                    if current_header != fieldnames:
                        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = f"{target_path}.bak_header_{stamp}"
                        os.replace(target_path, backup_path)
                        logger.info(
                            f"[ROI DRIFT] CSV header changed; old log moved to {backup_path}"
                        )
                        exists = False
                except Exception as e:
                    logger.warning(f"[ROI DRIFT] CSV header check failed: {e}")
            with open(target_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        try:
            write_one_csv(path)

        except Exception as e:
            logger.warning(f"[ROI DRIFT] CSV log append failed: {e}")

    def _now_iso(self):
        kst = pytz.timezone("Asia/Seoul")
        return datetime.datetime.now(kst).replace(microsecond=0).isoformat()

    def _camera_params(self, camera_key, camera_conf):
        params = dict(ROI_ALIGN_LEARNING_DEFAULTS)
        sys_cfg = SYS_CFG.get("roi_align_learning", {}) or {}
        params.update(sys_cfg.get("defaults", {}) or {})
        params.update((sys_cfg.get("cameras", {}) or {}).get(camera_key, {}) or {})
        params.update((camera_conf or {}).get("roi_align_learning", {}) or {})
        return params

    def _ensure_camera_locked(self, camera_key, camera_conf):
        state = self.data.setdefault("cameras", {}).setdefault(camera_key, {})
        params = self._camera_params(camera_key, camera_conf)
        params["confirm_count_required"] = int(params.get("confirm_count_required", ROI_DRIFT_CONFIRM_COUNT))
        params["disturbed_confirm_count_required"] = int(
            params.get("disturbed_confirm_count_required", GRID_DISTURBED_CONFIRM_COUNT)
        )
        params["abnormal_count_required"] = int(
            params.get("abnormal_count_required", GRID_ABNORMAL_CONFIRM_COUNT)
        )
        state.setdefault("consecutive_suspect", 0)
        state.setdefault("consecutive_disturbed", 0)
        state.setdefault("consecutive_abnormal", 0)
        state.setdefault("awaiting_roi_setup", False)
        state.setdefault("latched_abnormal_kind", "")
        state["params"] = params
        return state, params

    def reset_camera(self, camera_key, reason="roi_updated"):
        """관제 ROI가 적용된 카메라의 누적 판정 상태를 초기화한다."""
        with self.lock:
            state = self.data.setdefault("cameras", {}).get(camera_key)
            if state is None:
                return False
            state["consecutive_suspect"] = 0
            state["consecutive_disturbed"] = 0
            state["consecutive_abnormal"] = 0
            state["awaiting_roi_setup"] = False
            state["latched_abnormal_kind"] = ""
            state["last_decision"] = "normal"
            state["last_reset_at"] = self._now_iso()
            state["last_reset_reason"] = str(reason)
            return True

    def record_check(self, camera_key, camera_conf, moved, disturbed=False):
        """단순 3-상태 판정(normal / suspect / confirm) + disturbed(방향 흩어진 큰 변화).
          moved=False → normal (suspect 카운터를 0으로 리셋)
          moved=True  → suspect 카운터 +1
                        · 카운터 < confirm_count_required(기본 3) → 'suspect'
                        · 카운터 == confirm_count_required        → 'confirm' + 헬스체크(API) 1회 발사
                        · 카운터 >  confirm_count_required        → 'confirm' 유지(이미 발사했으므로 재발사 X)
          disturbed=True(전 칸 이동 + 방향 불일치: 큰 회전/줌/장면 전환) → disturbed 카운터 +1
                        · 연속 disturbed_confirm_count_required(기본 3)회 도달 시 disturbed 확정
          suspect 또는 disturbed이면 카메라별 abnormal_count +1, 요청 전 normal이면 0으로 초기화
                        · abnormal_count_required(기본 3) 도달 순간 헬스체크(API) 1회 발사
                        · 요청 후에는 관제 ROI가 적용될 때까지 confirm을 유지하며 검사마다 +1
                        · 자동보정은 하지 않음(방향이 흩어져 median 이동량 신뢰 불가 → 사람이 재설정)
        관제 ROI 적용 시 reset_camera()가 세 카운터와 pending 상태를 초기화한다."""
        with self.lock:
            state, params = self._ensure_camera_locked(camera_key, camera_conf)
            now_iso = self._now_iso()
            state["last_checked_at"] = now_iso
            confirm_required = max(1, int(params.get("confirm_count_required", ROI_DRIFT_CONFIRM_COUNT)))
            disturbed_required = max(
                1,
                int(params.get("disturbed_confirm_count_required", GRID_DISTURBED_CONFIRM_COUNT))
            )
            abnormal_required = max(
                1,
                int(params.get("abnormal_count_required", GRID_ABNORMAL_CONFIRM_COUNT))
            )

            # 알림 발생 후에는 새 ROI 수신 전까지 판정을 잠근다. 현재 화면 상태와 관계없이
            # abnormal_count를 검사 주기마다 증가시켜 대기 지속 시간을 로그에서 확인한다.
            if bool(state.get("awaiting_roi_setup", False)):
                abnormal_count = int(state.get("consecutive_abnormal", 0)) + 1
                state["consecutive_abnormal"] = abnormal_count
                latched_kind = str(state.get("latched_abnormal_kind", ""))
                if latched_kind not in ("suspect", "disturbed"):
                    latched_kind = (
                        "disturbed"
                        if int(state.get("consecutive_disturbed", 0)) >= int(state.get("consecutive_suspect", 0))
                        else "suspect"
                    )
                    state["latched_abnormal_kind"] = latched_kind
                if latched_kind == "disturbed":
                    disturbed_count = int(state.get("consecutive_disturbed", 0)) + 1
                    state["consecutive_disturbed"] = disturbed_count
                    suspect_count = 0
                    state["consecutive_suspect"] = 0
                else:
                    suspect_count = int(state.get("consecutive_suspect", 0)) + 1
                    state["consecutive_suspect"] = suspect_count
                    disturbed_count = 0
                    state["consecutive_disturbed"] = 0
                state["last_decision"] = "confirm"
                observed = "disturbed" if disturbed else ("suspect" if moved else "normal")
                return {
                    "decision": "confirm",
                    "observed_decision": observed,
                    "latched_abnormal_kind": latched_kind,
                    "suspect_count": suspect_count,
                    "disturbed_count": disturbed_count,
                    "abnormal_count": abnormal_count,
                    "confirmed": False,
                    "disturbed_confirmed": False,
                    "pending": True,
                    "healthcheck": False,
                    "confirm_count_required": confirm_required,
                    "disturbed_confirm_count_required": disturbed_required,
                    "abnormal_count_required": abnormal_required,
                }

            if disturbed:
                state["consecutive_suspect"] = 0
                disturbed_count = int(state.get("consecutive_disturbed", 0)) + 1
                state["consecutive_disturbed"] = disturbed_count
                abnormal_count = int(state.get("consecutive_abnormal", 0)) + 1
                state["consecutive_abnormal"] = abnormal_count
                state["last_decision"] = "disturbed"
                healthcheck = (abnormal_count == abnormal_required)
                if healthcheck:
                    state["last_healthcheck_at"] = now_iso
                    state["awaiting_roi_setup"] = True
                    state["latched_abnormal_kind"] = "disturbed"
                return {"decision": "confirm" if healthcheck else "disturbed",
                        "observed_decision": "disturbed",
                        "suspect_count": 0, "disturbed_count": disturbed_count,
                        "abnormal_count": abnormal_count,
                        "confirmed": False, "disturbed_confirmed": disturbed_count >= disturbed_required,
                        "pending": healthcheck,
                        "healthcheck": healthcheck, "confirm_count_required": confirm_required,
                        "disturbed_confirm_count_required": disturbed_required,
                        "abnormal_count_required": abnormal_required}

            if not moved:
                state["consecutive_suspect"] = 0
                state["consecutive_disturbed"] = 0
                state["consecutive_abnormal"] = 0
                state["last_decision"] = "normal"
                return {"decision": "normal", "suspect_count": 0, "disturbed_count": 0,
                        "abnormal_count": 0,
                        "confirmed": False, "disturbed_confirmed": False,
                        "pending": False,
                        "healthcheck": False, "confirm_count_required": confirm_required,
                        "disturbed_confirm_count_required": disturbed_required,
                        "abnormal_count_required": abnormal_required}

            state["consecutive_disturbed"] = 0
            suspect_count = int(state.get("consecutive_suspect", 0)) + 1
            state["consecutive_suspect"] = suspect_count
            abnormal_count = int(state.get("consecutive_abnormal", 0)) + 1
            state["consecutive_abnormal"] = abnormal_count
            healthcheck = (abnormal_count == abnormal_required)

            if suspect_count < confirm_required:
                state["last_decision"] = "suspect"
                if healthcheck:
                    state["last_healthcheck_at"] = now_iso
                    state["awaiting_roi_setup"] = True
                    state["latched_abnormal_kind"] = "suspect"
                return {"decision": "confirm" if healthcheck else "suspect",
                        "observed_decision": "suspect",
                        "suspect_count": suspect_count, "disturbed_count": 0,
                        "abnormal_count": abnormal_count,
                        "confirmed": False, "disturbed_confirmed": False,
                        "pending": healthcheck,
                        "healthcheck": healthcheck, "confirm_count_required": confirm_required,
                        "disturbed_confirm_count_required": disturbed_required,
                        "abnormal_count_required": abnormal_required}

            # suspect_count >= confirm_required → confirm. API는 '막 도달한 순간'(==)에만 1회 발사.
            state["last_decision"] = "confirm"
            if healthcheck:
                state["last_healthcheck_at"] = now_iso
                state["awaiting_roi_setup"] = True
                state["latched_abnormal_kind"] = "suspect"
            return {"decision": "confirm", "observed_decision": "suspect",
                    "suspect_count": suspect_count, "disturbed_count": 0,
                    "abnormal_count": abnormal_count,
                    "confirmed": True, "disturbed_confirmed": False,
                    "pending": healthcheck,
                    "healthcheck": healthcheck, "confirm_count_required": confirm_required,
                    "disturbed_confirm_count_required": disturbed_required,
                    "abnormal_count_required": abnormal_required}

    def was_roi_setup_reported(self):
        with self.lock:
            return bool(self.roi_setup_reported)

    def mark_roi_setup_reported(self):
        with self.lock:
            self.roi_setup_reported = True

ROI_ALIGN_LEARNING_STORE = ROIAlignLearningStore()

class AnchorTrackingROIAligner:
    """전체 화면 3×3 격자 phaseCorrelate 기반 화각 흔들림 감지기.
    앵커 슬롯(BASE=원본 보존, UPDATED=주기 갱신)에 전체 프레임 gray만 보관한다."""
    def __init__(self):
        self.anchor_slots = {}                 # {ANCHOR_BASE/UPDATED: {"gray", "shape", "created_at", "updated_at"}}
        self.last_debug = {"status": "not_initialized", "method": "grid_phase"}
        self.last_grid_result = None           # 마지막 detect_grid_camera_motion 결과(칸별 진단 포함, 외부 조회용)

    def _gray_plain(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _grid_cells(self, h, w):
        """전체 프레임을 GRID_ROWS×GRID_COLS로 나눈 셀의 (i, j, y1, y2, x1, x2)를 yield."""
        rh = max(1, h // GRID_ROWS)
        rw = max(1, w // GRID_COLS)
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                y1 = i * rh
                y2 = (i + 1) * rh if i < GRID_ROWS - 1 else h
                x1 = j * rw
                x2 = (j + 1) * rw if j < GRID_COLS - 1 else w
                yield i, j, y1, y2, x1, x2

    def _grid_textured_cell_count(self, gray):
        """텍스처가 충분한(표준편차 >= GRID_CELL_MIN_STD) 셀 개수."""
        n = 0
        h, w = gray.shape[:2]
        for _, _, y1, y2, x1, x2 in self._grid_cells(h, w):
            if float(gray[y1:y2, x1:x2].std()) >= GRID_CELL_MIN_STD:
                n += 1
        return n

    def _cell_phase(self, a, b):
        """두 동일 크기 셀의 평행이동 벡터를 phaseCorrelate로 측정."""
        try:
            if a.shape != b.shape or a.size == 0:
                return None
            win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
            return {"dx": float(dx), "dy": float(dy), "shift": float(math.hypot(dx, dy))}
        except Exception:
            return None

    def set_grid_anchor(self, frame):
        """전체 프레임 gray를 격자 앵커로 저장(BASE+UPDATED). 텍스처 셀이 부족하면 실패."""
        if frame is None:
            self.last_debug = {"status": "grid_anchor_no_frame", "method": "grid_phase"}
            return False
        gray = self._gray_plain(frame)
        n_tex = self._grid_textured_cell_count(gray)
        if n_tex < GRID_QUORUM_FLOOR:
            self.last_debug = {"status": f"grid_anchor_low_texture:{n_tex}/{GRID_ROWS*GRID_COLS}",
                               "method": "grid_phase"}
            return False
        now_iso = ROI_ALIGN_LEARNING_STORE._now_iso()
        for slot in (ANCHOR_BASE, ANCHOR_UPDATED):
            self.anchor_slots[slot] = {
                "gray": gray, "shape": frame.shape[:2],
                "created_at": now_iso, "updated_at": now_iso,
            }
        self.last_debug = {"status": "grid_anchor_set", "method": "grid_phase"}
        return True

    def refresh_grid_anchor(self, frame):
        """흔들림 없음이 확정된 상태에서 UPDATED 격자 앵커(gray)를 현재 프레임으로 갱신.
        BASE는 보존."""
        if frame is None:
            return "skip_refresh_no_frame"
        slot = self.anchor_slots.get(ANCHOR_UPDATED)
        if slot is None:
            return "skip_refresh_no_slot"
        gray = self._gray_plain(frame)
        n_tex = self._grid_textured_cell_count(gray)
        if n_tex < GRID_QUORUM_FLOOR:
            return f"skip_refresh_low_texture:{n_tex}"
        slot["gray"] = gray
        slot["shape"] = frame.shape[:2]
        slot["updated_at"] = ROI_ALIGN_LEARNING_STORE._now_iso()
        return "grid_refresh"

    def detect_grid_camera_motion(self, frame):
        """전체 화면 3×3 격자에서 각 칸의 평행이동을 측정해 '카메라 틀어짐'을 판정.
        측정 성공한 칸이 모두 10px(GRID_SHAKE_THRESHOLD_PX)를 초과해 움직이고,
        그중 대표 방향과 코사인 유사도 >= GRID_DIRECTION_COS_MIN인 칸이
        round(n_moving × GRID_QUORUM_FRACTION) 이상이면 moved=True.
        반환 dict: moved, n_measurable, n_moving, n_textured, quorum, consistent, consistent_quorum, frame_std, cells, status."""
        res = {"moved": False, "disturbed": False, "n_measurable": 0, "n_moving": 0, "n_textured": 0,
               "quorum": GRID_QUORUM_FLOOR, "consistent": 0, "consistent_quorum": 0,
               "all_measured_moving": False, "frame_std": 0.0,
               "median_dx": 0.0, "median_dy": 0.0,   # 움직인 칸들의 대표 평행이동(roi_change_apply 보정용)
               "cells": [], "status": "grid_not_initialized"}
        self.last_grid_result = res  # 외부(test 등)에서 칸별 수치 조회에 사용
        anchor = self.anchor_slots.get(ANCHOR_UPDATED) or self.anchor_slots.get(ANCHOR_BASE)
        if not anchor or anchor.get("gray") is None or frame is None:
            return res
        anchor_gray = anchor["gray"]
        cur = self._gray_plain(frame)
        h, w = cur.shape[:2]
        if anchor_gray.shape[:2] != (h, w):
            res["status"] = "grid_shape_mismatch"
            return res

        vecs = []          # 모든 측정칸 벡터(정족수 계산용 n_meas)
        moving_cells = []  # '움직인 칸'(>임계)의 cell dict 참조(같은 방향 판정 + 칸별 cos 기록용)
        n_moving = 0
        n_textured = 0  # std(텍스처) 통과 칸 수 = 측정 가능한 칸. 적응형 정족수의 기준.
        cells = []      # 칸별 진단(격자 순서 9개). 모든 칸에 std, 측정칸은 shift도 기록.
        res["frame_std"] = float(cur.std())  # 전체 프레임 표준편차(텍스처/대비)
        for i, j, y1, y2, x1, x2 in self._grid_cells(h, w):
            a = anchor_gray[y1:y2, x1:x2].astype(np.float32)
            b = cur[y1:y2, x1:x2].astype(np.float32)
            astd = float(a.std())
            bstd = float(b.std())
            # std 게이트가 비교하는 값(앵커·현재 중 작은 쪽). 이게 GRID_CELL_MIN_STD 미만이면 텍스처 없음.
            cell_std = min(astd, bstd)
            if cell_std < GRID_CELL_MIN_STD:
                cells.append({"m": False, "why": "lowstd", "std": cell_std})
                continue
            n_textured += 1
            c = self._cell_phase(a, b)
            if c is None:
                cells.append({"m": False, "why": "phase_fail", "std": cell_std})
                continue
            moving = c["shift"] > GRID_SHAKE_THRESHOLD_PX
            cell = {"m": True, "shift": c["shift"], "std": cell_std,
                    "dx": c["dx"], "dy": c["dy"], "moving": moving}
            cells.append(cell)
            vecs.append((c["dx"], c["dy"]))
            if moving:
                n_moving += 1
                moving_cells.append(cell)   # 나중에 cos/consistent를 이 dict에 직접 기록

        n_meas = len(vecs)
        # 적응형 정족수: 이 프레임의 텍스처 칸 수에 비례. 카메라별 장면 차이를 자동 보정.
        quorum = max(GRID_QUORUM_FLOOR, int(round(n_textured * GRID_QUORUM_FRACTION)))
        res["cells"] = cells
        res["n_measurable"] = n_meas
        res["n_textured"] = n_textured
        res["quorum"] = quorum
        res["n_moving"] = n_moving

        # '움직인 칸(>임계)'들의 대표 방향(median 벡터)과, 그 방향과 코사인 유사도가 높은 칸 수(consistent).
        #   consistent = "10px 이상 움직였고 + 대표 방향과 cos >= GRID_DIRECTION_COS_MIN" 인 칸 수 → 판정의 핵심.
        #   거리(px)가 아니라 방향(각도)으로 보므로, 같은 방향이면 이동 크기가 달라도 함께 묶인다.
        #   각 움직인 칸 dict에 cos(코사인)·consistent(통과 여부)를 기록 → 어느 칸이 방향 조건을 통과했는지 확인 가능.
        if moving_cells:
            arr = np.array([(mc["dx"], mc["dy"]) for mc in moving_cells], dtype=np.float32)
            mdx = float(np.median(arr[:, 0]))
            mdy = float(np.median(arr[:, 1]))
            res["median_dx"] = mdx   # 움직인 칸들의 대표 이동벡터(ROI 자동 보정에 사용)
            res["median_dy"] = mdy
            ref_mag = float(math.hypot(mdx, mdy))
            if ref_mag > 1e-6:
                mags = np.hypot(arr[:, 0], arr[:, 1])
                cos_sim = (arr[:, 0] * mdx + arr[:, 1] * mdy) / (mags * ref_mag + 1e-6)
                for mc, cs in zip(moving_cells, cos_sim):
                    mc["cos"] = float(cs)
                    mc["consistent"] = bool(cs >= GRID_DIRECTION_COS_MIN)
                res["consistent"] = int(np.sum(cos_sim >= GRID_DIRECTION_COS_MIN))
                # 보정용 대표 이동벡터(median_dx/dy)는 '방향 일치 칸'만으로 재계산.
                # (반대 방향으로 측정된 아웃라이어 칸(내용 변화)이 median을 오염시키는 것 방지.
                #  실측: cos=-0.83으로 17.7px 측정된 칸이 전체 median을 1.7px 끌어내렸음)
                cons_vecs = [(mc["dx"], mc["dy"]) for mc in moving_cells if mc.get("consistent")]
                if cons_vecs:
                    arr_c = np.array(cons_vecs, dtype=np.float32)
                    res["median_dx"] = float(np.median(arr_c[:, 0]))
                    res["median_dy"] = float(np.median(arr_c[:, 1]))

        if n_meas < quorum:
            # 측정칸이 정족수 미달(주로 저텍스처/야간) → 판단 보류(moved=False, 알람 안 함).
            res["status"] = f"grid_low_texture:meas={n_meas}/tex={n_textured}/q={quorum}"
            res["moved"] = False
            return res

        # [판정] 측정 성공한 칸이 모두 10px 초과로 움직였고,
        #   그중 같은 방향인 칸이 움직인 칸 수의 GRID_QUORUM_FRACTION 이상이면 카메라 틀어짐.
        consistent_quorum = int(round(n_moving * GRID_QUORUM_FRACTION))
        # n_meas >= quorum 은 위 low_texture 체크에서 이미 보장됨 → '모든 측정칸이 움직였나'만 확인.
        all_measured_moving = (n_moving == n_meas)
        res["consistent_quorum"] = int(consistent_quorum)
        res["all_measured_moving"] = bool(all_measured_moving)
        res["moved"] = bool(all_measured_moving and res["consistent"] >= consistent_quorum)
        # 전 칸 이동했지만 방향이 흩어짐(정족수 미달) = 평행이동으로 설명 안 되는 큰 변화(회전/줌/장면 전환)
        res["disturbed"] = bool(all_measured_moving and res["consistent"] < consistent_quorum)
        tag = "grid_moved" if res["moved"] else ("grid_disturbed" if res["disturbed"] else "grid_still")
        res["status"] = (f"{tag}:consistent={res['consistent']}/q={consistent_quorum}"
                         f"/moving={n_moving}/meas={n_meas}/all_moving={int(all_measured_moving)}")
        return res

def transform_roi_points_h(points, H):
    """ROI 점 리스트를 homography H로 변환한 새 리스트를 반환(roi_change_apply homography 보정용)."""
    if not points:
        return []
    arr = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(arr, H).reshape(-1, 2)
    return [[int(round(float(x))), int(round(float(y)))] for x, y in out]

def estimate_alignment_homography(anchor_gray, cur_gray, expected_shift):
    """앵커(틀어지기 전) gray ↔ 현재 gray를 ORB 특징점 매칭 + RANSAC으로 정합해 homography를 추정.
    렌즈 왜곡으로 지역별 이동량이 다른 경우까지 반영하므로 평행이동(median)보다 ROI 위치에서
    정확한 보정이 가능하다. confirm 시점에 1회만 호출된다.
    아래 게이트를 하나라도 통과 못 하면 (None, 사유)를 반환 → 호출부가 평행이동 보정으로 폴백.
      게이트 1: RANSAC 인라이어 수 >= GRID_HOMOGRAPHY_MIN_INLIERS (매칭 신뢰성)
      게이트 2: H의 화면중심 이동량 ≈ 격자 median 측정(expected_shift) (교차검증, 오매칭 방어)
      게이트 3: 스케일/원근 성분 상한 (ROI가 찌그러지는 비정상 변환 방어)
    반환: (H(3x3 np.ndarray) 또는 None, 상태 문자열)"""
    try:
        if anchor_gray is None or cur_gray is None or anchor_gray.shape != cur_gray.shape:
            return None, "homography_bad_input"
        orb = cv2.ORB_create(nfeatures=GRID_HOMOGRAPHY_MAX_FEATURES)
        kp1, des1 = orb.detectAndCompute(anchor_gray, None)
        kp2, des2 = orb.detectAndCompute(cur_gray, None)
        if des1 is None or des2 is None:
            return None, "homography_no_features"
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        if len(matches) < GRID_HOMOGRAPHY_MIN_INLIERS:
            return None, f"homography_low_matches:{len(matches)}"
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, GRID_HOMOGRAPHY_RANSAC_REPROJ_PX)
        if H is None:
            return None, "homography_ransac_fail"
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < GRID_HOMOGRAPHY_MIN_INLIERS:
            return None, f"homography_low_inliers:{inliers}"

        # 게이트 2: 화면 중심의 이동량이 격자 측정과 대략 일치해야 함(전혀 다른 곳에 매칭된 경우 방어)
        h, w = anchor_gray.shape[:2]
        center = np.float32([[[w / 2.0, h / 2.0]]])
        moved = cv2.perspectiveTransform(center, H)[0][0]
        tdx = float(moved[0]) - w / 2.0
        tdy = float(moved[1]) - h / 2.0
        edx, edy = float(expected_shift[0]), float(expected_shift[1])
        if math.hypot(tdx - edx, tdy - edy) > GRID_HOMOGRAPHY_SHIFT_TOL_PX:
            return None, (f"homography_shift_mismatch:H=({tdx:.1f},{tdy:.1f})"
                          f"/grid=({edx:.1f},{edy:.1f})")

        # 게이트 3: 과도한 스케일/원근 변형 방지. 렌즈 왜곡 때문에 스케일이 1에서 다소 벗어나는 건
        # 정상이므로 상한을 여유 있게 둔다(GRID_HOMOGRAPHY_SCALE_MIN/MAX 주석 참고).
        h33 = float(H[2, 2]) if abs(float(H[2, 2])) > 1e-9 else 1.0
        A = np.array(H[:2, :2], dtype=np.float64) / h33
        sv = np.linalg.svd(A, compute_uv=False)
        if float(sv[0]) > GRID_HOMOGRAPHY_SCALE_MAX or float(sv[1]) < GRID_HOMOGRAPHY_SCALE_MIN:
            return None, f"homography_scale_out:sv=({float(sv[0]):.2f},{float(sv[1]):.2f})"
        if (abs(float(H[2, 0])) > GRID_HOMOGRAPHY_PERSPECTIVE_MAX
                or abs(float(H[2, 1])) > GRID_HOMOGRAPHY_PERSPECTIVE_MAX):
            return None, "homography_perspective_excessive"

        return H, f"homography_ok:inliers={inliers} center_shift=({tdx:.1f},{tdy:.1f}) sv=({float(sv[0]):.2f},{float(sv[1]):.2f})"
    except Exception as e:
        return None, f"homography_error:{e}"

def refine_roi_local_residual(anchor_gray, cur_gray, H, roi_center,
                              patch_px=None):
    """homography 적용 후 ROI 지역에 남는 잔차 평행이동을 측정한다(roi_change_apply 정밀 보정).
    앵커를 H로 워핑하면 '보정이 완벽할 때의 현재 화면 예측'이 되므로, ROI 중심 주변 패치에서
    예측(워핑 앵커)과 실제(현재 프레임)의 차이를 phaseCorrelate로 1회 측정해 반환한다.
    부호 규약은 격자 측정과 동일: 반환값 = 그 지역 내용물이 예측 대비 이동한 방향/거리
    → ROI 점들에 그대로 더하면 된다.
    측정 불가(텍스처 부족)거나 잔차가 비정상적으로 크면 (None, 사유)를 반환한다.
    반환: ((rdx, rdy) 또는 None, 상태 문자열)"""
    try:
        if patch_px is None:
            patch_px = GRID_APPLY_REFINE_PATCH_PX
        h, w = cur_gray.shape[:2]
        half = max(32, int(patch_px) // 2)
        # 패치가 화면 안에 완전히 들어오도록 중심을 클램프
        cx = min(max(float(roi_center[0]), half), w - half)
        cy = min(max(float(roi_center[1]), half), h - half)
        x1 = int(round(cx - half)); x2 = x1 + 2 * half
        y1 = int(round(cy - half)); y2 = y1 + 2 * half
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None, "refine_patch_out_of_frame"
        warped = cv2.warpPerspective(anchor_gray, H, (w, h))
        a = warped[y1:y2, x1:x2].astype(np.float32)
        b = cur_gray[y1:y2, x1:x2].astype(np.float32)
        # 워핑 경계의 검은 영역/무늬 없는 패치는 측정 불가
        if min(float(a.std()), float(b.std())) < GRID_CELL_MIN_STD:
            return None, "refine_low_texture"
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (rdx, rdy), _ = cv2.phaseCorrelate(a, b, win)
        if math.hypot(rdx, rdy) > GRID_APPLY_REFINE_MAX_PX:
            return None, f"refine_residual_too_big:({rdx:.1f},{rdy:.1f})"
        return (float(rdx), float(rdy)), f"refine_ok:({rdx:.1f},{rdy:.1f})"
    except Exception as e:
        return None, f"refine_error:{e}"

class FrameReader:
    def __init__(self, url, ip):
        self.url = sanitize_camera_url(url)
        self.ip = ip
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_t = time.time()
        self.lock = threading.Lock()
        self.decode_cfg = SYS_CFG.get("video_decode", {})
        self._decode_mode_logged = False
        self._decode_mode = "init"
        self._decode_pid = "-"
        self._decode_shape = "-"
        self._decode_restarts = 0
        self._decode_read_failures = 0
        self._decode_frame_count = 0
        self._decode_window_frames = 0
        self._decode_window_bytes = 0
        self._decode_window_start = time.time()
        self._gst_check_logged = False

        threading.Thread(target=self._run, daemon=True).start()

    def _decode_log_interval(self):
        try:
            return max(1.0, float(self.decode_cfg.get("log_interval_sec", 10.0)))
        except Exception:
            return 10.0

    def _decode_sleep_sec(self, key, default, max_sec=0.05):
        try:
            value = float(self.decode_cfg.get(key, default))
        except Exception:
            value = float(default)
        return min(max(0.0, value), max(0.0, float(max_sec)))

    def _decode_verbose_logs(self):
        value = self.decode_cfg.get("verbose_logs", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _emit_decode_log(self, *args, **kwargs):
        """
        [최적화 및 에러 방어] 파라미터 순서와 키워드 호출이 혼용되는 
        기존 코드의 모든 케이스를 스마트하게 파싱하여 처리합니다.
        """
        # 1. 시스템 설정에서 verbose_logs가 명시적으로 켜져 있지 않으면 즉시 드랍
        if not SYS_CFG.get("verbose_logs", False):
            return

        if not args:
            return

        # 2. 첫 번째 인자가 'level'인지 'msg'인지 스마트 판별
        first_arg = args[0]
        # 들어온 값이 숫자(int)이거나 알려진 레벨 문자열(debug, info 등)이면 level이 먼저 온 것으로 간주
        is_level_first = isinstance(first_arg, int) or (isinstance(first_arg, str) and first_arg.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

        if is_level_first:
            level = first_arg
            msg = args[1] if len(args) > 1 else ""
            fmt_args = args[2:]
        else:
            msg = first_arg
            # kwargs에서 level을 빼내고, 없으면 INFO로 기본값 처리
            level = kwargs.pop('level', logging.INFO)
            fmt_args = args[1:]

        # 3. 레벨 정규화 (문자열 'debug' -> logging.DEBUG 정수형)
        if isinstance(level, str):
            level_int = getattr(logging, level.upper(), logging.INFO)
        else:
            level_int = level

        # 4. 로거 레벨 필터링 (불필요한 조립 방지)
        if not logger.isEnabledFor(level_int):
            return

        # 5. 안전한 kwarg 추출 (표준 로깅 모듈이 모르는 인자는 버림)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['exc_info', 'stack_info', 'extra']}

        # 6. 최종 로깅 실행 (지연 평가 방식)
        logger.log(level_int, msg, *fmt_args, **valid_kwargs)

    def _emit_gst_check_once(self, message, level="info"):
        if self._gst_check_logged:
            return
        self._gst_check_logged = True
        self._emit_decode_log(message, level=level)

    def _mask_pipeline_text(self, text):
        return re.sub(r"(?i)(rtsp://)([^/@\s]+)@", r"\1***@", str(text))

    def _short_external_output(self, text, limit=800):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        text = self._mask_pipeline_text(str(text or "").strip())
        text = re.sub(r"\s+", " ", text)
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _ensure_clean_url(self, context):
        clean_url = sanitize_camera_url(self.url)
        if clean_url != self.url:
            logger.warning(
                f"[CAM:{self.ip}] {context} sanitized runtime stream URL: "
                f"{self._mask_pipeline_text(repr(self.url))} -> "
                f"{self._mask_pipeline_text(repr(clean_url))}"
            )
            self.url = clean_url
        return self.url

    def _drain_binary_log_pipe(self, pipe, line_buffer):
        sleep_sec = self._decode_sleep_sec("gstreamer_log_drain_sleep_sec", 0.002)
        try:
            for raw_line in iter(pipe.readline, b""):
                line = self._short_external_output(raw_line)
                if line:
                    line_buffer.append(line)
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _emit_buffered_stderr(self, label, line_buffer):
        if not line_buffer:
            return
        self._emit_decode_log(
            f"[{label} STDERR] CAM:{self.ip} {' | '.join(line_buffer)}",
            level="warning"
        )

    def _cmd_text(self, cmd):
        return self._mask_pipeline_text(" ".join(str(part) for part in cmd))

    def _set_decode_pipeline(self, mode, shape="-", pid="-", cmd=None, extra=""):
        self._decode_mode = mode
        self._decode_pid = str(pid)
        self._decode_shape = shape
        self._decode_restarts += 1
        self._decode_frame_count = 0
        self._decode_window_frames = 0
        self._decode_window_bytes = 0
        self._decode_window_start = time.time()

        detail = f" extra={extra}" if extra else ""
        self._emit_decode_log(
            f"[DECODE PIPELINE] CAM:{self.ip} restart={self._decode_restarts} "
            f"mode={mode} pid={self._decode_pid} shape={shape}{detail}"
        )
        if cmd:
            self._emit_decode_log(f"[DECODE PIPELINE] CAM:{self.ip} cmd={self._cmd_text(cmd)}")

    def _note_decode_frame(self, frame_bytes, shape):
        self._decode_frame_count += 1
        self._decode_window_frames += 1
        self._decode_window_bytes += int(frame_bytes or 0)
        self._decode_shape = shape

        now = time.time()
        elapsed = now - self._decode_window_start
        # 1. 쿨타임(기본 10초)이 안 지났으면 아무 연산 없이 즉시 복귀 (CPU 방어)
        if elapsed < self._decode_log_interval():
            return

        fps = self._decode_window_frames / max(0.001, elapsed)
        mbps = (self._decode_window_bytes * 8.0) / max(0.001, elapsed) / 1_000_000.0
        
        # 2. [핵심 최적화] 레벨을 DEBUG로 낮춰 CLI 출력을 원천 차단.
        # f-string을 쓰지 않고 % 포맷을 사용해 실제 파일에 쓸 때만 백그라운드에서 조립되게 유도.
        self._emit_decode_log(
            logging.DEBUG,
            "[DECODE FPS] CAM:%s mode=%s pid=%s fps=%.2f frames=%d shape=%s pipe_mbps=%.1f read_failures=%d restarts=%d connected=%s",
            self.ip, self._decode_mode, self._decode_pid, fps, self._decode_frame_count,
            self._decode_shape, mbps, self._decode_read_failures, self._decode_restarts, self.connected
        )
        
        self._decode_window_frames = 0
        self._decode_window_bytes = 0
        self._decode_window_start = now

    def _note_decode_failure(self, reason, level="warning"):
        self._decode_read_failures += 1
        self._emit_decode_log(
            f"[DECODE FAIL] CAM:{self.ip} mode={self._decode_mode} pid={self._decode_pid} "
            f"reason={reason} failures={self._decode_read_failures} "
            f"frames={self._decode_frame_count} restarts={self._decode_restarts}",
            level=level
        )

    def _open_capture(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            shape = f"{src_w}x{src_h}" if src_w > 0 and src_h > 0 else "unknown"
            self._set_decode_pipeline(
                "opencv_ffmpeg_cpu",
                shape=shape,
                extra=f"backend=CAP_FFMPEG source_fps={src_fps:.2f}"
            )
            self._decode_mode_logged = True
        return cap

    def _should_use_ffmpeg_vaapi_pipe(self):
        backend = str(self.decode_cfg.get("backend", "auto")).strip().lower()
        if backend in ("opencv", "cv2", "ffmpeg_opencv", "gstreamer", "gst", "gst_vaapi"):
            return False

        mode = str(self.decode_cfg.get("hw_acceleration", "auto")).strip().lower()
        if mode in ("", "none", "off", "cpu", "false", "0"):
            return False

        if not sys.platform.startswith("linux"):
            return False

        hw_device = str(self.decode_cfg.get("hw_device", "/dev/dri/renderD128")).strip()
        if mode not in ("auto", "vaapi") or not hw_device or not os.path.exists(hw_device):
            return False

        return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

    def _should_use_gstreamer_pipe(self):
        backend = str(self.decode_cfg.get("backend", "auto")).strip().lower()
        if backend in ("opencv", "cv2", "ffmpeg_opencv", "ffmpeg", "ffmpeg_vaapi", "vaapi"):
            return False

        if backend not in ("auto", "gstreamer", "gst", "gst_vaapi"):
            self._emit_gst_check_once(
                f"[GSTREAMER CHECK] CAM:{self.ip} skip reason=unsupported_backend backend={backend}"
            )
            return False

        if not sys.platform.startswith("linux"):
            self._emit_gst_check_once(
                f"[GSTREAMER CHECK] CAM:{self.ip} skip reason=non_linux platform={sys.platform} backend={backend}"
            )
            return False

        gst_launch = shutil.which("gst-launch-1.0")
        gst_inspect = shutil.which("gst-inspect-1.0")
        if not gst_launch or not gst_inspect:
            self._emit_gst_check_once(
                f"[GSTREAMER CHECK] CAM:{self.ip} skip reason=missing_gstreamer_tools "
                f"backend={backend} gst-launch={gst_launch or '-'} gst-inspect={gst_inspect or '-'}",
                level="warning"
            )
            return False

        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            self._emit_gst_check_once(
                f"[GSTREAMER CHECK] CAM:{self.ip} skip reason=missing_ffprobe backend={backend}",
                level="warning"
            )
            return False

        self._emit_gst_check_once(
            f"[GSTREAMER CHECK] CAM:{self.ip} enabled backend={backend} "
            f"gst-launch={gst_launch} gst-inspect={gst_inspect} ffprobe={ffprobe_path}"
        )
        return True

    def _probe_stream_info(self):
        probe_url = self._ensure_clean_url("ffprobe")
        cmd = ["ffprobe", "-v", "error"]
        if probe_url.lower().startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp", "-stimeout", "3000000"])
        cmd.extend([
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            probe_url,
        ])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=8,
                text=True
            )
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
        except Exception as e:
            logger.warning(f"[CAM:{self.ip}] ffprobe stream info failed: {e}")
            return None

        if result.returncode != 0:
            logger.warning(
                f"[CAM:{self.ip}] ffprobe stream info failed: rc={result.returncode} "
                f"stderr={self._short_external_output(err)} stdout={self._short_external_output(out)}"
            )
            return None

        try:
            data = json.loads(out)
            streams = data.get("streams", [])
            if not streams:
                raise ValueError("no video stream")
            stream = streams[0]
            width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
            codec = str(stream.get("codec_name") or "").strip().lower()
            if width > 0 and height > 0:
                return width, height, codec
        except Exception as e:
            logger.warning(
                f"[CAM:{self.ip}] ffprobe stream info parse failed: {e} "
                f"stdout={self._short_external_output(out)} stderr={self._short_external_output(err)}"
            )

        logger.warning(
            f"[CAM:{self.ip}] ffprobe returned no usable video info: "
            f"stdout={self._short_external_output(out)} stderr={self._short_external_output(err)}"
        )
        return None

    def _probe_stream_shape(self):
        info = self._probe_stream_info()
        if info is None:
            return None
        width, height, _ = info
        return width, height

    def _scaled_output_shape(self, width, height):
        if width <= 7200:
            return width, height
        ratio = 720.0 / float(width)
        out_height = max(2, int(round((height * ratio) / 2.0) * 2))
        return 720, out_height

    def _decode_fps_limit(self):
        try:
            fps_limit = float(self.decode_cfg.get("fps_limit", 15.0) or 0.0)
        except Exception:
            fps_limit = 15.0
        return fps_limit if fps_limit > 0 else None

    def _gst_element_exists(self, element_name):
        try:
            subprocess.run(
                ["gst-inspect-1.0", element_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=True
            )
            return True
        except Exception:
            return False

    def _first_gst_element(self, candidates):
        for element_name in candidates:
            if self._gst_element_exists(element_name):
                return element_name
        return None

    def _gst_framerate_caps(self, fps_limit):
        if fps_limit is None:
            return None
        fraction = Fraction(float(fps_limit)).limit_denominator(1000)
        return f"{fraction.numerator}/{fraction.denominator}"

    def _select_gstreamer_decoder(self, codec):
        codec = str(codec or "").strip().lower()
        mode = str(self.decode_cfg.get("hw_acceleration", "auto")).strip().lower()
        allow_hw = mode not in ("", "none", "off", "cpu", "false", "0")

        if codec in ("h264", "avc1"):
            depay, parser = "rtph264depay", "h264parse"
            hw_candidates = ["vaapih264dec", "vah264dec"]
            cpu_candidates = ["avdec_h264", "openh264dec"]
        elif codec in ("hevc", "h265"):
            depay, parser = "rtph265depay", "h265parse"
            hw_candidates = ["vaapih265dec", "vah265dec"]
            cpu_candidates = ["avdec_h265"]
        else:
            return None

        decoder = None
        decoder_kind = "cpu"
        if allow_hw:
            decoder = self._first_gst_element(hw_candidates)
            decoder_kind = "vaapi" if decoder else "cpu"
        if decoder is None:
            decoder = self._first_gst_element(cpu_candidates)
        if decoder is None:
            return None

        return depay, parser, decoder, decoder_kind

    def _run_gstreamer_pipe(self):
        info = self._probe_stream_info()
        if info is None:
            return False

        in_w, in_h, codec = info
        out_w, out_h = self._scaled_output_shape(in_w, in_h)
        
        # [핵심 최적화] 파이프 전송량 50% 감축 (BGR: 3 bytes -> NV12: 1.5 bytes)
        frame_size = int(out_w * out_h * 1.5)
        
        fps_limit = self._decode_fps_limit()
        loop_sleep_sec = self._decode_sleep_sec("gstreamer_loop_sleep_sec", 0.001)
        decoder_info = self._select_gstreamer_decoder(codec)
        if decoder_info is None:
            self._note_decode_failure(f"gstreamer_unsupported_codec_{codec or 'unknown'}")
            return False

        depay, parser, decoder, decoder_kind = decoder_info
        latency_ms = int(self.decode_cfg.get("gstreamer_latency_ms", 50) or 50)
        protocols = str(self.decode_cfg.get("gstreamer_protocols", "tcp") or "tcp").strip().lower()
        tcp_timeout_us = 3000000
        drop_on_latency_text = "true"

        cmd = [
            "gst-launch-1.0", "-q",
            "rtspsrc", f"location={self.url}", f"protocols={protocols}",
            f"latency={latency_ms}", f"drop-on-latency={drop_on_latency_text}", f"tcp-timeout={tcp_timeout_us}",
            "!", depay,
            "!", parser,
            "!", decoder
        ]

        # -------------------------------------------------------------
        # [수정 핵심 1] GStreamer 문법 오류 해결 (15.0/1 -> 15/1 분수 형태 변환)
        # -------------------------------------------------------------
        framerate_caps = self._gst_framerate_caps(fps_limit)
        framerate_str = f",framerate={framerate_caps}" if framerate_caps else ""

        # -------------------------------------------------------------
        # [수정 핵심 2] 스트라이드 패딩 찌그러짐 원천 차단
        # vaapipostproc 직후에 videoconvert를 배치하여 GPU 패딩 메모리를 
        # 파이썬이 읽기 좋은 촘촘한(Dense) NV12 메모리로 쫙 펴줍니다.
        # -------------------------------------------------------------
        if decoder_kind == "vaapi":
            cmd.extend([
                "!", "vaapipostproc",
                "!", f"video/x-raw,format=NV12,width={out_w},height={out_h}",
                "!", "videoconvert",
                "!", "videorate",
                "!", f"video/x-raw{framerate_str}"
            ])
        else:
            cmd.extend([
                "!", "videoconvert",
                "!", "videoscale",
                "!", "videorate",
                "!", f"video/x-raw,format=NV12,width={out_w},height={out_h}{framerate_str}"
            ])

        cmd.extend([
            "!", "fdsink", "fd=1", "sync=false"
        ])

        env = os.environ.copy()
        proc = None
        gst_failed = False
        gst_stderr_lines = deque(maxlen=30)
        gst_stderr_thread = None
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, bufsize=frame_size * 2)
            if proc.stderr is not None:
                gst_stderr_thread = threading.Thread(target=self._drain_binary_log_pipe, args=(proc.stderr, gst_stderr_lines), daemon=True)
                gst_stderr_thread.start()
            if proc.stdout is None:
                return False

            self.connected = True
            self._set_decode_pipeline("gstreamer_pipe_nv12", shape=f"{in_w}x{in_h}->{out_w}x{out_h}", pid=proc.pid, cmd=cmd)
            self.last_t = time.time()
            first_frame_logged = False

            while self.running:
                if time.time() - self.last_t > WATCHDOG_TIMEOUT:
                    self._note_decode_failure(f"gstreamer_timeout_{WATCHDOG_TIMEOUT:.0f}s", level="error")
                    break

                raw = proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break

                # [핵심 최적화] Python GIL을 100% 우회하는 OpenCV C++ 초고속 BGR 변환
                yuv_img = np.frombuffer(raw, dtype=np.uint8).reshape((int(out_h * 1.5), out_w))
                fr = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV12)
                
                with self.lock:
                    self.frame = fr
                    self.fid += 1
                    self.last_t = time.time()
                    
                self._note_decode_frame(frame_size, f"{out_w}x{out_h}")
                if not first_frame_logged:
                    first_frame_logged = True
                if loop_sleep_sec > 0.0:
                    time.sleep(loop_sleep_sec)

            return True
        except Exception as e:
            logger.warning(f"[CAM:{self.ip}] GStreamer pipe failed: {e}")
            return False
        finally:
            self.connected = False
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            
            if gst_stderr_thread is not None:
                try:
                    gst_stderr_thread.join(timeout=0.2)
                except Exception:
                    pass
            if gst_failed:
                self._emit_buffered_stderr("GSTREAMER", gst_stderr_lines)

    def _run_ffmpeg_vaapi_pipe(self):
        shape = self._probe_stream_shape()
        if shape is None:
            return False

        in_w, in_h = shape
        out_w, out_h = self._scaled_output_shape(in_w, in_h)
        frame_size = out_w * out_h * 3
        hw_device = str(self.decode_cfg.get("hw_device", "/dev/dri/renderD128")).strip()
        fps_limit = self._decode_fps_limit()

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if self.url.lower().startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp", "-stimeout", "3000000", "-fflags", "nobuffer", "-flags", "low_delay"])
        cmd.extend(["-hwaccel", "vaapi", "-hwaccel_device", hw_device, "-i", self.url, "-an"])

        vf_chain = []
        if (out_w, out_h) != (in_w, in_h):
            vf_chain.append(f"scale={out_w}:{out_h}")
        if fps_limit is not None:
            vf_chain.append(f"fps={fps_limit:g}")
        if vf_chain:
            cmd.extend(["-vf", ",".join(vf_chain)])

        cmd.extend(["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"])

        env = os.environ.copy()
        vaapi_driver = str(self.decode_cfg.get("vaapi_driver", "")).strip()
        if vaapi_driver:
            env.setdefault("LIBVA_DRIVER_NAME", vaapi_driver)

        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=frame_size * 2)
            if proc.stdout is None:
                return False

            self.connected = True
            self._set_decode_pipeline(
                "ffmpeg_vaapi_pipe",
                shape=f"{in_w}x{in_h}->{out_w}x{out_h}",
                pid=proc.pid,
                cmd=cmd,
                extra=f"device={hw_device} driver={vaapi_driver or '-'} fps_limit={fps_limit or '-'} frame_bytes={frame_size}"
            )
            self._decode_mode_logged = True
            self.last_t = time.time()

            while self.running:
                if time.time() - self.last_t > WATCHDOG_TIMEOUT:
                    self._note_decode_failure(f"vaapi_timeout_{WATCHDOG_TIMEOUT:.0f}s", level="error")
                    break

                raw = proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    self._note_decode_failure(f"vaapi_short_read_{len(raw)}_of_{frame_size}", level="error")
                    break

                fr = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
                with self.lock:
                    self.frame = fr
                    self.fid += 1
                    self.last_t = time.time()
                self._note_decode_frame(frame_size, f"{out_w}x{out_h}")

            return True
        except Exception as e:
            logger.warning(f"[CAM:{self.ip}] FFmpeg VAAPI pipe failed: {e}")
            return False
        finally:
            self.connected = False
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def _run(self):
        while self.running:
            if self._should_use_gstreamer_pipe():
                used_gstreamer_pipe = self._run_gstreamer_pipe()
                if used_gstreamer_pipe:
                    continue
                if not self.decode_cfg.get("fallback_to_cpu", True):
                    time.sleep(5)
                    continue
                logger.warning(f"[CAM:{self.ip}] GStreamer pipe unavailable; trying FFmpeg/OpenCV fallback")

            if self._should_use_ffmpeg_vaapi_pipe():
                used_vaapi_pipe = self._run_ffmpeg_vaapi_pipe()
                if used_vaapi_pipe:
                    continue
                if not self.decode_cfg.get("fallback_to_cpu", True):
                    time.sleep(5)
                    continue
                logger.warning(f"[CAM:{self.ip}] VAAPI pipe unavailable; trying OpenCV FFmpeg reader")

            cap = self._open_capture()
            if not cap.isOpened():
                #  [수정] 초기 연결 실패 로깅 (디버그 모드일때만 빈도수 조절하여 출력하도록 권장하나, 연결 실패는 중요하므로 error 처리)
                logger.error(f" [CAM:{self.ip}] RTSP 연결 실패. 5초 후 재시도합니다.")
                time.sleep(5)
                continue

            self.connected = True
            logger.info(f"[CAM:{self.ip}] 카메라 스트림 연결 성공.")
            self.last_t = time.time()

            while self.running and cap.isOpened():
                if time.time() - self.last_t > WATCHDOG_TIMEOUT:
                    #  [수정] 타임아웃 로깅 레벨 격상
                    logger.error(f" [CAM:{self.ip}] 카메라 수신 타임아웃({WATCHDOG_TIMEOUT}s). 재연결을 시도합니다.")
                    break

                ret, fr = cap.read()
                if not ret:
                    self._note_decode_failure("opencv_read_failed", level="error")
                    break

                if fr is not None:
                    if fr.shape[1] > 7200:
                        ratio = 720 / fr.shape[1]
                        fr = cv2.resize(fr, (720, int(fr.shape[0] * ratio)), interpolation=cv2.INTER_NEAREST)
                    with self.lock:
                        self.frame = fr
                        self.fid += 1
                        self.last_t = time.time()
                    self._note_decode_frame(fr.nbytes, f"{fr.shape[1]}x{fr.shape[0]}")
                time.sleep(0.005)

            self.connected = False
            try: cap.release()
            except Exception as e: logger.error(f"카메라 리소스 해제 중 예외: {e}")

    def read(self):
        with self.lock:
            return self.frame, self.fid, self.connected

class Camera:
    def __init__(self, ip, conf, det_main, det_helmet, det_face, det_plate, cam_id, event_inference_mode="main"):
        self.ip = ip
        self.camera_key = ip
        self.conf = sanitize_camera_config(conf)
        self.cam_id = cam_id
        self.event_inference_mode = event_inference_mode
        self.events = list(self.conf.get("events", []))

        self.det_main = det_main
        self.det_helmet = det_helmet
        self.det_face = det_face
        self.det_plate = det_plate

        self.trk_main = SimpleTracker()
        self.trk_helmet = SimpleTracker()

        self.reader = FrameReader(self.conf.get("url", ""), ip)
        self.recorder = VideoRecorder(ip, cam_id=self.cam_id)

        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}
        self.fps_queue = deque(maxlen=30)
        self.current_fps = 0.0

        self.roi_poly_norm = self.conf.get("roi_poly_norm", [])
        self.roi_lines_norm = []
        self.abnormal_drive_zones_norm = self.conf.get("abnormal_drive_zones_norm", [])
        self.roi_poly = []
        self.roi_lines = []
        self.abnormal_drive_zones = []

        self.base_roi_poly = []
        self.base_roi_lines = []
        self.base_abnormal_drive_zones = []
        self.aligned_roi_poly = []
        self.aligned_roi_lines = []
        self.aligned_abnormal_drive_zones = []

        self.roi_frame_shape = None
        self.status_history = deque(maxlen=10)
        self._reset_alignment_state("ALIGN INIT")
        self._rebuild_handlers()

    def _denormalize_abnormal_drive_zones(self, width, height):
        zones = []
        for zone in self.abnormal_drive_zones_norm or []:
            polygon = denormalize_roi_points(zone.get("roi_poly_norm", []), width, height)
            direction = denormalize_roi_points(zone.get("direction_points_norm", []), width, height)
            if len(polygon) >= 3 and len(direction) == 2:
                zones.append({"roi_poly": polygon, "direction_points": direction})
        return zones

    @staticmethod
    def _shift_abnormal_drive_zones(zones, shift):
        dx = int(round(float(shift[0])))
        dy = int(round(float(shift[1])))
        shifted = []
        for zone in zones or []:
            shifted.append({
                "roi_poly": [[int(point[0]) + dx, int(point[1]) + dy] for point in zone.get("roi_poly", [])],
                "direction_points": [[int(point[0]) + dx, int(point[1]) + dy] for point in zone.get("direction_points", [])],
            })
        return shifted

    def _reset_alignment_state(self, status_text="ALIGN RESET"):
        self.aligner = AnchorTrackingROIAligner()
        self.anchor_set = False

        self.base_roi_poly = []
        self.base_roi_lines = []
        self.base_abnormal_drive_zones = []
        self.aligned_roi_poly = []
        self.aligned_roi_lines = []
        self.aligned_abnormal_drive_zones = []
        self.roi_shift = [0.0, 0.0]
        self.roi_auto_corrected = False
        self.roi_setup_pending = False

        self.last_align_time = 0.0
        self.last_anchor_attempt_time = 0.0
        self.anchor_startup_wait_started_at = 0.0
        self.align_status_text = status_text
        self.align_ok = False
        self.align_shifted = False

    def _rebuild_handlers(self):
        self.handlers = {}
        for event_name in self.events:
            detector_class = EVENT_REGISTRY.get(event_name)
            if detector_class is None:
                continue
            self.handlers[event_name] = detector_class(
                SYS_CFG.get("event_config", {}).get(event_name, {}),
                self.roi_poly,
                self.roi_lines,
                self.abnormal_drive_zones,
            )

    def update_config(self, new_conf):
        old_events = list(self.events)
        ROI_ALIGN_LEARNING_STORE.reset_camera(self.camera_key, reason="camera_config_updated")

        self.conf = sanitize_camera_config(new_conf)
        self.events = list(self.conf.get("events", []))
        self.roi_poly_norm = self.conf.get("roi_poly_norm", [])
        self.roi_lines_norm = []
        self.abnormal_drive_zones_norm = self.conf.get("abnormal_drive_zones_norm", [])
        self.roi_poly = []
        self.roi_lines = []
        self.abnormal_drive_zones = []
        self.roi_frame_shape = None

        self._reset_alignment_state("ALIGN RESET")
        self._rebuild_handlers()
        self.status_history.clear()
        logger.info(f"[CAM:{self.ip}] 무중단 설정 리로드 완료: {old_events} -> {self.events} | ROI aligner reset")
        logger.debug(f"[CCTV_Aligner] CAM {self.cam_id} aligner reset after config reload")

    def _initialize_base_roi_if_needed(self, frame):
        if frame is None:
            return False
        height, width = frame.shape[:2]
        need_init = self.roi_frame_shape != frame.shape[:2]
        need_init = need_init or bool(self.roi_poly_norm and not self.base_roi_poly)
        need_init = need_init or bool(self.abnormal_drive_zones_norm and not self.base_abnormal_drive_zones)
        if not need_init:
            return True

        self.base_roi_poly = denormalize_roi_points(self.roi_poly_norm, width, height) if self.roi_poly_norm else []
        self.base_roi_lines = []
        self.base_abnormal_drive_zones = self._denormalize_abnormal_drive_zones(width, height)
        self.roi_shift = [0.0, 0.0]
        self.roi_auto_corrected = False
        self.aligned_roi_poly = list(self.base_roi_poly)
        self.aligned_roi_lines = []
        self.aligned_abnormal_drive_zones = self._shift_abnormal_drive_zones(self.base_abnormal_drive_zones, self.roi_shift)
        self.roi_frame_shape = frame.shape[:2]
        self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)
        logger.info(
            f"[CAM:{self.cam_id}] base ROI init | poly={len(self.base_roi_poly)} "
            f"abnormal_zones={len(self.base_abnormal_drive_zones)} shape={frame.shape[:2]}"
        )
        return True

    def _inject_roi_to_handlers(self, roi_poly, roi_lines):
        self.roi_poly = roi_poly or []
        self.roi_lines = []
        self.aligned_abnormal_drive_zones = self._shift_abnormal_drive_zones(self.base_abnormal_drive_zones, self.roi_shift)
        self.abnormal_drive_zones = self.aligned_abnormal_drive_zones

        for event_name, handler in self.handlers.items():
            handler.roi_poly = np.array(self.roi_poly, dtype=np.int32) if len(self.roi_poly) >= 3 else np.empty((0, 2), dtype=np.int32)
            handler.roi_lines = []
            if event_name == "abnormal_drive":
                handler.abnormal_drive_zones = list(self.abnormal_drive_zones)

    def _shift_roi_points(self, points, shift):
        """ROI 점들을 (dx, dy)만큼 평행이동한 새 리스트로 반환(roi_change_apply 보정용).homography가
        안됬을때 (조건미달등 이유로) 사용되는 fallback 함수."""
        dx = int(round(shift[0]))
        dy = int(round(shift[1]))
        shifted = []
        for p in (points or []):
            shifted.append([int(p[0]) + dx, int(p[1]) + dy])
        return shifted

    def _log_align_blocked(self, decision, detail):
        try:
            now_b = time.time()
            if now_b - getattr(self, "_last_blocked_csv_time", 0.0) < ALIGN_INTERVAL_SEC:
                return
            self._last_blocked_csv_time = now_b
            csv_row = {
                "timestamp": ROI_ALIGN_LEARNING_STORE._now_iso(),
                "camera_key": self.camera_key,
                "decision": "normal",
                "suspect_count": 0,
                "disturbed_count": 0,
                "abnormal_count": 0,
                "cells_measurable": "",
                "cells_moving": "",
                "cells_consistent": "",
                "consistent_quorum": "",
                "grid_cells": "",
                "grid_cells_std": "",
                "frame_std": "",
                "anchor_refreshed": False,
                "healthcheck": bool(getattr(self, "roi_setup_pending", False)),
                "reason": (
                    f"{decision}:{detail} "
                    f"applied_shift=({getattr(self, 'roi_shift', [0.0, 0.0])[0]:.1f},"
                    f"{getattr(self, 'roi_shift', [0.0, 0.0])[1]:.1f})"
                ),
            }
            ROI_ALIGN_LEARNING_STORE.append_csv_log(csv_row)
        except Exception as e:
            logger.debug(f"[CAM:{getattr(self,'cam_id','?')}] blocked-state log failed: {e}")

    def _update_alignment(self, frame):
        if frame is None:
            return

        self._initialize_base_roi_if_needed(frame)

        # 격자(화각변경) 감지는 이벤트 지정(cameras.json events)된 카메라만 동작
        #   roi_change       = 감지 + 알림(사람이 재설정)
        #   roi_change_apply = 감지 + ROI 자동 보정(둘 중 하나만 있어도 감지는 켜짐)
        if ROI_CHANGE_EVENT not in self.events and ROI_CHANGE_APPLY_EVENT not in self.events:
            self.align_status_text = "ROI CHANGE OFF"
            return

        if not self.anchor_set:
            now = time.time()
            if getattr(self, "anchor_startup_wait_started_at", 0.0) <= 0.0:
                self.anchor_startup_wait_started_at = now
                self.align_status_text = "ANCHOR WAIT STABILIZE"
                return
            startup_elapsed = now - float(getattr(self, "anchor_startup_wait_started_at", now) or now)
            if startup_elapsed < ANCHOR_STARTUP_DELAY_SEC:
                self.align_status_text = f"ANCHOR WAIT {ANCHOR_STARTUP_DELAY_SEC - startup_elapsed:.1f}s"
                self._log_align_blocked("blocked_anchor_wait", f"anchor_startup_wait:{ANCHOR_STARTUP_DELAY_SEC - startup_elapsed:.1f}s")
                return
            if now - getattr(self, "last_anchor_attempt_time", 0.0) < ANCHOR_RETRY_INTERVAL_SEC:
                return
            self.last_anchor_attempt_time = now

            if self.aligner.set_grid_anchor(frame):
                self.anchor_set = True
                self.last_align_time = now
                self.align_status_text = "ANCHOR SET"
                self.align_ok = True
                self.align_shifted = False
                logger.info(f"[CAM:{self.cam_id}] grid anchor set | ip={self.ip}")
            else:
                self.align_status_text = "ANCHOR FAIL"
                self.align_ok = False
                dbg = getattr(self.aligner, "last_debug", {}) or {}
                self._log_align_blocked("anchor_fail", f"grid_anchor_fail:{dbg.get('status', 'unknown')}")
            return

        now = time.time()
        if now - self.last_align_time < ALIGN_INTERVAL_SEC:
            return

        grid = self.aligner.detect_grid_camera_motion(frame)
        moved = bool(grid["moved"])
        disturbed = bool(grid.get("disturbed", False))
        n_meas = int(grid["n_measurable"])
        n_mov = int(grid["n_moving"])
        quorum = int(grid.get("quorum", GRID_QUORUM_FLOOR))
        consistent = int(grid.get("consistent", 0))
        consistent_quorum = int(grid.get("consistent_quorum", 0))
        self.align_ok = (n_meas >= quorum)

        refresh_allowed = (
            (not moved)
            and self.align_ok
            and (n_mov < quorum)
            and not self.roi_setup_pending
        )
        anchor_refreshed = False
        if refresh_allowed:
            action = self.aligner.refresh_grid_anchor(frame)
            anchor_refreshed = str(action).startswith("grid_refresh")

        decision = ROI_ALIGN_LEARNING_STORE.record_check(self.camera_key, self.conf, moved, disturbed=disturbed)
        decision_name = str(decision.get("decision", "normal"))
        observed_decision = str(decision.get("observed_decision", decision_name))
        decision_pending = bool(decision.get("pending", False))
        suspect_count = int(decision.get("suspect_count", 0))
        disturbed_count = int(decision.get("disturbed_count", 0))
        abnormal_count = int(decision.get("abnormal_count", 0))
        confirm_required = int(decision.get("confirm_count_required", ROI_DRIFT_CONFIRM_COUNT))
        disturbed_required = int(decision.get("disturbed_confirm_count_required", GRID_DISTURBED_CONFIRM_COUNT))
        abnormal_required = int(decision.get("abnormal_count_required", GRID_ABNORMAL_CONFIRM_COUNT))
        if decision_pending:
            self.roi_setup_pending = True
        self.align_shifted = bool(decision.get("confirmed", False))

        # ---- ROI 자동 보정 (roi_change_apply 카메라 전용) ------------------------------
        # confirm 시점에 [1순위] homography 보정을 시도한다:
        #   앵커(틀어지기 전) gray ↔ 현재 프레임을 ORB 특징점 매칭으로 정합해, 렌즈 왜곡에 의한
        #   지역별 이동량 차이까지 반영해 ROI 점들을 변환한다(전역 평행이동보다 정확).
        #   검증 게이트(estimate_alignment_homography)를 통과 못 하면
        #   [2순위] 격자 median 평행이동 보정으로 폴백한다. 시도 결과(h=...)는 CSV reason에 기록.
        # 보정 후 현재 프레임으로 재앵커한다(→ 다음 검사는 새 위치 기준 → 이중 보정 방지).
        # 보정은 관제센터가 ROI를 내려줄 때까지 '1회만' 한다(roi_auto_corrected 래치).
        #   보정 후 추가 틀어짐이 감지돼도 다시 보정하지 않고 setup required 보고만 유지하며,
        #   관제센터가 ROI를 내려주면 update_config → _reset_alignment_state에서 래치가 풀린다.
        # 보정 성공 여부와 무관하게 confirm이면 아래에서 서버에 setup required를 보고한다.
        #   (pending 플래그는 관제센터가 헬스체크 응답으로 ROI를 내려줄(확인) 때까지 계속 true로 전송됨)
        # 이동량이 상한 초과(평행이동으로 설명 안 되는 큰 변화)면 보정 없이 보고만 한다.
        roi_corrected = False
        roi_correct_method = ""
        h_status = ""
        mdx = float(grid.get("median_dx", 0.0))
        mdy = float(grid.get("median_dy", 0.0))
        shift_mag = math.hypot(mdx, mdy)
        can_auto_correct = (
            ROI_CHANGE_APPLY_EVENT in self.events
            and not self.roi_auto_corrected
            and decision.get("confirmed", False)
            and not disturbed
            and 0.0 < shift_mag <= GRID_APPLY_MAX_SHIFT_PX
            and (self.base_roi_poly or self.base_roi_lines or self.base_abnormal_drive_zones)
        )
        if can_auto_correct:
            # [1순위] homography 보정 시도 (앵커 gray는 이미 aligner에 보관돼 있음)
            new_poly = None
            new_lines = None
            h_status = "homography_no_anchor"
            anchor_slot = (self.aligner.anchor_slots.get(ANCHOR_UPDATED)
                           or self.aligner.anchor_slots.get(ANCHOR_BASE))
            anchor_gray = anchor_slot.get("gray") if anchor_slot else None
            if anchor_gray is not None:
                cur_gray = self.aligner._gray_plain(frame)
                H, h_status = estimate_alignment_homography(
                    anchor_gray,
                    cur_gray,
                    expected_shift=(GRID_APPLY_SHIFT_SIGN * mdx, GRID_APPLY_SHIFT_SIGN * mdy),
                )
                if H is not None:
                    cand_poly = transform_roi_points_h(self.base_roi_poly, H)
                    cand_lines = transform_roi_points_h(self.base_roi_lines, H)
                    # ROI 점 단위 최종 검증: 변위가 비정상적으로 크면 폴백
                    base_all = list(self.base_roi_poly) + list(self.base_roi_lines)
                    cand_all = cand_poly + cand_lines
                    disps = [(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
                             for b, a in zip(base_all, cand_all)]
                    max_disp = max((math.hypot(dx, dy) for dx, dy in disps), default=0.0)
                    if 0.0 < max_disp <= GRID_APPLY_MAX_SHIFT_PX * 1.5:
                        # [정밀화] ROI 지역 잔차 보정: H는 전 화면 최적 근사라 ROI 지점에는
                        # 몇 px 잔차가 남을 수 있음 → ROI 중심 패치에서 잔차를 1회 더 측정해 반영
                        roi_center = (
                            sum(float(p[0]) for p in cand_all) / len(cand_all),
                            sum(float(p[1]) for p in cand_all) / len(cand_all),
                        )
                        residual, refine_status = refine_roi_local_residual(
                            anchor_gray, cur_gray, H, roi_center)
                        h_status = f"{h_status} {refine_status}"
                        if residual is not None:
                            rdx = int(round(residual[0]))
                            rdy = int(round(residual[1]))
                            cand_poly = [[p[0] + rdx, p[1] + rdy] for p in cand_poly]
                            cand_lines = [[p[0] + rdx, p[1] + rdy] for p in cand_lines]
                            cand_all = cand_poly + cand_lines
                            disps = [(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
                                     for b, a in zip(base_all, cand_all)]
                        new_poly, new_lines = cand_poly, cand_lines
                        # 오버레이/로그용 유효 평행이동 = ROI 점들의 평균 변위(잔차 반영 후)
                        self.roi_shift = [
                            sum(d[0] for d in disps) / len(disps),
                            sum(d[1] for d in disps) / len(disps),
                        ]
                    else:
                        h_status = f"homography_point_disp_out:max={max_disp:.1f}"

            if new_poly is not None or new_lines is not None:
                roi_correct_method = "homography"
                self.aligned_roi_poly = new_poly or []
                self.aligned_roi_lines = new_lines or []
            else:
                # [2순위] 평행이동(격자 median) 폴백
                roi_correct_method = "translation"
                self.roi_shift[0] += GRID_APPLY_SHIFT_SIGN * mdx
                self.roi_shift[1] += GRID_APPLY_SHIFT_SIGN * mdy
                self.aligned_roi_poly = self._shift_roi_points(self.base_roi_poly, self.roi_shift)
                self.aligned_roi_lines = self._shift_roi_points(self.base_roi_lines, self.roi_shift)

            self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)
            self.aligner.refresh_grid_anchor(frame)   # 보정 후 현재 프레임을 새 기준 앵커로
            anchor_refreshed = True                   # CSV 반영: 보정하면서 재앵커함
            self.align_shifted = False                # 보정 완료 → confirm 상태 해제
            self.roi_auto_corrected = True            # 래치 잠금: 관제센터 ROI 수신 전까지 추가 보정 금지
            roi_corrected = True
            logger.warning(
                f"[ROI AUTO-CORRECT] cam={self.cam_id} ip={self.ip} method={roi_correct_method} "
                f"grid_shift=({mdx:.1f},{mdy:.1f}) mag={shift_mag:.1f}px "
                f"applied_shift=({self.roi_shift[0]:.1f},{self.roi_shift[1]:.1f}) "
                f"h={h_status} consistent={consistent}/{consistent_quorum}"
            )
        # -----------------------------------------------------------------------------

        healthcheck_requested = False
        healthcheck_reason = ""
        if decision.get("healthcheck", False):
            # confirm/disturbed 확정 시 서버에 ROI 재설정 필요를 보고. 자동 보정 성공 여부와 무관하게 보내며,
            # pending 플래그는 관제센터가 헬스체크 응답으로 ROI를 내려줄 때까지 유지된다(계속 true 전송).
            healthcheck_requested = True
            self.roi_setup_pending = True   # 관제 확인(update_config) 전까지 계속 true로 전송/기록
            if observed_decision == "disturbed":
                healthcheck_reason = (
                    f"disturbed camera={self.camera_key} cam_id={self.cam_id} "
                    f"consistent={consistent}/q={consistent_quorum} "
                    f"moving={n_mov}/{n_meas} disturbed={disturbed_count}/{disturbed_required} "
                    f"abnormal={abnormal_count}/{abnormal_required} "
                    f"auto_corrected=False"
                )
            elif observed_decision == "suspect" and not decision.get("confirmed", False):
                healthcheck_reason = (
                    f"abnormal camera={self.camera_key} cam_id={self.cam_id} "
                    f"current=suspect consistent={consistent}/q={consistent_quorum} "
                    f"moving={n_mov}/{n_meas} suspect={suspect_count}/{confirm_required} "
                    f"abnormal={abnormal_count}/{abnormal_required} "
                    f"auto_corrected=False"
                )
            else:
                healthcheck_reason = (
                    f"confirm camera={self.camera_key} cam_id={self.cam_id} "
                    f"consistent={consistent}/q={consistent_quorum} "
                    f"moving={n_mov}/{n_meas} suspect={suspect_count}/{confirm_required} "
                    f"abnormal={abnormal_count}/{abnormal_required} "
                    f"auto_corrected={roi_corrected} method={roi_correct_method or '-'} "
                    f"grid_shift=({mdx:.1f},{mdy:.1f}) mag={shift_mag:.1f} "
                    f"applied_shift=({self.roi_shift[0]:.1f},{self.roi_shift[1]:.1f}) "
                    f"h={h_status or '-'}"
                )
            request_terminal_roi_setup_required(reason=healthcheck_reason)
            if observed_decision == "disturbed":
                self.align_status_text = (
                    f"ROI SETUP REQUIRED disturbed={disturbed_count}/{disturbed_required} "
                    f"abnormal={abnormal_count}/{abnormal_required} "
                    f"consistent={consistent}/{consistent_quorum} moving={n_mov}/{n_meas}"
                )
            elif observed_decision == "suspect" and not decision.get("confirmed", False):
                self.align_status_text = (
                    f"ROI SETUP REQUIRED abnormal={abnormal_count}/{abnormal_required} "
                    f"current=suspect={suspect_count}/{confirm_required} "
                    f"consistent={consistent}/{consistent_quorum} moving={n_mov}/{n_meas}"
                )
            elif roi_corrected:
                self.align_status_text = (
                    f"ROI AUTO-CORRECT[{roi_correct_method}] + SETUP REQUIRED "
                    f"shift=({self.roi_shift[0]:.1f},{self.roi_shift[1]:.1f}) "
                    f"mag={shift_mag:.1f}px consistent={consistent}/{consistent_quorum} moving={n_mov}/{n_meas}"
                )
            else:
                self.align_status_text = (
                    f"ROI SETUP REQUIRED confirm consistent={consistent}/{consistent_quorum} moving={n_mov}/{n_meas}"
                )
        elif roi_corrected:
            healthcheck_reason = (
                f"auto_correct method={roi_correct_method} grid_shift=({mdx:.1f},{mdy:.1f}) mag={shift_mag:.1f}px "
                f"applied_shift=({self.roi_shift[0]:.1f},{self.roi_shift[1]:.1f}) h={h_status or '-'}"
            )
            self.align_status_text = (
                f"ROI AUTO-CORRECT[{roi_correct_method}] "
                f"shift=({self.roi_shift[0]:.1f},{self.roi_shift[1]:.1f}) mag={shift_mag:.1f}px"
            )
        elif decision_pending:
            self.align_status_text = (
                f"ROI SETUP PENDING confirm abnormal={abnormal_count}/{abnormal_required} "
                f"observed={observed_decision} moving={n_mov}/{n_meas} "
                f"consistent={consistent}/{consistent_quorum}"
            )
        else:
            self.align_status_text = (
                f"GRID {decision_name} suspect={suspect_count}/{confirm_required} disturbed={disturbed_count}/{disturbed_required} "
                f"abnormal={abnormal_count}/{abnormal_required} "
                f"moving={n_mov}/{n_meas} consistent={consistent}/{consistent_quorum}"
            )

        # [화각 변경 → 관제센터 빨간불] confirm(suspect>=3) 또는 pending(자동보정 후 관제 확인 대기)
        #   상태가 유지되는 동안, 검사 주기(300초)마다 /cctv/roi/img 로
        #   isReqRoiSetup=True + 현재 스냅샷을 전송한다.
        #   roi_change_apply 자동보정이 성공해도(align_shifted=False) 관제센터가 ROI를 내려줄
        #   때까지는 계속 true를 보낸다(roi_setup_pending). 관제센터가 관리자 설정 ROI를
        #   health 응답(roiSettings)으로 내려주면 HealthCheckDaemon._apply_roi_settings_from_response
        #   → update_config → _reset_alignment_state 에서 두 플래그가 모두 풀려 전송이 멈춘다.
        if self.align_shifted or self.roi_setup_pending:
            try:
                snap_img = create_roi_snapshot(self, frame)
                if snap_img is not None:
                    _sh, _sw = snap_img.shape[:2]
                    roi_info = {
                        "roi_poly_norm": self.roi_poly_norm,
                        "roi_lines_norm": [],
                        "abnormal_drive_zones_norm": self.abnormal_drive_zones_norm,
                        "roi_change_poly_norm": []
                    }
                    IMAGE_SAVER_POOL.submit(
                        _send_roi_snapshot_task,
                        self.cam_id,
                        SYS_CFG.get("terminal_id", "99999"),
                        snap_img,
                        json.dumps(roi_info),
                        _sw, _sh,
                        True,  # is_req_roi_setup
                        "roi_check_5min"
                    )
                    logger.info(
                        f"[CAM:{self.cam_id}] ROI 재설정 요청 전송(queued) isReqRoiSetup=True "
                        f"suspect={suspect_count} disturbed={disturbed_count} abnormal={abnormal_count}"
                    )
            except Exception as e:
                logger.error(f"[CAM:{self.cam_id}] ROI 재설정 요청 전송 실패: {e}")

        shift_reason = (
            f"grid_shift=({mdx:.1f},{mdy:.1f}) "
            f"mag={shift_mag:.1f}px "
            f"applied_shift=({self.roi_shift[0]:.1f},{self.roi_shift[1]:.1f}) "
            f"method={roi_correct_method or '-'}"
        )
        if decision_pending:
            shift_reason = (
                f"awaiting_roi_setup observed={observed_decision} "
                f"abnormal={abnormal_count}/{abnormal_required} {shift_reason}"
            )
        csv_reason = healthcheck_reason
        if not csv_reason:
            csv_reason = shift_reason
        elif "applied_shift=" not in csv_reason:
            csv_reason = f"{csv_reason} {shift_reason}"

        csv_row = {
            "timestamp": ROI_ALIGN_LEARNING_STORE._now_iso(),
            "camera_key": self.camera_key,
            "decision": decision_name,
            "suspect_count": suspect_count,
            "disturbed_count": disturbed_count,
            "abnormal_count": abnormal_count,
            "cells_measurable": n_meas,
            "cells_moving": n_mov,
            "cells_consistent": consistent,
            "consistent_quorum": consistent_quorum,
            "grid_cells": "|".join(_format_grid_cell_diag(c) for c in grid.get("cells", [])),
            "grid_cells_std": "|".join(_format_grid_cell_std(c) for c in grid.get("cells", [])),
            "frame_std": round(float(grid.get("frame_std", 0.0)), 1),
            "anchor_refreshed": anchor_refreshed,
            # pending 상태를 기록: confirm 확정 순간부터 관제센터가 ROI를 내려줄 때까지 계속 True.
            # (발사 '순간'은 reason 컬럼이 채워진 행으로 구분 가능)
            "healthcheck": bool(self.roi_setup_pending),
            "reason": csv_reason,
        }
        ROI_ALIGN_LEARNING_STORE.append_csv_log(csv_row)
        self.status_history.append(self.align_status_text)
        self.last_align_time = now
        logger.info(f"[CAM:{self.cam_id}] {self.align_status_text}")
        return


    def process_frame(self):
        fr, fid, connected = self.reader.read()
        return fr, fid, connected

    def apply_face_blur(self, frame, person_boxes, return_meta=False):
        if frame is None or self.det_face is None or not person_boxes:
            return (frame, []) if return_meta else frame

        blur_img = frame.copy()
        blurred_faces = []
        try:
            face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
            face_detections = self.det_face.infer(blur_img, conf_override=face_conf)
            for detection in face_detections:
                fx1, fy1, fx2, fy2 = map(int, detection[:4])
                face_width, face_height = fx2 - fx1, fy2 - fy1
                if face_width <= 0 or face_height <= 0 or face_width > blur_img.shape[1] * 0.4:
                    continue
                face_center_x = fx1 + face_width / 2.0
                face_center_y = fy1 + face_height / 2.0
                matched_person_tid = -1
                for person in person_boxes:
                    px1, py1, px2, py2 = map(int, person[:4])
                    person_width, person_height = px2 - px1, py2 - py1
                    if (
                        px1 - person_width * 0.15 <= face_center_x <= px2 + person_width * 0.15
                        and py1 - person_height * 0.25 <= face_center_y <= py2 + person_height * 0.05
                    ):
                        matched_person_tid = int(person[4]) if len(person) > 4 else -1
                        break
                if matched_person_tid < 0:
                    continue
                roi = blur_img[fy1:fy2, fx1:fx2]
                if roi.size == 0:
                    continue
                small = cv2.resize(roi, (max(1, face_width // 15), max(1, face_height // 15)), interpolation=cv2.INTER_LINEAR)
                blur_img[fy1:fy2, fx1:fx2] = cv2.resize(small, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
                blurred_faces.append({
                    "box": [fx1, fy1, fx2, fy2],
                    "score": round(float(detection[4]), 4) if len(detection) > 4 else 0.0,
                    "class_id": int(detection[5]) if len(detection) > 5 else -1,
                    "matched_person_tid": matched_person_tid,
                })
        except Exception as exc:
            logger.error(f"모자이크 처리 실패: {exc}")
        return (blur_img, blurred_faces) if return_meta else blur_img

    def apply_plate_blur(self, frame, vehicle_boxes=None, return_meta=False):
        if frame is None or self.det_plate is None:
            return (frame, []) if return_meta else frame

        blur_img = frame.copy()
        blurred_plates = []

        try:
            plate_conf = SYS_CFG.get("model_confidences", {}).get("PLATE", 0.1)
            p_dets = self.det_plate.infer(blur_img, conf_override=plate_conf)
            h_img, w_img = blur_img.shape[:2]

            for p in p_dets:
                px1, py1, px2, py2 = map(int, p[:4])
                px1 = max(0, min(px1, w_img - 1))
                py1 = max(0, min(py1, h_img - 1))
                px2 = max(0, min(px2, w_img))
                py2 = max(0, min(py2, h_img))
                pw = px2 - px1
                ph = py2 - py1

                if pw <= 0 or ph <= 0:
                    continue
                if pw > w_img * 0.6 or ph > h_img * 0.3:
                    continue

                pcx = px1 + pw / 2.0
                pcy = py1 + ph / 2.0

                matched_vehicle_tid = -1
                if vehicle_boxes is not None and len(vehicle_boxes) > 0:
                    for v in vehicle_boxes:
                        vx1, vy1, vx2, vy2 = map(int, v[:4])
                        vw = vx2 - vx1
                        vh = vy2 - vy1
                        pad_x = vw * 0.10
                        pad_y = vh * 0.10
                        if (vx1 - pad_x) <= pcx <= (vx2 + pad_x) and (vy1 - pad_y) <= pcy <= (vy2 + pad_y):
                            matched_vehicle_tid = int(v[4]) if len(v) > 4 else -1
                            break

                roi = blur_img[py1:py2, px1:px2]
                if roi.size > 0:
                    small_w = max(1, pw // 12)
                    small_h = max(1, ph // 12)
                    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    blur_img[py1:py2, px1:px2] = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
                    blurred_plates.append({
                        "box": [px1, py1, px2, py2],
                        "score": round(float(p[4]), 4) if len(p) > 4 else 0.0,
                        "class_id": int(p[5]) if len(p) > 5 else -1,
                        "matched_vehicle_tid": matched_vehicle_tid
                    })

        except Exception as e:
            logger.error(f"번호판 모자이크 처리 실패: {e}")

        return (blur_img, blurred_plates) if return_meta else blur_img

    def apply_privacy_blur(self, frame, t_main, t_helmet=None, blur_face=True, blur_plate=True):
        privacy_meta = {
            "blur_face": bool(blur_face),
            "blur_plate": bool(blur_plate),
            "face": [],
            "plate": [],
        }
        if frame is None:
            return frame, privacy_meta

        blurred_img = frame.copy()
        person_boxes = [track for track in t_main if int(track[6]) == ID_PCTC_PERSON]
        vehicle_boxes = [track for track in t_main if int(track[6]) in PLATE_PRIVACY_CLASS_IDS]
        if blur_face:
            blurred_img, privacy_meta["face"] = self.apply_face_blur(blurred_img, person_boxes, return_meta=True)
        if blur_plate:
            blurred_img, privacy_meta["plate"] = self.apply_plate_blur(blurred_img, vehicle_boxes, return_meta=True)
        privacy_meta["applied"] = bool(privacy_meta["face"] or privacy_meta["plate"])
        return blurred_img, privacy_meta

    def _privacy_tracks_from_event_objects(self, objects):
        label_to_class = {name.lower(): class_id for class_id, name in enumerate(PCTC_CLASS_NAMES)}
        label_to_class.update({"low_body": ID_PCTC_TRUCK, "yt": ID_PCTC_YT, "yc": ID_PCTC_YC})
        allowed_classes = {ID_PCTC_PERSON, *PLATE_PRIVACY_CLASS_IDS}
        tracks = []
        for obj in objects or []:
            try:
                box = obj.get("box", [])
                if len(box) < 4:
                    continue
                try:
                    class_id = int(obj.get("class_id"))
                except Exception:
                    class_id = label_to_class.get(str(obj.get("class_name") or obj.get("label") or "").lower())
                if class_id not in allowed_classes:
                    continue
                tracks.append([
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                    int(obj.get("tid", -1)), float(obj.get("score", 0.95)), int(class_id),
                ])
            except Exception:
                continue
        return tracks

    def _serialize_detection(self, det):
        class_id = int(det[5])
        return {
            "box": [int(round(float(value))) for value in det[:4]],
            "score": round(float(det[4]), 4),
            "class_id": class_id,
            "class_name": pctc_class_name(class_id),
        }

    def _serialize_track(self, track):
        class_id = int(track[6])
        return {
            "box": [int(round(float(value))) for value in track[:4]],
            "tid": int(track[4]),
            "score": round(float(track[5]), 4),
            "class_id": class_id,
            "class_name": pctc_class_name(class_id),
        }

    def _serialize_event_objects(self, objects):
        safe_objects = []
        for obj in objects or []:
            class_id = int(obj.get("class_id", -1))
            safe_objects.append({
                "label": str(obj.get("label", "")),
                "class_name": str(obj.get("class_name") or pctc_class_name(class_id)),
                "box": [int(round(float(value))) for value in obj.get("box", [])],
                "score": round(float(obj.get("score", 0.0)), 4),
                "tid": int(obj.get("tid", -1)),
                "class_id": class_id,
            })
        return safe_objects

    def build_inference_log(self, fid, frame, d_main_res, d_helmet_res, t_main, t_helmet, alarms, new_events):
        height, width = frame.shape[:2] if frame is not None else (0, 0)
        return {
            "ts": now_kst().isoformat(),
            "fid": int(fid),
            "cam_id": int(self.cam_id),
            "ip": str(self.ip),
            "frame_shape": [int(height), int(width)],
            "inference_mode": str(self.event_inference_mode),
            "events": list(self.events),
            "roi_poly": [[int(point[0]), int(point[1])] for point in (self.roi_poly or [])],
            "roi_lines": [],
            "abnormal_drive_zones": to_json_safe(self.abnormal_drive_zones),
            "detections": {
                "main": [self._serialize_detection(det) for det in d_main_res],
                "helmet": [self._serialize_detection(det) for det in d_helmet_res],
            },
            "tracks": {
                "main": [self._serialize_track(track) for track in t_main],
                "helmet": [self._serialize_track(track) for track in t_helmet],
            },
            "alarms": {str(int(tid)): event_name for tid, event_name in (alarms or {}).items()},
            "new_events": [{
                "event_id": str(event.get("event_id", "")),
                "ts": str(event.get("ts", "")),
                "event_name": str(event.get("event_name", "")),
                "objects": self._serialize_event_objects(event.get("objects", [])),
                "privacy_blur": to_json_safe(event.get("privacy_blur", {})),
                "decision_trace": to_json_safe(event.get("decision_trace", {})),
            } for event in (new_events or [])],
        }

    def run_logic(self, frame, fid, d_main_res, d_helmet_res):
        if frame is None:
            return [], [], {}, []

        now_value = time.time()
        self.fps_queue.append(now_value)
        if len(self.fps_queue) > 1:
            elapsed = self.fps_queue[-1] - self.fps_queue[0]
            self.current_fps = len(self.fps_queue) / elapsed if elapsed > 0 else 0.0

        self._update_alignment(frame)
        main_filtered = [det for det in d_main_res if int(det[5]) in PCTC_MAIN_TRACK_CLASS_IDS]
        t_main = self.trk_main.update(main_filtered)
        t_helmet = self.trk_helmet.update(d_helmet_res)
        track_map_main = {int(track[4]): int(track[6]) for track in t_main}
        score_map_main = {int(track[4]): round(float(track[5]), 4) for track in t_main}
        current_alarms = {}
        newly_triggered_events = []

        for event_name, handler in self.handlers.items():
            kwargs = {"privacy_tracks": t_main}
            if event_name == "no_helmet":
                kwargs["helmet_tracks"] = t_helmet
            try:
                triggered_events = handler.process(t_main, track_map_main, None, frame, fid, **kwargs)
            except Exception as exc:
                logger.error(f"[CAM:{self.ip}] {event_name} 핸들러 처리 중 예외 발생: {exc}\n{traceback.format_exc()}")
                continue

            for event in triggered_events:
                tid = int(event["tid"])
                bbox = event["bbox"]
                event_frame = event.get("frame") if event.get("frame") is not None else frame
                event_config = SYS_CFG.get("event_config", {}).get(event_name, {})
                cooldown = float(event_config.get("cooldown_sec", 600))
                class_id = int(track_map_main.get(tid, -1))
                actual_score = score_map_main.get(tid, 0.95)
                objects_meta = event.get("objects") or [{
                    "label": pctc_class_name(class_id),
                    "class_name": pctc_class_name(class_id),
                    "class_id": class_id,
                    "box": [int(round(float(value))) for value in bbox],
                    "score": actual_score,
                    "tid": tid,
                }]
                privacy_reference_tracks = event.get("privacy_tracks")
                privacy_reference_fid = event.get("privacy_fid", event.get("fid", fid))
                if privacy_reference_tracks is not None and len(privacy_reference_tracks) > 0:
                    privacy_scope = "event_frame_tracks"
                else:
                    privacy_reference_tracks = t_main
                    privacy_scope = "current_tracks"

                decision_trace = to_json_safe(event.get("decision_trace", {
                    "detector": handler.__class__.__name__,
                    "reason": "event_triggered_without_detail",
                }))

                if now_value - self.last_evt_t.get(event_name, 0.0) >= cooldown:
                    event_ts_dt = now_kst()
                    event_ts = event_ts_dt.isoformat()
                    event_fid = int(event.get("fid", fid))
                    event_id = make_event_id(self.cam_id, self.ip, event_name, tid, event_fid, event_ts_dt)
                    objects_log = " | ".join(
                        f"{obj.get('class_name') or obj.get('label')}({float(obj.get('score', 0.0)):.2f}): {obj.get('box')}"
                        for obj in objects_meta
                    )
                    logger.warning(
                        f"[EVENT TRIGGERED] event_id={event_id} CAM:{self.cam_id}({self.ip}) | Event:{event_name} | "
                        f"TermID:{SYS_CFG.get('terminal_id', '99999')} | TID:{tid} | FID:{event_fid} | "
                        f"FPS:{self.current_fps:.1f} | Reason:{decision_trace.get('reason', '-')} | Objects -> {objects_log}"
                    )

                    blur_face = bool(event_config.get("blur_face", True))
                    blur_plate = bool(event_config.get("blur_plate", True))
                    saved_image, privacy_meta = self.apply_privacy_blur(
                        event_frame,
                        privacy_reference_tracks,
                        t_helmet=t_helmet,
                        blur_face=blur_face,
                        blur_plate=blur_plate,
                    )
                    privacy_meta.update({
                        "scope": "event_snapshot",
                        "reference_tracks": privacy_scope,
                        "reference_fid": int(privacy_reference_fid) if privacy_reference_fid is not None else None,
                    })

                    trajectories = {}
                    for obj in objects_meta:
                        object_tid = obj.get("tid")
                        if object_tid in self.trk_main.tracks:
                            trajectories[object_tid] = list(self.trk_main.tracks[object_tid]["history"])
                        elif object_tid in self.trk_helmet.tracks:
                            trajectories[object_tid] = list(self.trk_helmet.tracks[object_tid]["history"])

                    event_meta = {
                        "event_id": event_id,
                        "ts": event_ts,
                        "event_name": event_name,
                        "terminal_id": str(SYS_CFG.get("terminal_id", "99999")),
                        "cctv_id": int(self.cam_id),
                        "ip": str(self.ip),
                        "tid": tid,
                        "bbox": int_box(bbox),
                        "fid": event_fid,
                        "objects": self._serialize_event_objects(objects_meta),
                        "trajectories": to_json_safe(trajectories),
                        "privacy_blur": to_json_safe(privacy_meta),
                        "decision_trace": decision_trace,
                    }
                    evidence_paths = save_event_image_with_mark(
                        frame=saved_image,
                        ip=self.ip,
                        event_type=event_name,
                        bbox=bbox,
                        tid=tid,
                        terminal_id=SYS_CFG.get("terminal_id", "99999"),
                        cctv_id=self.cam_id,
                        objects_meta=objects_meta,
                        trajectories=trajectories,
                        event_id=event_id,
                        event_ts=event_ts,
                    )
                    if evidence_paths:
                        event_meta.update(evidence_paths)
                    self.recorder.trigger(
                        event_name,
                        objects_meta=objects_meta,
                        event_meta=event_meta,
                        current_fps=SYS_CFG.get("video_decode", {}).get("fps_limit", 10.0),
                    )
                    self.last_evt_t[event_name] = now_value
                    newly_triggered_events.append({
                        "event_id": event_id,
                        "ts": event_ts,
                        "event_name": event_name,
                        "objects": objects_meta,
                        "privacy_blur": privacy_meta,
                        "decision_trace": decision_trace,
                    })
                else:
                    cooldown_remaining = max(0.0, cooldown - (now_value - self.last_evt_t.get(event_name, 0.0)))
                    logger.debug(
                        f"[EVENT SUPPRESSED] cam={self.cam_id} ip={self.ip} event={event_name} tid={tid} "
                        f"fid={int(event.get('fid', fid))} cooldown_remaining={cooldown_remaining:.1f}s"
                    )
                current_alarms[tid] = event_name

        alarm_duration = float(SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0))
        for tid, event_name in current_alarms.items():
            self.visual_alarms[tid] = {"evt": event_name, "expire": now_value + alarm_duration}
        for tid in list(self.visual_alarms):
            if now_value > self.visual_alarms[tid]["expire"]:
                self.visual_alarms.pop(tid, None)

        active_visual_alarms = {tid: info["evt"] for tid, info in self.visual_alarms.items()}
        return t_main, t_helmet, active_visual_alarms, newly_triggered_events

    def draw(self, frame, t_main, t_helmet, alarms, connected=True):
        if frame is None or not connected:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(blank, self.ip, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return blank

        height, width = frame.shape[:2]
        if alarms:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 20)
        if len(self.roi_poly) > 2:
            cv2.polylines(frame, [np.array(self.roi_poly, np.int32)], True, (0, 255, 255), 2)
        draw_abnormal_drive_zones(frame, self.abnormal_drive_zones, thickness=2)

        visible_class_ids = get_display_pctc_class_ids(self.events)
        for track in t_main:
            tid = int(track[4])
            class_id = int(track[6])
            is_alarmed = tid in alarms
            if not is_alarmed and class_id not in visible_class_ids:
                continue
            color = (0, 0, 255) if is_alarmed else (0, 255, 0)
            if tid in self.trk_main.tracks:
                history = list(self.trk_main.tracks[tid]["history"])
                if len(history) > 1:
                    cv2.polylines(frame, [np.array(history, np.int32)], False, color, 1, cv2.LINE_AA)
            label = f"{pctc_class_name(class_id)} [{tid}]"
            if is_alarmed:
                label = f"ALARM: {label}"
            cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 1)
            cv2.putText(frame, label, (int(track[0]), max(15, int(track[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        if "no_helmet" in self.events:
            for track in t_helmet:
                tid = int(track[4])
                class_id = int(track[6])
                if class_id == ID_H_HELMET:
                    color, label, thickness = (0, 255, 0), f"Helmet [{tid}]", 1
                elif class_id == ID_H_HEAD:
                    color = (0, 0, 255)
                    label = f"Head [{tid}]"
                    thickness = 3 if "no_helmet" in alarms.values() else 1
                else:
                    continue
                cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, thickness)
                cv2.putText(frame, label, (int(track[0]), max(15, int(track[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.putText(frame, f"CAM {self.cam_id}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        fps_color = (0, 0, 0) if self.current_fps >= 10.0 else (0, 0, 255)
        cv2.putText(frame, f"AI FPS: {self.current_fps:.1f}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, fps_color, 1)

        active_events = set(alarms.values())
        y_pos = 15
        for event_name in self.events:
            if event_name in (ROI_CHANGE_EVENT, ROI_CHANGE_APPLY_EVENT):
                continue
            display_name = EVENT_REGISTRY[event_name].gui_name if event_name in EVENT_REGISTRY else event_name.upper()
            color = (0, 0, 255) if event_name in active_events else (0, 255, 0)
            prefix = "[!] " if event_name in active_events else " -  "
            cv2.putText(frame, f"{prefix}{display_name}", (width - 175, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_pos += 20
        return frame
# ==========================================
# [11]  Platform 송수신 모듈
# ==========================================
def get_system_temperature():
    """OS 환경(Linux, Edge Device 등)에 맞게 시스템 온도를 안전하게 수집합니다."""
    try:
        # 1차 시도: psutil을 통한 센서 온도 읽기
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        return float(entry.current)

        # 2차 시도: 리눅스/엣지 단말(Jetson, Raspberry Pi 등)의 하드웨어 파일 직접 참조
        temp_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                return float(f.read().strip()) / 1000.0
    except Exception as e:
        logger.debug(f"온도 센서 읽기 실패 (해당 OS 미지원): {e}")

    return 0.0 # 센서가 없는 PC 환경 등의 폴백(Fallback)

def get_npu_temperature():
    """dxrt-cli 명령어를 통해 DeepX NPU의 최대 온도를 파싱하여 반환합니다."""
    if not HAS_DX_ENGINE:
        return 0.0
    try:
        # 터미널 명령어 실행 (응답 지연 방지를 위해 2초 타임아웃)
        output = subprocess.check_output(["dxrt-cli", "-s"], stderr=subprocess.DEVNULL, text=True, timeout=2)
        # 정규식으로 "temperature XX'C" 패턴을 모두 검색
        temps = re.findall(r"temperature\s+(\d+)'C", output)
        if temps:
            return float(max(int(t) for t in temps))
    except Exception:
        pass
    return 0.0

class HealthCheckDaemon:
    def __init__(self, terminal_id, version="v1.1.0", interval_sec=60, cams=None, config_file=CONFIG_CAMERAS_FILE):
        self.terminal_id = terminal_id
        self.version = version
        self.interval = interval_sec
        self.running = True
        self.url = "1https://tmlsafety.hudaters.net/receiver/api/v1/cctv/health"
        self.cams = list(cams or [])
        self.config_file = config_file
        self._config_lock = threading.Lock()

        self._roi_setup_required_pending = False
        self._roi_setup_required_reason = ""
        self._roi_setup_required_true_sent_count = 0
        self._roi_setup_required_lock = threading.Lock()
        self._roi_snapshot_refresh_cctv_ids = set()
        self._roi_snapshot_refresh_lock = threading.Lock()
        self._consecutive_failures = 0

        # 데몬 스레드로 실행하여 메인 프로세스 종료 시 강제 종료되도록 허용
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"[Health Check] 백그라운드 헬스 체크 데몬 시작 (주기: {self.interval}초)")




    def request_roi_setup_required(self, reason=""):
        with self._roi_setup_required_lock:
            if self._roi_setup_required_pending:
                return False

            self._roi_setup_required_pending = True
            self._roi_setup_required_true_sent_count = 0
            self._roi_setup_required_reason = str(reason or "")

        logger.warning(
            f"[Health Check] ROI setup required flagged | "
            f"terminalId={self.terminal_id} reason={reason or 'unspecified'}"
        )
        return True

    def _should_send_roi_setup_required(self):
        if not bool(SYS_CFG.get("ROI_SETUP_REQUIRED_API_ENABLED", False)):
            return False
        with self._roi_setup_required_lock:
            return bool(self._roi_setup_required_pending)

    def _mark_roi_setup_required_sent(self):
        with self._roi_setup_required_lock:
            if not self._roi_setup_required_pending:
                return

            self._roi_setup_required_true_sent_count += 1
            sent_count = self._roi_setup_required_true_sent_count
            reason = self._roi_setup_required_reason

        logger.warning(
            f"[Health Check] ROI setup required true sent (persistent) | "
            f"terminalId={self.terminal_id} count={sent_count} "
            f"reason={reason or 'unspecified'}"
        )

    def clear_roi_setup_required(self, reason=""):
        with self._roi_setup_required_lock:
            if not self._roi_setup_required_pending:
                return False

            self._roi_setup_required_pending = False
            self._roi_setup_required_true_sent_count = 0
            old_reason = self._roi_setup_required_reason
            self._roi_setup_required_reason = ""

        logger.info(
            f"[Health Check] ROI setup required cleared | "
            f"terminalId={self.terminal_id} reason={reason or old_reason or 'unspecified'}"
        )
        return True

    def request_roi_snapshot_refresh(self, cctv_ids=None, reason=""):
        ids = {
            str(cctv_id).strip()
            for cctv_id in (cctv_ids or [])
            if str(cctv_id or "").strip()
        }
        if not ids:
            return False

        with self._roi_snapshot_refresh_lock:
            before = set(self._roi_snapshot_refresh_cctv_ids)
            self._roi_snapshot_refresh_cctv_ids.update(ids)
            pending = set(self._roi_snapshot_refresh_cctv_ids)

        logger.info(
            f"[Health Check] ROI snapshot refresh requested | "
            f"terminalId={self.terminal_id} cctvIds={','.join(sorted(ids))} "
            f"pending={','.join(sorted(pending))} reason={reason or 'unspecified'}"
        )
        return pending != before

    def get_roi_snapshot_refresh_cctv_ids(self):
        with self._roi_snapshot_refresh_lock:
            return set(self._roi_snapshot_refresh_cctv_ids)

    def clear_roi_snapshot_refresh(self, cctv_ids=None, reason=""):
        ids = {
            str(cctv_id).strip()
            for cctv_id in (cctv_ids or [])
            if str(cctv_id or "").strip()
        }
        if not ids:
            return False

        with self._roi_snapshot_refresh_lock:
            before = set(self._roi_snapshot_refresh_cctv_ids)
            self._roi_snapshot_refresh_cctv_ids.difference_update(ids)
            cleared = before - set(self._roi_snapshot_refresh_cctv_ids)
            pending = set(self._roi_snapshot_refresh_cctv_ids)

        if cleared:
            logger.info(
                f"[Health Check] ROI snapshot refresh cleared | "
                f"terminalId={self.terminal_id} cctvIds={','.join(sorted(cleared))} "
                f"pending={','.join(sorted(pending)) or '-'} reason={reason or 'unspecified'}"
            )
            return True
        return False

    @staticmethod
    def _decode_jsonish(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            return json.loads(text)
        return value

    @staticmethod
    def _coerce_roi_norm_points(value):
        value = HealthCheckDaemon._decode_jsonish(value)
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"ROI norm value must be a list, got {type(value).__name__}")

        points = []
        for point in value:
            if isinstance(point, dict):
                x = point.get("x")
                y = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = point[0], point[1]
            else:
                raise ValueError(f"Invalid ROI point: {point!r}")
            points.append([round(float(x), 6), round(float(y), 6)])
        return points

    @classmethod
    def _coerce_abnormal_drive_zones_norm(cls, value):
        value = cls._decode_jsonish(value)
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"abnormal_drive_zones_norm must be a list, got {type(value).__name__}")
        zones = []
        for zone in value:
            if not isinstance(zone, dict):
                raise ValueError(f"Invalid abnormal_drive zone: {zone!r}")
            polygon = cls._coerce_roi_norm_points(zone.get("roi_poly_norm"))
            direction = cls._coerce_roi_norm_points(zone.get("direction_points_norm"))
            if len(polygon) < 3 or len(direction) != 2:
                raise ValueError("abnormal_drive zone requires polygon >=3 and direction points ==2")
            zones.append({"roi_poly_norm": polygon, "direction_points_norm": direction})
        return zones

    @classmethod
    def _extract_roi_norm_updates(cls, roi_json):
        payload = cls._decode_jsonish(roi_json)
        if not isinstance(payload, dict):
            return None
        candidates = [payload]
        for nested_key in ("roiInfo", "roi_info"):
            if nested_key in payload:
                nested = cls._decode_jsonish(payload.get(nested_key))
                if isinstance(nested, dict):
                    candidates.append(nested)
        for candidate in candidates:
            updates = {}
            if "roi_poly_norm" in candidate:
                updates["roi_poly_norm"] = cls._coerce_roi_norm_points(candidate.get("roi_poly_norm"))
            if "roi_lines_norm" in candidate:
                updates["roi_lines_norm"] = []
            if "abnormal_drive_zones_norm" in candidate:
                updates["abnormal_drive_zones_norm"] = cls._coerce_abnormal_drive_zones_norm(candidate.get("abnormal_drive_zones_norm"))
            if updates:
                return updates
        return None

    def _find_camera_by_cctv_id(self, cctv_id):
        cctv_text = str(cctv_id or "").strip()
        if not cctv_text:
            return None

        for cam in self.cams:
            if cctv_text == str(getattr(cam, "cam_id", "")):
                return cam
            if cctv_text == str(getattr(cam, "ip", "")):
                return cam
            if cctv_text == str(getattr(cam, "camera_key", "")):
                return cam
        return None

    def _apply_roi_settings_from_response(self, response_payload):
        if not isinstance(response_payload, dict):
            return []

        data = response_payload.get("data")
        if not isinstance(data, dict):
            return []

        roi_settings = data.get("roiSettings")
        if not roi_settings:
            logger.debug("[Health Check] roiSettings empty; no ROI update")
            return []
        if not isinstance(roi_settings, list):
            logger.warning(f"[Health Check] roiSettings ignored because it is not a list: {type(roi_settings).__name__}")
            return []

        handled_cctv_ids = []
        changed = False
        runtime_updates = []

        with self._config_lock:
            try:
                if os.path.exists(self.config_file):
                    with open(self.config_file, "r", encoding="utf-8") as f:
                        camera_configs = json.load(f)
                else:
                    camera_configs = {}
            except Exception as e:
                logger.error(f"[Health Check] failed to load cameras config for ROI update: {e}")
                return []

            if not isinstance(camera_configs, dict):
                logger.error("[Health Check] cameras config is not an object; ROI update skipped")
                return []

            for item in roi_settings:
                if not isinstance(item, dict):
                    logger.warning(f"[Health Check] invalid roiSettings item ignored: {item!r}")
                    continue

                cam = self._find_camera_by_cctv_id(item.get("cctvId"))
                if cam is None:
                    logger.warning(f"[Health Check] ROI settings camera not found: cctvId={item.get('cctvId')!r}")
                    continue

                try:
                    roi_updates = self._extract_roi_norm_updates(item.get("roiJson"))
                except Exception as e:
                    logger.warning(
                        f"[Health Check] ROI settings parse failed: "
                        f"cctvId={item.get('cctvId')!r} error={e}"
                    )
                    continue

                if not roi_updates:
                    logger.info(
                        f"[Health Check] ROI settings has no norm keys; treated as initial setup: "
                        f"cctvId={item.get('cctvId')!r}"
                    )
                    continue

                disk_conf = camera_configs.get(cam.ip)
                new_conf = dict(getattr(cam, "conf", {}) or {})
                if isinstance(disk_conf, dict):
                    new_conf.update(disk_conf)

                item_changed = False
                for key, value in roi_updates.items():
                    key_was_present = key in new_conf
                    old_value = new_conf.get(key)
                    try:
                        old_norm = self._coerce_roi_norm_points(old_value) if key_was_present else None
                    except Exception:
                        old_norm = old_value

                    if (not key_was_present) or old_norm != value:
                        new_conf[key] = value
                        item_changed = True

                if item_changed:
                    camera_configs[cam.ip] = new_conf
                    runtime_updates.append((cam, new_conf, roi_updates))
                    handled_cctv_ids.append(str(cam.cam_id))
                    changed = True
                else:
                    logger.info(
                        f"[Health Check] ROI settings unchanged; pending kept: "
                        f"cctvId={item.get('cctvId')!r} keys={','.join(sorted(roi_updates.keys()))}"
                    )

            if changed:
                try:
                    temp_path = f"{self.config_file}.tmp"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(camera_configs, f, indent=4, ensure_ascii=False)
                    os.replace(temp_path, self.config_file)
                except Exception as e:
                    logger.error(f"[Health Check] failed to write cameras config for ROI update: {e}")
                    return []

        for cam, new_conf, roi_updates in runtime_updates:
            try:
                cam.update_config(new_conf)
                logger.info(
                    f"[Health Check] ROI settings applied: "
                    f"cctvId={cam.cam_id} camera={cam.ip} keys={','.join(sorted(roi_updates.keys()))} "
                    f"poly={len(new_conf.get('roi_poly_norm', []) or [])} "
                    f"lines={len(new_conf.get('roi_lines_norm', []) or [])}"
                )
            except Exception as e:
                logger.error(f"[Health Check] runtime ROI update failed: cctvId={cam.cam_id} error={e}")

        return handled_cctv_ids

    def _run(self):
        while self.running:
            try:
                # 1. 시스템 자원 및 하드웨어 정보 싹쓸이 수집
                cpu = psutil.cpu_percent(interval=1.0)
                mem = psutil.virtual_memory().percent
                sys_temp = get_system_temperature()
                npu_temp = get_npu_temperature()
                
                # 디스크 용량 수집
                disk_usage = shutil.disk_usage(PROJECT_ROOT)
                disk_free_gb = disk_usage.free / (1024 ** 3)
                disk_total_gb = disk_usage.total / (1024 ** 3)

                # 카메라 연결 상태 확인
                total_cams = len(self.cams)
                active_cams = 0
                zombie_cams = 0
                current_time_for_check = time.time()
                
                for cam in self.cams:
                    if getattr(cam.reader, "connected", False):
                        last_t = getattr(cam.reader, "last_t", 0.0)
                        # 마지막 프레임 수신 시간이 WATCHDOG_TIMEOUT(기본 30초)을 초과했다면 데드락(좀비)으로 간주
                        if current_time_for_check - last_t > WATCHDOG_TIMEOUT:
                            zombie_cams += 1
                        else:
                            active_cams += 1

                # ISO 8601 포맷 타임스탬프
                kst = pytz.timezone('Asia/Seoul')
                reported_at = datetime.datetime.now(kst).strftime('%Y-%m-%dT%H:%M:%S')
                is_roi_setup_required = self._should_send_roi_setup_required()

                # API 데이터 (API 서버가 NPU 온도를 모를 수 있으므로, 둘 중 가장 높은 위험 온도를 대표로 보냅니다)
                data = {
                    "terminalId": str(self.terminal_id),
                    "reportedAt": reported_at,
                    "cpuUsage": round(cpu, 1),
                    "memoryUsage": round(mem, 1),
                    "temperature": round(max(sys_temp, npu_temp), 1), 
                    "softwareVersion": self.version,
                    "isRoiSetupRequired": is_roi_setup_required
                }

                headers = {"accept": "application/json"}
                response = requests.post(self.url, headers=headers, data=data, timeout=10, verify=False)

                # 2. 결과 종합 및 로깅
                if response.status_code == 200:
                    api_status = "OK"
                    if self._consecutive_failures > 0:
                        api_status = f"RECOVERED({self._consecutive_failures})"
                    self._consecutive_failures = 0
                    
                    if is_roi_setup_required:
                        self._mark_roi_setup_required_sent()

                    try:
                        response_payload = response.json()
                        applied_roi_cctv_ids = self._apply_roi_settings_from_response(response_payload)
                        if applied_roi_cctv_ids:
                            self.clear_roi_setup_required(reason="roi_settings_applied_from_health_response")
                            self.request_roi_snapshot_refresh(
                                cctv_ids=applied_roi_cctv_ids,
                                reason="roi_settings_applied_from_health_response"
                            )
                    except Exception:
                        print("response_payload = response.json() failed, skipping ROI update")
                        pass
                else:
                    self._consecutive_failures += 1
                    api_status = f"FAIL({response.status_code})"

                # [핵심] 좀비 스레드(Deadlock) 상태를 명시적으로 모니터링 로그에 추가
                logger.info(
                    f"📊 [SYSTEM STATUS] API:{api_status} | CAM:{active_cams}/{total_cams} (Zombie:{zombie_cams}) | "
                    f"CPU:{cpu:.1f}% | MEM:{mem:.1f}% | DISK:{disk_free_gb:.1f}/{disk_total_gb:.1f}GB | "
                    f"TEMP(Sys/NPU):{sys_temp:.1f}'C/{npu_temp:.1f}'C"
                )

            except Exception as e:
                self._consecutive_failures += 1
                logger.error(f"🚨 [Health Check] 네트워크/데이터 수집 실패 (누적:{self._consecutive_failures}): {e}")

            # interval(기본 60초) 대기하되, 프로세스 종료 신호(running)를 1초마다 감시
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
                
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

HEALTH_DAEMON = None

def request_terminal_roi_setup_required(reason=""):
    if not bool(SYS_CFG.get("ROI_SETUP_REQUIRED_API_ENABLED", False)):
        logger.info(
            f"[Health Check] ROI setup required flag suppressed by config | "
            f"reason={reason or 'unspecified'}"
        )
        return False
    if HEALTH_DAEMON is None:
        logger.warning(f"[Health Check] ROI setup required could not be flagged because daemon is not ready | reason={reason or 'unspecified'}")
        return False
    return HEALTH_DAEMON.request_roi_setup_required(reason=reason)

def is_terminal_roi_setup_required_pending():
    if HEALTH_DAEMON is None:
        return False
    try:
        return HEALTH_DAEMON._should_send_roi_setup_required()
    except Exception:
        return False

def get_terminal_roi_snapshot_refresh_cctv_ids():
    if HEALTH_DAEMON is None:
        return set()
    try:
        return HEALTH_DAEMON.get_roi_snapshot_refresh_cctv_ids()
    except Exception:
        return set()

def clear_terminal_roi_snapshot_refresh(cctv_ids=None, reason=""):
    if HEALTH_DAEMON is None:
        return False
    try:
        return HEALTH_DAEMON.clear_roi_snapshot_refresh(cctv_ids=cctv_ids, reason=reason)
    except Exception:
        return False

        
def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge AI CCTV Event Detection")
    parser.add_argument("--gui", action="store_true", help="GUI 모드를 활성화하여 모니터에 영상을 렌더링합니다.")
    args = parser.parse_args()
    is_gui_mode = args.gui
    logger.info("[시스템 모드] GUI 모드" if is_gui_mode else "[시스템 모드] CLI (Headless) 모드")

    global DEBUG_MODE
    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    if not rtsp_list:
        logger.error(f"카메라 목록 파일({CAMERA_LIST_FILE})을 확인하십시오.")
        return

    config_file = CONFIG_CAMERAS_FILE
    camera_configs = {}
    debug_answer = guarded_input(">> CLI 디버그 출력을 활성화하시겠습니까? (파일 로그는 항상 상세히 기록됩니다) [y/N]: ").strip().lower()
    DEBUG_MODE = debug_answer == "y"
    logger.setLevel(logging.DEBUG)
    queue_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as config_handle:
                loaded_configs = json.load(config_handle)
            camera_configs = sanitize_camera_configs(loaded_configs)
            if camera_configs != loaded_configs:
                with open(config_file, "w", encoding="utf-8") as config_handle:
                    json.dump(camera_configs, config_handle, indent=4, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"cameras.json 로드 실패: {exc}")
            camera_configs = {}

        reset_answer = guarded_input(">> 기존 설정(cameras.json)을 무시하고 ROI 및 이벤트를 재설정하시겠습니까? [y/N]: ").strip().lower()
        if reset_answer == "y":
            camera_configs = run_wizard_batch_mode(rtsp_list, camera_configs)
            with open(config_file, "w", encoding="utf-8") as config_handle:
                json.dump(camera_configs, config_handle, indent=4, ensure_ascii=False)
    else:
        logger.warning("설정 파일(cameras.json)이 없어 터미널 마법사를 실행합니다.")
        camera_configs = run_wizard_batch_mode(rtsp_list, {})
        with open(config_file, "w", encoding="utf-8") as config_handle:
            json.dump(camera_configs, config_handle, indent=4, ensure_ascii=False)

    models_cfg = SYS_CFG.get("models", {})
    event_inference_mode = "main"
    main_model_path = resolve_model_path(models_cfg.get("MAIN", "pctc_v1.dxnn"))

    try:
        logger.info("DeepX PCTC MAIN 모델과 전용 privacy/helmet 모델을 VPU 메모리로 할당 중...")
        d_main = YoLoDeepX(
            main_model_path,
            output_format=get_main_model_output_format(main_model_path),
            pool_size=get_model_engine_pool_size("MAIN", default=2),
            model_key="MAIN",
            class_count=len(PCTC_CLASS_NAMES),
            ppu_box_format="auto",
            output_hint="ppu",
            input_shape=get_model_input_shape("MAIN", (640, 384)),
        )
        d_helmet = YoLoDeepX(
            resolve_model_path(models_cfg.get("HELMET", "helmet_260622.dxnn")),
            output_format=get_model_output_format("HELMET"),
            pool_size=get_model_engine_pool_size("HELMET", default=1),
            model_key="HELMET",
            class_count=2,
            input_shape=get_model_input_shape("HELMET", (640, 640)),
        )
        face_fmt = get_model_output_format("FACE")
        d_face = YoLoDeepX(
            resolve_model_path(models_cfg.get("FACE", "yolov8m-face_ppu.dxnn")),
            output_format=face_fmt,
            pool_size=get_model_engine_pool_size("FACE", default=1),
            model_key="FACE",
            class_count=1,
        )
        d_plate = YoLoDeepX(
            resolve_model_path(models_cfg.get("PLATE", "license_plate_detector_v2.dxnn")),
            output_format=get_model_output_format("PLATE"),
            pool_size=get_model_engine_pool_size("PLATE", default=1),
            model_key="PLATE",
            class_count=1,
        )
    except Exception as exc:
        logger.error(f"모델 로드 실패. 경로를 확인하십시오: {exc}")
        return

    cams = []
    for index, rtsp in enumerate(rtsp_list):
        ip = extract_ip(rtsp)
        conf = sanitize_camera_config(camera_configs.get(ip, {}))
        if not conf.get("events"):
            continue
        conf["url"] = rtsp
        camera_configs[ip] = conf
        cams.append(Camera(
            ip,
            conf,
            d_main,
            d_helmet,
            d_face,
            d_plate,
            cam_id=index + 1,
            event_inference_mode=event_inference_mode,
        ))
        logger.info(
            f"[CAMERA LOADED] cam={index + 1} ip={ip} events={','.join(conf.get('events', [])) or '-'} "
            f"roi_poly_points={len(conf.get('roi_poly_norm', []) or [])} "
            f"abnormal_zones={len(conf.get('abnormal_drive_zones_norm', []) or [])}"
        )

    if not cams:
        logger.error("[SYSTEM STARTUP] no active cameras loaded; check cameras.csv and cameras.json events.")
        return

    with open(config_file, "w", encoding="utf-8") as config_handle:
        json.dump(camera_configs, config_handle, indent=4, ensure_ascii=False)

    performance_cfg = SYS_CFG.get("system_performance", {})
    target_fps = float(performance_cfg.get("target_fps", 10.0))
    dynamic_cpu_adjust = bool(performance_cfg.get("dynamic_cpu_adjust_enabled", False))
    system_target_fps = target_fps
    confidence_cfg = SYS_CFG.get("model_confidences", {})
    main_conf = min(1.0, max(0.0, float(confidence_cfg.get("MAIN", 0.35))))
    helmet_conf = min(1.0, max(0.0, float(confidence_cfg.get("HELMET", 0.85))))
    person_conf = min(1.0, max(0.0, float(confidence_cfg.get("PERSON", 0.30))))
    helmet_person_conf = min(1.0, max(0.0, float(confidence_cfg.get("HELMET_PERSON", person_conf))))
    runtime_cfg = SYS_CFG.get("inference_runtime", {})
    display_all_pctc_objects = bool(runtime_cfg.get("display_all_pctc_objects", True))
    try:
        max_detection_area_ratio = min(1.0, max(0.01, float(runtime_cfg.get("max_detection_area_ratio", 0.95))))
    except Exception:
        max_detection_area_ratio = 0.95
    try:
        filter_diagnostic_interval_sec = max(1.0, float(runtime_cfg.get("filter_diagnostic_interval_sec", 10.0)))
    except Exception:
        filter_diagnostic_interval_sec = 10.0
    helmet_person_assist_enabled = bool(runtime_cfg.get("helmet_person_assist_enabled", True))
    try:
        helmet_person_class_id = int(runtime_cfg.get("helmet_person_class_id", ID_H_PERSON))
    except Exception:
        helmet_person_class_id = ID_H_PERSON
    try:
        person_merge_iou_threshold = min(1.0, max(0.0, float(runtime_cfg.get("person_merge_iou_threshold", 0.35))))
    except Exception:
        person_merge_iou_threshold = 0.35
    logger.info(
        f"[PCTC RUNTIME] model={os.path.basename(main_model_path)} "
        f"output_setting={get_main_model_output_format(main_model_path)} "
        f"main_conf={main_conf:.3f} person_conf={person_conf:.3f} "
        f"helmet_person_conf={helmet_person_conf:.3f} helmet_person_class_id={helmet_person_class_id} "
        f"person_merge_iou={person_merge_iou_threshold:.2f} helmet_person_assist={helmet_person_assist_enabled} "
        f"display_all={display_all_pctc_objects} max_area_ratio={max_detection_area_ratio:.3f}"
    )

    terminal_id = SYS_CFG.get("terminal_id", "99999")
    log_disk_health([("event_root", EVENT_ROOT_DIR), ("log_dir", LOG_DIR)])
    global HEALTH_DAEMON
    health_daemon = HealthCheckDaemon(
        terminal_id=terminal_id,
        version="v1.3.0-pctc-terminal",
        interval_sec=60,
        cams=cams,
        config_file=config_file,
    )
    HEALTH_DAEMON = health_daemon

    last_config_mtime = os.path.getmtime(config_file) if os.path.exists(config_file) else 0.0
    ram_disk_dir = "/dev/shm/cctv_frames"
    try:
        os.makedirs(ram_disk_dir, exist_ok=True)
    except Exception:
        ram_disk_dir = "./web_frames"

    output_retention_days = float(SYS_CFG.get("OUTPUT_RETENTION_DAYS", 14))
    output_cleanup_interval_sec = float(SYS_CFG.get("OUTPUT_CLEANUP_INTERVAL_SEC", 86400))
    last_output_cleanup_time = time.time()
    run_output_retention_cleanup(output_retention_days)

    def run_camera_inference(cam, frame):
        active_events = [event_name for event_name in cam.events if event_name in EVENT_REGISTRY]
        d_main_res = np.empty((0, 6))
        d_helmet_res = np.empty((0, 6))
        if not active_events:
            return d_main_res, d_helmet_res

        display_class_ids = get_display_pctc_class_ids(active_events)
        height, width = frame.shape[:2]
        max_area_threshold = height * width * max_detection_area_ratio
        inference_threshold = min(main_conf, person_conf) if ID_PCTC_PERSON in display_class_ids else main_conf
        raw_detections = cam.det_main.infer(frame, conf_override=inference_threshold)
        d_main_res = split_unified_event_detections(
            raw_detections,
            main_conf=main_conf,
            person_conf=person_conf,
            max_area_threshold=max_area_threshold,
            class_ids=display_class_ids,
        )

        raw_count = int(len(raw_detections)) if raw_detections is not None else 0
        kept_count = int(len(d_main_res))
        if kept_count and not getattr(cam, "_pctc_first_detection_logged", False):
            cam._pctc_first_detection_logged = True
            class_counts = defaultdict(int)
            for det in d_main_res:
                class_counts[pctc_class_name(int(det[5]))] += 1
            logger.info(
                f"[PCTC DETECTION] cam={cam.cam_id} ip={cam.ip} "
                f"raw={raw_count} kept={kept_count} classes={dict(class_counts)}"
            )
        elif raw_count and not kept_count:
            now_diag = time.monotonic()
            last_diag = float(getattr(cam, "_last_pctc_filter_log_at", 0.0) or 0.0)
            if now_diag - last_diag >= filter_diagnostic_interval_sec:
                cam._last_pctc_filter_log_at = now_diag
                class_counts = defaultdict(int)
                score_max = 0.0
                area_ratio_max = 0.0
                for det in raw_detections:
                    if len(det) < 6:
                        continue
                    class_counts[int(det[5])] += 1
                    score_max = max(score_max, float(det[4]))
                    det_area = max(0.0, float(det[2]) - float(det[0])) * max(0.0, float(det[3]) - float(det[1]))
                    area_ratio_max = max(area_ratio_max, det_area / max(1.0, float(height * width)))
                logger.warning(
                    f"[PCTC FILTER] cam={cam.cam_id} ip={cam.ip} decoder_raw={raw_count} kept=0 "
                    f"class_ids={dict(class_counts)} max_score={score_max:.3f} "
                    f"max_area_ratio_seen={area_ratio_max:.3f} allowed={sorted(display_class_ids)} "
                    f"main_conf={main_conf:.3f} person_conf={person_conf:.3f} "
                    f"max_area_ratio_allowed={max_detection_area_ratio:.3f}"
                )

        # HELMET model is also a secondary person detector (0=helmet, 1=head, 2=person).
        # Run it whenever the current terminal view needs person, even when PCTC detected none.
        person_assist_required = helmet_person_assist_enabled and ID_PCTC_PERSON in display_class_ids
        helmet_safety_required = "no_helmet" in cam.events
        if (person_assist_required or helmet_safety_required) and cam.det_helmet is not None:
            helmet_infer_conf = min(helmet_conf, helmet_person_conf) if person_assist_required else helmet_conf
            raw_helmet = cam.det_helmet.infer(frame, conf_override=helmet_infer_conf)
            helmet_safety_res, helmet_person_res = split_helmet_model_detections(
                raw_helmet,
                helmet_conf=helmet_conf,
                person_conf=helmet_person_conf,
                max_area_threshold=max_area_threshold,
                person_class_id=helmet_person_class_id,
            )
            d_helmet_res = helmet_safety_res if helmet_safety_required else np.empty((0, 6))
            if person_assist_required:
                d_main_res, fusion_stats = merge_person_detections(
                    d_main_res,
                    helmet_person_res,
                    iou_threshold=person_merge_iou_threshold,
                )
                if (fusion_stats["helmet_person"] or fusion_stats["fused_pairs"]) and not getattr(cam, "_helmet_person_fusion_logged", False):
                    cam._helmet_person_fusion_logged = True
                    logger.info(
                        f"[PERSON FUSION] cam={cam.cam_id} ip={cam.ip} "
                        f"main_person={fusion_stats['main_person']} helmet_person={fusion_stats['helmet_person']} "
                        f"fused_pairs={fusion_stats['fused_pairs']} "
                        f"unmatched_main={fusion_stats['unmatched_main']} "
                        f"unmatched_helmet={fusion_stats['unmatched_helmet']} "
                        f"iou_threshold={person_merge_iou_threshold:.2f}"
                    )

        return d_main_res, d_helmet_res

    system_runtime_state = {"target_fps": system_target_fps}

    class LatestItemBuffer:
        def __init__(self):
            self.item = None
            self.has_new = False
            self.lock = threading.Lock()

        def put(self, item):
            with self.lock:
                self.item = item
                self.has_new = True

        def get(self):
            with self.lock:
                if not self.has_new:
                    return None
                self.has_new = False
                return self.item

    class CameraWorker(threading.Thread):
        def __init__(self, cam):
            super().__init__(daemon=True)
            self.cam = cam
            self.frame_buffer = LatestItemBuffer()
            self.result_buffer = LatestItemBuffer()
            self.running = True
            self.last_inference_time = 0.0

        def run(self):
            while self.running:
                item = self.frame_buffer.get()
                if item is None:
                    time.sleep(0.005)
                    continue
                now_value = time.time()
                current_target = max(1.0, float(system_runtime_state.get("target_fps", 10.0)))
                if now_value - self.last_inference_time < 1.0 / current_target:
                    continue
                self.last_inference_time = now_value
                frame, fid, connected = item
                if not connected or frame is None or not self.cam.events:
                    self.result_buffer.put((frame, fid, connected, [], [], {}, [], None))
                    continue
                try:
                    d_main_res, d_helmet_res = run_camera_inference(self.cam, frame)
                    t_main, t_helmet, alarms, new_events = self.cam.run_logic(frame, fid, d_main_res, d_helmet_res)
                    infer_meta = self.cam.build_inference_log(fid, frame, d_main_res, d_helmet_res, t_main, t_helmet, alarms, new_events)
                    self.result_buffer.put((frame, fid, connected, t_main, t_helmet, alarms, new_events, infer_meta))
                except Exception as exc:
                    logger.error(f"[Worker Error] CAM {self.cam.cam_id}: {exc}\n{traceback.format_exc()}")
                    self.result_buffer.put((frame, fid, connected, [], [], {}, [], None))

    camera_workers = []
    last_rendered_frames = {}
    for cam in cams:
        worker = CameraWorker(cam)
        worker.start()
        camera_workers.append(worker)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "WAITING...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        last_rendered_frames[cam.ip] = blank

    last_roi_snapshot_times = {cam.ip: time.time() - 3590.0 for cam in cams}
    last_worker_active_times = {cam.ip: time.time() for cam in cams}
    roi_snapshot_interval_sec = 3600.0
    last_processed_fids = {}
    loop_count = 0
    fps_calc_interval = 30
    last_fps_time = time.time()

    try:
        psutil.cpu_percent(interval=None)
        while True:
            start_time = time.time()
            if output_cleanup_interval_sec > 0 and start_time - last_output_cleanup_time >= output_cleanup_interval_sec:
                run_output_retention_cleanup(output_retention_days)
                last_output_cleanup_time = start_time

            if loop_count > 0 and loop_count % 45 == 0 and os.path.exists(config_file):
                current_mtime = os.path.getmtime(config_file)
                if current_mtime > last_config_mtime:
                    try:
                        with open(config_file, "r", encoding="utf-8") as config_handle:
                            new_configs_raw = json.load(config_handle)
                        new_configs = sanitize_camera_configs(new_configs_raw)
                        for cam in cams:
                            if cam.ip in new_configs:
                                cam.update_config(new_configs[cam.ip])
                        camera_configs = new_configs
                        if new_configs != new_configs_raw:
                            with open(config_file, "w", encoding="utf-8") as config_handle:
                                json.dump(new_configs, config_handle, indent=4, ensure_ascii=False)
                        last_config_mtime = os.path.getmtime(config_file)
                    except Exception as exc:
                        logger.warning(f"카메라 설정 핫리로드 실패: {exc}")

            loop_count += 1
            if loop_count % fps_calc_interval == 0:
                current_time = time.time()
                elapsed = current_time - last_fps_time
                cpu_usage = psutil.cpu_percent(interval=None)
                if dynamic_cpu_adjust:
                    if cpu_usage < 75:
                        target_fps = min(system_target_fps, target_fps + 1.0)
                    elif cpu_usage > 90:
                        target_fps = max(8.0, target_fps - 1.0)
                else:
                    target_fps = system_target_fps
                system_runtime_state["target_fps"] = target_fps
                last_fps_time = current_time

            if loop_count % 300 == 0:
                gc.collect()
                log_disk_health([("event_root", EVENT_ROOT_DIR), ("log_dir", LOG_DIR)])

            for worker in camera_workers:
                frame, fid, connected = worker.cam.process_frame()
                if fid != last_processed_fids.get(worker.cam.ip, -1):
                    worker.frame_buffer.put((frame, fid, connected))
                    last_processed_fids[worker.cam.ip] = fid

            final_images = []
            now_value = time.time()
            roi_snapshot_refresh_ids = get_terminal_roi_snapshot_refresh_cctv_ids()
            refreshed_ids = set()

            for worker_index, worker in enumerate(camera_workers):
                cam = worker.cam
                result = worker.result_buffer.get()
                if result is None:
                    if now_value - last_worker_active_times.get(cam.ip, now_value) > WATCHDOG_TIMEOUT:
                        logger.error(f"[WATCHDOG] CAM:{cam.cam_id}({cam.ip}) worker stalled; hot-swap restart")
                        worker.running = False
                        cam.reader.running = False
                        cam.recorder.running = False
                        worker.join(timeout=1.0)
                        conf = sanitize_camera_config(camera_configs.get(cam.ip, cam.conf))
                        new_cam = Camera(cam.ip, conf, d_main, d_helmet, d_face, d_plate, cam_id=cam.cam_id, event_inference_mode=event_inference_mode)
                        new_worker = CameraWorker(new_cam)
                        new_worker.start()
                        camera_workers[worker_index] = new_worker
                        cams[cams.index(cam)] = new_cam
                        if HEALTH_DAEMON is not None:
                            HEALTH_DAEMON.cams = cams
                        last_worker_active_times[cam.ip] = time.time()
                        last_processed_fids[cam.ip] = -1
                    if is_gui_mode:
                        final_images.append(last_rendered_frames[cam.ip])
                    continue

                last_worker_active_times[cam.ip] = now_value
                frame, fid, connected, t_main, t_helmet, alarms, new_events, infer_meta = result
                cctv_id_text = str(cam.cam_id)
                force_snapshot = cctv_id_text in roi_snapshot_refresh_ids
                periodic_snapshot = now_value - last_roi_snapshot_times.get(cam.ip, 0.0) >= roi_snapshot_interval_sec
                if connected and frame is not None and (periodic_snapshot or force_snapshot):
                    blurred_snapshot, _ = cam.apply_privacy_blur(frame.copy(), t_main, blur_face=True, blur_plate=True)
                    cam._initialize_base_roi_if_needed(blurred_snapshot)
                    snapshot_image = create_roi_snapshot(cam, blurred_snapshot)
                    if snapshot_image is not None:
                        height, width = snapshot_image.shape[:2]
                        roi_info = {
                            "roi_poly_norm": cam.roi_poly_norm,
                            "roi_lines_norm": [],
                            "abnormal_drive_zones_norm": cam.abnormal_drive_zones_norm,
                        }
                        IMAGE_SAVER_POOL.submit(
                            _send_roi_snapshot_task,
                            cam.cam_id,
                            terminal_id,
                            snapshot_image,
                            json.dumps(roi_info),
                            width,
                            height,
                            bool(cam.align_shifted or cam.roi_setup_pending),
                            "roi_refresh" if force_snapshot else "hourly",
                        )
                        last_roi_snapshot_times[cam.ip] = now_value
                        if force_snapshot:
                            refreshed_ids.add(cctv_id_text)

                if connected and frame is not None and loop_count % 100 == 0:
                    try:
                        cv2.imwrite(os.path.join(ram_disk_dir, f"{cam.ip}.jpg"), cv2.resize(frame, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    except Exception:
                        pass

                if not connected or frame is None or not cam.events:
                    if is_gui_mode:
                        display_frame = cam.draw(None, [], [], {}, False)
                        last_rendered_frames[cam.ip] = display_frame
                        final_images.append(display_frame)
                    continue

                record_frame = frame.copy()
                cv2.putText(record_frame, f"Event Time: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if len(cam.roi_poly) > 2:
                    cv2.polylines(record_frame, [np.array(cam.roi_poly, np.int32)], True, (0, 255, 255), 1)
                draw_abnormal_drive_zones(record_frame, cam.abnormal_drive_zones, thickness=1)
                visible_ids = get_display_pctc_class_ids(cam.events)
                for track in t_main:
                    tid, class_id = int(track[4]), int(track[6])
                    if class_id not in visible_ids and tid not in alarms:
                        continue
                    color = (0, 0, 255) if tid in alarms else (0, 255, 0)
                    x1, y1, x2, y2 = map(int, track[:4])
                    cv2.rectangle(record_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(record_frame, f"{pctc_class_name(class_id)} [{tid}]", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if infer_meta:
                    cam.recorder.update(record_frame, infer_meta, timestamp=now_value)

                if is_gui_mode:
                    display_frame = cam.draw(frame.copy(), t_main, t_helmet, alarms, True)
                    last_rendered_frames[cam.ip] = display_frame
                    final_images.append(display_frame)

            if refreshed_ids:
                clear_terminal_roi_snapshot_refresh(cctv_ids=refreshed_ids, reason="roi_snapshot_sent")
            if is_gui_mode:
                if final_images:
                    #h, w, c = create_mosaic_image(final_images).shape
                    #print(f"width: {w}, height: {h}")
                    cv2.imshow("Monitor", create_mosaic_image(final_images))
                if cv2.waitKey(1) == ord("q"):
                    break
            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("[종료] 사용자에 의해 시스템이 중단되었습니다.")
    except Exception as exc:
        logger.error(f"[치명적 오류] {exc}\n{traceback.format_exc()}")
    finally:
        health_daemon.stop()
        for cam in cams:
            cam.reader.running = False
            cam.recorder.running = False
        for worker in camera_workers:
            worker.running = False
        for model_name in ("d_main", "d_helmet", "d_face", "d_plate"):
            model = locals().get(model_name)
            if model is not None and hasattr(model, "release"):
                try:
                    model.release()
                except Exception:
                    pass
        if is_gui_mode:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
