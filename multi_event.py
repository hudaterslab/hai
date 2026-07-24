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

# YOLOv8 클래스 ID 정의 (신규 13개 클래스 통합 모델 기준 hanjin_cctv_v3.dxnn)
# 0: helmet (MAIN 무시, 전용 모델 사용 유지용 상수)
# 1: head (MAIN 무시, 전용 모델 사용 유지용 상수)
ID_H_HELMET = 0
ID_H_NO_HELMET = 1

ID_G_PERSON = 2
ID_G_CAR = 3          # normalvehicle
ID_PERSON_LOW = 4     # lowerbody
ID_SIGNALMAN = 5      # signalman (검은가방 포함)
ID_REFLECTIVE_VEST = 5 # 기존 코드 하위 호환성 유지용 (5번으로 통합)
ID_G_TRUCK = 6        # trunklinetruck

# 7: xband (무시)

ID_RAINCOAT = 8       # raincoat -> 신호수(5)와 동일 취급
ID_SIGNALFLAG = 9     # signalflag -> 신호수(5)와 동일 취급

# 10: box, 11: soft_package, 12: sack (무시)

TARGET_VEHICLES = [ID_G_CAR, ID_G_TRUCK]

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

def load_system_config():
    default_config = {
        "terminal_id": "99999",
        "system_performance": {
            "target_fps": 10.0,
            "dynamic_cpu_adjust_enabled": False
        },
        "logging": {
            "dir": "./logs",
            "level": "INFO",
            "file_level": "INFO",
            "console_level": "INFO",
            "debug_file_level": "DEBUG",
            "retention_days": 14,
            "event_audit_enabled": True,
            "disk_free_warn_gb": 5.0
        },
        "event_config": {
            "intrusion": {"enabled": False, "cooldown_sec": 600},
            "illegal_parking": {"enabled": False, "cooldown_sec": 600, "trigger_sec": 5.0, "move_threshold_ratio": 0.1, "blur_plate": True},
            "no_helmet": {"enabled": False, "cooldown_sec": 600, "blur_face": True, "blur_plate": True, "trigger_sec": 3.0},
            "conveyor_crossing": {
                "enabled": False, "cooldown_sec": 600, "snapshot_mode": "crossing_moment",
                "distance_ratio": 0.9, "min_crossing_angle": 20.0, "candidate_ttl_sec": 5.0,
                "blur_face": True, "blur_plate": True
            },
            "signal_vehicle": {
                "enabled": False, "cooldown_sec": 600, "motion_threshold_ratio": 0.30,
                "blur_plate": True,
                "line_truck_confirm_frames": 10,
                "line_truck_confirm_ratio": 0.7,
                "line_truck_car_veto_frames": 5,
                "line_truck_min_conf": 0.7,
                "line_truck_car_veto_iou": 0.10,
                "line_truck_car_veto_distance_ratio": 0.60,
                "state_inherit_distance_ratio": 0.5,
                "state_inherit_max_size_ratio": 2.5,
                "state_inherit_max_area_ratio": 6.0
            }
        },
        "models": {
            "MAIN_V2": "hanjin_cctv_v2.dxnn",
            "MAIN_V3": "hanjin_cctv_v3.dxnn",
            "FACE": "yolov8m-face_ppu.dxnn",
            "HELMET": "helmet_260622.dxnn",
            "PLATE": "license_plate_detector_v2.dxnn"
        },
        "model_confidences": {
            "MAIN_V2": 0.6,
            "MAIN_V3": 0.6,
            "FACE": 0.35,
            "HELMET": 0.55,
            "PERSON": 0.5,
            "SIGNALMAN": 0.5,
            "PLATE": 0.1
        },
        "model_output_formats": {
            "MAIN_V2": "ppu",
            "MAIN_V3": "ppu",
            "FACE": "auto",
            "HELMET": "auto",
            "PLATE": "yolo"
        },
        "model_engine_pool_sizes": {
            "MAIN_V2": 2,
            "MAIN_V3": 1,
            "FACE": 1,
            "HELMET": 1,
            "PLATE": 1
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
            "log_interval_sec": 10.0,
            "verbose_logs": False
        },
        "INFERENCE_MODE": "auto",
        "BATCH_SIZE": 9,
        "REC_FPS": 3,
        "LOOP_FPS": 10.0,
        "PERF_LOG_INTERVAL_SEC": 10.0,
        "REC_PRE_SEC": 10,
        "REC_POST_SEC": 10,
        "EVENT_FRAME_SAVE_DELAY_SEC": 10.0,
        "EVENT_FRAME_SAVE_MAX_COUNT": 0,
        "OUTPUT_RETENTION_DAYS": 14,
        "OUTPUT_CLEANUP_INTERVAL_SEC": 86400,
        "ROI_SETUP_REQUIRED_API_ENABLED": False,
        "INTERACTIVE_INPUT_GUARD_SEC": 0.35,
        "VISUAL_ALARM_DURATION": 5.0
    }

    if not os.path.exists(CONFIG_COMMON_FILE):
        try:
            with open(CONFIG_COMMON_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception:
            pass
        return default_config

    try:
        with open(CONFIG_COMMON_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            
        merged_config = deep_merge_dict(default_config, loaded_config)
        
        # [수정] 예전 버전 Config에 누락된 항목이 있다면 강제로 병합본을 파일에 덮어씀 (마이그레이션)
        try:
            with open(CONFIG_COMMON_FILE, 'w', encoding='utf-8') as f:
                json.dump(merged_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"설정 파일 마이그레이션 쓰기 실패: {e}")

        return merged_config
    except Exception as e:
        print(f"[Warning] 설정 파일 로드 실패. 기본값을 사용합니다: {e}")
        return default_config

SYS_CFG = load_system_config()
BATCH_SIZE = SYS_CFG.get("BATCH_SIZE", 9)
IMAGE_SAVER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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

def get_main_model_output_format(model_path):
    fmt = get_model_output_format("MAIN")
    model_name = os.path.basename(str(model_path or "")).lower()
    if model_name == "hanjin_cctv_v2.dxnn":
        if fmt != "ppu":
            logger.warning(
                f"[DeepX] MAIN model {model_name} requires PPU decode. "
                f"Ignoring configured output_format={fmt}."
            )
        return "ppu"
    return fmt

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

def split_unified_event_detections(raw_dets, events, main_conf, person_conf, helmet_conf, signalman_conf, max_area_threshold):
    # 단일 통합 모델 출력에서 이벤트별로 필요한 탐지 결과만 나눕니다.
    d_main_res_list = []
    d_helmet_res_list = []
    d_signalman_res_list = []

    for d in raw_dets:
        # 0. [오탐 방어] BBox 면적이 화면의 1/2을 초과하는 거대 객체(비/IR 노이즈 등) 차단
        obj_w = float(d[2]) - float(d[0])
        obj_h = float(d[3]) - float(d[1])
        if (obj_w * obj_h) > max_area_threshold:
            continue

        cls_id = int(d[5])
        conf = float(d[4])

        # 1. 무시(Drop) 처리할 클래스들: 0, 1 (헬멧 관련), 7, 10, 11, 12 (물류 및 기타)
        if cls_id in [0, 1, 7, 10, 11, 12]:
            continue

        # 2. 신호수 통합 처리 (5: signalman, 8: raincoat, 9: signalflag)
        if cls_id in [ID_SIGNALMAN, ID_RAINCOAT, ID_SIGNALFLAG]:
            # 트래커 혼동을 막고 단일 객체로 추적하기 위해 ID를 5(ID_SIGNALMAN)로 강제 덮어쓰기
            mod_d = list(d)
            mod_d[5] = ID_SIGNALMAN
            
            if conf >= person_conf:
                d_main_res_list.append(mod_d)
            if "signal_vehicle" in events and conf >= signalman_conf:
                d_signalman_res_list.append(mod_d)
                
        # 3. 사람 및 하반신 (2: person, 4: lowerbody)
        elif cls_id in [ID_G_PERSON, ID_PERSON_LOW]:
            if conf >= person_conf:
                d_main_res_list.append(d)
                
        # 4. 차량 (3: normalvehicle, 6: trunklinetruck)
        elif cls_id in [ID_G_CAR, ID_G_TRUCK]:
            if conf >= main_conf:
                d_main_res_list.append(d)
                
        # 5. 혹시 모를 그 외의 클래스에 대한 방어 로직
        else:
            if conf >= main_conf:
                d_main_res_list.append(d)

    return (
        detection_array(d_main_res_list),
        detection_array(d_helmet_res_list), # 헬멧은 전용 모델에서 처리하므로 빈 배열 유지
        detection_array(d_signalman_res_list),
    )

def create_roi_snapshot(cam, frame):
    """현재 카메라 프레임에 ROI와 설정된 이벤트 명만 그립니다."""
    if frame is None: return None
    img = frame.copy()

    # 1. ROI Polygon 그리기
    if cam.roi_poly and len(cam.roi_poly) > 2:
        cv2.polylines(img, [np.array(cam.roi_poly, np.int32)], True, (0, 255, 255), 2)

    # 2. ROI Line 그리기
    if cam.roi_lines:
        for i in range(0, len(cam.roi_lines), 2):
            if i + 1 < len(cam.roi_lines):
                cv2.line(img, tuple(cam.roi_lines[i]), tuple(cam.roi_lines[i+1]), (0, 0, 255), 2)

    # 3. 설정된 이벤트 명 좌측 상단에 표시
    y_pos = 30
    for evt in cam.events:
        # [수정] roi_change 이벤트는 관제 스냅샷 텍스트 렌더링에서 제외합니다.
        if evt == "roi_change" or evt == getattr(sys.modules[__name__], 'ROI_CHANGE_EVENT', 'roi_change'):
            continue

        display_name = EVENT_REGISTRY[evt].gui_name if evt in EVENT_REGISTRY else evt.upper()
        cv2.putText(img, f"Event: {display_name}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        y_pos += 30

    return img

def _send_roi_snapshot_task( cam_id, terminal_id, img, roi_info_str, w, h, is_req_roi_setup=False, send_type="hourly"):
    """관제 서버로 ROI 스냅샷을 백그라운드에서 전송합니다."""
    url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/roi/img"
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

    url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/img"
    event_type_mapping = {
        "conveyor_crossing": 1,
        "no_helmet": 2,
        "signal_vehicle": 3,
        "illegal_parking": 4,
        "intrusion": 5
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

def _draw_event_api_image(frame, event_type, bbox, tid, objects_meta=None, auth_tokens=None):
    """Create an API-only image with event boxes drawn on top."""
    api_img = frame.copy()
    target_tid = int(tid) if tid is not None else None
    drawn = False

    draw_items = objects_meta or [{'label': event_type, 'box': bbox, 'tid': tid}]
    for obj in draw_items:

        try:
            x1, y1, x2, y2 = int_box(obj.get('box', bbox))
        except Exception:
            continue

        obj_tid_raw = obj.get('tid')
        obj_tid = None
        if obj_tid_raw is not None:
            try:
                obj_tid = int(obj_tid_raw)
            except Exception:
                obj_tid = None

        # conveyor_crossing 이벤트일 때 low_body 클래스는 그리지 않음
        if event_type == "conveyor_crossing" and obj.get('label') == 'low_body':
            continue

        is_target = target_tid is not None and obj_tid == target_tid
        color = (0, 0, 255) # 고객사 요청으로 그냥 빨간색 표시
        thickness = 1 if is_target else 2
        cv2.rectangle(api_img, (x1, y1), (x2, y2), color, thickness)

        label = str(event_type)
        # 고객사 요청으로 API 전송 이미지 라벨은 이벤트 명만 표시합니다.
        text_y = max(20, y1 - 8)
        cv2.putText(api_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        drawn = True

    if not drawn:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(api_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(api_img, str(event_type), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    # 관제 서버로 전송되는 증거 이미지 우측 하단에 Signalman 상태창 강제 베이킹 (유지)
    if event_type == "signal_vehicle":
        h_frame, w_frame = api_img.shape[:2]
        token_count = max(1, len(auth_tokens) if auth_tokens else 1)
        
        # [수정] 미정의 변수 display_items 제거 및 token_count로 교체
        box_w, box_h = 340, 35 + token_count * 40
        x_start, y_start = w_frame - box_w - 20, h_frame - box_h - 20

        # [수정] fr 객체 대신 api_img 참조로 변경하여 예외 발생 차단
        roi_sig = api_img[y_start:y_start + box_h, x_start:x_start + box_w]
        black_bg2 = np.zeros_like(roi_sig)
        cv2.addWeighted(black_bg2, 0.6, roi_sig, 0.4, 0, roi_sig)
        
        cv2.putText(api_img, "Signalman Auth", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
        if not auth_tokens:
            cv2.putText(api_img, "Status: UNAUTH (ALARM)", (x_start + 10, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(api_img, "Reason: Moving without Signalman", (x_start + 10, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            for i, tkn in enumerate(auth_tokens):
                remain = tkn.get('remain', 0)
                sig_tid = tkn.get('sig_tid', 'Unknown')
                cv2.putText(api_img, f"Auth Remain: {remain:.1f}s", (x_start + 10, y_start + 45 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(api_img, f"Auth by: Signalman [{sig_tid}]", (x_start + 10, y_start + 65 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return api_img

    if not drawn:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(api_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(api_img, str(event_type), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    # 관제 서버로 전송되는 증거 이미지 우측 하단에 Signalman 상태창 강제 베이킹 (유지)
    if event_type == "signal_vehicle":
        h_frame, w_frame = api_img.shape[:2]
        token_count = max(1, len(auth_tokens) if auth_tokens else 1)
        box_w, box_h = 340, 35 + token_count * 40
        x_start, y_start = w_frame - box_w - 20, h_frame - box_h - 20

        overlay = api_img.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + box_w, y_start + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, api_img, 0.4, 0, api_img)
        cv2.putText(api_img, "Last Signalman Checked", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if not auth_tokens:
            cv2.putText(api_img, "Status: UNAUTH (ALARM)", (x_start + 10, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(api_img, "Reason: Moving without Signalman", (x_start + 10, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            for i, tkn in enumerate(auth_tokens):
                remain = tkn.get('remain', 0)
                sig_tid = tkn.get('sig_tid', 'Unknown')
                cv2.putText(api_img, f"Auth Remain: {remain:.1f}s", (x_start + 10, y_start + 45 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(api_img, f"Auth by: Signalman [{sig_tid}]", (x_start + 10, y_start + 65 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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

def save_event_image_with_mark(frame, ip, event_type, bbox, tid, terminal_id="99999", cctv_id=1, objects_meta=None, trajectories=None, auth_tokens=None, event_id=None, event_ts=None):
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
        
        # [수정] auth_tokens 데이터를 _draw_event_api_image 로 전달
        api_img = _draw_event_api_image(img, event_type, [x1, y1, x2, y2], tid, objects_meta, auth_tokens)

        h, w = frame.shape[:2]

        if objects_meta:
            ai_detected_bboxes = []
            for o in objects_meta:
                if event_type == "conveyor_crossing" and str(o.get('label', '')).lower() == 'low_body':
                    continue
                item = {
                    "box": [int(b) for b in o['box']],
                    "label": event_type,
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
    def __init__(self, engine_path, output_format="auto", pool_size=1):
        if not HAS_DX_ENGINE:
            raise RuntimeError("dx_engine is not installed; YoLoDeepX can only run on a DeepX/NPU runtime.")

        self.engine_path = engine_path
        self.requested_output_format = str(output_format or "auto").strip().lower()
        self.output_format = self._normalize_configured_output_format(output_format) or "yolo"
        self.pool_size = max(1, int(pool_size or 1))
        self.engine_pool = queue.Queue(maxsize=self.pool_size)
        self.engines_ref = []
        self.input_height = 640
        self.input_width = 640
        self.input_layout = "hwc"
        self.input_has_batch = False

        try:
            io = InferenceOption()
            for _ in range(self.pool_size):
                engine = InferenceEngine(self.engine_path, io)
                self.engine_pool.put(engine)
                self.engines_ref.append(engine)

            self._load_input_shape(self.engines_ref[0])
            self.output_format = self._resolve_output_format(output_format, self.engines_ref[0])
            logger.info(
                f"[DeepX] 모델 로드 성공: {os.path.basename(self.engine_path)} "
                f"(output={self.output_format}, pool={self.pool_size}, "
                f"input={self.input_width}x{self.input_height}, layout={self.input_layout})"
            )
        except Exception as e:
            logger.error(f"[DeepX Load Fail] 엔진 초기화 실패 ({engine_path}): {e}")
            raise e

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
        if fmt in ["", "auto"]:
            return None
        if fmt in ["ppu", "deepx_ppu", "yolov8_ppu"]:
            return "ppu"
        if fmt in ["yolo_xyxy", "xyxy"]:
            return "yolo_xyxy"
        if fmt in ["yolo_tlwh", "tlwh"]:
            return "yolo_tlwh"
        if fmt in ["yolo", "yolov8", "raw", "standard", "raw_yolo"]:
            return "yolo"
        logger.warning(f"[DeepX] 알 수 없는 모델 출력 포맷({output_format})입니다. yolo 후처리로 동작합니다.")
        return "yolo"

    def postprocess_xyxy(self, output_tensor, conf_thres=0.40, iou_thres=0.45):
        """출력이 [x1, y1, x2, y2, score, class_id...] 형태일 때의 후처리"""
        try:
            pred = np.array(output_tensor[0])
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
                pred = pred.transpose((0, 2, 1))
            if pred.ndim == 3:
                pred = pred[0]

            class_scores = pred[:, 4:]
            if class_scores.shape[1] == 1:
                scores = class_scores[:, 0]
                class_ids = np.zeros(scores.shape, dtype=np.int32)
            else:
                scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)

            mask = scores > conf_thres
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if len(pred) == 0: return []

            boxes_xywh = np.zeros((len(pred), 4), dtype=np.float32)
            boxes_xywh[:, 0] = pred[:, 0]               
            boxes_xywh[:, 1] = pred[:, 1]               
            boxes_xywh[:, 2] = pred[:, 2] - pred[:, 0]  
            boxes_xywh[:, 3] = pred[:, 3] - pred[:, 1]  

            max_wh = 7680
            class_offset = class_ids * max_wh
            boxes_shifted = boxes_xywh.copy()
            boxes_shifted[:, 0] += class_offset
            boxes_shifted[:, 1] += class_offset

            indices = cv2.dnn.NMSBoxes(boxes_shifted.tolist(), scores.tolist(), conf_thres, iou_thres)

            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x_min, y_min, w, h = boxes_xywh[i]
                    results.append([[x_min, y_min, x_min + w, y_min + h], scores[i], class_ids[i]])
            return results
        except Exception as e:
            logger.error(f"NPU XYXY Postprocess Error ({os.path.basename(self.engine_path)}): {e}")
            return []

    def postprocess_tlwh(self, output_tensor, conf_thres=0.40, iou_thres=0.45):
        """출력이 [TopLeft_X, TopLeft_Y, Width, Height, score, class_id...] 형태일 때의 후처리"""
        try:
            pred = np.array(output_tensor[0])
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
                pred = pred.transpose((0, 2, 1))
            if pred.ndim == 3:
                pred = pred[0]

            class_scores = pred[:, 4:]
            if class_scores.shape[1] == 1:
                scores = class_scores[:, 0]
                class_ids = np.zeros(scores.shape, dtype=np.int32)
            else:
                scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)

            mask = scores > conf_thres
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if len(pred) == 0: return []

            # 이미 좌상단 좌표이므로 크기의 절반을 빼는 연산 생략
            boxes_xywh = pred[:, :4].copy()

            max_wh = 7680
            class_offset = class_ids * max_wh
            boxes_shifted = boxes_xywh.copy()
            boxes_shifted[:, 0] += class_offset
            boxes_shifted[:, 1] += class_offset

            indices = cv2.dnn.NMSBoxes(boxes_shifted.tolist(), scores.tolist(), conf_thres, iou_thres)

            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x_min, y_min, w, h = boxes_xywh[i]
                    results.append([[x_min, y_min, x_min + w, y_min + h], scores[i], class_ids[i]])
            return results
        except Exception as e:
            logger.error(f"NPU TLWH Postprocess Error ({os.path.basename(self.engine_path)}): {e}")
            return []

    def infer(self, img, conf_override=None):
        if img is None:
            return np.empty((0,6))

        h_orig, w_orig = img.shape[:2]
        npu_input, scale, offset = self.letter_box(img)
        input_tensor = self._prepare_input_tensor(npu_input)
        engine = self.engine_pool.get()

        try:
            output_tensor = engine.run([input_tensor])
            thres = conf_override if conf_override is not None else 0.40
            
            # [수정] 출력 포맷에 맞춘 라우팅 적용
            if self.output_format == "ppu":
                raw_dets = self.postprocess_ppu(output_tensor, conf_thres=thres)
            elif self.output_format == "yolo_xyxy":
                raw_dets = self.postprocess_xyxy(output_tensor, conf_thres=thres)
            elif self.output_format == "yolo_tlwh":
                raw_dets = self.postprocess_tlwh(output_tensor, conf_thres=thres)
            else:
                raw_dets = self.postprocess(output_tensor, conf_thres=thres)

            if not raw_dets:
                return np.empty((0,6))

            res = []
            dw, dh = offset

            for box, score, cls_id in raw_dets:
                x1 = np.clip((box[0] - dw) / scale, 0, w_orig)
                y1 = np.clip((box[1] - dh) / scale, 0, h_orig)
                x2 = np.clip((box[2] - dw) / scale, 0, w_orig)
                y2 = np.clip((box[3] - dh) / scale, 0, h_orig)
                res.append([x1, y1, x2, y2, score, cls_id])

            return np.array(res, dtype=float)
        except Exception as e:
            logger.error(f"NPU Inference Error: {e}")
            return np.empty((0,6))
        finally:
            self.engine_pool.put(engine)

    def _resolve_output_format(self, output_format, engine=None):
        configured = self._normalize_configured_output_format(output_format)
        if configured:
            return configured

        detected, detail = self._detect_output_format_from_engine(engine)
        if detected:
            logger.info(
                f"[DeepX] output_format=auto detected {detected}: "
                f"{os.path.basename(self.engine_path)} ({detail})"
            )
            return detected

        fallback = self._detect_output_format_from_filename()
        logger.warning(
            f"[DeepX] output_format=auto could not inspect output tensors for "
            f"{os.path.basename(self.engine_path)} ({detail or 'no metadata'}). "
            f"Using filename fallback: {fallback}."
        )
        return fallback

    def _detect_output_format_from_filename(self):
        model_name = os.path.basename(str(self.engine_path or "")).lower()
        return "ppu" if "_ppu" in model_name or "-ppu" in model_name else "yolo"

    def _detect_output_format_from_engine(self, engine):
        if engine is None:
            return None, "engine unavailable"

        info_reader = None
        for method_name in ["get_output_tensors_info", "get_outputs_info", "get_output_tensor_info"]:
            if hasattr(engine, method_name):
                info_reader = getattr(engine, method_name)
                break
        if info_reader is None:
            return None, "output tensor metadata API unavailable"

        try:
            output_info = info_reader()
        except Exception as e:
            return None, f"output tensor metadata read failed: {e}"

        entries = self._tensor_info_entries(output_info)
        if not entries:
            return None, "empty output tensor metadata"

        metadata_text = self._safe_json_text(entries).lower()
        shapes = [shape for shape in (self._tensor_shape(entry) for entry in entries) if shape]
        dtypes = [self._tensor_dtype(entry).lower() for entry in entries if self._tensor_dtype(entry)]
        detail = f"shapes={shapes or '-'} dtypes={dtypes or '-'}"

        if "ppu" in metadata_text or "postprocess" in metadata_text or "bbox" in metadata_text:
            return "ppu", detail
        if self._metadata_looks_raw_yolo(shapes, dtypes):
            return "yolo", detail
        if self._metadata_looks_ppu(shapes, dtypes):
            return "ppu", detail

        return None, detail

    def _tensor_info_entries(self, value):
        if value is None:
            return []
        if isinstance(value, dict):
            for key in ["outputs", "output", "tensors", "tensor_info"]:
                nested = value.get(key)
                if isinstance(nested, (list, tuple)):
                    return list(nested)
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _safe_json_text(self, value):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _tensor_shape(self, entry):
        shape = None
        if isinstance(entry, dict):
            for key in ["shape", "dims", "dimension", "tensor_shape"]:
                if key in entry:
                    shape = entry.get(key)
                    break
        else:
            for key in ["shape", "dims", "dimension", "tensor_shape"]:
                if hasattr(entry, key):
                    shape = getattr(entry, key)
                    break

        if shape is None:
            return []
        try:
            return [int(x) for x in list(shape)]
        except Exception:
            numbers = re.findall(r"-?\d+", str(shape))
            return [int(x) for x in numbers]

    def _tensor_dtype(self, entry):
        if isinstance(entry, dict):
            for key in ["dtype", "data_type", "type", "format"]:
                if key in entry and entry.get(key) is not None:
                    return str(entry.get(key))
        else:
            for key in ["dtype", "data_type", "type", "format"]:
                if hasattr(entry, key):
                    return str(getattr(entry, key))
        return ""

    def _shape_without_ones(self, shape):
        return [int(x) for x in shape if int(x) > 1]

    def _metadata_looks_raw_yolo(self, shapes, dtypes):
        for shape in shapes:
            dims = self._shape_without_ones(shape)
            if len(dims) < 2:
                continue
            has_candidate_axis = max(dims) >= 1000
            has_class_axis = any(5 <= dim <= 256 for dim in dims)
            if has_candidate_axis and has_class_axis:
                return True
        return False

    def _metadata_looks_ppu(self, shapes, dtypes):
        has_byte_output = any(dtype in ["uint8", "byte", "bytes"] or "uint8" in dtype for dtype in dtypes)
        if has_byte_output and any(self._shape_looks_ppu_rows(shape) for shape in shapes):
            return True
        return any(self._shape_looks_ppu_rows(shape) for shape in shapes) and not self._metadata_looks_raw_yolo(shapes, dtypes)

    def _shape_looks_ppu_rows(self, shape):
        dims = self._shape_without_ones(shape)
        if not dims:
            return False
        if len(dims) == 1:
            return dims[0] % 32 == 0 and dims[0] <= 65536
        return dims[-1] in [6, 7, 8, 32] and max(dims) < 1000

    def _load_input_shape(self, engine):
        try:
            input_info = engine.get_input_tensors_info()
            shape = list(input_info[0].get("shape", []))
        except Exception as e:
            logger.warning(f"[DeepX] 입력 텐서 shape 확인 실패. 640x640 기본값을 사용합니다: {e}")
            return

        if len(shape) == 4:
            self.input_has_batch = True
            if shape[-1] in [1, 3, 4]:
                self.input_layout = "nhwc"
                self.input_height, self.input_width = int(shape[1]), int(shape[2])
            elif shape[1] in [1, 3, 4]:
                self.input_layout = "nchw"
                self.input_height, self.input_width = int(shape[2]), int(shape[3])
        elif len(shape) == 3:
            self.input_has_batch = False
            if shape[-1] in [1, 3, 4]:
                self.input_layout = "hwc"
                self.input_height, self.input_width = int(shape[0]), int(shape[1])
            elif shape[0] in [1, 3, 4]:
                self.input_layout = "chw"
                self.input_height, self.input_width = int(shape[1]), int(shape[2])

    def letter_box(self, img, new_shape=None):
        if new_shape is None:
            new_shape = (self.input_height, self.input_width)
        h, w = img.shape[:2]
        scale = min(new_shape[0]/h, new_shape[1]/w)
        nw, nh = int(w*scale), int(h*scale)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)

        dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized

        return canvas, scale, (dw, dh)

    def _prepare_input_tensor(self, npu_input):
        input_tensor = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)

        # Keep the historical HWC path for standard YOLO models. PPU models are
        # stricter about matching the SDK-reported input shape.
        if self.output_format == "ppu":
            if self.input_layout in ["nchw", "chw"]:
                input_tensor = np.transpose(input_tensor, (2, 0, 1))
            if self.input_has_batch:
                input_tensor = np.expand_dims(input_tensor, axis=0)

        return np.ascontiguousarray(input_tensor, dtype=np.uint8)

    def postprocess(self, output_tensor, conf_thres=0.40, iou_thres=0.45):
        try:
            pred = np.array(output_tensor[0])

            # YOLOv8 배열 형태 보정
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
                pred = pred.transpose((0, 2, 1))
            if pred.ndim == 3:
                pred = pred[0]

            class_scores = pred[:, 4:]
            if class_scores.shape[1] == 1:
                # Single-class YOLO heads expose one score column after xywh.
                scores = class_scores[:, 0]
                class_ids = np.zeros(scores.shape, dtype=np.int32)
            else:
                scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)

            # Confidence 필터링
            mask = scores > conf_thres
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if len(pred) == 0:
                return []

            # NMSBoxes 포맷 맞춤 (x_min, y_min, width, height)
            boxes_xywh = pred[:, :4].copy()
            boxes_xywh[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # 중심 X -> 최소 X
            boxes_xywh[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # 중심 Y -> 최소 Y

            # Class-Aware NMS
            max_wh = 7680
            class_offset = class_ids * max_wh
            boxes_shifted = boxes_xywh.copy()
            boxes_shifted[:, 0] += class_offset
            boxes_shifted[:, 1] += class_offset

            indices = cv2.dnn.NMSBoxes(boxes_shifted.tolist(), scores.tolist(), conf_thres, iou_thres)

            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x_min, y_min, w, h = boxes_xywh[i]
                    results.append([[x_min, y_min, x_min + w, y_min + h], scores[i], class_ids[i]])

            return results
        except Exception as e:
            logger.error(f"NPU Postprocess Error ({os.path.basename(self.engine_path)}): {e}")
            return []

    def postprocess_ppu(self, output_tensor, conf_thres=0.40, iou_thres=0.45):
        try:
            raw_data = output_tensor[0]
            if isinstance(raw_data, bytes):
                flat = np.frombuffer(raw_data, dtype=np.uint8).copy()
            else:
                flat = np.ascontiguousarray(raw_data).view(np.uint8).ravel().copy()

            if len(flat) == 0:
                return []

            stride = 32
            if len(flat) % stride != 0:
                logger.error(f"[DeepX PPU] 출력 버퍼 길이({len(flat)})가 {stride}의 배수가 아닙니다.")
                return []

            flat_stride = flat.reshape(len(flat) // stride, stride)
            boxes_raw = np.ascontiguousarray(flat_stride[:, :16]).view(np.float32).reshape(-1, 4)
            scores = np.ascontiguousarray(flat_stride[:, 20:24]).view(np.float32).flatten()
            labels = np.ascontiguousarray(flat_stride[:, 24:28]).view(np.uint32).flatten()

            mask = scores >= conf_thres
            if not np.any(mask):
                return []

            boxes_raw = boxes_raw[mask]
            scores = scores[mask]
            labels = labels[mask]

            cx = boxes_raw[:, 0]
            cy = boxes_raw[:, 1]
            bw = boxes_raw[:, 2]
            bh = boxes_raw[:, 3]

            x1 = cx - bw * 0.5
            y1 = cy - bh * 0.5
            x2 = cx + bw * 0.5
            y2 = cy + bh * 0.5

            max_wh = 7680
            class_offset = labels.astype(np.float32) * max_wh
            boxes_shifted = np.column_stack([x1 + class_offset, y1 + class_offset, bw, bh])

            indices = cv2.dnn.NMSBoxes(boxes_shifted.tolist(), scores.tolist(), conf_thres, iou_thres)
            if indices is None or len(indices) == 0:
                return []

            results = []
            for i in np.array(indices).reshape(-1):
                results.append([[x1[i], y1[i], x2[i], y2[i]], scores[i], labels[i]])

            return results
        except Exception as e:
            logger.error(f"NPU PPU Postprocess Error ({os.path.basename(self.engine_path)}): {e}")
            return []
        
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
            if trk['lost'] == 0:
                res_tracks.append([*trk['bbox'], tid, trk.get('conf', 1.0), trk['cls']])

        return np.array(res_tracks)

class VideoRecorder:
    def __init__(self, ip, cam_id=None):
        self.ip = ip
        self.cam_id = cam_id if cam_id is not None else "-"
        # [수정] 지연 횡단을 고려하여 최대 40초 분량(과거 30초+사전 10초)의 프레임 캐싱 보장
        self.pre_sec = 40.0
        self.max_buffer_len = int(15 * self.pre_sec) # Max 15FPS 기준
        self.buffer = deque(maxlen=self.max_buffer_len)
        self.write_queue = queue.Queue()

        self.recording = False
        self.record_end_time = 0
        self.current_event = "unknown"
        self.current_meta = None
        self.current_record_started_at = None
        self.recorded_fps = 3.0
        self.running = True

        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def update(self, frame, infer_meta=None, timestamp=None):
        if frame is None: return
        self.buffer.append((frame.copy(), infer_meta, timestamp or time.time()))

        if self.recording:
            if time.time() > self.record_end_time:
                self.recording = False
                self.write_queue.put(None)
                logger.info(f" [녹화종료] {self.ip} - {self.current_event}")
            else:
                self.write_queue.put((frame.copy(), infer_meta, timestamp or time.time()))

    def trigger(self, event_name, objects_meta=None, event_meta=None, current_fps=3.0):
        now = time.time()
        post_sec = SYS_CFG.get("REC_POST_SEC", 10.0)
        
        # [수정] 선에 처음 닿은 시점 역산 (Crossing Delay 반영)
        candidate_age_sec = 0.0
        if isinstance(event_meta, dict) and 'decision_trace' in event_meta:
            candidate_age_sec = float(event_meta['decision_trace'].get('candidate_age_sec', 0.0))
        
        pre_sec = SYS_CFG.get("REC_PRE_SEC", 10.0)
        target_start_time = now - candidate_age_sec - pre_sec

        if self.recording:
            self.record_end_time = max(self.record_end_time, now + post_sec)
        else:
            logger.info(f"? [녹화시작] {self.ip} - {event_name} (FPS: {current_fps:.1f})")
            self.recording = True
            self.record_end_time = now + post_sec
            self.current_event = event_name
            self.current_meta = event_meta
            self.current_record_started_at = now
            self.recorded_fps = max(1.0, float(current_fps))

            for item in list(self.buffer):
                if item[2] >= target_start_time:
                    self.write_queue.put(item)

    def _writer_loop(self):
        writer = None
        fpath = None
        infer_log_file = None
        video_frame_index = 0
        
        while self.running:
            try: item = self.write_queue.get(timeout=1.0)
            except queue.Empty: continue

            if item is None:
                if writer: writer.release()
                if infer_log_file: infer_log_file.close()
                writer, infer_log_file = None, None
                continue

            frame, infer_meta, timestamp = item

            if writer is None:
                # [수정] 모든 산출물을 하나의 videos 디렉토리로 통합
                dpath = os.path.join(EVENT_ROOT_DIR, "events", self.ip, "videos", self.current_event)
                os.makedirs(dpath, exist_ok=True)
                
                time_str = datetime.datetime.fromtimestamp(self.current_record_started_at).strftime('%Y%m%d_%H%M%S')
                fname = f"{time_str}_{self.ip}_{self.current_event}.mp4"
                fpath = os.path.join(dpath, fname)
                infer_log_path = os.path.join(dpath, f"{time_str}_{self.ip}_{self.current_event}.infer.jsonl")
                meta_path = os.path.join(dpath, f"{time_str}_{self.ip}_{self.current_event}.meta.json")

                if isinstance(self.current_meta, dict):
                    self.current_meta.update({"video_path": fpath, "infer_log_path": infer_log_path, "recorded_fps": self.recorded_fps})
                    try:
                        with open(meta_path, 'w', encoding='utf-8') as f_meta:
                            json.dump(to_json_safe(self.current_meta), f_meta, indent=4, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"메타데이터 저장 실패: {e}")

                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(fpath, fourcc, self.recorded_fps, (w, h))

                try: infer_log_file = open(infer_log_path, 'w', encoding='utf-8')
                except Exception: pass

            if writer:
                if infer_log_file and infer_meta is not None:
                    try:
                        log_record = dict(infer_meta)
                        log_record["video_frame_index"] = video_frame_index
                        infer_log_file.write(json.dumps(to_json_safe(log_record), ensure_ascii=False) + "\n")
                    except Exception: pass

                writer.write(frame)
                video_frame_index += 1

class MotionDetector:
    def __init__(self, sensitivity=5):
        self.threshold = 100 - ((sensitivity - 1) * 9)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame):
        if frame is None:
            return None
        small_frame = cv2.resize(frame, (640, 360))
        fg_mask = self.bg_subtractor.apply(small_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        return fg_mask

# ==========================================
# [8] 정밀 이벤트 감지 로직 (OOP 구조)
# ==========================================
class BaseEventDetector:
    gui_name = "BASE"
    def __init__(self, config, roi_poly=None, roi_lines=None):
        self.config = config
        self.roi_poly = np.array(roi_poly, dtype=np.int32) if roi_poly and len(roi_poly) >= 3 else np.empty((0, 2), dtype=np.int32)
        self.roi_lines = roi_lines or []
        self.fps = SYS_CFG.get("REC_FPS", 3)

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        return []

class IntrusionDetector(BaseEventDetector):
    gui_name = "INTRUSION"

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        privacy_tracks = kwargs.get('privacy_tracks', []) # [추가] 전체 객체 트랙 수신
        if self.roi_poly.size == 0:
            return triggered

        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) == ID_G_PERSON:
                if cv2.pointPolygonTest(self.roi_poly, get_foot_point(*t[:4]), False) >= 0:
                    triggered.append({
                        'tid': tid,
                        'bbox': t[:4],
                        'frame': frame.copy() if frame is not None else None,
                        'fid': fid,
                        'privacy_tracks': privacy_tracks, # [추가] 페이로드에 전체 객체 포함
                        'privacy_fid': fid            # [추가]
                    })

        return triggered

class ParkingDetector(BaseEventDetector):
    gui_name = "PARKING"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.states = defaultdict(lambda: {'start_time': 0.0, 'pos': None})

        self.trigger_sec = config.get("trigger_sec", 5.0)
        self.move_threshold_ratio = config.get("move_threshold_ratio", 0.1)

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        current_time = time.time()
        privacy_tracks = kwargs.get('privacy_tracks', []) # [추가]

        if self.roi_poly.size == 0:
            return triggered

        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) in TARGET_VEHICLES:
                if cv2.pointPolygonTest(self.roi_poly, get_check_point(*t[:4]), False) >= 0:
                    curr_ids.add(tid)

                    x1, y1, x2, y2 = t[:4]
                    c = get_center_point(x1, y1, x2, y2)
                    vehicle_size = max(x2 - x1, y2 - y1)

                    dynamic_move_threshold = vehicle_size * self.move_threshold_ratio

                    if self.states[tid]['start_time'] == 0.0 or get_distance(c, self.states[tid]['pos']) > dynamic_move_threshold:
                        self.states[tid].update({
                            'start_time': current_time,
                            'pos': c,
                            'bbox': t[:4],
                            'frame': frame.copy() if frame is not None else None,
                            'fid': fid,
                            'privacy_tracks': privacy_tracks,
                            'triggered': False
                        })
                    else:
                        duration_sec = current_time - self.states[tid]['start_time']

                        if not self.states[tid].get('triggered', False) and duration_sec >= self.trigger_sec:
                            triggered.append({
                                'tid': tid,
                                'bbox': self.states[tid]['bbox'],
                                'frame': self.states[tid]['frame'],
                                'fid': self.states[tid]['fid'],
                                'decision_trace': {
                                    'detector': 'ParkingDetector',
                                    'reason': 'stationary_duration_exceeded',
                                    'class_id': int(track_map.get(tid, -1)),
                                    'roi_check_point': int_point(get_check_point(*t[:4])),
                                    'anchor_center': int_point(self.states[tid]['pos']),
                                    'current_center': int_point(c),
                                    'duration_sec': round(float(duration_sec), 3),
                                    'trigger_sec': round(float(self.trigger_sec), 3),
                                    'move_threshold_ratio': round(float(self.move_threshold_ratio), 4),
                                    'dynamic_move_threshold': round(float(dynamic_move_threshold), 3),
                                    'vehicle_size': round(float(vehicle_size), 3)
                                }
                            })
                            self.states[tid]['triggered'] = True

        for tid in list(self.states.keys()):
            if tid not in curr_ids:
                del self.states[tid]

        return triggered

class CrossingDetector(BaseEventDetector):
    gui_name = "CROSSING"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.lines = []
        for i in range(0, len(self.roi_lines), 2):
            if i + 1 < len(self.roi_lines):
                self.lines.append((self.roi_lines[i], self.roi_lines[i+1]))

        self.prev = {}
        self.candidates = {}

        self.min_crossing_angle = config.get("min_crossing_angle", 20.0)
        self.distance_ratio = config.get("distance_ratio", 0.2)

        self.candidate_ttl_sec = config.get("candidate_ttl_sec", 30.0)

    def _is_intersect(self, p1, p2, p3, p4):
        c1 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
        c2 = ccw(p3, p4, p1) * ccw(p3, p4, p2)
        return c1 <= 0 and c2 <= 0

    def _get_perpendicular_distance(self, p1, p2, pt):
        den = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if den == 0: return 0
        return abs((p2[0] - p1[0]) * (p1[1] - pt[1]) - (p1[0] - pt[0]) * (p2[1] - p1[1])) / den

    def _get_angle_between_lines(self, line1, line2):
        dx1 = line1[1][0] - line1[0][0]
        dy1 = line1[1][1] - line1[0][1]
        dx2 = line2[1][0] - line2[0][0]
        dy2 = line2[1][1] - line2[0][1]

        dot_product = dx1 * dx2 + dy1 * dy2
        mag1 = math.sqrt(dx1**2 + dy1**2)
        mag2 = math.sqrt(dx2**2 + dy2**2)

        if mag1 * mag2 == 0:
            return 0.0

        cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_theta))

        if angle > 90:
            angle = 180 - angle
        return angle

    def _get_intersection_over_lowbody_area(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

        return inter_area / float(area1) if area1 > 0 else 0.0

    def _update_computed_lines(self):
        """
        [최적화] ROI 라인이 변경되었을 때만 최초 1회 선분의 기울기를 계산하여
        해당 선분의 트리거 앵커 타입(TOP, CENTER, FOOT)을 캐싱합니다.
        """
        if not hasattr(self, '_cached_lines') or self._cached_lines != self.lines:
            self.computed_lines = []
            for p1, p2 in self.lines:
                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                
                if dx == 0:
                    atype = 'TOP'
                else:
                    slope = abs(dy / dx)
                    if slope < 0.57: atype = 'TOP'       # 완만한 가로선 (컨베이어 횡단 방어)
                    elif slope > 1.73: atype = 'FOOT'    # 가파른 세로선
                    else: atype = 'CENTER'               # 대각선
                    
                self.computed_lines.append({'p1': p1, 'p2': p2, 'anchor_type': atype})
            self._cached_lines = list(self.lines)

    def _get_dynamic_anchor(self, box, anchor_type):
        """
        [최적화] 사전 연산된 anchor_type을 받아 즉시 좌표만 반환합니다. (연산 부하 0)
        """
        bx1, by1, bx2, by2 = box
        bcx = (bx1 + bx2) / 2.0
        
        if anchor_type == 'TOP':
            return (bcx, float(by1))
        elif anchor_type == 'FOOT':
            return (bcx, float(by2))
        else:
            return (bcx, (by1 + by2) / 2.0)

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        # [핵심] ROI 라인 정보가 캐싱되어 있는지 확인 (CPU 부하 최소화)
        self._update_computed_lines()
        
        triggered = []
        curr_ids = set()
        current_time = time.time()
        privacy_tracks = kwargs.get('privacy_tracks', [])
        
        persons = [t for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        low_bodies = [t for t in tracks if track_map.get(int(t[4])) == ID_PERSON_LOW]

        for p in persons:
            p_tid = int(p[4])
            curr_ids.add(p_tid)

            px1, py1, px2, py2 = p[:4]
            person_height = max(1, py2 - py1)

            best_low_track = None
            max_ioa = 0

            for lb in low_bodies:
                lx1, ly1, lx2, ly2 = lb[:4]
                lcy = (ly1 + ly2) / 2

                if lcy < py1 + person_height * 0.4:
                    continue

                ioa = self._get_intersection_over_lowbody_area(lb[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa
                    best_low_track = lb

            curr_objects = [{'label': 'person', 'box': [int(x) for x in p[:4]], 'score': float(p[5]), 'tid': p_tid, 'class_id': ID_G_PERSON}]
            
            if max_ioa >= 0.4 and best_low_track is not None:
                curr_box = best_low_track[:4]
                curr_objects.append({'label': 'low_body', 'box': [int(x) for x in best_low_track[:4]], 'score': float(best_low_track[5]), 'tid': int(best_low_track[4]), 'class_id': ID_PERSON_LOW})
            else:
                if p_tid in self.candidates and current_time - self.candidates[p_tid]['timestamp_time'] > self.candidate_ttl_sec:
                    del self.candidates[p_tid]
                continue

            # 이전 박스가 없으면 등록 후 스킵
            if p_tid not in self.prev:
                self.prev[p_tid] = curr_box
                continue
                
            prev_box = self.prev[p_tid]
            
            # 점프 방어는 BBox 중심점을 기준으로 튀는지 검사
            prev_cx, prev_cy = (prev_box[0]+prev_box[2])/2, (prev_box[1]+prev_box[3])/2
            curr_cx, curr_cy = (curr_box[0]+curr_box[2])/2, (curr_box[1]+curr_box[3])/2
            jump_dist = get_distance((prev_cx, prev_cy), (curr_cx, curr_cy))
            
            if jump_dist > person_height * 0.2:
                self.prev[p_tid] = curr_box
                continue

            # 횡단 후보군 등록 (사전 연산된 computed_lines 사용)
            if p_tid not in self.candidates:
                for c_line in self.computed_lines:
                    p1, p2 = c_line['p1'], c_line['p2']
                    atype = c_line['anchor_type']
                    
                    # 캐싱된 앵커 타입을 사용하여 연산 없이 즉각 좌표 추출
                    prev_pos = self._get_dynamic_anchor(prev_box, atype)
                    curr_pos = self._get_dynamic_anchor(curr_box, atype)
                    trajectory = (prev_pos, curr_pos)
                    
                    if self._is_intersect(p1, p2, trajectory[0], trajectory[1]):
                        cross_angle = self._get_angle_between_lines((p1, p2), trajectory)
                        if cross_angle >= self.min_crossing_angle:
                            self.candidates[p_tid] = {
                                'person_height': person_height,
                                'timestamp_time': current_time,
                                'line': (p1, p2),
                                'anchor_type': atype, # [핵심] 판정 시 사용할 앵커 타입 함께 보관
                                'entry_side': ccw(p1, p2, trajectory[0]),
                                'cross_angle': cross_angle,
                                'candidate_trajectory': [trajectory[0], trajectory[1]],
                                'crossed_pos': curr_pos, 
                                'bbox': tuple(curr_box),
                                'frame': frame.copy() if frame is not None else None,
                                'fid': fid,
                                'privacy_tracks': privacy_tracks,
                                'objects': curr_objects
                            }
                        break

            # 최종 트리거 판별
            if p_tid in self.candidates:
                cand = self.candidates[p_tid]
                p1, p2 = cand['line']
                atype = cand['anchor_type']
                
                # 저장해둔 앵커 타입으로 현재 위치 즉각 추출
                curr_pos = self._get_dynamic_anchor(curr_box, atype)
                curr_side = ccw(p1, p2, curr_pos)

                if cand['entry_side'] != 0 and curr_side != 0 and cand['entry_side'] != curr_side:
                    perp_dist = self._get_perpendicular_distance(p1, p2, curr_pos)
                    post_cross_dist = get_distance(cand['crossed_pos'], curr_pos)

                    dx = abs(p2[0] - p1[0])
                    dy = abs(p2[1] - p1[1])
                    line_tilt_angle = math.degrees(math.atan2(dy, dx))

                    tilt_factor = 1.0 + (math.sin(math.radians(line_tilt_angle)) * 0.5)
                    dynamic_threshold = cand['person_height'] * self.distance_ratio * tilt_factor

                    if perp_dist >= dynamic_threshold and post_cross_dist >= (dynamic_threshold * 0.6):
                        triggered.append({
                            'tid': p_tid,
                            'bbox': cand['bbox'],
                            'frame': cand['frame'],
                            'fid': cand['fid'],
                            'privacy_tracks': cand.get('privacy_tracks', []), # [추가]
                            'privacy_fid': cand['fid'],
                            'objects': cand['objects'],
                            'decision_trace': {
                                'detector': 'CrossingDetector',
                                'reason': 'line_crossed_after_candidate',
                                'line': [int_point(p1), int_point(p2)],
                                'anchor_type': atype,
                                'entry_side': int(cand['entry_side']),
                                'current_side': int(curr_side),
                                'candidate_age_sec': round(float(current_time - cand['timestamp_time']), 3),
                                'crossed_pos': int_point(cand['crossed_pos']),
                                'current_pos': int_point(curr_pos),
                                'perp_dist': round(float(perp_dist), 3),
                                'post_cross_dist': round(float(post_cross_dist), 3),
                            }
                        })
                        del self.candidates[p_tid]
                
                elif current_time - cand['timestamp_time'] > self.candidate_ttl_sec:
                    del self.candidates[p_tid]

            self.prev[p_tid] = curr_box

        for tid in list(self.prev.keys()):
            if tid not in curr_ids:
                del self.prev[tid]
                if tid in self.candidates: del self.candidates[tid]

        return triggered

class HelmetDetector(BaseEventDetector):
    gui_name = "NO-HELMET"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.sessions = []

        self.min_streak_sec = config.get("min_streak_sec", 2.0)
        self.trigger_total_sec = config.get("trigger_total_sec", 4.0)
        self.max_gap_sec = config.get("max_gap_sec", 1.5)

        self.window_sec = config.get("window_sec", 30.0)

        self.ignore_top_ratio = config.get("ignore_top_ratio", 0.2)
        self.red_helmet_tids = set()

    def _get_roi_crop(self, frame, box):
        if frame is None:
            return None

        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        h_box = y2 - y1
        if h_box <= 0 or (x2 - x1) <= 0:
            return None

        roi_y2 = y1 + int(h_box * 0.5)
        roi = frame[y1:roi_y2, x1:x2]

        if roi.size == 0:
            return None

        return roi.copy()

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

        med_r = np.median(r_means)
        med_h = np.median(h_means)
        med_s = np.median(s_means)

        return (10 <= med_h <= 40) and (med_s >= 60) and (med_r >= 100)

    def _get_intersection_over_head_area(self, head_box, person_box):
        inter_w = max(0, min(head_box[2], person_box[2]) - max(head_box[0], person_box[0]))
        inter_h = max(0, min(head_box[3], person_box[3]) - max(head_box[1], person_box[1]))
        inter_area = inter_w * inter_h

        head_area = max(1, (head_box[2] - head_box[0]) * (head_box[3] - head_box[1]))
        return inter_area / head_area

    def _is_no_helmet_in_roi(self, head_box):
        if self.roi_poly is None or self.roi_poly.size == 0:
            return True
        return cv2.pointPolygonTest(self.roi_poly, get_center_point(*head_box), False) >= 0

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        helmet_tracks = kwargs.get('helmet_tracks', [])
        privacy_tracks = kwargs.get('privacy_tracks', [])
        current_time = time.time()

        unhelmeted_heads = [t for t in helmet_tracks if int(t[6]) == ID_H_NO_HELMET]
        current_nh_persons = []

        ignore_y_thresh = 0
        if frame is not None:
            ignore_y_thresh = frame.shape[0] * self.ignore_top_ratio

        for p in tracks:
            p_tid = int(p[4])

            if p_tid in self.red_helmet_tids:
                continue
            if track_map.get(p_tid) != ID_G_PERSON:
                continue

            px1, py1, px2, py2 = p[:4]

            if py1 <= ignore_y_thresh:
                continue

            if self.roi_poly is not None and self.roi_poly.size > 0:
                foot_pt = get_foot_point(*p[:4])
                if cv2.pointPolygonTest(self.roi_poly, foot_pt, False) < 0:
                    continue

            person_height = max(1, py2 - py1)
            person_width = max(1, px2 - px1)

            max_ioa = 0
            nh_track_match = None

            for head in unhelmeted_heads:
                hx1, hy1, hx2, hy2 = head[:4]
                hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

                if hcy > py1 + person_height * 0.4: continue
                margin = person_width * 0.15
                if hcx < px1 - margin or hcx > px2 + margin: continue

                ioa = self._get_intersection_over_head_area(head[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa
                    nh_track_match = head

            if max_ioa >= 0.5 and nh_track_match is not None:
                if not self._is_no_helmet_in_roi(nh_track_match[:4]):
                    continue

                hx1, hy1, hx2, hy2 = nh_track_match[:4]
                head_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
                current_nh_persons.append({
                    'tid': p_tid,
                    'head_bbox': nh_track_match[:4],
                    'person_bbox': p[:4],
                    'decision_context': {
                        'person_tid': p_tid,
                        'no_helmet_tid': int(nh_track_match[4]),
                        'no_helmet_score': float(nh_track_match[5]),
                        'ioa_with_person': float(max_ioa),
                        'min_ioa': 0.5,
                        'head_center': int_point(head_center),
                        'person_top40_y': int(round(float(py1 + person_height * 0.4))),
                        'person_width': int(round(float(person_width))),
                        'ignore_top_ratio': round(float(self.ignore_top_ratio), 4),
                        'roi_checked': bool(self.roi_poly is not None and self.roi_poly.size > 0),
                        'roi_passed': True
                    },
                    'objects': [
                        # [수정] 사람(Person) BBox를 페이로드에서 제외하고, 미착용 머리 객체만 전송
                        {'label': 'no_helmet', 'box': [int(x) for x in nh_track_match[:4]], 'score': float(nh_track_match[5]), 'tid': int(nh_track_match[4]), 'class_id': ID_H_NO_HELMET}
                    ],
                    'privacy_objects': [
                        {'label': 'person', 'box': [int(x) for x in p[:4]], 'score': float(p[5]), 'tid': p_tid, 'class_id': ID_G_PERSON}
                    ]
                })

        for nh_p in current_nh_persons:
            matched_session = None
            for session in self.sessions:
                if session['last_tid'] == nh_p['tid'] or calculate_iou(nh_p['person_bbox'], session['last_person_bbox']) > 0.3:
                    matched_session = session
                    break

            roi_crop = self._get_roi_crop(frame, nh_p['head_bbox'])

            if matched_session:
                gap_sec = current_time - matched_session['last_seen_time']
                if gap_sec <= self.max_gap_sec:
                    matched_session['streaks'][-1]['end_time'] = current_time
                else:
                    matched_session['streaks'].append({'start_time': current_time, 'end_time': current_time})

                matched_session['last_seen_time'] = current_time
                matched_session['last_tid'] = nh_p['tid']
                matched_session['last_person_bbox'] = nh_p['person_bbox']
                matched_session['bbox'] = nh_p['head_bbox']
                matched_session['fid'] = fid
                matched_session['objects'] = nh_p['objects']
                matched_session['privacy_objects'] = nh_p.get('privacy_objects', [])
                matched_session['decision_context'] = nh_p.get('decision_context', {})

                if roi_crop is not None:
                    matched_session['roi_buffer'].append(roi_crop)
            else:
                new_buffer = deque(maxlen=5)
                if roi_crop is not None:
                    new_buffer.append(roi_crop)

                self.sessions.append({
                    'start_time': current_time,
                    'last_seen_time': current_time,
                    'streaks': [{'start_time': current_time, 'end_time': current_time}],
                    'last_tid': nh_p['tid'],
                    'last_person_bbox': nh_p['person_bbox'],
                    'bbox': nh_p['head_bbox'],
                    'fid': fid,
                    'triggered': False,
                    'roi_buffer': new_buffer,
                    'objects': nh_p['objects'],
                    'privacy_objects': nh_p.get('privacy_objects', []),
                    'decision_context': nh_p.get('decision_context', {})
                })

        active_sessions = []
        for session in self.sessions:
            if session['last_tid'] in self.red_helmet_tids: continue
            if current_time - session['start_time'] > self.window_sec: continue

            total_valid_sec = 0.0
            valid_streaks = []
            for streak in session['streaks']:
                streak_duration = streak['end_time'] - streak['start_time']
                if streak_duration >= self.min_streak_sec:
                    total_valid_sec += streak_duration
                    valid_streaks.append({
                        'duration_sec': round(float(streak_duration), 3)
                    })

            if not session['triggered'] and total_valid_sec >= self.trigger_total_sec:
                is_red_helmet = self._is_red_helmet_median(session['roi_buffer'])
                if is_red_helmet:
                    self.red_helmet_tids.add(session['last_tid'])
                else:
                    triggered.append({
                        'tid': session['last_tid'],
                        'bbox': session['bbox'],
                        'frame': frame.copy() if frame is not None else None,
                        'fid': session['fid'],
                        'privacy_tracks': privacy_tracks, # [추가]
                        'privacy_fid': fid,
                        'objects': session['objects'],
                        'privacy_objects': session.get('privacy_objects', []),
                        'decision_trace': {
                            'detector': 'HelmetDetector',
                            'reason': 'no_helmet_duration_exceeded',
                            'session_age_sec': round(float(current_time - session['start_time']), 3),
                            'total_valid_sec': round(float(total_valid_sec), 3),
                            'trigger_total_sec': round(float(self.trigger_total_sec), 3),
                            'min_streak_sec': round(float(self.min_streak_sec), 3),
                            'max_gap_sec': round(float(self.max_gap_sec), 3),
                            'valid_streaks': valid_streaks,
                            'red_helmet_veto': False,
                            'roi_buffer_size': int(len(session.get('roi_buffer', []))),
                            **session.get('decision_context', {})
                        }
                    })
                session['triggered'] = True

            active_sessions.append(session)

        self.sessions = active_sessions
        return triggered

class SignalVehicleDetector(BaseEventDetector):
    gui_name = "NO-SIGNAL"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.history = defaultdict(lambda: deque(maxlen=30))
        
        # [복구 및 최적화] MOG2 모션 감지기 초기화 
        # 그림자 감지(detectShadows=False)를 꺼서 CPU 부하를 대폭 줄입니다.
        self.mog = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
        
        self.auth_grace_sec = config.get("auth_grace_sec", 120.0)
        self.presence_threshold_sec = config.get("presence_threshold_sec", 3.0)
        self.parked_threshold_sec = config.get("parked_threshold_sec", 60.0)
        self.prox_ratio_x = config.get("prox_ratio_x", 1.0)
        self.prox_ratio_y = config.get("prox_ratio_y", 1.0)

        self.line_truck_confirm_frames = max(1, int(config.get("line_truck_confirm_frames", 10)))
        self.line_truck_confirm_ratio = min(1.0, max(0.0, float(config.get("line_truck_confirm_ratio", 0.7))))
        self.line_truck_car_veto_frames = max(0, int(config.get("line_truck_car_veto_frames", 5)))
        self.line_truck_min_conf = float(config.get("line_truck_min_conf", 0.7))
        self.line_truck_car_veto_iou = float(config.get("line_truck_car_veto_iou", 0.10))
        self.line_truck_car_veto_distance_ratio = float(config.get("line_truck_car_veto_distance_ratio", 0.60))
        self.state_inherit_distance_ratio = float(config.get("state_inherit_distance_ratio", 0.5))
        self.state_inherit_max_size_ratio = max(1.0, float(config.get("state_inherit_max_size_ratio", 2.5)))
        self.state_inherit_max_area_ratio = max(1.0, float(config.get("state_inherit_max_area_ratio", 6.0)))
        self.line_truck_votes = defaultdict(lambda: deque(maxlen=self.line_truck_confirm_frames))
        self.recent_car_observations = deque(maxlen=max(30, self.line_truck_car_veto_frames * 10))
        self.process_seq = 0
        self.confirmed_line_truck_ids = set()

        self.presence_start_time = {}
        self.last_auth_time = {}
        self.last_auth_signalman = {}
        self.last_signalman_seen_time = {}
        self.stationary_anchor = {}
        self.stationary_start_time = {}

        self.last_seen_bbox = {}
        self.last_seen_time = {}
        self.is_parked = set()
        self.state_inherit_sources = {}

    def _remember_recent_cars(self, tracks, track_map):
        for t in tracks:
            if track_map.get(int(t[4])) != ID_G_CAR:
                continue
            self.recent_car_observations.append({
                "seq": self.process_seq,
                "bbox": np.array(t[:4], dtype=float),
                "foot": get_foot_point(*t[:4])
            })

        while self.recent_car_observations:
            oldest = self.recent_car_observations[0]
            if self.process_seq - oldest["seq"] <= self.line_truck_car_veto_frames:
                break
            self.recent_car_observations.popleft()

    def _has_recent_car_veto(self, truck_box):
        if self.line_truck_car_veto_frames <= 0:
            return False

        truck_box = np.array(truck_box, dtype=float)
        truck_foot = get_foot_point(*truck_box)
        truck_size = max(truck_box[2] - truck_box[0], truck_box[3] - truck_box[1])

        for obs in self.recent_car_observations:
            if self.process_seq - obs["seq"] > self.line_truck_car_veto_frames:
                continue

            car_box = obs["bbox"]
            car_size = max(car_box[2] - car_box[0], car_box[3] - car_box[1])
            near_distance = max(truck_size, car_size) * self.line_truck_car_veto_distance_ratio

            if calculate_iou(truck_box, car_box) >= self.line_truck_car_veto_iou:
                return True
            if get_distance(truck_foot, obs["foot"]) <= near_distance:
                return True

        return False

    def _box_size_meta(self, box):
        w = max(1.0, float(box[2] - box[0]))
        h = max(1.0, float(box[3] - box[1]))
        return w, h, max(w, h), w * h

    def _can_inherit_track_state(self, curr_box, old_box, iou, dist):
        _, _, curr_size, curr_area = self._box_size_meta(curr_box)
        _, _, old_size, old_area = self._box_size_meta(old_box)

        distance_gate = min(curr_size, old_size) * self.state_inherit_distance_ratio
        spatial_match = iou > 0.3 or dist < distance_gate
        if not spatial_match:
            return False, "spatial_mismatch"

        size_ratio = max(curr_size / old_size, old_size / curr_size)
        area_ratio = max(curr_area / old_area, old_area / curr_area)
        if size_ratio > self.state_inherit_max_size_ratio or area_ratio > self.state_inherit_max_area_ratio:
            return False, f"size_jump:size={size_ratio:.2f},area={area_ratio:.2f}"

        return True, "ok"

    def _is_confirmed_line_truck(self, track):
        tid = int(track[4])
        score = float(track[5])
        box = track[:4]
        high_conf_truck = score >= self.line_truck_min_conf
        car_veto = self._has_recent_car_veto(box)

        self.line_truck_votes[tid].append(high_conf_truck and not car_veto)
        votes = list(self.line_truck_votes[tid])
        required_votes = max(1, int(math.ceil(self.line_truck_confirm_frames * self.line_truck_confirm_ratio)))

        return len(votes) >= self.line_truck_confirm_frames and sum(votes) >= required_votes

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        current_time = time.time()
        privacy_tracks = kwargs.get('privacy_tracks', [])
        self.process_seq += 1

        if self.roi_poly.size == 0 or frame is None: return triggered

        h_frame, w_frame = frame.shape[:2]
        
        # [복구 및 최적화] 320x180 초소형 해상도로 MOG 연산 수행 (CPU 연산량 1/16 수준 방어)
        # 멈춰있던 배경(트럭)이 픽셀 수준에서 진짜 움직였는지 교차 검증하는 용도
        small_frame = cv2.resize(frame, (320, 180))
        small_motion_mask = self.mog.apply(small_frame)

        prox_x_thresh, prox_y_thresh = w_frame * self.prox_ratio_x, h_frame * self.prox_ratio_y
        signalman_tracks = kwargs.get('signalman_tracks', [])
        signalmen_info = [{'tid': int(t[4]), 'pt': get_foot_point(*t[:4])} for t in signalman_tracks]

        self._remember_recent_cars(tracks, track_map)
        self.confirmed_line_truck_ids = set()
        
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) == ID_G_TRUCK and self._is_confirmed_line_truck(t):
                curr_ids.add(tid)
                self.confirmed_line_truck_ids.add(tid)

        missing_tids = [tid for tid in self.last_seen_bbox.keys() if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) < 3.0]

        for curr_tid in curr_ids:
            curr_box = next((t[:4] for t in tracks if int(t[4]) == curr_tid), None)
            if curr_box is None: continue
            curr_fc = get_foot_point(*curr_box)

            for old_tid in missing_tids:
                old_box = self.last_seen_bbox[old_tid]
                iou = calculate_iou(curr_box, old_box)
                old_fc = get_foot_point(*old_box)
                dist = get_distance(curr_fc, old_fc)
                can_inherit, inherit_reason = self._can_inherit_track_state(curr_box, old_box, iou, dist)

                if can_inherit:
                    self.state_inherit_sources[curr_tid] = {
                        'from_tid': int(old_tid), 'to_tid': int(curr_tid),
                        'iou': round(float(iou), 4), 'distance': round(float(dist), 3),
                        'reason': inherit_reason
                    }
                    if old_tid in self.last_auth_time:
                        self.last_auth_time[curr_tid] = self.last_auth_time[old_tid]
                        del self.last_auth_time[old_tid]
                    if old_tid in self.last_auth_signalman:
                        self.last_auth_signalman[curr_tid] = self.last_auth_signalman[old_tid]
                        del self.last_auth_signalman[old_tid]
                    if old_tid in self.history:
                        self.history[curr_tid] = self.history[old_tid]
                        del self.history[old_tid]
                    if old_tid in self.presence_start_time:
                        self.presence_start_time[curr_tid] = self.presence_start_time[old_tid]
                        del self.presence_start_time[old_tid]
                    if old_tid in self.last_signalman_seen_time:
                        self.last_signalman_seen_time[curr_tid] = self.last_signalman_seen_time[old_tid]
                        del self.last_signalman_seen_time[old_tid]
                    if old_tid in self.stationary_anchor:
                        self.stationary_anchor[curr_tid] = self.stationary_anchor[old_tid]
                        del self.stationary_anchor[old_tid]
                    if old_tid in self.stationary_start_time:
                        self.stationary_start_time[curr_tid] = self.stationary_start_time[old_tid]
                        del self.stationary_start_time[old_tid]
                    if old_tid in self.is_parked:
                        self.is_parked.add(curr_tid)
                        self.is_parked.remove(old_tid)

                    missing_tids.remove(old_tid)
                    break
                elif inherit_reason.startswith("size_jump"):
                    pass 

        for t in tracks:
            tid = int(t[4])
            if tid not in self.confirmed_line_truck_ids: continue

            x1, y1, x2, y2 = t[:4]
            fc, c_pt = get_foot_point(*t[:4]), get_center_point(*t[:4])
            v_size = max(x2 - x1, y2 - y1)

            self.last_seen_bbox[tid] = t[:4]
            self.last_seen_time[tid] = current_time

            is_in_roi = cv2.pointPolygonTest(self.roi_poly, c_pt, False) >= 0
            if is_in_roi:
                if tid not in self.stationary_anchor:
                    self.stationary_anchor[tid] = fc
                    self.stationary_start_time[tid] = current_time
                else:
                    dist_from_anchor = get_distance(self.stationary_anchor[tid], fc)
                    tolerance = max(v_size * 0.1, 15.0)

                    if dist_from_anchor > tolerance:
                        self.stationary_anchor[tid] = fc
                        self.stationary_start_time[tid] = current_time
                    else:
                        if current_time - self.stationary_start_time[tid] >= self.parked_threshold_sec:
                            self.is_parked.add(tid)
            else:
                if tid in self.stationary_anchor: del self.stationary_anchor[tid]
                if tid in self.stationary_start_time: del self.stationary_start_time[tid]
                if tid in self.is_parked: self.is_parked.remove(tid)

            if len(self.history[tid]) > 0 and get_distance(self.history[tid][-1], fc) > v_size * 0.6:
                self.history[tid].clear()
            self.history[tid].append(fc)

            has_signalman, matched_sig_tid = False, -1
            for s_info in signalmen_info:
                if abs(fc[0] - s_info['pt'][0]) <= prox_x_thresh and abs(fc[1] - s_info['pt'][1]) <= prox_y_thresh:
                    has_signalman, matched_sig_tid = True, s_info['tid']
                    break

            if has_signalman:
                self.last_signalman_seen_time[tid] = current_time
                if tid not in self.presence_start_time: self.presence_start_time[tid] = current_time
                if current_time - self.presence_start_time[tid] >= self.presence_threshold_sec:
                    self.last_auth_time[tid] = current_time
                    self.last_auth_signalman[tid] = matched_sig_tid
            else:
                last_seen = self.last_signalman_seen_time.get(tid, 0.0)
                if current_time - last_seen > 1.5:
                    if tid in self.presence_start_time: del self.presence_start_time[tid]

        # BBox 궤적 거리 + MOG 픽셀 변화량 교차 검증 로직
        for t in tracks:
            tid = int(t[4])
            if tid not in self.confirmed_line_truck_ids: continue
            if tid not in self.is_parked: continue

            x1, y1, x2, y2 = t[:4]
            v_size = max(x2 - x1, y2 - y1)
            c_pt = get_center_point(*t[:4])
            is_in_roi = cv2.pointPolygonTest(self.roi_poly, c_pt, False) >= 0
            h_list = list(self.history[tid])

            if len(h_list) > 5:
                start_p = (sum(p[0] for p in h_list[:3])/3, sum(p[1] for p in h_list[:3])/3)
                end_p = (sum(p[0] for p in h_list[-3:])/3, sum(p[1] for p in h_list[-3:])/3)
                dist = get_distance(start_p, end_p)
                min_movement = max(v_size * 0.15, 10.0)

                # 1차 검증: BBox 중심점이 충분히 이동했는가?
                if dist >= min_movement and is_in_roi:
                    
                    # 2차 검증 (ID 스위칭 오탐 방어): 실제 해당 영역 픽셀에 움직임(MOG)이 있었는가?
                    sx1, sy1 = max(0, int(x1 * 320 / w_frame)), max(0, int(y1 * 180 / h_frame))
                    sx2, sy2 = min(320, int(x2 * 320 / w_frame)), min(180, int(y2 * 180 / h_frame))
                    
                    truck_motion = small_motion_mask[sy1:sy2, sx1:sx2]
                    motion_area = max(1, (sx2 - sx1) * (sy2 - sy1))
                    motion_ratio = cv2.countNonZero(truck_motion) / motion_area
                    
                    # BBox 영역 내 픽셀 변화가 5% 미만이라면 트럭이 이동한게 아니라 트래커 ID가 튄 것으로 간주!
                    if motion_ratio < 0.05:
                        self.history[tid].clear() # 튀어버린 잘못된 궤적 초기화
                        continue

                    last_auth = self.last_auth_time.get(tid, 0.0)
                    time_since_auth = current_time - last_auth

                    if last_auth == 0.0 or time_since_auth > self.auth_grace_sec:
                        recent_auths = []
                        for a_tid, auth_t in self.last_auth_time.items():
                            remain = max(0, self.auth_grace_sec - (current_time - auth_t))
                            sig_tid = self.last_auth_signalman.get(a_tid, "Unknown")
                            recent_auths.append({'tid': a_tid, 'remain': remain, 'auth_t': auth_t, 'sig_tid': sig_tid})

                        recent_auths.sort(key=lambda x: x['auth_t'], reverse=True)
                        votes = list(self.line_truck_votes.get(tid, []))
                        stationary_start = self.stationary_start_time.get(tid, current_time)
                        triggered.append({
                            'tid': tid,
                            'bbox': t[:4],
                            'frame': frame.copy() if frame is not None else None,
                            'fid': fid,
                            'confidence': float(t[5]),
                            'auth_tokens': recent_auths[:1],
                            'privacy_tracks': privacy_tracks, # [추가]
                            'privacy_fid': fid,
                            'objects': [{
                                'label': 'LineTruck',
                                'box': [int(x) for x in t[:4]],
                                'score': float(t[5]),
                                'tid': tid,
                                'class_id': ID_G_TRUCK
                            }],
                            'decision_trace': {
                                'detector': 'SignalVehicleDetector',
                                'reason': 'moving_confirmed_line_truck_without_recent_signalman',
                                'confirmed_line_truck': True,
                                'line_truck_vote_count': int(sum(votes)),
                                'line_truck_vote_window': int(len(votes)),
                                'line_truck_confirm_frames': int(self.line_truck_confirm_frames),
                                'line_truck_confirm_ratio': round(float(self.line_truck_confirm_ratio), 4),
                                'line_truck_min_conf': round(float(self.line_truck_min_conf), 4),
                                'is_parked': bool(tid in self.is_parked),
                                'stationary_sec': round(float(current_time - stationary_start), 3),
                                'parked_threshold_sec': round(float(self.parked_threshold_sec), 3),
                                'movement_dist': round(float(dist), 3),
                                'min_movement': round(float(min_movement), 3),
                                'roi_passed': bool(is_in_roi),
                                'last_auth_age_sec': None if last_auth == 0.0 else round(float(time_since_auth), 3),
                                'auth_grace_sec': round(float(self.auth_grace_sec), 3),
                                'recent_auths': recent_auths[:1],
                                'state_inherited': self.state_inherit_sources.get(tid),
                                'history_len': int(len(h_list))
                            }
                        })

                        self.history[tid].clear()
                        if tid in self.last_auth_time: del self.last_auth_time[tid]
                        if tid in self.last_auth_signalman: del self.last_auth_signalman[tid]
                        if tid in self.is_parked: self.is_parked.remove(tid)
                        if tid in self.state_inherit_sources: del self.state_inherit_sources[tid]

        for tid in list(self.history.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0:
                del self.history[tid]

        for tid in list(self.last_auth_time.keys()):
            if current_time - self.last_auth_time[tid] > self.auth_grace_sec:
                del self.last_auth_time[tid]
                if tid in self.last_auth_signalman: del self.last_auth_signalman[tid]

        for tid in list(self.last_seen_bbox.keys()):
            if tid not in curr_ids and current_time - self.last_seen_time.get(tid, 0) > 5.0:
                del self.last_seen_bbox[tid]
                if tid in self.last_seen_time: del self.last_seen_time[tid]
                if tid in self.is_parked: self.is_parked.remove(tid)
                if tid in self.line_truck_votes: del self.line_truck_votes[tid]
                if tid in self.state_inherit_sources: del self.state_inherit_sources[tid]

        for tid in list(self.presence_start_time.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0:
                del self.presence_start_time[tid]

        for tid in list(self.last_signalman_seen_time.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0:
                del self.last_signalman_seen_time[tid]

        for tid in list(self.stationary_start_time.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0:
                del self.stationary_start_time[tid]

        for tid in list(self.stationary_anchor.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0:
                del self.stationary_anchor[tid]

        visible_tids = {int(t[4]) for t in tracks}
        for tid in list(self.line_truck_votes.keys()):
            if tid not in visible_tids and tid not in curr_ids:
                del self.line_truck_votes[tid]

        return triggered

EVENT_REGISTRY = {
    "intrusion": IntrusionDetector,
    "illegal_parking": ParkingDetector,
    "conveyor_crossing": CrossingDetector,
    "no_helmet": HelmetDetector,
    "signal_vehicle": SignalVehicleDetector
}

# ==========================================
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

def run_wizard_batch_mode(rtsp_list, existing_configs=None):
    logger.info("=== 설정 마법사 시작 ===")
    # 기존 설정을 그대로 복사하여 기반으로 삼음
    configs = existing_configs.copy() if existing_configs else {}

    for i in range(0, len(rtsp_list), BATCH_SIZE):
        batch = rtsp_list[i : i + BATCH_SIZE]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            frames = list(executor.map(capture_snapshot, batch))

        display = []
        for idx, frm in enumerate(frames):
            if frm is None:
                blk = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blk, "Conn Fail", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                display.append(blk)
            else:
                display.append(frm)

        mosaic = create_mosaic_image(display)
        cols = max(1, math.ceil(math.sqrt(len(display))))
        rows = max(1, math.ceil(len(display) / cols))
        cw = SCREEN_WIDTH // cols
        ch = SCREEN_HEIGHT // rows

        for idx in range(len(display)):
            r, c = divmod(idx, cols)
            cx, cy = c * cw, r * ch
            cv2.rectangle(mosaic, (cx, cy), (cx + 50, cy + 50), (255, 255, 255), -1)
            
            # [요구사항 반영 2] 배치(Batch) 상대 번호가 아닌 CSV 전체 기준 절대 순차 번호 생성
            abs_cam_id = i + idx + 1 
            cv2.putText(mosaic, str(abs_cam_id), (cx + 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

        cv2.imshow("Select Cameras", mosaic)
        cv2.waitKey(1)

        # [요구사항 반영 3] 프롬프트 안내 메시지도 절대 번호 기준으로 변경
        example_ids = f"{i+1},{i+2}" if len(batch) > 1 else f"{i+1}"
        sel = guarded_input(f">> [Batch {i//BATCH_SIZE + 1}] 설정할 카메라 번호 (예: {example_ids} / 건너뛰기: 엔터): ").strip()
        if not sel:
            continue

        try:
            nums = [int(s.strip()) for s in sel.split(',')]
            for n in nums:
                # [요구사항 반영 3] 사용자가 입력한 절대 번호를 다시 배치 내 로컬 인덱스로 변환하여 처리
                local_idx = n - i - 1 
                if 0 <= local_idx < len(batch) and frames[local_idx] is not None:
                    url = batch[local_idx]
                    ip = extract_ip(url)

                    print(
                        f"[{ip}] 1.침입 2.주정차 3.안전모 4.횡단 5.신호수차량 "
                        f"6.roi화각변경 7.roi화각변경+자동보정"
                    )
                    evts = guarded_input(f"[{ip}] 이벤트 선택 (예: 1,4,7): ")
                    events = []

                    selected_events = {s.strip() for s in evts.split(',') if s.strip()}

                    if '1' in selected_events: events.append("intrusion")
                    if '2' in selected_events: events.append("illegal_parking")
                    if '3' in selected_events: events.append("no_helmet")
                    if '4' in selected_events: events.append("conveyor_crossing")
                    if '5' in selected_events: events.append("signal_vehicle")
                    if '6' in selected_events: events.append(ROI_CHANGE_EVENT)
                    if '7' in selected_events: events.append(ROI_CHANGE_APPLY_EVENT)

                    roi_p = []
                    roi_l = []

                    if any(e in events for e in ["intrusion", "illegal_parking", "no_helmet", "signal_vehicle"]):
                        roi_p = get_roi_points_scaled(frames[local_idx], f"Polygon - CAM: {ip}")

                    if "conveyor_crossing" in events:
                        while True:
                            l = get_roi_points_scaled(frames[local_idx], f"Line - CAM: {ip}", mode="line")
                            if len(l) == 2:
                                roi_l.extend(l)
                            if guarded_input("횡단 라인을 추가하시겠습니까? (y/n): ") != 'y':
                                break

                    configs[ip] = {
                        "url": url,
                        "events": events,
                        "roi_poly_norm": roi_p,
                        "roi_lines_norm": roi_l
                    }
        except Exception as e:
            logger.error(f"마법사 설정 중 오류 발생: {e}")
            pass

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
        try:
            for raw_line in iter(pipe.readline, b""):
                line = self._short_external_output(raw_line)
                if line:
                    line_buffer.append(line)
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
        if width <= 720:
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
                    if fr.shape[1] > 720:
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
    def __init__(self, ip, conf, det_main_v2, det_main_v3, det_helmet, det_face, det_signalman, det_plate, cam_id, event_inference_mode="separate"):
        self.ip = ip
        self.camera_key = ip
        self.conf = conf
        self.cam_id = cam_id
        self.event_inference_mode = event_inference_mode
        self.events = conf.get('events', [])

        # [수정] V2, V3 모델 모두 수용
        self.det_main_v2 = det_main_v2
        self.det_main_v3 = det_main_v3
        self.det_helmet = det_helmet
        self.det_face = det_face
        self.det_signalman = det_signalman
        self.det_plate = det_plate

        self.trk_main = SimpleTracker()
        self.trk_helmet = SimpleTracker()
        self.trk_signalman = SimpleTracker()

        self.reader = FrameReader(conf.get('url', ''), ip)
        self.recorder = VideoRecorder(ip, cam_id=self.cam_id)
        # self.motion_det = MotionDetector() 전면 제거 완료

        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}

        self.fps_queue = deque(maxlen=30)
        self.current_fps = 0.0

        self.roi_poly_norm = conf.get('roi_poly_norm', [])
        self.roi_lines_norm = conf.get('roi_lines_norm', [])
        self.roi_poly = []
        self.roi_lines = []

        self.base_roi_poly = []
        self.base_roi_lines = []
        self.aligned_roi_poly = []
        self.aligned_roi_lines = []

        self.roi_frame_shape = None 
        self.status_history = deque(maxlen=10)
        self._reset_alignment_state("ALIGN INIT")
        self._rebuild_handlers()

    def _reset_alignment_state(self, status_text="ALIGN RESET"):
        self.aligner = AnchorTrackingROIAligner()
        self.anchor_set = False

        self.base_roi_poly = []
        self.base_roi_lines = []
        self.aligned_roi_poly = []
        self.aligned_roi_lines = []
        self.roi_shift = [0.0, 0.0]      # roi_change_apply: base ROI에 적용된 평행이동(px)
        self.roi_auto_corrected = False  # roi_change_apply 1회 보정 래치. True면 관제센터 ROI 수신(update_config) 전까지 추가 보정 금지
        self.roi_setup_pending = False   # confirm/disturbed 확정 후 관제센터 ROI 수신 전까지 True(=서버에 true 전송 중인 상태). CSV healthcheck 컬럼에 기록

        self.last_align_time = 0.0
        self.last_anchor_attempt_time = 0.0
        self.anchor_startup_wait_started_at = 0.0
        self.align_status_text = status_text
        self.align_ok = False
        self.align_shifted = False

    def _rebuild_handlers(self):
        self.handlers = {}

        for ename in self.events:
            if ename in EVENT_REGISTRY:
                self.handlers[ename] = EVENT_REGISTRY[ename](
                    SYS_CFG.get("event_config", {}).get(ename, {}),
                    self.roi_poly,
                    self.roi_lines
                )

    def update_config(self, new_conf):
        old_events = self.events.copy()

        # 관제센터에서 새 ROI가 적용되면 해당 카메라의 pending/count를 해제한다.
        ROI_ALIGN_LEARNING_STORE.reset_camera(self.camera_key, reason="camera_config_updated")

        self.conf = new_conf
        self.events = new_conf.get('events', [])
        self.roi_poly_norm = new_conf.get('roi_poly_norm', [])
        self.roi_lines_norm = new_conf.get('roi_lines_norm', [])

        self.roi_poly = []
        self.roi_lines = []
        self.roi_frame_shape = None

        self._reset_alignment_state("ALIGN RESET")
        self._rebuild_handlers()

        try:
            self.status_history.clear()
        except Exception:
            pass

        logger.info(
            f" [CAM:{self.ip}] 무중단 설정 리로드 완료: "
            f"{old_events} -> {self.events} | ROI aligner reset"
        )
        logger.debug(f"[CCTV_Aligner] CAM {self.cam_id} aligner reset after config reload")

    def _initialize_base_roi_if_needed(self, frame):
        if frame is None:
            return False

        h, w = frame.shape[:2]

        need_init = False
        if self.roi_frame_shape != frame.shape[:2]:
            need_init = True
        if self.roi_poly_norm and not self.base_roi_poly:
            need_init = True
        if self.roi_lines_norm and not self.base_roi_lines:
            need_init = True

        if not need_init:
            return True

        self.base_roi_poly = denormalize_roi_points(self.roi_poly_norm, w, h) if self.roi_poly_norm else []
        self.base_roi_lines = denormalize_roi_points(self.roi_lines_norm, w, h) if self.roi_lines_norm else []

        self.roi_shift = [0.0, 0.0]      # 새 base 기준이므로 보정량·1회 보정 래치 초기화
        self.roi_auto_corrected = False
        self.aligned_roi_poly = list(self.base_roi_poly)
        self.aligned_roi_lines = list(self.base_roi_lines)
        self.roi_frame_shape = frame.shape[:2]

        self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)
        logger.info(f"[CAM:{self.cam_id}] base ROI init | poly={len(self.base_roi_poly)} lines={len(self.base_roi_lines)} shape={frame.shape[:2]}")
        return True

    def _inject_roi_to_handlers(self, roi_poly, roi_lines):
        self.roi_poly = roi_poly or []
        self.roi_lines = roi_lines or []

        for ename in self.events:
            if ename not in self.handlers: continue
            handler = self.handlers[ename]

            if self.roi_poly and len(self.roi_poly) >= 3:
                handler.roi_poly = np.array(self.roi_poly, dtype=np.int32)
            else:
                handler.roi_poly = np.empty((0, 2), dtype=np.int32)

            if hasattr(handler, "roi_lines"):
                handler.roi_lines = self.roi_lines or []

            if hasattr(handler, "lines"):
                new_lines = []
                lines = self.roi_lines or []
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        new_lines.append((lines[i], lines[i + 1]))
                handler.lines = new_lines

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
            and (self.base_roi_poly or self.base_roi_lines)
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
                        "roi_lines_norm": self.roi_lines_norm,
                        "roi_change_poly_norm": []  # 폐기 필드. 관제 서버 호환용 빈 배열.
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

    def apply_face_blur(self, frame, person_boxes, helmet_tracks=None, return_meta=False):
        if frame is None:
            return (frame, []) if return_meta else frame

        blur_img = frame.copy()
        blurred_faces = []
        blurred_person_tids = set()
        h_img, w_img = blur_img.shape[:2]

        try:
            # [1단계] AI 얼굴 모델 적용 (동적 15% 패딩)
            if getattr(self, 'det_face', None) is not None:
                face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
                f_dets = self.det_face.infer(blur_img, conf_override=face_conf)

                for f in f_dets:
                    orig_fx1, orig_fy1, orig_fx2, orig_fy2 = map(int, f[:4])
                    orig_fw, orig_fh = orig_fx2 - orig_fx1, orig_fy2 - orig_fy1

                    if orig_fw > w_img * 0.4: continue

                    pad_x, pad_y = int(orig_fw * 0.15), int(orig_fh * 0.15)
                    fx1 = max(0, orig_fx1 - pad_x)
                    fy1 = max(0, orig_fy1 - pad_y)
                    fx2 = min(w_img, orig_fx2 + pad_x)
                    fy2 = min(h_img, orig_fy2 + pad_y)
                    fw, fh = fx2 - fx1, fy2 - fy1

                    fcx, fcy = orig_fx1 + (orig_fw / 2.0), orig_fy1 + (orig_fh / 2.0)
                    matched_person_tid = -1

                    for p in person_boxes:
                        px1, py1, px2, py2 = map(int, p[:4])
                        pw, ph = px2 - px1, py2 - py1
                        person_pad_x, person_pad_y_top, person_pad_y_bottom = pw * 0.15, ph * 0.25, ph * 0.05

                        if (px1 - person_pad_x) <= fcx <= (px2 + person_pad_x) and (py1 - person_pad_y_top) <= fcy <= (py2 + person_pad_y_bottom):
                            matched_person_tid = int(p[4]) if len(p) > 4 else -1
                            if matched_person_tid != -1:
                                blurred_person_tids.add(matched_person_tid)
                            break

                    if matched_person_tid != -1:
                        roi = blur_img[fy1:fy2, fx1:fx2]
                        if roi.size > 0:
                            small = cv2.resize(roi, (max(1, fw//15), max(1, fh//15)), interpolation=cv2.INTER_LINEAR)
                            blur_img[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
                            blurred_faces.append({
                                "box": [fx1, fy1, fx2, fy2],
                                "score": round(float(f[4]), 4) if len(f) > 4 else 0.0,
                                "class_id": int(f[5]) if len(f) > 5 else -1,
                                "matched_person_tid": matched_person_tid
                            })

            # [2단계] AI 헬멧/머리 트래킹 데이터 활용 (얼굴 모델 누락자 대상)
            if helmet_tracks is not None and len(helmet_tracks) > 0:
                for p in person_boxes:
                    p_tid = int(p[4]) if len(p) > 4 else -1
                    if p_tid in blurred_person_tids or p_tid == -1: continue

                    px1, py1, px2, py2 = map(int, p[:4])
                    pw, ph = px2 - px1, py2 - py1
                    best_match = None
                    max_ioa = 0

                    for h_track in helmet_tracks:
                        hx1, hy1, hx2, hy2 = map(int, h_track[:4])
                        hcx, hcy = hx1 + (hx2 - hx1) / 2.0, hy1 + (hy2 - hy1) / 2.0
                        
                        if hcy > py1 + ph * 0.4: continue
                        if hcx < px1 - pw * 0.15 or hcx > px2 + pw * 0.15: continue
                        
                        inter_w = max(0, min(hx2, px2) - max(hx1, px1))
                        inter_h = max(0, min(hy2, py2) - max(hy1, py1))
                        head_area = max(1, (hx2 - hx1) * (hy2 - hy1))
                        ioa = (inter_w * inter_h) / head_area

                        if ioa > max_ioa:
                            max_ioa = ioa
                            best_match = h_track

                    if max_ioa > 0.3 and best_match is not None:
                        hx1, hy1, hx2, hy2 = map(int, best_match[:4])
                        # 헬멧/머리 영역도 15% 패딩
                        hw, hh = hx2 - hx1, hy2 - hy1
                        pad_x, pad_y = int(hw * 0.15), int(hh * 0.15)
                        hx1, hy1 = max(0, hx1 - pad_x), max(0, hy1 - pad_y)
                        hx2, hy2 = min(w_img, hx2 + pad_x), min(h_img, hy2 + pad_y)
                        hw, hh = hx2 - hx1, hy2 - hy1

                        roi = blur_img[hy1:hy2, hx1:hx2]
                        if roi.size > 0:
                            small = cv2.resize(roi, (max(1, hw//15), max(1, hh//15)), interpolation=cv2.INTER_LINEAR)
                            blur_img[hy1:hy2, hx1:hx2] = cv2.resize(small, (hw, hh), interpolation=cv2.INTER_NEAREST)
                            blurred_faces.append({
                                "box": [hx1, hy1, hx2, hy2],
                                "score": round(float(best_match[5]), 4) if len(best_match) > 5 else 0.0,
                                "class_id": int(best_match[6]) if len(best_match) > 6 else -1,
                                "matched_person_tid": p_tid
                            })
                            blurred_person_tids.add(p_tid)

            # [3단계] 최후의 휴리스틱 비율 추정 적용 (1, 2단계 모두 실패한 사람만)
            for p in person_boxes:
                p_tid = int(p[4]) if len(p) > 4 else -1
                if p_tid in blurred_person_tids or p_tid == -1: continue

                px1, py1, px2, py2 = map(int, p[:4])
                pw, ph = px2 - px1, py2 - py1
                if pw <= 0 or ph <= 0: continue

                fx1, fy1 = max(0, int(px1 + pw * 0.20)), max(0, int(py1 - ph * 0.05))
                fx2, fy2 = min(w_img, int(px2 - pw * 0.20)), min(h_img, int(py1 + ph * 0.25))
                fw, fh = fx2 - fx1, fy2 - fy1
                if fw <= 0 or fh <= 0: continue

                roi = blur_img[fy1:fy2, fx1:fx2]
                if roi.size > 0:
                    small = cv2.resize(roi, (max(1, fw//15), max(1, fh//15)), interpolation=cv2.INTER_LINEAR)
                    blur_img[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    blurred_faces.append({
                        "box": [fx1, fy1, fx2, fy2],
                        "score": 1.0, "class_id": -1, "matched_person_tid": p_tid
                    })

        except Exception as e:
            logger.error(f"모자이크 처리 실패: {e}")

        return (blur_img, blurred_faces) if return_meta else blur_img


    def apply_plate_blur(self, frame, vehicle_boxes=None, return_meta=False):
        if frame is None:
            return (frame, []) if return_meta else frame

        blur_img = frame.copy()
        blurred_plates = []
        blurred_vehicle_tids = set()
        h_img, w_img = blur_img.shape[:2]

        try:
            # [1단계] AI 번호판 모델 적용 (동적 15% 패딩)
            if getattr(self, 'det_plate', None) is not None:
                plate_conf = SYS_CFG.get("model_confidences", {}).get("PLATE", 0.1)
                p_dets = self.det_plate.infer(blur_img, conf_override=plate_conf)

                for p in p_dets:
                    orig_px1, orig_py1, orig_px2, orig_py2 = map(int, p[:4])
                    orig_pw, orig_ph = orig_px2 - orig_px1, orig_py2 - orig_py1

                    if orig_pw <= 0 or orig_ph <= 0 or orig_pw > w_img * 0.6 or orig_ph > h_img * 0.3: continue

                    pad_x, pad_y = int(orig_pw * 0.15), int(orig_ph * 0.15)
                    px1 = max(0, orig_px1 - pad_x)
                    py1 = max(0, orig_py1 - pad_y)
                    px2 = min(w_img, orig_px2 + pad_x)
                    py2 = min(h_img, orig_py2 + pad_y)
                    pw, ph = px2 - px1, py2 - py1

                    pcx, pcy = orig_px1 + (orig_pw / 2.0), orig_py1 + (orig_ph / 2.0)
                    matched_vehicle_tid = -1

                    if vehicle_boxes is not None:
                        for v in vehicle_boxes:
                            vx1, vy1, vx2, vy2 = map(int, v[:4])
                            vw, vh = vx2 - vx1, vy2 - vy1
                            if (vx1 - vw * 0.10) <= pcx <= (vx2 + vw * 0.10) and (vy1 - vh * 0.10) <= pcy <= (vy2 + vh * 0.10):
                                matched_vehicle_tid = int(v[4]) if len(v) > 4 else -1
                                if matched_vehicle_tid != -1:
                                    blurred_vehicle_tids.add(matched_vehicle_tid)
                                break

                    roi = blur_img[py1:py2, px1:px2]
                    if roi.size > 0:
                        small = cv2.resize(roi, (max(1, pw // 12), max(1, ph // 12)), interpolation=cv2.INTER_LINEAR)
                        blur_img[py1:py2, px1:px2] = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
                        blurred_plates.append({
                            "box": [px1, py1, px2, py2],
                            "score": round(float(p[4]), 4) if len(p) > 4 else 0.0,
                            "class_id": int(p[5]) if len(p) > 5 else -1,
                            "matched_vehicle_tid": matched_vehicle_tid
                        })

            # [2단계] 하단 휴리스틱 비율 추정 (AI 모델 누락 차량 대상)
            if vehicle_boxes is not None:
                for v in vehicle_boxes:
                    v_tid = int(v[4]) if len(v) > 4 else -1
                    if v_tid in blurred_vehicle_tids or v_tid == -1: continue

                    vx1, vy1, vx2, vy2 = map(int, v[:4])
                    vw, vh = vx2 - vx1, vy2 - vy1
                    if vw <= 0 or vh <= 0: continue
                    
                    px1, py1 = max(0, int(vx1 + vw * 0.25)), max(0, int(vy2 - vh * 0.30))
                    px2, py2 = min(w_img, int(vx2 - vw * 0.25)), min(h_img, int(vy2 - vh * 0.05))
                    pw, ph = px2 - px1, py2 - py1
                    if pw <= 0 or ph <= 0: continue
                    
                    roi = blur_img[py1:py2, px1:px2]
                    if roi.size > 0:
                        small = cv2.resize(roi, (max(1, pw // 12), max(1, ph // 12)), interpolation=cv2.INTER_LINEAR)
                        blur_img[py1:py2, px1:px2] = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
                        blurred_plates.append({
                            "box": [px1, py1, px2, py2],
                            "score": 1.0, "class_id": -1, "matched_vehicle_tid": v_tid
                        })

        except Exception as e:
            logger.error(f"번호판 모자이크 처리 실패: {e}")

        return (blur_img, blurred_plates) if return_meta else blur_img

    # 파라미터에 t_helmet 추가
    def apply_privacy_blur(self, frame, t_main, t_helmet=None, blur_face=True, blur_plate=True):
        privacy_meta = {
            "blur_face": bool(blur_face),
            "blur_plate": bool(blur_plate),
            "face": [],
            "plate": []
        }

        if frame is None:
            return frame, privacy_meta

        blurred_img = frame.copy()
        person_boxes = [t for t in t_main if int(t[6]) in [ID_G_PERSON, ID_PERSON_LOW, ID_REFLECTIVE_VEST]]
        vehicle_boxes = [t for t in t_main if int(t[6]) in TARGET_VEHICLES]

        if blur_face:
            # 헬멧 정보를 인자로 함께 넘김
            blurred_img, face_blurs = self.apply_face_blur(blurred_img, person_boxes, helmet_tracks=t_helmet, return_meta=True)
            privacy_meta["face"] = face_blurs

        if blur_plate:
            blurred_img, plate_blurs = self.apply_plate_blur(blurred_img, vehicle_boxes, return_meta=True)
            privacy_meta["plate"] = plate_blurs

        privacy_meta["applied"] = bool(privacy_meta["face"] or privacy_meta["plate"])
        return blurred_img, privacy_meta

    def _privacy_tracks_from_event_objects(self, objects):
        label_to_class = {
            "person": ID_G_PERSON,
            "low_body": ID_PERSON_LOW,
            "reflective_vest": ID_REFLECTIVE_VEST,
            "car": ID_G_CAR,
            "truck": ID_G_TRUCK,
            "vehicle": ID_G_CAR
        }
        allowed_classes = {ID_G_PERSON, ID_PERSON_LOW, ID_REFLECTIVE_VEST, *TARGET_VEHICLES}
        tracks = []

        for obj in objects or []:
            try:
                box = obj.get("box", [])
                if len(box) < 4:
                    continue

                class_id = obj.get("class_id")
                try:
                    class_id = int(class_id)
                except Exception:
                    class_id = None
                if class_id is None or class_id < 0:
                    class_id = label_to_class.get(str(obj.get("label", "")).lower())
                if class_id is None:
                    continue

                class_id = int(class_id)
                if class_id not in allowed_classes:
                    continue

                try:
                    obj_tid = int(obj.get("tid", -1))
                except Exception:
                    obj_tid = -1

                tracks.append([
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                    obj_tid,
                    float(obj.get("score", 0.95)),
                    class_id
                ])
            except Exception:
                continue

        return tracks

    def _serialize_detection(self, det):
        return {
            "box": [int(round(float(v))) for v in det[:4]],
            "score": round(float(det[4]), 4),
            "class_id": int(det[5])
        }

    def _serialize_track(self, track):
        return {
            "box": [int(round(float(v))) for v in track[:4]],
            "tid": int(track[4]),
            "score": round(float(track[5]), 4),
            "class_id": int(track[6])
        }

    def _serialize_event_objects(self, objects):
        safe_objects = []
        for obj in objects or []:
            safe_objects.append({
                "label": str(obj.get("label", "")),
                "box": [int(round(float(v))) for v in obj.get("box", [])],
                "score": round(float(obj.get("score", 0.0)), 4),
                "tid": int(obj.get("tid", -1)),
                "class_id": int(obj.get("class_id", -1))
            })
        return safe_objects

    def build_inference_log(self, fid, frame, d_main_res, d_helmet_res, t_main, t_helmet, alarms, new_events, d_signalman_res=None):
        h, w = frame.shape[:2] if frame is not None else (0, 0)
        kst = pytz.timezone('Asia/Seoul')
        if d_signalman_res is None:
            d_signalman_res = np.empty((0, 6))
        return {
            "ts": datetime.datetime.now(kst).isoformat(),
            "fid": int(fid),
            "cam_id": int(self.cam_id),
            "ip": str(self.ip),
            "frame_shape": [int(h), int(w)],
            "inference_mode": str(self.event_inference_mode),
            "events": list(self.events),
            "roi_poly": [[int(p[0]), int(p[1])] for p in (self.roi_poly or [])],
            "roi_lines": [[int(p[0]), int(p[1])] for p in (self.roi_lines or [])],
            "detections": {
                "main": [self._serialize_detection(d) for d in d_main_res],
                "helmet": [self._serialize_detection(d) for d in d_helmet_res],
                "signalman": [self._serialize_detection(d) for d in d_signalman_res]
            },
            "tracks": {
                "main": [self._serialize_track(t) for t in t_main],
                "helmet": [self._serialize_track(t) for t in t_helmet]
            },
            "alarms": {str(int(tid)): evt for tid, evt in (alarms or {}).items()},
            "new_events": [
                {
                    "event_id": str(ev.get("event_id", "")),
                    "ts": str(ev.get("ts", "")),
                    "event_name": str(ev.get("event_name", "")),
                    "objects": self._serialize_event_objects(ev.get("objects", [])),
                    "privacy_blur": to_json_safe(ev.get("privacy_blur", {})),
                    "decision_trace": to_json_safe(ev.get("decision_trace", {}))
                }
                for ev in (new_events or [])
            ]
        }

    def run_logic(self, fr, fid, d_main_res, d_helmet_res, d_signalman_res=None):
        if fr is None:
            return [], [], [], {}, []

        now_t = time.time()
        self.fps_queue.append(now_t)
        if len(self.fps_queue) > 1:
            time_diff = self.fps_queue[-1] - self.fps_queue[0]
            self.current_fps = len(self.fps_queue) / time_diff if time_diff > 0 else 0.0

        if d_signalman_res is None:
            d_signalman_res = np.empty((0, 6))

        self._update_alignment(fr)

        d_main_filtered = [d for d in d_main_res if int(d[5]) not in [ID_H_HELMET, ID_H_NO_HELMET]]
        t_main = self.trk_main.update(d_main_filtered)
        t_helmet = self.trk_helmet.update(d_helmet_res)
        t_signalman = self.trk_signalman.update(d_signalman_res)

        now = time.time()
        current_alarms = {}
        track_map_main = {int(t[4]): int(t[6]) for t in t_main}
        score_map_main = {int(t[4]): round(float(t[5]), 2) for t in t_main}
        track_map_helmet = {int(t[4]): int(t[6]) for t in t_helmet}
        score_map_helmet = {int(t[4]): round(float(t[5]), 2) for t in t_helmet}
        newly_triggered_events = []

        record_fr = None

        for ename, handler in self.handlers.items():
            # 1. 이벤트 핸들러에 전달할 인자 세팅
            if ename == "no_helmet":
                # [수정 핵심] 메인 객체(사람)가 분석 기준이 되어야 하므로 handler_tracks는 t_main이어야 합니다.
                kwargs = {'helmet_tracks': t_helmet, 'privacy_tracks': t_main}
                handler_tracks, handler_track_map, handler_score_map = t_main, track_map_main, score_map_main
            elif ename == "signal_vehicle":
                kwargs = {'signalman_tracks': t_signalman, 'privacy_tracks': t_main}
                handler_tracks, handler_track_map, handler_score_map = t_main, track_map_main, score_map_main
            else:
                kwargs = {'privacy_tracks': t_main}
                handler_tracks, handler_track_map, handler_score_map = t_main, track_map_main, score_map_main

            try:
                # 핸들러 실행 (내부에서 과거 시점의 privacy_tracks를 저장하고 ev에 담아 반환해야 함)
                triggered = handler.process(handler_tracks, handler_track_map, None, fr, fid, **kwargs)
            except Exception as e:
                logger.error(f"🚨 [CAM:{self.ip}] {ename} 핸들러 처리 중 예외 발생: {e}\n{traceback.format_exc()}")
                continue

            for ev in triggered:
                tid = ev['tid']
                bbox = ev['bbox']
                ev_frame = ev.get('frame') if ev.get('frame') is not None else fr
                cooldown = SYS_CFG.get("event_config", {}).get(ename, {}).get("cooldown_sec", 600)

                actual_score = handler_score_map.get(tid, score_map_main.get(tid, 0.95))
                objects_meta = ev.get('objects', [{'label': ename, 'box': [int(x) for x in bbox], 'score': actual_score, 'tid': tid}])
                
                # ---------------------------------------------------------
                # [핵심 로직] 과거 프레임의 모든 객체(privacy_tracks) 복원
                # ---------------------------------------------------------
                event_frame_privacy_tracks = ev.get('privacy_tracks')
                privacy_reference_fid = ev.get('privacy_fid', ev.get('fid', fid))
                
                # 타임캡슐(privacy_tracks)이 온전히 반환되었다면 그것을 사용 (과거의 전체 객체)
                if event_frame_privacy_tracks is not None and len(event_frame_privacy_tracks) > 0:
                    privacy_reference_tracks = event_frame_privacy_tracks
                    privacy_reference_tracks_label = "event_frame_tracks"
                else:
                    # [수정] 타임캡슐이 없을 경우 위반 객체로 축소하지 않고 무조건 현재 프레임의 전체 객체(t_main)를 마스킹 대상으로 삼음
                    privacy_reference_tracks = t_main
                    privacy_reference_tracks_label = "current_tracks"

                decision_trace = to_json_safe(ev.get('decision_trace', {
                    'detector': handler.__class__.__name__,
                    'reason': 'event_triggered_without_detail'
                }))

                if ename not in self.alerted[tid] and (now - self.last_evt_t.get(ename, 0) >= cooldown):
                    event_ts_dt = now_kst()
                    event_ts = event_ts_dt.isoformat()
                    event_fid = int(ev.get('fid', fid))
                    event_id = make_event_id(self.cam_id, self.ip, ename, tid, event_fid, event_ts_dt)
                    
                    objs_log_str = " | ".join([f"{o['label']}({o['score']:.2f}): {o['box']}" for o in objects_meta])
                    log_msg = (
                        f"🔥 [EVENT TRIGGERED] event_id={event_id} CAM:{self.cam_id}({self.ip}) | Event:{ename} | "
                        f"TermID:{SYS_CFG.get('terminal_id', '99999')} | TID:{tid} | FID:{event_fid} | FPS:{self.current_fps:.1f} | "
                        f"Reason:{decision_trace.get('reason', '-')} | "
                        f"Objects -> {objs_log_str}"
                    )
                    logger.warning(log_msg)

                    blur_face_option = SYS_CFG.get("event_config", {}).get(ename, {}).get("blur_face", True)
                    blur_plate_option = SYS_CFG.get("event_config", {}).get(ename, {}).get("blur_plate", True)

                    # 완벽하게 동기화된 트랙(privacy_reference_tracks) 및 t_helmet 으로 다단계 블러 적용
                    saved_img, privacy_blur_meta = self.apply_privacy_blur(
                        ev_frame, 
                        privacy_reference_tracks,
                        t_helmet=t_helmet,
                        blur_face=blur_face_option,
                        blur_plate=blur_plate_option
                    )
                    
                    # 블러 처리 메타데이터 로깅 (추적용)
                    privacy_blur_meta["scope"] = "event_snapshot"
                    privacy_blur_meta["reference_tracks"] = privacy_reference_tracks_label
                    try:
                        privacy_blur_meta["reference_fid"] = int(privacy_reference_fid)
                    except Exception:
                        privacy_blur_meta["reference_fid"] = None
                        
                    logger.info(
                        f"[PRIVACY BLUR] event_id={event_id} cam={self.cam_id} event={ename} "
                        f"face_enabled={blur_face_option} plate_enabled={blur_plate_option} "
                        f"face_count={len(privacy_blur_meta.get('face', []))} "
                        f"plate_count={len(privacy_blur_meta.get('plate', []))} "
                        f"reference_tracks={privacy_blur_meta.get('reference_tracks', '-')}"
                    )

                    event_trajectories = {}
                    for obj in objects_meta:
                        obj_tid = obj.get('tid')
                        if obj_tid in self.trk_main.tracks:
                            event_trajectories[obj_tid] = list(self.trk_main.tracks[obj_tid]['history'])
                        elif obj_tid in self.trk_helmet.tracks:
                            event_trajectories[obj_tid] = list(self.trk_helmet.tracks[obj_tid]['history'])

                    auth_tokens = ev.get('auth_tokens', None)
                    event_meta = {
                        'event_id': event_id,
                        'ts': event_ts,
                        'event_name': ename,
                        'terminal_id': str(SYS_CFG.get("terminal_id", "99999")),
                        'cctv_id': int(self.cam_id),
                        'ip': str(self.ip),
                        'tid': int(tid),
                        'bbox': int_box(bbox),
                        'fid': event_fid,
                        'objects': self._serialize_event_objects(objects_meta),
                        'trajectories': to_json_safe(event_trajectories),
                        'auth_tokens': to_json_safe(auth_tokens or []),
                        'privacy_blur': to_json_safe(privacy_blur_meta),
                        'decision_trace': decision_trace
                    }

                    evidence_paths = save_event_image_with_mark(
                        frame=saved_img, ip=self.ip, event_type=ename, bbox=bbox, tid=tid,
                        terminal_id=SYS_CFG.get("terminal_id", "99999"), cctv_id=self.cam_id,
                        objects_meta=objects_meta, trajectories=event_trajectories,
                        auth_tokens=auth_tokens,
                        event_id=event_id,
                        event_ts=event_ts
                    )
                    if evidence_paths:
                        event_meta.update(evidence_paths)

                    self.recorder.trigger(
                        ename,
                        objects_meta=objects_meta,
                        event_meta=event_meta,
                        current_fps=SYS_CFG.get("video_decode", {}).get("fps_limit", 15.0)
                    )
                    self.alerted[tid].add(ename)
                    self.last_evt_t[ename] = now

                    newly_triggered_events.append({
                        'event_id': event_id,
                        'ts': event_ts,
                        'event_name': ename,
                        'objects': objects_meta,
                        'privacy_blur': privacy_blur_meta,
                        'decision_trace': decision_trace
                    })

                else:
                    cooldown_remaining = max(0.0, cooldown - (now - self.last_evt_t.get(ename, 0)))
                    logger.debug(
                        f"[EVENT SUPPRESSED] cam={self.cam_id} ip={self.ip} event={ename} tid={tid} "
                        f"fid={int(ev.get('fid', fid))} alerted={ename in self.alerted[tid]} "
                        f"cooldown_remaining={cooldown_remaining:.1f}s"
                    )

                current_alarms[tid] = ename

        alarm_duration = SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0)
        for tid, ename in current_alarms.items():
            self.visual_alarms[tid] = {'evt': ename, 'expire': now + alarm_duration}

        for tid in list(self.visual_alarms.keys()):
            if now > self.visual_alarms[tid]['expire']:
                del self.visual_alarms[tid]

        if record_fr is not None:
            # 1. 메인 객체 렌더링 (사람, 차량 등)
            for t in t_main:
                t_id = int(t[4])
                is_alarmed = t_id in current_alarms
                color = (0, 0, 255) if is_alarmed else (0, 255, 0)
                thickness = 2 if is_alarmed else 1 # 알람 시 테두리 약간 강조
                bx1, by1, bx2, by2 = map(int, t[:4])
                cv2.rectangle(record_fr, (bx1, by1), (bx2, by2), color, thickness)
                if t_id in self.trk_main.tracks:
                    hist = list(self.trk_main.tracks[t_id]['history'])
                    if len(hist) > 1:
                        cv2.polylines(record_fr, [np.array(hist, np.int32)], False, color, thickness, cv2.LINE_AA)

            # 2. [추가] 안전모 객체 로컬 영상 렌더링
            if "no_helmet" in self.events:
                for t in t_helmet:
                    cls_id = int(t[6])
                    color = (0, 0, 255) if cls_id == ID_H_NO_HELMET else (0, 255, 0)
                    bx1, by1, bx2, by2 = map(int, t[:4])
                    cv2.rectangle(record_fr, (bx1, by1), (bx2, by2), color, 1)

            # 3. [추가] 신호수 객체 로컬 영상 렌더링
            if "signal_vehicle" in self.events:
                for t in t_signalman:
                    bx1, by1, bx2, by2 = map(int, t[:4])
                    cv2.rectangle(record_fr, (bx1, by1), (bx2, by2), (0, 255, 255), 1)

        return t_main, t_helmet, t_signalman, {t: info['evt'] for t, info in self.visual_alarms.items()}, newly_triggered_events

    def draw(self, fr, t_main, t_helmet, t_signalman, alarms, connected=True):
        if fr is None or not connected:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
            cv2.putText(blank, self.ip, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            return blank

        h_frame, w_frame = fr.shape[:2]

        if len(alarms) > 0:
            cv2.rectangle(fr, (0, 0), (w_frame, h_frame), (0, 0, 255), 20)

        if len(self.roi_poly) > 2:
            cv2.polylines(fr, [np.array(self.roi_poly, np.int32)], True, (0, 255, 255), 2)
        if self.roi_lines:
            for i in range(0, len(self.roi_lines), 2):
                if i + 1 < len(self.roi_lines):
                    cv2.line(fr, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0, 0, 255), 2)

        allowed_classes = set()
        if "signal_vehicle" in self.events:
            allowed_classes.add(ID_G_TRUCK)
        if "no_helmet" in self.events or "conveyor_crossing" in self.events or "intrusion" in self.events:
            allowed_classes.update([ID_G_PERSON, ID_PERSON_LOW])
        if "illegal_parking" in self.events or "intrusion" in self.events:
            allowed_classes.update(TARGET_VEHICLES)

        for t in t_main:
            tid = int(t[4])
            cls_id = int(t[6])
            is_alarmed = tid in alarms

            if not is_alarmed and cls_id not in allowed_classes:
                continue

            color = (0, 0, 255) if is_alarmed else (0, 255, 0)
            thickness = 1 if is_alarmed else 1

            if tid in self.trk_main.tracks:
                hist = list(self.trk_main.tracks[tid]['history'])
                if len(hist) > 1:
                    cv2.polylines(fr, [np.array(hist, np.int32)], False, color, 1, cv2.LINE_AA)

            if cls_id == ID_G_PERSON: label = f"Person [{tid}]"
            elif cls_id == ID_PERSON_LOW: label, color = f"LowBody [{tid}]", (0, 150, 0)
            elif cls_id == ID_G_CAR: label, color = f"Car [{tid}]", (255, 100, 0)
            elif cls_id == ID_G_TRUCK: label, color = f"LineTruck [{tid}]", (255, 100, 0)
            else: label = f"OBJ [{tid}]"

            if is_alarmed:
                color = (0, 0, 255)
                label = f"ALARM: {label}"

            cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if "signal_vehicle" in self.events:
            for t in t_signalman:
                tid = int(t[4])
                color, thickness = (0, 255, 255), 1
                cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
                cv2.putText(fr, f"Signalman [{tid}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if "no_helmet" in self.events:
            for t in t_helmet:
                tid = int(t[4])
                cls_id = int(t[6]) 

                if cls_id == ID_H_HELMET:
                    color = (0, 255, 0) 
                    label = f"Helmet [{tid}]"
                    thickness = 1
                elif cls_id == ID_H_NO_HELMET:
                    color = (0, 0, 255) 
                    label = f"Head [{tid}]"
                    # [수정] 해당 카메라에 안전모 알람이 하나라도 켜져 있으면 미착용 머리를 강하게 강조
                    is_alarming = "no_helmet" in alarms.values()
                    thickness = 3 if is_alarming else 1
                elif cls_id == ID_G_PERSON:
                    color = (0, 255, 0)
                    label = f"Person(H) [{tid}]"
                    thickness = 1
                elif cls_id == ID_PERSON_LOW:
                    color = (0, 150, 0)
                    label = f"LowBody(H) [{tid}]"
                    thickness = 1
                else:
                    color = (0, 165, 255)
                    label = f"HelmetObj {cls_id} [{tid}]"
                    thickness = 1

                cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
                cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        #cv2.rectangle(fr, (0, 0), (115, 40), (0, 0, 0), -1)
        cv2.putText(fr, f"CAM {self.cam_id}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        fps_color = (0, 0, 0) if self.current_fps >= 10.0 else (0, 0, 255)
        cv2.putText(fr, f"AI FPS: {self.current_fps:.1f}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, fps_color, 1)

        active_alarms = set(alarms.values())

        menu_height = len(self.events) * 20 + 10
        #overlay = fr.copy()
        #cv2.rectangle(overlay, (w_frame - 150, 0), (w_frame, menu_height), (0, 0, 0), -1)
        #cv2.addWeighted(overlay, 0.5, fr, 0.5, 0, fr)

        y_pos = 15
        for evt in self.events:
            if evt == "roi_change" or evt == getattr(sys.modules[__name__], 'ROI_CHANGE_EVENT', 'roi_change'):
                continue

            display_name = EVENT_REGISTRY[evt].gui_name if evt in EVENT_REGISTRY else evt.upper()
            color = (0, 0, 255) if evt in active_alarms else (0, 255, 0)
            prefix = "[!] " if evt in active_alarms else " -  "
            cv2.putText(fr, f"{prefix}{display_name}", (w_frame - 145, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_pos += 20

        if "signal_vehicle" in self.events and "signal_vehicle" in self.handlers:
            sv_handler = self.handlers["signal_vehicle"]
            current_time = time.time()
            display_items = []
            confirmed_line_truck_ids = set(getattr(sv_handler, "confirmed_line_truck_ids", set()))

            for a_tid, auth_t in list(sv_handler.last_auth_time.items()):
                remain = sv_handler.auth_grace_sec - (current_time - auth_t)
                if remain > 0:
                    auth_time_str = datetime.datetime.fromtimestamp(auth_t).strftime('%H:%M:%S')
                    is_visible = a_tid in confirmed_line_truck_ids
                    status_text = "Tracking" if is_visible else "Hidden"
                    sig_id = sv_handler.last_auth_signalman.get(a_tid, "Unknown")

                    sort_score = auth_t
                    if a_tid in alarms: sort_score = float('inf')

                    display_items.append({
                        'tid': a_tid, 'sort_val': sort_score,
                        'line1': f"Auth: {auth_time_str} | Remain: {remain:.1f}s ({status_text})",
                        'line2': f"Auth by: Signalman [{sig_id}]",
                        'color': (0, 255, 0) if is_visible else (0, 180, 0)
                    })

            auth_tids = [item['tid'] for item in display_items]

            current_trucks = set([int(t[4]) for t in t_main if int(t[6]) == ID_G_TRUCK and int(t[4]) in confirmed_line_truck_ids])
            alarmed_tids = set([tid for tid, evt in alarms.items() if evt == "signal_vehicle"])
            target_tids = current_trucks.union(alarmed_tids)

            for t_tid in target_tids:
                if t_tid not in auth_tids:
                    is_alarming = t_tid in alarms
                    base_sort = float('inf') if is_alarming else 0

                    if is_alarming:
                        display_items.append({
                            'tid': t_tid, 'sort_val': base_sort + 3,
                            'line1': "Status: UNAUTH (ALARM)",
                            'line2': "Reason: Moving without Signalman",
                            'color': (0, 0, 255)
                        })
                    elif t_tid in sv_handler.presence_start_time:
                        wait_sec = current_time - sv_handler.presence_start_time[t_tid]
                        display_items.append({
                            'tid': t_tid, 'sort_val': base_sort + 2,
                            'line1': f"Wait: {wait_sec:.1f}s / {sv_handler.presence_threshold_sec}s",
                            'line2': "Authenticating Signalman...",
                            'color': (0, 165, 255)
                        })
                    elif t_tid in sv_handler.is_parked:
                        display_items.append({
                            'tid': t_tid, 'sort_val': base_sort + 1,
                            'line1': "Status: PARKED (Monitoring)",
                            'line2': "Awaiting Signalman",
                            'color': (255, 150, 0)
                        })
                    else:
                        dwell_sec = current_time - sv_handler.stationary_start_time.get(t_tid, current_time)
                        display_items.append({
                            'tid': t_tid, 'sort_val': -1,
                            'line1': f"Status: ARRIVING (Stop: {dwell_sec:.0f}s / {sv_handler.parked_threshold_sec}s)",
                            'line2': "Ignoring Move (Parking in progress)",
                            'color': (180, 180, 180)
                        })

            display_items.sort(key=lambda x: x['sort_val'], reverse=True)
            display_items = display_items[:1]

            box_w, box_h = 340, 35 + max(1, len(display_items)) * 40
            x_start, y_start = w_frame - box_w - 20, h_frame - box_h - 20

            overlay2 = fr.copy()
            cv2.rectangle(overlay2, (x_start, y_start), (x_start + box_w, y_start + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, fr, 0.4, 0, fr)
            cv2.putText(fr, "Signalman Auth", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if not display_items:
                cv2.putText(fr, "No active tokens", (x_start + 10, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                for i, item in enumerate(display_items):
                    cv2.putText(fr, item['line1'], (x_start + 10, y_start + 45 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, item['color'], 1)
                    cv2.putText(fr, item['line2'], (x_start + 10, y_start + 65 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, item['color'], 1)

        return fr
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
        self.url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/health"
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
    def _extract_roi_norm_updates(cls, roi_json):
        payload = cls._decode_jsonish(roi_json)
        if not isinstance(payload, dict):
            return None

        candidates = [payload]
        for nested_key in ("roiInfo", "roi_info"):
            if nested_key not in payload:
                continue
            nested = cls._decode_jsonish(payload.get(nested_key))
            if isinstance(nested, dict):
                candidates.append(nested)

        for candidate in candidates:
            updates = {}
            if "roi_poly_norm" in candidate:
                updates["roi_poly_norm"] = cls._coerce_roi_norm_points(candidate.get("roi_poly_norm"))
            if "roi_lines_norm" in candidate:
                updates["roi_lines_norm"] = cls._coerce_roi_norm_points(candidate.get("roi_lines_norm"))
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
    parser.add_argument('--gui', action='store_true', help="GUI 모드를 활성화하여 모니터에 영상을 렌더링합니다.")
    args = parser.parse_args()

    is_gui_mode = args.gui

    if not is_gui_mode:
        logger.info("[시스템 모드] CLI (Headless) 모드로 동작합니다. (렌더링 생략으로 CPU 부하 최소화)")
    else:
        logger.info("[시스템 모드] GUI 모드로 동작합니다. (--gui 플래그 활성화됨)")

    global DEBUG_MODE
    logger.info("[System] 단일 스크립트 기반 YOLOv8 모듈화 시스템 초기화 완료")

    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    if not rtsp_list:
        logger.error(f"카메라 목록 파일({CAMERA_LIST_FILE})을 확인하십시오.")
        return

    config_file = os.path.join(PROJECT_ROOT, "cameras.json")
    camera_configs = {}

    debug_ans = guarded_input(">> CLI 디버그 출력을 활성화하시겠습니까? (파일 로그는 항상 상세히 기록됩니다) [y/N]: ").strip().lower()
    DEBUG_MODE = True if debug_ans == 'y' else False
    
    _runtime_log_cfg = SYS_CFG.get("logging", {})
    # [수정] 파일 로그는 무조건 가장 상세한 DEBUG 레벨 고정, 콘솔 출력만 사용자 선택에 따름
    _debug_file_level = logging.DEBUG 
    _debug_console_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    logger.setLevel(logging.DEBUG)
    queue_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(_debug_file_level)
    stream_handler.setLevel(_debug_console_level)
    
    if DEBUG_MODE:
        logger.debug("디버그 모드가 활성화되었습니다. 콘솔 상세 로깅이 시작됩니다.")

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                camera_configs = json.load(f)
        except Exception as e:
            logger.error(f"cameras.json 로드 실패: {e}")
            pass

        reset_ans = guarded_input(">> 기존 설정(cameras.json)을 무시하고 ROI 및 이벤트를 재설정하시겠습니까? [y/N]: ").strip().lower()
        if reset_ans == 'y':
            logger.info("기존 설정을 무시하고 터미널 마법사를 실행합니다.")
            camera_configs = run_wizard_batch_mode(rtsp_list, camera_configs)
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(camera_configs, f, indent=4)
            except: pass
    else:
        logger.warning("설정 파일(cameras.json)이 없어 터미널 마법사를 실행합니다.")
        camera_configs = run_wizard_batch_mode(rtsp_list, {})
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(camera_configs, f, indent=4)
        except: pass

    models_cfg = SYS_CFG.get("models", {})
    inference_mode = str(SYS_CFG.get("INFERENCE_MODE", "auto")).strip().lower()
    event_inference_mode = "main"
    
    # [수정] V2, V3 모델 분리 로드
    main_v2_model_path = resolve_model_path(models_cfg.get("MAIN_V2", "hanjin_cctv_v2.dxnn"))
    main_v3_model_path = resolve_model_path(models_cfg.get("MAIN_V3", "hanjin_cctv_v3.dxnn"))

    try:
        logger.info(f"DeepX 모델을 VPU 메모리로 할당 중... (event inference: {event_inference_mode})")

        # 풀사이즈 배분 (V2=2, V3=1)
        main_v2_pool_size = get_model_engine_pool_size("MAIN_V2", default=2)
        main_v3_pool_size = get_model_engine_pool_size("MAIN_V3", default=1)
        helmet_pool_size = min(2, get_model_engine_pool_size("HELMET", default=1))
        
        d_main_v2 = YoLoDeepX(
            main_v2_model_path,
            output_format=get_model_output_format("MAIN_V2"),
            pool_size=main_v2_pool_size
        )
        d_main_v3 = YoLoDeepX(
            main_v3_model_path,
            output_format=get_model_output_format("MAIN_V3"),
            pool_size=main_v3_pool_size
        )
        d_helmet = YoLoDeepX(
            resolve_model_path(models_cfg.get("HELMET")),
            output_format=get_model_output_format("HELMET"),
            pool_size=helmet_pool_size
        )
        
        # [복구 및 활성화] d_face 및 d_plate 모델 로드 (좌표 매칭 오류 픽스)
        face_fmt = get_model_output_format("FACE")
        if face_fmt in ["auto", "yolo"]: 
            face_fmt = "yolo_xyxy" # 사용 중이신 얼굴 모델의 출력 형식에 따라 xyxy 또는 yolo_tlwh 적용
            
        d_face = YoLoDeepX(
            resolve_model_path(models_cfg.get("FACE", "yolov8m-face.dxnn")),
            output_format=face_fmt,
            pool_size=get_model_engine_pool_size("FACE", default=1)
        )
        
        plate_fmt = get_model_output_format("PLATE")
        if plate_fmt in ["auto", "yolo"]:
            plate_fmt = "yolo_xyxy"
            
        d_plate = YoLoDeepX(
            resolve_model_path(models_cfg.get("PLATE", "license_plate_detector_v2.dxnn")),
            output_format=plate_fmt,
            pool_size=get_model_engine_pool_size("PLATE", default=1)
        )
        d_signalman = None
    except Exception as e:
        logger.error(f"모델 로드 실패. 경로를 확인하십시오: {e}")
        return

    cams = []
    for i, rtsp in enumerate(rtsp_list):
        ip = extract_ip(rtsp)
        conf = camera_configs.get(ip)

        if not conf or not conf.get('events'):
            continue
        conf['url'] = rtsp
        # [수정] d_main_v2, d_main_v3 모두 주입
        cams.append(Camera(
            ip, conf, d_main_v2, d_main_v3, d_helmet, d_face, d_signalman, d_plate,
            cam_id=i+1,
            event_inference_mode=event_inference_mode
        ))
        logger.info(
            f"[CAMERA LOADED] cam={i+1} ip={ip} events={','.join(conf.get('events', [])) or '-'} "
            f"roi_poly_points={len(conf.get('roi_poly_norm', []) or [])} "
            f"roi_line_points={len(conf.get('roi_lines_norm', []) or [])}"
        )

    if not cams:
        logger.error("[SYSTEM STARTUP] no active cameras loaded; check cameras.csv and cameras.json events.")
        return

    perf_cfg = SYS_CFG.get("system_performance", {})
    target_fps = float(perf_cfg.get("target_fps", 10.0))
    dynamic_cpu_adjust = bool(perf_cfg.get("dynamic_cpu_adjust_enabled", False))
    sys_target_fps = target_fps

    main_conf = SYS_CFG["model_confidences"].get("MAIN_V2", 0.6)
    helmet_conf = SYS_CFG["model_confidences"].get("HELMET", 0.55)
    person_conf = SYS_CFG.get("model_confidences", {}).get("PERSON", 0.5) 
    signalman_conf = SYS_CFG.get("model_confidences", {}).get("SIGNALMAN", person_conf)
    
    loop_count = 0
    fps_calc_interval = 30
    last_fps_time = time.time()
    cpu_usage = 0.0
    
    try:
        current_fps_log_interval_sec = max(1.0, float(SYS_CFG.get("CURRENT_FPS_LOG_INTERVAL_SEC", 10.0)))
    except Exception:
        current_fps_log_interval_sec = 10.0
    current_fps_last_print = {}

    terminal_id = SYS_CFG.get("terminal_id", "99999")
    software_version = "v1.1.0"
    
    log_disk_health([("event_root", EVENT_ROOT_DIR), ("log_dir", LOG_DIR)])
    global HEALTH_DAEMON
    health_daemon = HealthCheckDaemon(
        terminal_id=terminal_id,
        version=software_version,
        interval_sec=60,
        cams=cams,
        config_file=config_file
    )
    HEALTH_DAEMON = health_daemon

    last_config_mtime = 0
    if os.path.exists(config_file):
        last_config_mtime = os.path.getmtime(config_file)

    RAM_DISK_DIR = "/dev/shm/cctv_frames"
    if not os.path.exists(RAM_DISK_DIR):
        try: os.makedirs(RAM_DISK_DIR, exist_ok=True)
        except: RAM_DISK_DIR = "./web_frames"

    event_frame_save_delay_sec = float(SYS_CFG.get("EVENT_FRAME_SAVE_DELAY_SEC", 10.0))
    configured_event_frame_save_max_count = int(SYS_CFG.get("EVENT_FRAME_SAVE_MAX_COUNT", 0) or 0)
    event_frame_save_fps = max(1.0, float(SYS_CFG.get("REC_FPS", 3)))
    if configured_event_frame_save_max_count > 0:
        event_frame_save_max_count = configured_event_frame_save_max_count
    else:
        event_frame_save_max_count = int(math.ceil(event_frame_save_delay_sec * event_frame_save_fps * 1.5))
    event_frame_save_max_count = max(1, int(event_frame_save_max_count))
    event_save_queues = {c.ip: deque(maxlen=event_frame_save_max_count) for c in cams}
    last_event_times = {c.ip: 0.0 for c in cams}
    
    output_retention_days = float(SYS_CFG.get("OUTPUT_RETENTION_DAYS", 14))
    output_cleanup_interval_sec = float(SYS_CFG.get("OUTPUT_CLEANUP_INTERVAL_SEC", 86400))
    last_output_cleanup_time = time.time()
    run_output_retention_cleanup(output_retention_days)

    def run_camera_inference(cam, fr):
        # 안전모 이벤트는 MAIN_V2(사람 검출)와 HELMET(머리 검출) 연계가 필수이므로 반드시 포함되어야 합니다.
        active_detection_events = [
            evt for evt in cam.events
            if evt not in (
                getattr(sys.modules[__name__], 'ROI_CHANGE_EVENT', 'roi_change'), 
                getattr(sys.modules[__name__], 'ROI_CHANGE_APPLY_EVENT', 'roi_change_apply')
            )
        ]
        t_main_input = np.empty((0, 6))
        d_signalman_res = np.empty((0, 6))

        h, w = fr.shape[:2]
        max_area_threshold = (h * w) * 0.5

        if active_detection_events:
            base_conf = min(main_conf, person_conf, signalman_conf)
            
            # [수정] 하이브리드 라우팅 (V3는 신호수 관련, V2는 그 외)
            v3_required = "signal_vehicle" in active_detection_events
            v2_required = any(evt in active_detection_events for evt in ["intrusion", "illegal_parking", "conveyor_crossing", "no_helmet"])

            raw_dets_v2 = cam.det_main_v2.infer(fr, conf_override=base_conf) if v2_required else []
            raw_dets_v3 = cam.det_main_v3.infer(fr, conf_override=base_conf) if v3_required else []

            # [수정] 클래스 필터링(BBox 겹침 원천 차단)
            filtered_v2 = [d for d in raw_dets_v2 if int(d[5]) in [ID_G_PERSON, ID_PERSON_LOW, ID_G_CAR]] if len(raw_dets_v2) > 0 else []
            filtered_v3 = [d for d in raw_dets_v3 if int(d[5]) in [ID_G_TRUCK, ID_SIGNALMAN, ID_RAINCOAT, ID_SIGNALFLAG]] if len(raw_dets_v3) > 0 else []

            # 두 결과를 하나로 병합
            raw_dets = list(filtered_v2) + list(filtered_v3)

            t_main_input, _, d_signalman_res = split_unified_event_detections(
                raw_dets,
                active_detection_events,
                main_conf=main_conf,
                person_conf=person_conf,
                helmet_conf=helmet_conf,
                signalman_conf=signalman_conf,
                max_area_threshold=max_area_threshold 
            )

        d_helmet_res = np.empty((0, 6))
        
        if "no_helmet" in cam.events:
            has_person = False
            for d in t_main_input:
                if int(d[5]) in [ID_G_PERSON, ID_PERSON_LOW, ID_REFLECTIVE_VEST]:
                    has_person = True
                    break
                    
            if has_person:
                d_helmet_res = cam.det_helmet.infer(fr, conf_override=helmet_conf)

        return t_main_input, d_helmet_res, d_signalman_res

    # [추가] 메인 스레드와 워커 스레드 간 목표 FPS를 안전하게 공유하기 위한 상태 객체
    system_runtime_state = {"target_fps": sys_target_fps}
    
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
            self.last_inference_time = 0.0 # [추가] FPS 쓰로틀링용 타이머

        def run(self):
            while self.running:
                item = self.frame_buffer.get()
                
                # 새로운 프레임이 없다면 즉시 GIL을 해제(Release)하여 다른 스레드에 양보
                if item is None:
                    time.sleep(0.005)
                    continue

                # [핵심 추가] FPS 강제 제한 로직 (프레임 버리기)
                now = time.time()
                current_target = max(1.0, system_runtime_state.get("target_fps", 10.0))
                delay_required = 1.0 / current_target
                
                if (now - self.last_inference_time) < delay_required:
                    # 지정된 FPS(예: 10 FPS)보다 빠르게 들어온 프레임은 무시(Drop)하여 NPU 연산량 방어
                    continue
                
                self.last_inference_time = now
                fr, fid, connected = item

                if not connected or fr is None or not self.cam.events:
                    self.result_buffer.put((fr, fid, connected, [], [], [], {}, [], None))
                    continue

                try:
                    # 추론과 로직을 한 워커에서 순차 처리하여 스레드 통신 오버헤드 제거
                    t_main_input, d_helmet_res, d_signalman_res = run_camera_inference(self.cam, fr)
                    t_main, t_helmet, t_signalman, alarms, new_events = self.cam.run_logic(fr, fid, t_main_input, d_helmet_res, d_signalman_res)
                    infer_meta = self.cam.build_inference_log(fid, fr, t_main_input, d_helmet_res, t_main, t_helmet, alarms, new_events, d_signalman_res=d_signalman_res)
                    
                    self.result_buffer.put((fr, fid, connected, t_main, t_helmet, t_signalman, alarms, new_events, infer_meta))
                except Exception as e:
                    logger.error(f"[Worker Error] CAM {self.cam.cam_id}: {e}\n{traceback.format_exc()}")
                    self.result_buffer.put((fr, fid, connected, [], [], [], {}, [], None))

    camera_workers = []
    last_rendered_frames = {}

    for c in cams:
        worker = CameraWorker(c)
        worker.start()
        camera_workers.append(worker)
        
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "WAITING...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        last_rendered_frames[c.ip] = blank

    # [수정] 전역 타이머 삭제 및 개별 타이머/활성시간 딕셔너리 도입
    last_roi_snapshot_times = {c.ip: time.time() - 3590.0 for c in cams}
    last_worker_active_times = {c.ip: time.time() for c in cams}
    ROI_SNAPSHOT_INTERVAL_SEC = 3600.0

    target_fps = sys_target_fps
    dynamic_delay = 1.0 / target_fps
    last_processed_fids = {}
    
    try:
        psutil.cpu_percent(interval=None)

        while True:
            start_time = time.time()

            if output_cleanup_interval_sec > 0 and (start_time - last_output_cleanup_time) >= output_cleanup_interval_sec:
                run_output_retention_cleanup(output_retention_days)
                last_output_cleanup_time = start_time

            if loop_count > 0 and loop_count % 45 == 0 and os.path.exists(config_file):
                current_mtime = os.path.getmtime(config_file)
                if current_mtime > last_config_mtime:
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            new_configs = json.load(f)
                        for c in cams:
                            if c.ip in new_configs:
                                c.update_config(new_configs[c.ip])
                        last_config_mtime = current_mtime
                    except Exception as e:
                        pass

            loop_count += 1

            if loop_count % fps_calc_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - last_fps_time
                actual_fps = fps_calc_interval / elapsed_time
                cpu_usage = psutil.cpu_percent(interval=None)

                if dynamic_cpu_adjust:
                    if cpu_usage < 75:
                        target_fps = min(sys_target_fps, target_fps + 1.0)
                    elif cpu_usage > 90:
                        target_fps = max(8.0, target_fps - 1.0)
                else:
                    target_fps = sys_target_fps
                
                # [추가] 계산된 타겟 FPS를 상태 객체에 담아 워커 스레드들과 동기화
                system_runtime_state["target_fps"] = target_fps
                
                dynamic_delay = 1.0 / target_fps
                last_fps_time = current_time

            if loop_count % 300 == 0:
                gc.collect()
                mem_usage = psutil.virtual_memory().percent
                q_size = IMAGE_SAVER_POOL._work_queue.qsize() if hasattr(IMAGE_SAVER_POOL, '_work_queue') else 0
                log_disk_health([("event_root", EVENT_ROOT_DIR), ("log_dir", LOG_DIR)])

            # ---------------------------------------------------------
            # Stage 1 (Producer): 프레임 캡처 및 버퍼 공급
            # ---------------------------------------------------------
            for idx, worker in enumerate(camera_workers):
                fr, fid, connected = worker.cam.process_frame()
                last_fid = last_processed_fids.get(worker.cam.ip, -1)
                if fid != last_fid:
                    worker.frame_buffer.put((fr, fid, connected))
                    last_processed_fids[worker.cam.ip] = fid

            # ---------------------------------------------------------
            # Stage 2 (Consumer): 최신 결과 회수, 마스킹 스냅샷, 및 최적화 렌더링
            # ---------------------------------------------------------
            final_imgs = []
            now_time = time.time()
            
            roi_snapshot_refresh_cctv_ids = get_terminal_roi_snapshot_refresh_cctv_ids()
            refreshed_cctv_ids = set()
            
            for idx, worker in enumerate(camera_workers):
                c = worker.cam
                res = worker.result_buffer.get()
                
                if res is None:
                    # [수정] 좀비 스레드(Deadlock) 감지 및 핫스왑 복구
                    if (now_time - last_worker_active_times.get(c.ip, now_time)) > WATCHDOG_TIMEOUT:
                        logger.error(f"🚨 [WATCHDOG] CAM:{c.cam_id}({c.ip}) 스레드 데드락(Zombie) 감지! 핫스왑(Hot-Swap) 복구를 진행합니다.")
                        
                        worker.running = False
                        c.reader.running = False
                        c.recorder.running = False
                        try:
                            worker.join(timeout=1.0)
                        except Exception:
                            pass
                            
                        conf = camera_configs.get(c.ip, c.conf)
                        # [수정 핵심] d_main을 최신 아키텍처에 맞게 d_main_v2, d_main_v3로 변경
                        new_cam = Camera(
                            c.ip, conf, d_main_v2, d_main_v3, d_helmet, d_face, d_signalman, d_plate,
                            cam_id=c.cam_id,
                            event_inference_mode=event_inference_mode
                        )
                        new_worker = CameraWorker(new_cam)
                        new_worker.start()
                        
                        camera_workers[idx] = new_worker
                        for i_cam, old_cam in enumerate(cams):
                            if old_cam.ip == c.ip:
                                cams[i_cam] = new_cam
                                break
                                
                        if HEALTH_DAEMON is not None:
                            HEALTH_DAEMON.cams = cams
                            
                        last_worker_active_times[c.ip] = time.time()
                        last_processed_fids[c.ip] = -1
                        logger.info(f"✅ [WATCHDOG] CAM:{c.cam_id}({c.ip}) 핫스왑 복구 완료. 모니터링을 재개합니다.")

                    if is_gui_mode: final_imgs.append(last_rendered_frames[c.ip])
                    continue

                # 정상 수신 시 활성 시간 갱신
                last_worker_active_times[c.ip] = now_time
                fr, fid, connected, t_main, t_helmet, t_signalman, alarms, new_events, infer_meta = res

                # [수정] 개별 카메라 1시간 타이머 검사
                cctv_id_text = str(c.cam_id)
                force_camera_snapshot = cctv_id_text in roi_snapshot_refresh_cctv_ids
                periodic_roi_snapshot_due = (now_time - last_roi_snapshot_times.get(c.ip, 0.0)) >= ROI_SNAPSHOT_INTERVAL_SEC
                
                if connected and fr is not None and (periodic_roi_snapshot_due or force_camera_snapshot):
                    blurred_snap, _ = c.apply_privacy_blur(fr.copy(), t_main, blur_face=True, blur_plate=True)
                    c._initialize_base_roi_if_needed(blurred_snap)
                    snap_img = create_roi_snapshot(c, blurred_snap)
                    if snap_img is not None:
                        h, w = snap_img.shape[:2]
                        roi_info = {"roi_poly_norm": c.roi_poly_norm, "roi_lines_norm": c.roi_lines_norm}

                        # [통합] shpark-roi-final의 상세 스냅샷 전송 파라미터 + fixbug의 개별 타이머 갱신 로직 병합
                        snapshot_send_type = "roi_refresh" if force_camera_snapshot else "hourly"
                        IMAGE_SAVER_POOL.submit(
                            _send_roi_snapshot_task,
                            c.cam_id, terminal_id, snap_img, json.dumps(roi_info), w, h,
                            bool(getattr(c, "align_shifted", False) or getattr(c, "roi_setup_pending", False)),
                            snapshot_send_type,
                        )  # 틀어짐/보정후 관제확인 대기 중이면 True 유지
                        roi_snapshot_queued = True
                        
                        # [수정 - fixbug 유지] 전송 성공(큐 삽입) 즉시 해당 카메라 타이머만 갱신
                        last_roi_snapshot_times[c.ip] = now_time

                        if force_camera_snapshot: refreshed_cctv_ids.add(cctv_id_text)

                if connected and fr is not None and loop_count % 100 == 0:
                    try:
                        small_fr = cv2.resize(fr, (640, 360))
                        save_path = os.path.join(RAM_DISK_DIR, f"{c.ip}.jpg")
                        cv2.imwrite(save_path, small_fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    except Exception: pass

                if not connected or fr is None or not c.events:
                    if is_gui_mode:
                        display_fr = c.draw(None, [], [], [], {}, False)
                        last_rendered_frames[c.ip] = display_fr
                        final_imgs.append(display_fr)
                    continue

                cam_ip = c.ip
                if now_time - current_fps_last_print.get(cam_ip, 0.0) >= current_fps_log_interval_sec:
                    current_fps_last_print[cam_ip] = now_time

                record_fr = fr.copy()
                time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(record_fr, f"Event Time: {time_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if len(c.roi_poly) > 2:
                    cv2.polylines(record_fr, [np.array(c.roi_poly, np.int32)], True, (0, 255, 255), 1)
                if c.roi_lines:
                    for i in range(0, len(c.roi_lines), 2):
                        if i + 1 < len(c.roi_lines):
                            cv2.line(record_fr, tuple(c.roi_lines[i]), tuple(c.roi_lines[i+1]), (0, 0, 255), 1)
                            
                for t in t_main:
                    tid, cls_id = int(t[4]), int(t[6])
                    if cls_id not in [ID_G_PERSON, ID_PERSON_LOW, ID_G_CAR, ID_G_TRUCK, ID_SIGNALMAN]: continue
                    bx1, by1, bx2, by2 = map(int, t[:4])
                    is_alarmed = tid in alarms
                    color = (0, 0, 255) if is_alarmed else (0, 255, 0)
                    cv2.rectangle(record_fr, (bx1, by1), (bx2, by2), color, 1)
                    if tid in c.trk_main.tracks:
                        hist = list(c.trk_main.tracks[tid]['history'])
                        if len(hist) > 1:
                            cv2.polylines(record_fr, [np.array(hist, np.int32)], False, color, 1)

                if infer_meta:
                    c.recorder.update(record_fr, infer_meta, timestamp=now_time)

                if is_gui_mode:
                    display_fr = c.draw(fr.copy(), t_main, t_helmet, t_signalman, alarms, True)
                    last_rendered_frames[c.ip] = display_fr
                    final_imgs.append(display_fr)

            # [수정] 전역 리셋 로직 삭제 완료
            if refreshed_cctv_ids:
                clear_terminal_roi_snapshot_refresh(cctv_ids=refreshed_cctv_ids, reason="roi_snapshot_sent")

            if is_gui_mode:
                if final_imgs: cv2.imshow("Monitor", create_mosaic_image(final_imgs))
                if cv2.waitKey(1) == ord('q'): break

            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("[종료] 사용자에 의해 시스템이 중단되었습니다.")
    except Exception as e:
        logger.error(f"[치명적 오류] {e}\n{traceback.format_exc()}")
    finally:
        if 'health_daemon' in locals():
            health_daemon.stop()

        for c in cams:
            c.reader.running = False
            c.recorder.running = False

        if 'camera_workers' in locals():
            for w in camera_workers:
                w.running = False

        for model_name in ["d_main", "d_helmet", "d_signalman", "d_face", "d_plate"]:
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
