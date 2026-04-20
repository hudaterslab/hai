import os
import sys
import copy
import json
import csv
import cv2
import math
import numpy as np
import time
import datetime
import traceback
import logging
import re
import requests
import pytz
from logging.handlers import TimedRotatingFileHandler
from urllib.parse import urlsplit, unquote
import concurrent.futures

logger = logging.getLogger("VMS_SYSTEM")

# ==========================================
# 상수 및 파일 경로 설정
# ==========================================
CONFIG_COMMON_FILE = os.path.join("config", "system_config.json")
CONFIG_CAMERAS_FILE = os.path.join("config", "cameras.json")
CAMERA_LIST_FILE = "cameras.csv"
EVENT_ROOT_DIR = "./CCTV_EVENT_ALERT" 

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BATCH_SIZE = 9

STREAM_WARMUP_FRAMES = 20
STREAM_READ_FAIL_THRESHOLD = 5
STREAM_RECONNECT_DELAY_SEC = 2.0
STREAM_BROKEN_RETRY_DELAY_SEC = 1.0
WATCHDOG_TIMEOUT = 30.0

ID_H_HELMET = 0
ID_H_NO_HELMET = 1
ID_H_PERSON = 2
ID_G_PERSON = 0
ID_G_CAR = 2
ID_G_BUS = 5
ID_G_TRUCK = 7
TARGET_VEHICLES = [ID_G_CAR, ID_G_BUS, ID_G_TRUCK]

IMAGE_SAVER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def load_system_config():
    # 💡 [핵심] NPU 모델 경로를 models/ 폴더 하위로 변경했습니다.
    default_config = {
        "terminal_id": "1",
        "logging": {"dir": "./logs", "level": "INFO", "retention_days": 14},
        "event_config": {
            "intrusion": {"enabled": False, "cooldown_sec": 600},
            "illegal_parking": {"enabled": False, "cooldown_sec": 600},
            "no_helmet": {"enabled": False, "cooldown_sec": 600, "blur_face": True},
            "conveyor_crossing": {
                "enabled": False, "cooldown_sec": 600, "snapshot_mode": "current_frame", 
                "distance_ratio": 0.5, "min_distance_px": 15, "candidate_ttl_sec": 5.0, "direction_check": True
            },
            "signal_vehicle": {"enabled": False, "cooldown_sec": 600}
        },
        "models": {
            "HELMET": "models/helmet_3cls_v8.dxnn",
            "GENERAL": "models/YOLOV8M-1.dxnn",
            "FACE": "models/YOLOV7_Face-1.dxnn"
        },
        "tower_lamp": {
            "enabled": False,
            "host": "",
            "port": 20000,
            "timeout_sec": 2.0,
            "reset_after_sec": 5.0,
            "default_pattern": "default_alarm",
            "patterns": {
                "default_alarm": {
                    "red": "BLINK",
                    "yellow": "OFF",
                    "green": "OFF",
                    "blue": "OFF",
                    "white": "OFF",
                    "sound_channel": 0,
                    "sound_group": 0,
                    "hold_sec": 5.0
                },
                "signal_vehicle": {
                    "red": "OFF",
                    "yellow": "BLINK",
                    "green": "OFF",
                    "blue": "OFF",
                    "white": "OFF",
                    "sound_channel": 1,
                    "sound_group": 0,
                    "hold_sec": 5.0
                },
                "illegal_parking": {
                    "red": "ON",
                    "yellow": "OFF",
                    "green": "OFF",
                    "blue": "OFF",
                    "white": "OFF",
                    "sound_channel": 2,
                    "sound_group": 0,
                    "hold_sec": 5.0
                },
                "conveyor_crossing": {
                    "red": "BLINK",
                    "yellow": "OFF",
                    "green": "OFF",
                    "blue": "ON",
                    "white": "OFF",
                    "sound_channel": 3,
                    "sound_group": 0,
                    "hold_sec": 5.0
                },
                "no_helmet": {
                    "red": "BLINK",
                    "yellow": "OFF",
                    "green": "OFF",
                    "blue": "OFF",
                    "white": "ON",
                    "sound_channel": 4,
                    "sound_group": 0,
                    "hold_sec": 5.0
                },
                "intrusion": {
                    "red": "BLINK",
                    "yellow": "BLINK",
                    "green": "OFF",
                    "blue": "OFF",
                    "white": "OFF",
                    "sound_channel": 5,
                    "sound_group": 0,
                    "hold_sec": 5.0
                }
            }
        },
        "SKIP_FRAMES": 4,
        "REC_FPS": 30,
        "REC_PRE_SEC": 3,
        "REC_POST_SEC": 4,
        "WATCHDOG_TIMEOUT": 30.0,
        "VISUAL_ALARM_DURATION": 5.0
    }
    
    if not os.path.exists(CONFIG_COMMON_FILE):
        os.makedirs(os.path.dirname(CONFIG_COMMON_FILE), exist_ok=True)
        try:
            with open(CONFIG_COMMON_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 시스템 설정 파일 자동 생성 실패: {e}")
        return default_config

    try:
        with open(CONFIG_COMMON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default_config

SYS_CFG = load_system_config()

# ==========================================
# 공통 유틸리티 함수
# ==========================================
def sanitize_camera_url(url: str) -> str:
    return re.sub(r'\s+', '', (url or '').strip())

def deep_merge_dict(base, override):
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

def load_json_file(path, default, expected_type=None, return_meta=False):
    if not os.path.exists(path):
        return (copy.deepcopy(default), "missing") if return_meta else copy.deepcopy(default)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        return (loaded, "ok") if return_meta else loaded
    except Exception:
        return (copy.deepcopy(default), "invalid") if return_meta else copy.deepcopy(default)

def save_json_file(path, data):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def setup_logging(common_conf):
    log_conf = common_conf.get("logging", {})
    log_dir = str(log_conf.get("dir", "./logs") or "./logs")
    level_name = str(log_conf.get("level", "INFO")).upper()
    
    try:
        retention_days = int(log_conf.get("retention_days", 14))
    except Exception:
        retention_days = 14
        
    level = getattr(logging, level_name, logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
            
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = TimedRotatingFileHandler(os.path.join(log_dir, "cctv.log"), when="midnight", interval=1, backupCount=max(retention_days, 0), encoding="utf-8")
    fh.suffix = "%Y%m%d"
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    logger.addHandler(sh)

def parse_camera_endpoint(rtsp_url: str):
    clean_url = sanitize_camera_url(rtsp_url)
    if "://" not in clean_url:
        clean_url = f"rtsp://{clean_url}"
    parsed = urlsplit(clean_url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    host_port = netloc.strip("[]")
    
    if ":" in host_port and host_port.count(":") == 1:
        host, port = host_port.split(":", 1)
    else:
        host, port = host_port, ""
    return clean_url, unquote(host.strip()), port.strip()

def extract_ip(rtsp_url: str) -> str:
    try:
        _, host, port = parse_camera_endpoint(rtsp_url)
        host_tail = host.split(".")[-1]
        return f"{host_tail}_{port}" if port else host_tail
    except Exception:
        return "unknown_cam"

def load_rtsp_list_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
    rtsp_list = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            sample = f.read(2048)
            f.seek(0)
            if csv.Sniffer().has_header(sample):
                for row in csv.DictReader(f):
                    url = sanitize_camera_url(row.get('url') or row.get('rtsp') or row.get('rtsp_url') or row.get('camera_url'))
                    if url: rtsp_list.append(url)
            else:
                for row in csv.reader(f):
                    if row and not row[0].startswith('#'):
                        rtsp_list.append(sanitize_camera_url(row[0]))
    except Exception:
        pass
        
    unique = []
    for u in rtsp_list:
        if u not in unique:
            unique.append(u)
    return unique

def calculate_iou(box1, box2):
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

def clean_overlapping_detections(detections, is_helmet_model=True):
    if len(detections) == 0:
        return detections
        
    keep = [True] * len(detections)
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i == j or not keep[j]:
                continue
            iou = calculate_iou(detections[i][:4], detections[j][:4])
            
            if iou > 0.85 and int(detections[i][5]) == int(detections[j][5]) and detections[i][4] < detections[j][4]:
                keep[i] = False
                
            if is_helmet_model:
                c_i = int(detections[i][5])
                c_j = int(detections[j][5])
                if (c_i == ID_H_HELMET and c_j == ID_H_NO_HELMET) or (c_i == ID_H_NO_HELMET and c_j == ID_H_HELMET):
                    if iou > 0.6 and c_i == ID_H_HELMET:
                        keep[i] = False
    return detections[keep]

def get_foot_point(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int(y1 + (y2 - y1) * (2/3)))

def get_check_point(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int(y2))

def get_center_point(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def ccw(p1, p2, p3):
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val > 0: return 1
    if val < 0: return -1
    return 0

def normalize_roi_points(points, width, height):
    if points and width > 0 and height > 0:
        return [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in points]
    return []

def denormalize_roi_points(points, width, height):
    if points and width > 0 and height > 0:
        return [[int(round(float(x) * width)), int(round(float(y) * height))] for x, y in points]
    return []

def create_mosaic_image(images, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT):
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
            cv2.putText(cell_img, "No Signal", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cell_img = cv2.resize(img, (cell_w, cell_h))
            
        mosaic[y:y+cell_h, x:x+cell_w] = cell_img
        cv2.rectangle(mosaic, (x, y), (x+cell_w, y+cell_h), (100, 100, 100), 1)
        
    return mosaic

def send_event_image_to_receiver(image_path, event_name, terminal_id, cctv_id, bboxes, img_width=None, img_height=None):
    url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/img"
    mapping = {
        "conveyor_crossing": 1,
        "no_helmet": 2,
        "signal_vehicle": 3,
        "illegal_parking": 4,
        "intrusion": 5
    }
    
    if event_name not in mapping:
        return
        
    kst = pytz.timezone('Asia/Seoul')
    data = {
        "collectedAt": datetime.datetime.now(kst).strftime('%Y-%m-%dT%H:%M:%S'),
        "eventType": mapping[event_name],
        "terminalId": str(terminal_id),
        "cctvId": int(cctv_id),
        "detectedObjects": json.dumps(bboxes) if bboxes else "[]"
    }
    
    if img_width: data["imageWidth"] = img_width
    if img_height: data["imageHeight"] = img_height
    
    try:
        with open(image_path, 'rb') as f:
            requests.post(url, data=data, files={"image": (os.path.basename(image_path), f, "image/jpeg")})
    except Exception as e:
        logger.error(f"[API 예외 발생]: {e}")

def _save_and_send_task(img, img_path, api_params):
    try:
        cv2.imwrite(img_path, img)
    except Exception:
        return
    send_event_image_to_receiver(
        img_path,
        api_params['event_name'],
        api_params['terminal_id'],
        api_params['cctv_id'],
        api_params['bboxes'],
        api_params['img_width'],
        api_params['img_height']
    )

def save_event_image_with_mark(frame, ip, event_type, bbox, tid, terminal_id="3", cctv_id=1):
    if IMAGE_SAVER_POOL._work_queue.qsize() > 50:
        return
    try:
        img = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        now = datetime.datetime.now()
        
        cv2.putText(img, f"{event_type} ID:{tid} {now.strftime('%H:%M:%S')}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if str(terminal_id) == "99999":
            cv2.putText(img, "[ TEST IMAGE ]", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
        dpath = os.path.join(EVENT_ROOT_DIR, "events", ip, "images", str(event_type))
        os.makedirs(dpath, exist_ok=True)
        img_path = os.path.join(dpath, f"{now.strftime('%Y%m%d_%H%M%S')}_{ip}_{event_type}_{tid}.jpg")
        
        params = {
            'event_name': event_type,
            'terminal_id': str(terminal_id),
            'cctv_id': int(cctv_id),
            'bboxes': [{"id": tid, "box": [x1, y1, x2, y2], "label": event_type}],
            'img_width': frame.shape[1],
            'img_height': frame.shape[0]
        }
        IMAGE_SAVER_POOL.submit(_save_and_send_task, img, img_path, params)
    except Exception:
        pass

class ConfigManager:
    def __init__(self, common_path, cameras_path):
        self.common_path = common_path
        self.cameras_path = cameras_path
        self.common_config = self.load_common()
        self.camera_configs = self.load_cameras()
        self.config = self.build_runtime_config()

    def load_common(self):
        return deep_merge_dict(SYS_CFG, load_json_file(self.common_path, {}, expected_type=dict))

    def load_cameras(self):
        return load_json_file(self.cameras_path, {}, expected_type=dict)

    def build_runtime_config(self):
        runtime = {}
        for ip, info in self.camera_configs.items():
            conf = deep_merge_dict(self.common_config, info or {})
            event_overrides = conf.get("event_config", {})
            effective_event_config = deep_merge_dict(self.common_config.get("event_config", {}), event_overrides)
            
            for event_name in conf.get("events", []):
                effective_event_config.setdefault(event_name, {})
                effective_event_config[event_name]["enabled"] = True
                
            conf["event_config"] = effective_event_config
            conf["events"] = [name for name, evt_conf in effective_event_config.items() if evt_conf.get("enabled", False)]
            
            if "url" in conf:
                conf["url"] = sanitize_camera_url(conf["url"])
            conf["terminal_id"] = str(conf.get("terminal_id", self.common_config.get("terminal_id", "3")))
            
            try:
                conf["cctv_id"] = int(conf.get("cctv_id", 1))
            except Exception:
                conf["cctv_id"] = 1
                
            runtime[ip] = conf
        return runtime

    def save(self):
        save_json_file(self.common_path, self.common_config)
        save_json_file(self.cameras_path, self.camera_configs)

    def get_config(self, ip):
        return self.config.get(ip, None)

    def set_config(self, ip, data):
        camera_only = copy.deepcopy(data)
        camera_only.pop("terminal_id", None)
        camera_only.pop("event_config", None)
        
        if "url" in camera_only:
            camera_only["url"] = sanitize_camera_url(camera_only["url"])
            
        try:
            camera_only["cctv_id"] = int(camera_only.get("cctv_id", 1))
        except Exception:
            camera_only["cctv_id"] = 1
            
        self.camera_configs[ip] = camera_only
        self.config = self.build_runtime_config()
        self.save()

    def clear_all(self):
        self.camera_configs = {}
        self.config = {}
        self.save()
