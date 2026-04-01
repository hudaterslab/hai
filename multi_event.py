import os
import sys
import gc
import json
import csv
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
from collections import deque, defaultdict
import concurrent.futures
import pytz
import requests
import re
from urllib.parse import urlsplit, unquote

# ==========================================
# [1] 전문적인 로깅 시스템 구축
# ==========================================
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = datetime.datetime.now().strftime("cctv_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

logger = logging.getLogger("CCTV_SYSTEM")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ==========================================
# [2] 환경 변수 설정
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["QT_QPA_PLATFORM"] = "xcb"

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"

# [3] DeepX NPU 엔진 임포트
try:
    from dx_engine import InferenceEngine, InferenceOption
except ImportError:
    logger.error("dx_engine 모듈을 찾을 수 없습니다. DeepX SDK가 올바르게 설치되었는지 확인하십시오.")
    sys.exit(1)

# ==========================================
# 0. 설정 및 상수
# ==========================================
CONFIG_FILE = "cctv_config.json"
CAMERA_LIST_FILE = "cameras.csv"
EVENT_ROOT_DIR = "./CCTV_EVENT_ALERT" 

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BATCH_SIZE = 9
GC_INTERVAL = 300           
MAX_SAVE_QUEUE = 50
WATCHDOG_TIMEOUT = 30.0

REC_FPS = 30             
REC_PRE_SEC = 10         
REC_POST_SEC = 10        
REC_BUFFER_SIZE = REC_FPS * REC_PRE_SEC
VISUAL_ALARM_DURATION = 5.0 
EVENT_COOLDOWN_SEC = 600

MODEL_HELMET_PATH   = "helmet_3cls_v8.dxnn"
MODEL_GENERAL_PATH = "YOLOV8M-1.dxnn"
MODEL_FACE_PATH = "YOLOV7_Face-1.dxnn"

ID_H_HELMET = 0; ID_H_NO_HELMET = 1; ID_H_PERSON = 2
ID_G_PERSON = 0; ID_G_CAR = 2; ID_G_BUS = 5; ID_G_TRUCK = 7
TARGET_VEHICLES = [ID_G_CAR, ID_G_BUS, ID_G_TRUCK]

IMAGE_SAVER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ==========================================
# 유틸리티 함수
# ==========================================
def sanitize_camera_url(url: str) -> str:
    return re.sub(r'\s+', '', (url or '').strip())

def parse_camera_endpoint(rtsp_url: str):
    clean_url = sanitize_camera_url(rtsp_url)
    if not clean_url:
        raise ValueError("빈 RTSP URL")

    if "://" not in clean_url:
        clean_url = f"rtsp://{clean_url}"

    parsed = urlsplit(clean_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"잘못된 RTSP URL 형식: {rtsp_url!r}")

    netloc = parsed.netloc.rsplit("@", 1)[-1]
    host_port = netloc.strip("[]")
    if not host_port:
        raise ValueError(f"호스트를 찾을 수 없음: {rtsp_url!r}")

    if ":" in host_port and host_port.count(":") == 1:
        host, port = host_port.split(":", 1)
    else:
        host, port = host_port, ""

    host = host.strip()
    port = port.strip()
    if not host:
        raise ValueError(f"호스트를 찾을 수 없음: {rtsp_url!r}")

    return clean_url, unquote(host), port

def extract_ip(rtsp_url: str) -> str:
    try:
        _clean_url, host, port = parse_camera_endpoint(rtsp_url)
        host_tail = host.split(".")[-1]
        return f"{host_tail}_{port}" if port else host_tail
    except Exception as e: 
        logger.warning(f"IP 추출 실패: {e} | input={rtsp_url!r}")
        return "unknown_cam"

def normalize_roi_points(points, width, height):
    if not points or width <= 0 or height <= 0:
        return []
    return [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in points]

def denormalize_roi_points(points, width, height):
    if not points or width <= 0 or height <= 0:
        return []
    return [[int(round(float(x) * width)), int(round(float(y) * height))] for x, y in points]

def roi_points_are_normalized(points):
    if not points:
        return False
    try:
        return all(0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0 for x, y in points)
    except (TypeError, ValueError):
        return False

def load_rtsp_list_from_csv(csv_path):
    if not os.path.exists(csv_path):
        logger.error(f"카메라 목록 CSV를 찾을 수 없습니다: {csv_path}")
        return []

    rtsp_list = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            sample = f.read(2048)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample)

            if has_header:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    url = (
                        row.get('url')
                        or row.get('rtsp')
                        or row.get('rtsp_url')
                        or row.get('camera_url')
                    )
                    url = sanitize_camera_url(url)
                    if url:
                        rtsp_list.append(url)
                    else:
                        logger.warning(f"CSV {row_idx}행에 URL 컬럼이 비어 있어 건너뜁니다.")
            else:
                reader = csv.reader(f)
                for row_idx, row in enumerate(reader, start=1):
                    if not row:
                        continue
                    first_col = row[0] if row else ''
                    url = sanitize_camera_url(first_col)
                    if not url or url.startswith('#'):
                        continue
                    rtsp_list.append(url)
    except csv.Error as e:
        logger.error(f"카메라 CSV 파싱 실패: {e}")
        return []
    except Exception as e:
        logger.error(f"카메라 CSV 로드 실패: {e}")
        return []

    unique_rtsp_list = []
    seen = set()
    for url in rtsp_list:
        if url in seen:
            logger.warning(f"중복 카메라 URL 건너뜀: {url}")
            continue
        seen.add(url)
        unique_rtsp_list.append(url)

    logger.info(f"카메라 CSV 로드 완료: {len(unique_rtsp_list)}대")
    return unique_rtsp_list

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def clean_overlapping_detections(detections, is_helmet_model=True):
    if len(detections) == 0: return detections
    keep = [True] * len(detections)
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i == j: continue
            if not keep[j]: continue
            iou = calculate_iou(detections[i][:4], detections[j][:4])
            if iou > 0.85 and int(detections[i][5]) == int(detections[j][5]):
                if detections[i][4] < detections[j][4]: keep[i] = False
            if is_helmet_model:
                c_i, c_j = int(detections[i][5]), int(detections[j][5])
                if ((c_i == ID_H_HELMET and c_j == ID_H_NO_HELMET) or 
                    (c_i == ID_H_NO_HELMET and c_j == ID_H_HELMET)):
                    if iou > 0.6 and c_i == ID_H_HELMET: keep[i] = False
    return detections[keep]

def get_foot_point(x1, y1, x2, y2):
    foot_y = int(y1 + (y2 - y1) * (2/3))
    center_x = int((x1 + x2) / 2)
    return (center_x, foot_y)

def get_check_point(x1, y1, x2, y2): return (int((x1+x2)/2), int(y2))
def get_center_point(x1, y1, x2, y2): return (int((x1+x2)/2), int((y1+y2)/2))
def get_distance(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def ccw(p1, p2, p3):
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val > 0: return 1
    elif val < 0: return -1
    return 0

def create_mosaic_image(images, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT):
    if not images: return None
    count = len(images)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    if cols == 0: cols = 1
    if rows == 0: rows = 1
    cell_w = screen_w // cols; cell_h = screen_h // rows
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        if img is None:
            cell_img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(cell_img, "No Signal", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else: cell_img = cv2.resize(img, (cell_w, cell_h))
        mosaic[y:y+cell_h, x:x+cell_w] = cell_img
        cv2.rectangle(mosaic, (x, y), (x+cell_w, y+cell_h), (100, 100, 100), 1)
    return mosaic

# ==========================================
# 이미지 저장 핸들러
# ==========================================
def send_event_image_to_receiver(image_path, event_name, terminal_id, cctv_id, bboxes, img_width=None, img_height=None):
    url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/img"
    event_type_mapping = {
        "conveyor_crossing": 1, "no_helmet": 2, "signal_vehicle": 3, "illegal_parking": 4, "intrusion": 5          
    }
    if event_name not in event_type_mapping:
        logger.info(f"[API 스킵] 정의되지 않은 이벤트 타입: {event_name}")
        return
        
    api_event_type = event_type_mapping[event_name]
    kst = pytz.timezone('Asia/Seoul')
    collected_at = datetime.datetime.now(kst).strftime('%Y-%m-%dT%H:%M:%S')
    detected_objects_json = json.dumps(bboxes) if bboxes else "[]"

    data = {
        "collectedAt": collected_at,
        "eventType": api_event_type,
        "terminalId": str(terminal_id),
        "cctvId": int(cctv_id),
        "detectedObjects": detected_objects_json
    }
    
    if img_width: data["imageWidth"] = img_width
    if img_height: data["imageHeight"] = img_height

    if not os.path.exists(image_path):
        logger.error(f"[API 에러] 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        with open(image_path, 'rb') as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, data=data, files=files)
            if response.status_code == 200:
                logger.info(f"[API 전송 성공] {os.path.basename(image_path)}")
            else:
                logger.error(f"[API 전송 실패] 코드: {response.status_code}, 메시지: {response.text}")
    except Exception as e:
        logger.error(f"[API 예외 발생]: {e}")

def _save_and_send_task(img, img_path, api_params):
    try:
        cv2.imwrite(img_path, img)
    except Exception as e:
        logger.error(f"[이미지 저장 실패] {e}")
        return
    try:
        send_event_image_to_receiver(
            image_path=img_path,
            event_name=api_params['event_name'],
            terminal_id=api_params['terminal_id'],
            cctv_id=api_params['cctv_id'],
            bboxes=api_params['bboxes'],
            img_width=api_params['img_width'],
            img_height=api_params['img_height']
        )
    except Exception as e:
        logger.error(f"[Task 내부 API 호출 에러] {e}")

def save_event_image_with_mark(frame, ip, event_type, bbox, tid):
    if IMAGE_SAVER_POOL._work_queue.qsize() > MAX_SAVE_QUEUE: return
    try:
        img = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        now = datetime.datetime.now()
        msg = f"{event_type} ID:{tid} {now.strftime('%H:%M:%S')}"
        cv2.putText(img, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        dpath = os.path.join(EVENT_ROOT_DIR, "events", ip, "images", str(event_type))
        if not os.path.exists(dpath): 
            os.makedirs(dpath)
            
        fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{ip}_{event_type}_{tid}.jpg"
        img_path = os.path.join(dpath, fname)
        h, w = frame.shape[:2]
        
        ai_detected_bboxes = [{"id": tid, "box": [x1, y1, x2, y2], "label": event_type}]
        api_params = {
            'event_name': event_type,
            'terminal_id': "3",
            'cctv_id': 1,            
            'bboxes': ai_detected_bboxes,
            'img_width': w,
            'img_height': h
        }
        IMAGE_SAVER_POOL.submit(_save_and_send_task, img, img_path, api_params)
    except Exception as e: 
        logger.error(f"[EventLogic Error] {e}")

# ==========================================
# 영상 녹화기
# ==========================================
class VideoRecorder:
    def __init__(self, ip):
        self.ip = ip
        self.buffer = deque(maxlen=REC_BUFFER_SIZE) 
        self.write_queue = queue.Queue()
        self.recording = False
        self.record_end_time = 0
        self.current_event = "unknown"
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def update(self, frame):
        if frame is None: return
        self.buffer.append(frame)
        if self.recording:
            if time.time() > self.record_end_time:
                self.recording = False
                self.write_queue.put(None)
                logger.info(f"🎬 [녹화종료] {self.ip} - {self.current_event}")
            else:
                self.write_queue.put(frame)

    def trigger(self, event_name):
        now = time.time()
        if self.recording:
            self.record_end_time = now + REC_POST_SEC
        else:
            logger.info(f"🎥 [녹화시작] {self.ip} - {event_name}")
            self.recording = True
            self.record_end_time = now + REC_POST_SEC
            self.current_event = event_name
            temp_buffer = list(self.buffer)
            for f in temp_buffer: self.write_queue.put(f)

    def _writer_loop(self):
        writer = None
        while self.running:
            try:
                frame = self.write_queue.get(timeout=1.0)
            except queue.Empty: continue

            if frame is None:
                if writer: writer.release(); writer = None
                continue

            if writer is None:
                dpath = os.path.join(EVENT_ROOT_DIR, "events", self.ip, "videos", self.current_event)
                if not os.path.exists(dpath): os.makedirs(dpath)
                fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.ip}_{self.current_event}.mp4"
                fpath = os.path.join(dpath, fname)
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(fpath, fourcc, REC_FPS, (w, h))
                if not writer.isOpened():
                    logger.error(f"[녹화에러] 파일을 열 수 없습니다: {fpath}")
                    writer = None; continue
            if writer: writer.write(frame)

# ==========================================
# 트래커 & 설정 관리
# ==========================================
class SimpleTracker:
    def __init__(self, max_lost=50, is_helmet=True): 
        self.next_id = 1; self.tracks = {}; self.max_lost = max_lost; self.is_helmet = is_helmet
    def update(self, detections, frame=None):
        detections = clean_overlapping_detections(detections, self.is_helmet)
        used_dets = set()
        for tid, trk in self.tracks.items():
            best_iou = 0; best_idx = -1
            for i, det in enumerate(detections):
                if i in used_dets: continue
                if int(det[5]) != trk['cls']: continue
                iou = calculate_iou(trk['bbox'], det[:4])
                if iou > best_iou: best_iou = iou; best_idx = i
            if best_iou > 0.2:
                self.tracks[tid].update({'bbox': detections[best_idx][:4], 'lost': 0})
                used_dets.add(best_idx)
            else: self.tracks[tid]['lost'] += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.max_lost}
        res_tracks = []
        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks[self.next_id] = {'bbox': det[:4], 'lost': 0, 'cls': int(det[5])}
                self.next_id += 1
        for tid, trk in self.tracks.items():
            if trk['lost'] == 0: res_tracks.append([*trk['bbox'], tid, 1.0, trk['cls']])
        return np.array(res_tracks)

class ConfigManager:
    def __init__(self, filepath): self.filepath = filepath; self.config = self.load()
    def load(self):
        if not os.path.exists(self.filepath): return {}
        try:
            data = json.load(open(self.filepath, 'r', encoding='utf-8'))
            for ip, info in data.items():
                if 'url' in info: info['url'] = info['url'].strip()
            return data
        except: return {}
    def save(self): json.dump(self.config, open(self.filepath, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    def get_config(self, ip): return self.config.get(ip, None)
    def set_config(self, ip, data): self.config[ip] = data; self.save()
    def clear_all(self): self.config = {}; self.save()

# ==========================================
# 딥엑스 NPU 엔진 (YOLO v7 / v8 자동 파싱 탑재)
# ==========================================
class YoLoDeepX:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        # 💡 [핵심] 파일명에 'v7'이 포함되어 있으면 YOLOv7 후처리 로직(Objectness 분리)을 사용
        self.is_yolov7 = "v7" in os.path.basename(self.engine_path).lower()
        try:
            io = InferenceOption()
            self.engine = InferenceEngine(self.engine_path, io)
            logger.info(f"[DeepX] 모델 로드 성공: {os.path.basename(self.engine_path)} (YOLOv{'7' if self.is_yolov7 else '8'} 타입)")
        except Exception as e:
            logger.error(f"[DeepX Load Fail] {e}")
            raise e

    def letter_box(self, img, new_shape=(640,640)):
        h, w = img.shape[:2]
        scale = min(new_shape[0]/h, new_shape[1]/w)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
        dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        return canvas, scale, (dw, dh)

    def postprocess(self, output_tensor, conf_thres=0.45, iou_thres=0.45):
        try:
            pred = np.array(output_tensor[0])
            
            # YOLOv8의 경우 [batch, num_classes+4, num_boxes] 배열일 수 있으므로 동적 트랜스포즈
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]: 
                pred = pred.transpose((0, 2, 1))
            
            if pred.ndim == 3:
                pred = pred[0] # batch 차원 제거 -> [N, C]
            
            C = pred.shape[1]
            
            # 💡 [완벽한 해결책] YOLOv7과 YOLOv8의 Tensor 배열 구조를 구분하여 파싱
            if self.is_yolov7 or C == 6 or C == 85:
                # YOLOv7 Format: 5번째 인덱스가 Objectness(객체 존재 여부)
                obj_conf = pred[:, 4]
                if C > 5:
                    cls_conf = pred[:, 5:]
                    scores = obj_conf * np.max(cls_conf, axis=1)
                    class_ids = np.argmax(cls_conf, axis=1)
                else:
                    # 완벽한 1-Class 전용 모델일 경우
                    scores = obj_conf
                    class_ids = np.zeros(len(scores), dtype=int)
            else:
                # YOLOv8 Format: Objectness 없이 4번부터 바로 Class Confidence 시작
                scores = np.max(pred[:, 4:], axis=1)
                class_ids = np.argmax(pred[:, 4:], axis=1)

            mask = scores > conf_thres
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            if len(pred) == 0: return []

            boxes = pred[:, :4]
            
            boxes_xyxy = boxes.copy()
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            
            indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_thres, iou_thres)
            
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    results.append([boxes_xyxy[i], scores[i], class_ids[i]])
            return results
        except Exception as e:
            logger.error(f"NPU Postprocess Error ({os.path.basename(self.engine_path)}): {e}")
            return []

    def infer(self, img):
        if img is None: return np.empty((0,6))
        
        h_orig, w_orig = img.shape[:2]
        npu_input, scale, offset = self.letter_box(img)
        npu_input_rgb = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)
        
        try:
            output_tensor = self.engine.run([npu_input_rgb])
            # 💡 v7 얼굴 모델은 보수적 감지를 위해 임계값을 살짝 낮춥니다 (오탐 방지는 모자이크부에서 수행)
            raw_dets = self.postprocess(output_tensor, conf_thres=0.30 if self.is_yolov7 else 0.45)
            if not raw_dets: return np.empty((0,6))
            
            res = []
            dw, dh = offset
            for box, score, cls_id in raw_dets:
                x1 = (box[0] - dw) / scale
                y1 = (box[1] - dh) / scale
                x2 = (box[2] - dw) / scale
                y2 = (box[3] - dh) / scale
                
                x1 = np.clip(x1, 0, w_orig)
                y1 = np.clip(y1, 0, h_orig)
                x2 = np.clip(x2, 0, w_orig)
                y2 = np.clip(y2, 0, h_orig)
                
                res.append([x1, y1, x2, y2, score, cls_id])
                
            return np.array(res)
        except Exception as e:
            logger.error(f"NPU Inference Error: {e}")
            return np.empty((0,6))

    def destroy(self):
        pass

class MotionDetector:
    def __init__(self, sensitivity):
        self.threshold = 100 - ((sensitivity - 1) * 9) 
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    def apply(self, frame):
        if frame is None: return None
        small_frame = cv2.resize(frame, (640, 360))
        fg_mask = self.bg_subtractor.apply(small_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        return fg_mask 

class IntrusionDetector:
    def __init__(self, roi):
        self.roi = np.array(roi, dtype=np.int32)
        if self.roi.shape[0] < 3: self.roi = np.empty((0, 2), dtype=np.int32)
    def process(self, tracks, track_map, motion_mask=None):
        triggered = []
        if self.roi.size == 0: return triggered
        for t in tracks:
            if track_map.get(int(t[4])) == ID_H_PERSON:
                if cv2.pointPolygonTest(self.roi, get_check_point(*t[:4]), False) >= 0: triggered.append(int(t[4]))
        return triggered

class ParkingDetector:
    def __init__(self, roi):
        self.roi = np.array(roi, dtype=np.int32)
        if self.roi.shape[0] < 3: self.roi = np.empty((0, 2), dtype=np.int32)
        self.states = defaultdict(lambda: {'start': 0, 'pos': None})
    def process(self, tracks, track_map, motion_mask=None):
        triggered = []
        if self.roi.size == 0: return triggered
        now = time.time(); curr_ids = set()
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) in TARGET_VEHICLES:
                pt = get_check_point(*t[:4])
                if cv2.pointPolygonTest(self.roi, pt, False) >= 0:
                    curr_ids.add(tid); c = get_center_point(*t[:4])
                    if self.states[tid]['start'] == 0:
                        self.states[tid]['start'] = now; self.states[tid]['pos'] = c
                    else:
                        if get_distance(c, self.states[tid]['pos']) > 30: 
                            self.states[tid]['start'] = now; self.states[tid]['pos'] = c
                        elif now - self.states[tid]['start'] > 5.0:
                            triggered.append(tid)
        for tid in list(self.states.keys()):
            if tid not in curr_ids: del self.states[tid]
        return triggered


class CrossingDetector:
    def __init__(self, roi_lines):
        self.lines = []
        for i in range(0, len(roi_lines), 2):
            if i + 1 < len(roi_lines): self.lines.append((roi_lines[i], roi_lines[i + 1]))
        self.prev = {}
        self.candidates = {}

        # 💡 [핵심 수정] 무한 연장선이 아닌 '실제 그려진 유한 선분'만 통과했는지 검사하는 양방향 판정

    def _is_intersect(self, p1, p2, p3, p4):
        # p1, p2: 사용자가 그린 ROI 선분
        # p3, p4: 사람의 발 이동 궤적 (과거 -> 현재)
        c1 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
        c2 = ccw(p3, p4, p1) * ccw(p3, p4, p2)
        # 두 선분이 서로를 가로지를 때만 True 반환 (가상의 연장선 오탐 완벽 차단)
        return c1 < 0 and c2 < 0

    def process(self, tracks, track_map, target_cls, motion_mask=None):
        triggered = []
        curr_ids = set()
        now = time.time()

        for t in tracks:
            tid = int(t[4])
            curr_ids.add(tid)
            if track_map.get(tid) != target_cls: continue

            curr_pos = get_foot_point(*t[:4])
            obj_width = t[2] - t[0]

            if tid in self.prev:
                pp = self.prev[tid]

                # 1. 교차 지점 후보 등록
                if tid not in self.candidates:
                    for p1, p2 in self.lines:
                        # 💡 기존의 절반짜리 판정식을 완벽한 유한 선분 판정식으로 교체
                        if self._is_intersect(p1, p2, pp, curr_pos):
                            self.candidates[tid] = {
                                'crossing_pt': curr_pos,
                                'width': obj_width,
                                'timestamp': now
                            }
                            break

            # 2. 바운딩 박스 떨림(Jittering) 방어 로직
            if tid in self.candidates:
                cand = self.candidates[tid]
                moved_dist = get_distance(cand['crossing_pt'], curr_pos)

                # 라인을 밟은 후, 객체 가로폭의 60% 이상 확실하게 넘어가야만 최종 알람 발생 (노이즈 방지)
                if moved_dist > (cand['width'] * 0.6):
                    triggered.append(tid)
                    del self.candidates[tid]
                # 5초가 지나도록 완전히 넘어가지 않고 맴돌면 오탐으로 간주하고 후보에서 삭제
                elif now - cand['timestamp'] > 5.0:
                    del self.candidates[tid]

            self.prev[tid] = curr_pos

        # 프레임에서 사라진 객체들 메모리 정리
        for tid in list(self.prev.keys()):
            if tid not in curr_ids:
                del self.prev[tid]
                if tid in self.candidates: del self.candidates[tid]

        return triggered

class HelmetDetector:
    def process(self, tracks, track_map, motion_mask=None):
        triggered = []
        nh = [t for t in tracks if track_map.get(int(t[4])) == ID_H_NO_HELMET]
        for n in nh: triggered.append(int(n[4]))
        return triggered

class SignalVehicleDetector:
    def __init__(self, roi):
        self.roi = np.array(roi, dtype=np.int32)
        if self.roi.shape[0] < 3: self.roi = np.empty((0, 2), dtype=np.int32)
        self.motion_threshold_ratio = 0.10
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30)) 

    def _get_distance_point_to_rect(self, point, bbox):
        px, py = point; bx1, by1, bx2, by2 = bbox
        dx = max(bx1 - px, 0, px - bx2)
        dy = max(by1 - py, 0, py - by2)
        return math.sqrt(dx*dx + dy*dy)

    def process(self, tracks, track_map, motion_mask):
        triggered = []
        if self.roi.size == 0 or motion_mask is None: return triggered
        
        scale_x = 640 / SCREEN_WIDTH; scale_y = 360 / SCREEN_HEIGHT
        people_points = [get_center_point(*t[:4]) for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        current_vehicle_ids = set()
        
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) not in TARGET_VEHICLES: continue
                
            current_vehicle_ids.add(tid)
            x1, y1, x2, y2 = t[:4]
            center = get_center_point(x1, y1, x2, y2)
            self.vehicle_history[tid].append(center)
            
            if len(self.vehicle_history[tid]) > 5:
                if get_distance(self.vehicle_history[tid][0], self.vehicle_history[tid][-1]) < 40.0: continue 
            else: continue 

            mx1 = max(0, int(x1 * scale_x)); my1 = max(0, int(y1 * scale_y))
            mx2 = min(640, int(x2 * scale_x)); my2 = min(360, int(y2 * scale_y))
            
            if mx2 > mx1 and my2 > my1:
                car_roi_mask = motion_mask[my1:my2, mx1:mx2]
                _, motion_only = cv2.threshold(car_roi_mask, 250, 255, cv2.THRESH_BINARY)
                total_pixels = (mx2 - mx1) * (my2 - my1)
                
                if total_pixels > 0 and (cv2.countNonZero(motion_only) / total_pixels) > self.motion_threshold_ratio:
                    if cv2.pointPolygonTest(self.roi, center, False) >= 0:
                        safe_radius = y2 - y1
                        has_signalman = any(self._get_distance_point_to_rect(pp, (x1, y1, x2, y2)) < safe_radius for pp in people_points)
                        if not has_signalman: triggered.append(tid)

        for tid in list(self.vehicle_history.keys()):
            if tid not in current_vehicle_ids: del self.vehicle_history[tid]
        return triggered

# ==========================================
# 안정적인 FFMPEG 백엔드 기반 프레임 리더
# ==========================================
class FrameReader:
    def __init__(self, url, ip):
        self.url = url.replace(" ","").strip()
        self.ip = ip
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_frame_time = time.time()
        self.is_stuck = False
        self.resolution_checked = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                time.sleep(5)
                continue

            self.connected = True
            self.last_frame_time = time.time()
            self.is_stuck = False
            
            while self.running and cap.isOpened():
                if time.time() - self.last_frame_time > WATCHDOG_TIMEOUT:
                    logger.warning(f"[{self.ip}] Timeout. Force Reconnecting...")
                    self.is_stuck = True 
                    break
                
                ret, fr = cap.read()
                if not ret: 
                    logger.warning(f"[{self.ip}] Stream broken.")
                    time.sleep(1)
                    break
                
                if fr is not None:
                    if not self.resolution_checked:
                        h, w = fr.shape[:2]
                        logger.info(f"[{self.ip}] 수신된 해상도: {w} x {h}")
                        self.resolution_checked = True

                    if fr.shape[1] > 720:
                        scale =720 / fr.shape[1]
                        fr = cv2.resize(fr, (720, int(fr.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

                with self.lock:
                    self.frame = fr
                    self.fid += 1
                    self.last_frame_time = time.time()
                
                time.sleep(0.005)
            
            self.connected = False
            try: cap.release()
            except: pass
            if self.running: time.sleep(2)

    def read(self):
        with self.lock:
            if self.is_stuck or (time.time() - self.last_frame_time > WATCHDOG_TIMEOUT):
                return None, self.fid, False
            return self.frame, self.fid, self.connected

class Camera:
    def __init__(self, ip, conf, det_h, det_g, det_f, npu_id, cam_id, sensitivity):
        self.ip = ip; self.conf = conf 
        self.reader = FrameReader(conf['url'], ip)
        self.roi_poly = []
        self.roi_lines = []
        self.roi_poly_norm = conf.get('roi_poly_norm', [])
        self.roi_lines_norm = conf.get('roi_lines_norm', [])
        self.using_normalized_roi = bool(self.roi_poly_norm or self.roi_lines_norm)
        if not self.using_normalized_roi:
            self.roi_poly = conf.get('roi_poly', [])
            self.roi_lines = conf.get('roi_lines', [])
        self.events = conf.get('events', [])
        self.det_h = det_h; self.det_g = det_g; self.det_f = det_f
        self.npu_id = npu_id; self.cam_id = cam_id 
        self.trk_h = SimpleTracker(is_helmet=True); self.trk_g = SimpleTracker(is_helmet=False)
        self.last_draw = None
        self.alerted = defaultdict(set); self.last_evt_t = {}
        self.visual_alarms = {} 
        self.face_blur_cache = {}
        self.roi_frame_shape = None

        self.config_lock = threading.Lock() 
        self.motion_det = MotionDetector(sensitivity)
        self.recorder = VideoRecorder(ip)
        self.init_handlers()

    def _update_runtime_roi(self, frame_shape):
        if not self.using_normalized_roi:
            return
        if self.roi_frame_shape == frame_shape[:2]:
            return
        height, width = frame_shape[:2]
        self.roi_poly = denormalize_roi_points(self.roi_poly_norm, width, height)
        self.roi_lines = denormalize_roi_points(self.roi_lines_norm, width, height)
        self.roi_frame_shape = frame_shape[:2]
        self.init_handlers()

    def init_handlers(self):
        self.handlers = {}
        if "intrusion" in self.events: self.handlers['intrusion'] = IntrusionDetector(self.roi_poly)
        if "illegal_parking" in self.events: self.handlers['illegal_parking'] = ParkingDetector(self.roi_poly)
        if "no_helmet" in self.events: self.handlers['no_helmet'] = HelmetDetector()
        if "conveyor_crossing" in self.events: self.handlers['conveyor_crossing'] = CrossingDetector(self.roi_lines)
        if "signal_vehicle" in self.events: self.handlers['signal_vehicle'] = SignalVehicleDetector(self.roi_poly)

    def update_config(self, new_events, new_poly=None, new_lines=None):
        with self.config_lock:
            self.events = new_events
            if new_poly is not None: self.roi_poly = new_poly
            if new_lines is not None: self.roi_lines = new_lines
            self.using_normalized_roi = False
            self.roi_frame_shape = None
            self.init_handlers()

    def process_frame(self):
        fr, fid, connected = self.reader.read()
        if fr is None and not connected:
            if time.time() - self.reader.last_frame_time > (WATCHDOG_TIMEOUT + 2.0):
                logger.error(f"[{self.ip}] Reader thread dead. Spawning NEW thread.")
                self.reader.running = False 
                self.reader = FrameReader(self.conf['url'], self.ip)
                time.sleep(0.5)
        if fr is not None:
            with self.config_lock:
                self._update_runtime_roi(fr.shape)
            self.recorder.update(fr)
        return fr, fid, connected

    def run_logic(self, fr, fid, d_h, d_g):
        with self.config_lock:
            motion_mask = self.motion_det.apply(fr) 

            t_h = self.trk_h.update(d_h); t_g = self.trk_g.update(d_g)
            current_alarms = {} 
            now = time.time()

            for ename, h in self.handlers.items():
                if ename == "illegal_parking":
                    tm = {int(t[4]): int(t[6]) for t in t_g}
                    ids = h.process(t_g, tm)
                    for i in ids: 
                        draw_tid = i + 10000
                        self._trigger(fr, fid, draw_tid, ename, t_g, now)
                        current_alarms[draw_tid] = ename
                elif ename == "conveyor_crossing":
                    tm = {int(t[4]): int(t[6]) for t in t_g}
                    ids = h.process(t_g, tm, target_cls=ID_G_PERSON)
                    for i in ids: 
                        draw_tid = i + 10000
                        self._trigger(fr, fid, draw_tid, ename, t_g, now)
                        current_alarms[draw_tid] = ename
                elif ename == "signal_vehicle":
                    tm = {int(t[4]): int(t[6]) for t in t_g}
                    ids = h.process(t_g, tm, motion_mask=motion_mask)
                    for i in ids: 
                        draw_tid = i + 10000
                        self._trigger(fr, fid, draw_tid, ename, t_g, now)
                        current_alarms[draw_tid] = ename
                else: 
                    tm = {int(t[4]): int(t[6]) for t in t_h}
                    ids = h.process(t_h, tm)
                    for i in ids: 
                        draw_tid = i
                        self._trigger(fr, fid, draw_tid, ename, t_h, now)
                        current_alarms[draw_tid] = ename
            
            for tid, ename in current_alarms.items():
                self.visual_alarms[tid] = {'evt': ename, 'expire': now + VISUAL_ALARM_DURATION}
            
            for tid in list(self.visual_alarms.keys()):
                if now > self.visual_alarms[tid]['expire']:
                    del self.visual_alarms[tid]
            
            final_alarms = {}
            for tid, info in self.visual_alarms.items():
                final_alarms[tid] = info['evt']

            return t_h, t_g, final_alarms

    def draw(self, fr, t_h, t_g, alarms, connected=True):
        if fr is None or not connected:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(blank, self.ip, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            self.last_draw = blank
            return blank

        h_frame, w_frame = fr.shape[:2]
        if len(alarms) > 0: cv2.rectangle(fr, (0, 0), (w_frame, h_frame), (0, 0, 255), 20)

        if self.roi_poly: cv2.polylines(fr, [np.array(self.roi_poly, np.int32)], True, (0,255,255), 2)
        if self.roi_lines:
            for i in range(0, len(self.roi_lines), 2):
                if i+1 < len(self.roi_lines): cv2.line(fr, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0,0,255), 2)

        for t in t_h:
            tid = int(t[4]); cls_id = int(t[6])
            
            if cls_id == ID_H_HELMET:
                color = (255, 0, 0) 
                label = f"Helmet [{tid}]"
            elif cls_id == ID_H_NO_HELMET:
                color = (0, 0, 255) 
                label = f"No-Helmet [{tid}]"
            elif cls_id == ID_H_PERSON:
                color = (0, 255, 0) 
                label = f"Person [{tid}]"
            else:
                color = (0, 255, 255) 
                label = f"Unknown({cls_id}) [{tid}]"

            if tid in alarms:
                color = (0, 0, 255)
                label = f"ALARM: {label}"
                cv2.rectangle(fr, (int(t[0]),int(t[1])), (int(t[2]),int(t[3])), color, 3)
            else:
                cv2.rectangle(fr, (int(t[0]),int(t[1])), (int(t[2]),int(t[3])), color, 2)
                
            cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            ft = get_foot_point(*t[:4])
            if "conveyor_crossing" in self.events: cv2.circle(fr, ft, 5, (255,0,255), -1)

        for t in t_g:
            tid = int(t[4])+10000
            cls_id = int(t[6])
            color = (0,0,255) if tid in alarms else (255,100,0)
            
            label_g = f"OBJ [{tid}]"
            if cls_id == ID_G_PERSON: label_g = f"Person [{tid}]"
            elif cls_id == ID_G_CAR: label_g = f"Car [{tid}]"
            elif cls_id == ID_G_BUS: label_g = f"Bus [{tid}]"
            elif cls_id == ID_G_TRUCK: label_g = f"Truck [{tid}]"
            
            cv2.rectangle(fr, (int(t[0]),int(t[1])), (int(t[2]),int(t[3])), color, 2)
            cv2.putText(fr, label_g, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            ft = get_foot_point(*t[:4])
            if "conveyor_crossing" in self.events: cv2.circle(fr, ft, 5, (255,0,255), -1)

        id_str = f"{self.cam_id}"
        cv2.rectangle(fr, (0, 0), (100, 100), (0, 0, 0), -1) 
        cv2.putText(fr, id_str, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 6)
        cv2.putText(fr, f"{self.ip}", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.recorder.recording:
            cv2.circle(fr, (w_frame - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(fr, "REC", (w_frame - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        active_alarms = set(alarms.values())
        list_h = len(self.events) * 40 + 10; list_w = 250
        overlay = fr.copy()
        cv2.rectangle(overlay, (w_frame - list_w, 0), (w_frame, list_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, fr, 0.5, 0, fr)

        y_pos = 35
        for evt in self.events:
            display_name = {
                "intrusion": "INTRUSION",
                "illegal_parking": "PARKING",
                "no_helmet": "NO-HELMET",
                "conveyor_crossing": "CROSSING",
                "signal_vehicle": "NO-SIGNALMAN"
            }.get(evt, evt.upper())
            color = (0, 0, 255) if evt in active_alarms else (0, 255, 0)
            text = f"[!] {display_name}" if evt in active_alarms else f" -  {display_name}"
            cv2.putText(fr, text, (w_frame - list_w + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_pos += 40
        self.last_draw = fr
        return fr

    def _trigger(self, fr, fid, tid, ename, tracks, now):
        real_tid = tid if tid < 10000 else tid - 10000
        if ename in self.alerted[tid]: return
        if now - self.last_evt_t.get(ename, 0) < EVENT_COOLDOWN_SEC: return
        
        bb = next((t[:4] for t in tracks if int(t[4]) == real_tid), None)
        if bb is not None:
            logger.warning(f"🚨 [CAM {self.cam_id}] {ename} Detected! ID:{real_tid}")
            
            # 💡 [보강] 단일/다중 클래스에 상관없이 모든 검출 객체(얼굴)만 정밀하게 모자이크
            if fid not in self.face_blur_cache:
                blur_img = fr.copy()
                if self.det_f is not None:
                    try:
                        f_dets = self.det_f.infer(blur_img)
                        for fx1, fy1, fx2, fy2, fscore, fcls in f_dets:
                            # 얼굴 인식 전용 모델이므로, 신뢰도 0.3 이상이면 클래스 ID를 무시하고 렌더링
                            if fscore > 0.3:
                                fx1, fy1, fx2, fy2 = max(0, int(fx1)), max(0, int(fy1)), int(fx2), int(fy2)
                                fh, fw = fy2 - fy1, fx2 - fx1
                                
                                # 자동차나 벽 전체를 잡는 터무니없는 오탐(화면의 80% 이상)만 차단
                                if fw > blur_img.shape[1] * 0.8 or fh > blur_img.shape[0] * 0.8:
                                    continue 
                                    
                                roi = blur_img[fy1:fy2, fx1:fx2]
                                if roi.size > 0:
                                    small = cv2.resize(roi, (fw//15 + 1, fh//15 + 1), interpolation=cv2.INTER_LINEAR)
                                    blur_img[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        logger.error(f"얼굴 모자이크 처리 중 오류: {e}")
                
                self.face_blur_cache[fid] = blur_img
                if len(self.face_blur_cache) > 5:
                    self.face_blur_cache.pop(next(iter(self.face_blur_cache)))
            
            saved_img = self.face_blur_cache[fid]
            
            save_event_image_with_mark(saved_img, self.ip, ename, bb, real_tid)
            self.recorder.trigger(ename)
            self.alerted[tid].add(ename)
            self.last_evt_t[ename] = now
    
    def stop(self): 
        self.reader.running = False
        self.recorder.running = False 

# ==========================================
# Main
# ==========================================
def capture_snapshot(url):
    clean_url = re.sub(r'\s+', '', url)
    cap = cv2.VideoCapture(clean_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened(): return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_roi_points_scaled(frame, title, mode="poly"):
    pts = []
    orig_h, orig_w = frame.shape[:2]
    disp_w = 960
    scale = disp_w / orig_w
    disp_h = int(orig_h * scale)
    disp_frame = cv2.resize(frame, (disp_w, disp_h))
    
    wname = f"Config: {title}"
    cv2.namedWindow(wname)
    def mouse_cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            if mode=="line" and len(pts)>=2: return
            pts.append([int(x/scale), int(y/scale)])
    cv2.setMouseCallback(wname, mouse_cb)
    logger.info(f"'{title}' 그리기 모드. 점을 찍고 Enter(완료) 또는 ESC(취소). Line 모드는 2점.")
    while True:
        temp = disp_frame.copy()
        dp = [[int(p[0]*scale), int(p[1]*scale)] for p in pts]
        if mode == "line":
            if len(dp)==1: cv2.circle(temp, tuple(dp[0]), 5, (0,0,255),-1)
            elif len(dp)==2: cv2.line(temp, tuple(dp[0]), tuple(dp[1]), (0,0,255), 2)
        else:
            if len(dp)>0: cv2.polylines(temp, [np.array(dp, np.int32)], True, (0,255,0), 2)
        cv2.imshow(wname, temp)
        k = cv2.waitKey(1)
        if k==13: break 
        if k==27: pts=[]; break 
        if mode=="line" and len(pts)==2: cv2.waitKey(500); break
    cv2.destroyWindow(wname)
    return normalize_roi_points(pts, orig_w, orig_h)

def run_wizard_batch_mode(mgr, rtsp_list):
    logger.info("=== CCTV 일괄 설정 마법사 (Batch Mode) ===")
    selected_indices = []
    total = len(rtsp_list)
    if total == 0:
        logger.warning("설정할 카메라가 없습니다. cameras.csv 내용을 확인하십시오.")
        return
    for i in range(0, total, BATCH_SIZE):
        batch_urls = rtsp_list[i : i + BATCH_SIZE]
        logger.info(f"[Batch {i//BATCH_SIZE + 1}] 카메라 {i+1} ~ {min(i+BATCH_SIZE, total)} 로딩 중...")
        frames = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            frames = list(executor.map(capture_snapshot, batch_urls))
        display_frames = []
        for idx, frm in enumerate(frames):
            if frm is None:
                blk = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blk, "Conn Fail", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                display_frames.append(blk)
            else: display_frames.append(frm)
        mosaic = create_mosaic_image(display_frames)
        cols = math.ceil(math.sqrt(len(display_frames)))
        if cols == 0: cols = 1
        cw = SCREEN_WIDTH // cols; ch = SCREEN_HEIGHT // math.ceil(len(display_frames)/cols)
        for idx in range(len(display_frames)):
            r, c = divmod(idx, cols)
            cx, cy = c * cw, r * ch
            cv2.rectangle(mosaic, (cx, cy), (cx+50, cy+50), (255,255,255), -1)
            cv2.putText(mosaic, str(idx+1), (cx+10, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
        cv2.imshow("Select Cameras", mosaic)
        cv2.waitKey(1)
        sel = input(">> 선택 (예: 1,3,5): ").strip()
        if sel:
            try:
                nums = [int(s.strip()) for s in sel.split(',')]
                for n in nums:
                    if 1 <= n <= len(batch_urls): selected_indices.append(i + (n - 1))
            except: pass
    cv2.destroyWindow("Select Cameras")
    
    if not selected_indices: 
        logger.info("선택된 카메라가 없습니다.")
        return

    logger.info(f"총 {len(selected_indices)}대 설정 시작")
    for idx in selected_indices:
        url = rtsp_list[idx].strip(); ip = extract_ip(url)
        logger.info(f"[{ip}] 설정 중...")
        frame = capture_snapshot(url)
        if frame is None: 
            logger.error(" -> 캡처 실패")
            continue
        h, w = frame.shape[:2]; ratio = 960 / w
        preview = cv2.resize(frame, (960, int(h*ratio)))
        win_name = f"Check Cam: {ip}"
        cv2.namedWindow(win_name); cv2.imshow(win_name, preview); cv2.moveWindow(win_name, 100, 100); cv2.waitKey(1)
        print("1.침입 2.주정차 3.안전모 4.횡단 5.신호수차량감지")
        sel = input(">> 이벤트 선택 (예: 1,4,5): ")
        cv2.destroyWindow(win_name)
        events = []
        if '1' in sel: events.append("intrusion")
        if '2' in sel: events.append("illegal_parking")
        if '3' in sel: events.append("no_helmet")
        if '4' in sel: events.append("conveyor_crossing")
        if '5' in sel: events.append("signal_vehicle")
        
        roi_p = []; roi_l = []
        if "intrusion" in events or "illegal_parking" in events or "signal_vehicle" in events: 
            roi_p = get_roi_points_scaled(frame, "Polygon")
        if "conveyor_crossing" in events:
            while True:
                l = get_roi_points_scaled(frame, "Line", mode="line")
                if len(l)==2: roi_l.extend(l)
                if input("    라인 추가? (y/n): ")!='y': break
        mgr.set_config(ip, {"url": url, "roi_poly_norm": roi_p, "roi_lines_norm": roi_l, "events": events})

def main():
    logger.info("[System] DeepX NPU 환경으로 초기화를 시작합니다.")
    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    if not rtsp_list:
        logger.error(f"카메라 목록을 불러오지 못했습니다. {CAMERA_LIST_FILE} 파일을 확인하십시오.")
        return
    
    sensitivity = 5
    try:
        val = input(">> 움직임 감지 민감도 설정 (1-10, 엔터시 기본값 5): ")
        if val.strip():
            sensitivity = int(val)
            sensitivity = max(1, min(10, sensitivity))
    except: pass
    logger.info(f"민감도 설정: {sensitivity}")

    detectors = {} 
    active_npu_ids = [0]
    cams = []
    loop_count = 0 
    
    try:
        mgr = ConfigManager(CONFIG_FILE)
        if not os.path.exists(CONFIG_FILE):
            logger.warning("설정 파일이 없습니다. 초기 설정 마법사는 디스플레이 환경이 필수입니다.")
            run_wizard_batch_mode(mgr, rtsp_list)
        else:
            is_reset = input(">> 설정 초기화 마법사를 실행하시겠습니까? (y/n): ").strip().lower()
            if is_reset == 'y':
                logger.warning("설정 마법사는 디스플레이 환경이 필수입니다.")
                mgr.clear_all()
                run_wizard_batch_mode(mgr, rtsp_list)

        val_disp = input(">> 모니터링 화면(GUI)을 출력하시겠습니까? (y/n, 기본값 y): ").strip().lower()
        use_display = False if val_disp == 'n' else True
        
        use_drawing = True
        if use_display:
            val_draw = input(">> 화면에 박스 및 텍스트(시각화)를 그리시겠습니까? (y/n, 기본값 y): ").strip().lower()
            use_drawing = False if val_draw == 'n' else True
        else:
            logger.info("디스플레이가 꺼져 있으므로 실시간 화면 그리기도 비활성화됩니다.")
            use_drawing = False

        if not os.path.exists(MODEL_HELMET_PATH) or not os.path.exists(MODEL_GENERAL_PATH) or not os.path.exists(MODEL_FACE_PATH): 
            logger.error("모델 파일(.dxnn)을 찾을 수 없습니다. Face 모델을 포함하여 3개의 경로를 모두 확인하십시오.")
            return

        logger.info("Loading DeepX Models (Helmet, General, Face)...")
        
        d_h = YoLoDeepX(MODEL_HELMET_PATH)
        d_g = YoLoDeepX(MODEL_GENERAL_PATH)
        d_f = YoLoDeepX(MODEL_FACE_PATH)
        detectors[0] = {'h': d_h, 'g': d_g, 'f': d_f}

        load_count = 0
        num_active = len(active_npu_ids)
        skipped_missing_config = []
        skipped_no_events = []
        skipped_invalid_ip = []

        for rtsp in rtsp_list:
            ip = extract_ip(rtsp)
            conf = mgr.get_config(ip)
            if ip == "unknown_cam":
                skipped_invalid_ip.append(rtsp)
                logger.warning(f"[카메라 스킵] IP 추출 실패로 설정 매칭 불가: {rtsp!r}")
                continue
            if not conf:
                skipped_missing_config.append((ip, rtsp))
                logger.warning(f"[카메라 스킵] 설정 없음: ip_key={ip}, url={rtsp}")
                continue
            if not conf.get('events'):
                skipped_no_events.append((ip, rtsp))
                logger.warning(f"[카메라 스킵] 이벤트 설정 없음: ip_key={ip}, url={rtsp}")
                continue

            if conf and conf.get('events'):
                if 'url' not in conf: conf['url'] = rtsp.strip()
                cam_id = load_count + 1 
                target_npu_idx = active_npu_ids[load_count % num_active]
                target_dets = detectors[target_npu_idx]
                
                cams.append(Camera(ip, conf, target_dets['h'], target_dets['g'], target_dets['f'], target_npu_idx, cam_id, sensitivity))
                logger.info(f"Load [CAM {cam_id}]: {ip} -> NPU {target_npu_idx}")
                load_count += 1
        
        if not cams: 
            logger.warning(
                f"카메라 없음 | csv={len(rtsp_list)} config_keys={len(mgr.config)} "
                f"missing_config={len(skipped_missing_config)} no_events={len(skipped_no_events)} "
                f"invalid_ip={len(skipped_invalid_ip)}"
            )
            if skipped_missing_config:
                logger.warning(f"설정 키 예시: {list(mgr.config.keys())[:10]}")
                logger.warning(f"미매칭 CSV 예시: {[ip for ip, _url in skipped_missing_config[:10]]}")
            return
        
        logger.info("모니터링 시작 (상시 녹화 모드 / 종료: Ctrl+C 또는 'q')")

        target_fps = 15
        dynamic_delay = 1.0 / target_fps

        while True:
            start_time = time.time()
            
            cpu_usage = psutil.cpu_percent(interval=None)
            if cpu_usage > 85:
                target_fps = max(5, target_fps - 2)
            elif cpu_usage < 60:
                target_fps = min(15, target_fps + 1)
            dynamic_delay = 1.0 / target_fps

            if loop_count % 100 == 0:
                time.sleep(0.1)

            loop_count += 1
            if loop_count % GC_INTERVAL == 0: gc.collect()
            
            raw_data = [c.process_frame() for c in cams]
            processed_results = [None] * len(cams)
            valid_frame_count = 0 
            
            current_active_npus = list(active_npu_ids)
            
            for npu_idx in current_active_npus:
                cam_indices = [i for i, c in enumerate(cams) if c.npu_id == npu_idx]
                if not cam_indices: continue
                
                try:
                    for idx in cam_indices:
                        c = cams[idx]
                        fr, fid, connected = raw_data[idx]
                        if fr is None or not connected:
                            processed_results[idx] = (None, fid, [], [], False)
                            continue
                        valid_frame_count += 1
                        
                        run_helmet = any(e in ["no_helmet", "intrusion"] for e in c.events)
                        run_general = any(e in ["illegal_parking", "conveyor_crossing", "signal_vehicle"] for e in c.events)
                        
                        d_h_res, d_g_res = [], []
                        if run_helmet: d_h_res = c.det_h.infer(fr)
                        if run_general: d_g_res = c.det_g.infer(fr)
                        
                        processed_results[idx] = (fr, fid, d_h_res, d_g_res, True)

                except Exception as e:
                    logger.error(f"[FATAL ERROR] NPU {npu_idx} 에서 오류 발생: {e}")
                    raise e 
            
            if valid_frame_count == 0: 
                time.sleep(0.01)

            final_imgs = []
            for idx, res in enumerate(processed_results):
                if res is None: 
                    if use_display and use_drawing:
                        final_imgs.append(cams[idx].draw(None, [], [], {}, connected=False))
                    elif use_display:
                        final_imgs.append(np.zeros((360, 640, 3), dtype=np.uint8))
                    continue
                
                fr, fid, d_h_res, d_g_res, connected = res
                if not connected:
                    if use_display and use_drawing:
                        final_imgs.append(cams[idx].draw(None, [], [], {}, connected=False))
                    elif use_display:
                        final_imgs.append(np.zeros((360, 640, 3), dtype=np.uint8))
                else:
                    t_h, t_g, alarms = cams[idx].run_logic(fr, fid, d_h_res, d_g_res)
                    if use_display:
                        if use_drawing:
                            img = cams[idx].draw(fr, t_h, t_g, alarms, connected=True)
                        else:
                            img = cv2.resize(fr, (640, 360)) 
                        final_imgs.append(img)

            if use_display:
                valid_imgs = [img for img in final_imgs if img is not None]
                if valid_imgs:
                    mosaic = create_mosaic_image(valid_imgs)
                    if mosaic is not None: cv2.imshow("Monitor", mosaic)
                
                key = cv2.waitKey(1)
                if key == ord('q'): break

            elapsed = time.time() - start_time
            sleep_time = dynamic_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("[종료] 사용자에 의해 모니터링이 중단되었습니다 (Ctrl+C).")
    except Exception as e: 
        logger.error(f"예외 발생: {e}")
        traceback.print_exc()
    finally:
        logger.info("[종료 절차 시작] 리소스 정리 중...")
        for c in cams: c.stop()
        cv2.destroyAllWindows()
        logger.info("[종료 완료]")

if __name__ == "__main__":
    main()
