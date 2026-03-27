import os
import sys
import gc
import json
import cv2
import math
import numpy as np
import time
import datetime
import traceback
import threading
import queue
from collections import deque, defaultdict
import concurrent.futures
import msvcrt
import pytz
import requests

# [1] 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;stimeout=3000000;buffer_size=1024;max_delay=500000"

# [GPU 설정]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# [2] PyCUDA & TensorRT
import pycuda.driver as cuda
import tensorrt as trt

# ==========================================
# 0. 설정 및 상수
# ==========================================
CONFIG_FILE = "cctv_config.json"
EVENT_ROOT_DIR = r"C:\CCTV_EVENT_ALERT"

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BATCH_SIZE = 9
GC_INTERVAL = 300           
MAX_SAVE_QUEUE = 50
WATCHDOG_TIMEOUT = 10.0

# [녹화 및 알림 설정]
REC_FPS = 30             
REC_PRE_SEC = 10         
REC_POST_SEC = 10        
REC_BUFFER_SIZE = REC_FPS * REC_PRE_SEC
VISUAL_ALARM_DURATION = 5.0 

# [RTSP URL 리스트]
RTSP_LIST = [
    "rtsp://192.168.1.170:9001/S.mp4",
    "rtsp://192.168.1.170:9002/s.mp4"
]
# RTSP_LIST = [
#     "rtsp://admin1:11qqaa..@192.168.100.2:554/h264",
#     "rtsp://admin1:11qqaa..@192.168.100.5:554/h264",
#     "rtsp://admin1:11qqaa..@192.168.100.7:554/h264",
#     "rtsp://admin1:11qqaa..@192.168.100.8:554/h264",
#     "rtsp://admin1:11qqaa..@192.168.100.9:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.16:554/stream1",
#     "rtsp://ca37bba7:qwert12@@192.168.100.78:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.79:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.83:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.90:554/stream1",
#     "rtsp://ca37bba7:qwert12@@192.168.100.91:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.27:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.65:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.67:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.68:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.69:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.70:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.71:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.72:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.73:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.74:554/stream1",
#     "rtsp://ca37bba7:qwert12@@192.168.100.77:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.80:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.81:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.82:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.84:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.85:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.86:554/h264",
#     "rtsp://ca37bba7:qwert12@@192.168.100.87:554/h264",
#     "rtsp://e138e933:qwert12@@192.168.100.26:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.28:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.29:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.75:554/stream1",
#     "rtsp://e138e933:qwert12@@192.168.100.76:554/stream1",
# ]

MODEL_HELMET_PATH   = "helmet_3cls_v8.engine"
MODEL_GENERAL_PATH = "yolov9-e.engine"

# Model IDs
ID_H_HELMET = 0; ID_H_NO_HELMET = 1; ID_H_PERSON = 2
ID_G_PERSON = 0; ID_G_CAR = 2; ID_G_BUS = 5; ID_G_TRUCK = 7
TARGET_VEHICLES = [ID_G_CAR, ID_G_BUS, ID_G_TRUCK]

IMAGE_SAVER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ==========================================
# 유틸리티 함수
# ==========================================
def extract_ip(rtsp_url: str) -> str:
    try:
        # 1. 인증 정보 제외하고 IP:Port 부분 추출
        if "@" in rtsp_url:
            rest = rtsp_url.split("@")[-1]
        else:
            rest = rtsp_url.split("//")[1]
            
        ip_port = rest.split("/")[0]  # 예: "192.168.1.170:9001"
        
        # 2. IP 마지막 자리와 포트를 결합하여 반환
        if ":" in ip_port:
            ip_part, port_part = ip_port.split(":")
            last_octet = ip_part.split(".")[-1] # "170" 추출
            return f"{last_octet}_{port_part}"  # "170_9001" 반환
        else:
            # 포트가 없는 경우 마지막 자리만 반환
            return ip_port.split(".")[-1]
            
    except Exception: 
        return "unknown_cam"

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
        "conveyor_crossing": 1, 
        "no_helmet": 2,         
        "signal_vehicle": 3,    
        "illegal_parking": 4,
        "intrusion": 5          # 필요하다면 매핑 추가
    }
    
    if event_name not in event_type_mapping:
        print(f"⏩ [API 스킵] 정의되지 않은 이벤트 타입: {event_name}")
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
        print(f"❌ [API 에러] 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        with open(image_path, 'rb') as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, data=data, files=files)
            
            if response.status_code == 200:
                print(f"✅ [API 전송 성공] {os.path.basename(image_path)}")
            else:
                print(f"❌ [API 전송 실패] 코드: {response.status_code}, 메시지: {response.text}")
    except Exception as e:
        print(f"❌ [API 예외 발생]: {e}")


# ---------------------------------------------------------
# 2. 이미지 저장 + API 전송을 동시에 수행하는 새로운 Task 함수
# ---------------------------------------------------------
def _save_and_send_task(img, img_path, api_params):
    # 1) 기존 _save_task의 역할 (이미지 쓰기)
    try:
        cv2.imwrite(img_path, img)
    except Exception as e:
        print(f"❌ [이미지 저장 실패] {e}")
        return
    
    # 2) 저장이 완료되면 API 전송 실행
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
        print(f"❌ [Task 내부 API 호출 에러] {e}")


# ---------------------------------------------------------
# 3. 기존 이벤트 로직 수정
# ---------------------------------------------------------
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
        
        # 이전 턴에서 만들었던 JSON 로컬 저장 로직 유지 (원치 않으시면 삭제 무방)
        h, w = frame.shape[:2]
        
        # ==========================================
        # 💡 API 전송을 위한 파라미터 패키징
        # ==========================================
        ai_detected_bboxes = [
            {"id": tid, "box": [x1, y1, x2, y2], "label": event_type}
        ]
        
        api_params = {
            'event_name': event_type,
            'terminal_id': "2",  # 샘플에 맞춰 고정
            'cctv_id': 1,            # 다중 카메라 환경이라면 ip를 기반으로 맵핑 추천
            'bboxes': ai_detected_bboxes,
            'img_width': w,
            'img_height': h
        }
        
        # 기존 _save_task 대신, 저장과 전송을 묶어둔 _save_and_send_task를 스레드 풀에 던집니다.
        IMAGE_SAVER_POOL.submit(_save_and_send_task, img, img_path, api_params)
        
    except Exception as e: 
        print(f"❌ [EventLogic Error] {e}")

# ==========================================
# 영상 녹화기 (Video Recorder)
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
                print(f"🎬 [녹화종료] {self.ip} - {self.current_event}")
            else:
                self.write_queue.put(frame)

    def trigger(self, event_name):
        now = time.time()
        if self.recording:
            self.record_end_time = now + REC_POST_SEC
        else:
            print(f"🎥 [녹화시작] {self.ip} - {event_name}")
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
                    print(f"❌ [녹화에러] 파일을 열 수 없습니다: {fpath}")
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
# YOLO TRT
# ==========================================
class YoLoTRT:
    def __init__(self, engine_path, cuda_ctx):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.cuda_ctx = cuda_ctx 
        try:
            self.cuda_ctx.push() 
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.allocations = [], [], []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                if shape[0] < 0: shape[0] = 1
                size = trt.volume(shape) * dtype.itemsize
                alloc = cuda.mem_alloc(size)
                self.allocations.append(alloc)
                binding = {'index': i, 'name': name, 'shape': list(shape), 'alloc': alloc}
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: self.inputs.append(binding)
                else: self.outputs.append(binding)
        except Exception as e:
            print(f"❌ [Model Load Fail] {e}")
            raise e
        finally: 
            try: self.cuda_ctx.pop()
            except: pass

    def infer(self, img):
        if img is None: return np.empty((0,6))
        try:
            self.cuda_ctx.push()
            h, w = img.shape[:2]
            r = min(640/h, 640/w)
            nw, nh = int(w*r), int(h*r)
            resized = cv2.resize(img, (nw, nh))
            dw, dh = (640-nw)/2, (640-nh)/2
            padded = cv2.copyMakeBorder(resized, int(dh), int(640-nh-int(dh)), int(dw), int(640-nw-int(dw)), cv2.BORDER_CONSTANT, value=(114,114,114))
            inp = np.ascontiguousarray(padded.transpose(2,0,1)[::-1]).astype(np.float32) / 255.0
            
            cuda.memcpy_htod(self.inputs[0]['alloc'], inp)
            self.context.execute_v2(self.allocations)
            out = np.zeros(self.outputs[0]['shape'], dtype=np.float32)
            cuda.memcpy_dtoh(out, self.outputs[0]['alloc'])
            
            if out.shape[1] < out.shape[2]: out = out.transpose(0,2,1)
            scores = np.max(out[0,:,4:], axis=1)
            mask = scores >= 0.5
            pred = out[0][mask]
            if len(pred) == 0: return np.empty((0,6))
            cx = pred[:, 0]; cy = pred[:, 1]; bw = pred[:, 2]; bh = pred[:, 3]
            x1 = (cx - bw/2 - dw) / r; y1 = (cy - bh/2 - dh) / r
            x2 = (cx + bw/2 - dw) / r; y2 = (cy + bh/2 - dh) / r
            res = []
            cls = np.argmax(pred[:,4:], axis=1)
            for i in range(len(pred)):
                res.append([np.clip(x1[i],0,w), np.clip(y1[i],0,h), np.clip(x2[i],0,w), np.clip(y2[i],0,h), scores[mask][i], cls[i]])
            return np.array(res)
        except Exception as e:
            raise e 
        finally:
            try: self.cuda_ctx.pop()
            except: pass

    def destroy(self):
        try:
            self.cuda_ctx.push()
            for alloc in self.allocations:
                try: alloc.free()
                except: pass
            if self.context: del self.context
            if self.engine: del self.engine
        except: pass
        finally: 
            try: self.cuda_ctx.pop()
            except: pass

# ==========================================
# 움직임 디텍터 (배경 차분) - 그림자 제거 설정 적용
# ==========================================
class MotionDetector:
    def __init__(self, sensitivity):
        self.threshold = 100 - ((sensitivity - 1) * 9) 
        # [수정] detectShadows=True로 변경하여 그림자(127)와 실제 움직임(255) 구분
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame):
        if frame is None: return None
        small_frame = cv2.resize(frame, (640, 360))
        fg_mask = self.bg_subtractor.apply(small_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        return fg_mask 

# ==========================================
# 이벤트 디텍터
# ==========================================
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
            if i+1 < len(roi_lines): self.lines.append((roi_lines[i], roi_lines[i+1]))
        self.prev = {}
        self.candidates = {} 
    def process(self, tracks, track_map, target_cls, motion_mask=None):
        triggered = []
        curr_ids = set(); now = time.time()
        for t in tracks:
            tid = int(t[4]); curr_ids.add(tid)
            if track_map.get(tid) != target_cls: continue
            curr_pos = get_foot_point(*t[:4]); obj_width = t[2] - t[0]
            if tid in self.prev:
                pp = self.prev[tid]
                if tid not in self.candidates:
                    for p1, p2 in self.lines:
                        if (ccw(p1, p2, pp) * ccw(p1, p2, curr_pos) < 0):
                            self.candidates[tid] = {'crossing_pt': curr_pos, 'width': obj_width, 'timestamp': now}
                            break
            if tid in self.candidates:
                cand = self.candidates[tid]
                moved_dist = get_distance(cand['crossing_pt'], curr_pos)
                if moved_dist > (cand['width'] * 0.5):
                    triggered.append(tid); del self.candidates[tid]
                elif now - cand['timestamp'] > 5.0:
                    del self.candidates[tid]
            self.prev[tid] = curr_pos
        for tid in list(self.prev.keys()):
            if tid not in curr_ids: 
                del self.prev[tid]
                if tid in self.candidates: del self.candidates[tid]
        return triggered

class HelmetDetector:
    def process(self, tracks, track_map, motion_mask=None):
        triggered = []
        nh = [t for t in tracks if track_map.get(int(t[4])) == ID_H_NO_HELMET]
        
        for n in nh:
            # 사람 몸통(Person)과의 교차 영역(IoU/IoA)을 계산하는 조건을 완전히 배제했습니다.
            # 이로써 AI 모델이 사람 몸통을 놓치고 머리만 검출하더라도 무조건 이벤트가 트리거됩니다.
            triggered.append(int(n[4]))
                
        return triggered

class SignalVehicleDetector:
    def __init__(self, roi):
        self.roi = np.array(roi, dtype=np.int32)
        if self.roi.shape[0] < 3: self.roi = np.empty((0, 2), dtype=np.int32)
        self.motion_threshold_ratio = 0.10
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30)) 

    def _get_distance_point_to_rect(self, point, bbox):
        """
        사람(point)과 차량 박스(bbox) 사이의 최단 거리를 구합니다.
        사람이 차량 내부에 있으면 거리는 0입니다.
        사람이 차량 오른쪽에 있으면 '차량 오른쪽 면'과의 거리를 잽니다.
        """
        px, py = point
        bx1, by1, bx2, by2 = bbox
        
        # x축 거리 계산 (사람이 박스 안에 있으면 0)
        dx = max(bx1 - px, 0, px - bx2)
        # y축 거리 계산 (사람이 박스 안에 있으면 0)
        dy = max(by1 - py, 0, py - by2)
        
        # 유클리드 거리 (피타고라스)
        return math.sqrt(dx*dx + dy*dy)

    def process(self, tracks, track_map, motion_mask):
        triggered = []
        if self.roi.size == 0 or motion_mask is None: return triggered
        
        scale_x = 640 / SCREEN_WIDTH
        scale_y = 360 / SCREEN_HEIGHT

        # 1. 사람들의 중심점(발 쪽이 더 정확할 수 있으나 여기선 중심 사용) 리스트업
        people_points = []
        for t in tracks:
            if track_map.get(int(t[4])) == ID_G_PERSON:
                people_points.append(get_center_point(*t[:4])) # 필요시 get_foot_point로 변경 가능

        # 2. 차량 처리
        current_vehicle_ids = set()
        
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) not in TARGET_VEHICLES:
                continue
                
            current_vehicle_ids.add(tid)
            x1, y1, x2, y2 = t[:4]
            center = get_center_point(x1, y1, x2, y2)
            
            # 히스토리 업데이트
            self.vehicle_history[tid].append(center)
            
            # [필터링 1] "가려졌다 나타난 주차 차량" 방지 (실제 이동 여부 확인)
            # 최근 N프레임 동안의 좌표 이동 거리를 계산
            if len(self.vehicle_history[tid]) > 5:
                start_pt = self.vehicle_history[tid][0]
                curr_pt = self.vehicle_history[tid][-1]
                displacement = get_distance(start_pt, curr_pt)
                
                # 좌표가 거의 안 움직였으면(예: 10픽셀 미만), 
                # 픽셀 모션(앞차가 비켜서 생기는 변화)이 있어도 무시 (주차된 차로 간주)
                if displacement < 40.0: 
                    continue 
            else:
                # 트래킹 초기 단계(등장 직후)에는 판단 보류 혹은 패스
                # (너무 빨리 알람이 울리는 것을 방지하려면 continue 사용)
                continue 

            # [기존 로직] 픽셀 모션 체크
            mx1 = int(x1 * scale_x); my1 = int(y1 * scale_y)
            mx2 = int(x2 * scale_x); my2 = int(y2 * scale_y)
            
            mx1 = max(0, mx1); my1 = max(0, my1)
            mx2 = min(640, mx2); my2 = min(360, my2)
            
            if mx2 > mx1 and my2 > my1:
                car_roi_mask = motion_mask[my1:my2, mx1:mx2]
                _, motion_only = cv2.threshold(car_roi_mask, 250, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(motion_only)
                total_pixels = (mx2 - mx1) * (my2 - my1)
                
                if total_pixels > 0:
                    ratio = motion_pixels / total_pixels
                    
                    if ratio > self.motion_threshold_ratio:
                        # 차량이 ROI 안에 있는지 확인
                        if cv2.pointPolygonTest(self.roi, center, False) >= 0:
                            
                            # [개선된 로직] 신호수 거리 판단
                            vehicle_height = y2 - y1
                            safe_radius = vehicle_height # 차량 높이만큼을 안전 반경으로 설정
                            
                            has_signalman = False
                            for pp in people_points:
                                # 기존: 차량 '중심' vs 사람 '중심' 거리
                                # 변경: 차량 '박스(가장 가까운 면)' vs 사람 '중심' 거리
                                # 이렇게 하면 차량 중심이 멀어도, 차량 범퍼 옆에 사람이 있으면 거리 0~가까움으로 계산됨
                                dist = self._get_distance_point_to_rect(pp, (x1, y1, x2, y2))
                                
                                if dist < safe_radius:
                                    has_signalman = True
                                    break
                            
                            if not has_signalman:
                                triggered.append(tid)

        # 사라진 차량은 히스토리에서 제거 (메모리 관리)
        for tid in list(self.vehicle_history.keys()):
            if tid not in current_vehicle_ids:
                del self.vehicle_history[tid]

        return triggered

# ==========================================
# 비동기 프레임 리더
# ==========================================
class FrameReader:
    def __init__(self, url, ip, use_gstreamer=False):  # 기본값을 False로 변경
        self.url = url.strip()
        self.ip = ip
        self.use_gstreamer = use_gstreamer  
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_frame_time = time.time()
        self.is_stuck = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            if self.use_gstreamer:
                # uridecodebin 플러그인을 사용하여 H.264/H.265를 자동 식별 및 디코딩합니다.
                # OpenCV가 처리할 수 있도록 BGR 포맷으로 변환 후 앱싱크로 전달합니다.
                pipeline = (
                    f"uridecodebin uri={self.url} ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink drop=true max-buffers=1"
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(self.url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000) 
                except: pass

            if not cap.isOpened():
                time.sleep(5)
                continue

            self.connected = True
            self.last_frame_time = time.time()
            self.is_stuck = False
            
            while self.running and cap.isOpened():
                if time.time() - self.last_frame_time > WATCHDOG_TIMEOUT:
                    print(f"⚠️ [Watchdog] {self.ip}: Timeout. Force Reconnecting...")
                    self.is_stuck = True 
                    break
                
                ret, fr = cap.read()
                if not ret: 
                    print(f"⚠️ [Reader] {self.ip}: Stream broken.")
                    time.sleep(1)
                    break
                
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
    def __init__(self, ip, conf, det_h, det_g, gpu_id, cam_id, sensitivity):
        self.ip = ip; self.conf = conf 
        # use_gstreamer=False로 설정하여 Windows 환경 로컬 테스트를 정상화합니다.
        self.reader = FrameReader(conf['url'], ip, use_gstreamer=False)
        self.roi_poly = conf.get('roi_poly', [])
        self.roi_lines = conf.get('roi_lines', [])
        self.events = conf.get('events', [])
        self.det_h = det_h; self.det_g = det_g
        self.gpu_id = gpu_id; self.cam_id = cam_id 
        self.trk_h = SimpleTracker(is_helmet=True); self.trk_g = SimpleTracker(is_helmet=False)
        self.last_draw = None
        self.alerted = defaultdict(set); self.last_evt_t = {}
        
        self.visual_alarms = {} 

        self.config_lock = threading.Lock() 
        self.motion_det = MotionDetector(sensitivity)
        self.recorder = VideoRecorder(ip)
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
            self.init_handlers()

    def process_frame(self):
        fr, fid, connected = self.reader.read()
        if fr is None and not connected:
            if time.time() - self.reader.last_frame_time > (WATCHDOG_TIMEOUT + 2.0):
                print(f"💀 [Fatal] {self.ip}: Reader thread dead. Spawning NEW thread.")
                self.reader.running = False 
                self.reader = FrameReader(self.conf['url'], self.ip, use_gstreamer=False)
                time.sleep(0.5)
        if fr is not None: self.recorder.update(fr)
        return fr, connected

    def run_logic(self, fr, d_h, d_g):
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
                        self._trigger(fr, draw_tid, ename, t_g, now)
                        current_alarms[draw_tid] = ename
                elif ename == "conveyor_crossing":
                    tm = {int(t[4]): int(t[6]) for t in t_g}
                    ids = h.process(t_g, tm, target_cls=ID_G_PERSON)
                    for i in ids: 
                        draw_tid = i + 10000
                        self._trigger(fr, draw_tid, ename, t_g, now)
                        current_alarms[draw_tid] = ename
                elif ename == "signal_vehicle":
                    tm = {int(t[4]): int(t[6]) for t in t_g}
                    ids = h.process(t_g, tm, motion_mask=motion_mask)
                    for i in ids: 
                        draw_tid = i + 10000
                        self._trigger(fr, draw_tid, ename, t_g, now)
                        current_alarms[draw_tid] = ename
                else: 
                    tm = {int(t[4]): int(t[6]) for t in t_h}
                    ids = h.process(t_h, tm)
                    for i in ids: 
                        draw_tid = i
                        self._trigger(fr, draw_tid, ename, t_h, now)
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

        # 1. 안전모 모델 트래커 시각화 (클래스 텍스트 추가 및 색상 재배치)
        for t in t_h:
            tid = int(t[4]); cls_id = int(t[6])
            
            # 클래스별 색상 및 디버깅용 라벨 지정
            if cls_id == ID_H_HELMET:
                color = (255, 0, 0) # 파란색
                label = f"Helmet [{tid}]"
            elif cls_id == ID_H_NO_HELMET:
                color = (0, 0, 255) # 빨간색 (이상 상황 명시)
                label = f"No-Helmet [{tid}]"
            elif cls_id == ID_H_PERSON:
                color = (0, 255, 0) # 초록색
                label = f"Person [{tid}]"
            else:
                color = (0, 255, 255) # 노란색 (예상치 못한 클래스 확인용)
                label = f"Unknown({cls_id}) [{tid}]"

            # 알람 발생 시 시각적 효과 강제 적용
            if tid in alarms:
                color = (0, 0, 255)
                label = f"ALARM: {label}"
                cv2.rectangle(fr, (int(t[0]),int(t[1])), (int(t[2]),int(t[3])), color, 3)
            else:
                cv2.rectangle(fr, (int(t[0]),int(t[1])), (int(t[2]),int(t[3])), color, 2)
                
            # 디버깅을 위해 바운딩 박스 위에 라벨을 그립니다.
            cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            ft = get_foot_point(*t[:4])
            if "conveyor_crossing" in self.events: cv2.circle(fr, ft, 5, (255,0,255), -1)

        # 2. 일반 모델 트래커 시각화 (텍스트 라벨 추가)
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

    def _trigger(self, fr, tid, ename, tracks, now):
        real_tid = tid if tid < 10000 else tid - 10000
        if ename in self.alerted[tid]: return
        if now - self.last_evt_t.get(ename, 0) < 60: return
        
        bb = next((t[:4] for t in tracks if int(t[4]) == real_tid), None)
        if bb is not None:
            print(f"🚨 [CAM {self.cam_id}] {ename} Detected! ID:{real_tid}")
            save_event_image_with_mark(fr, self.ip, ename, bb, real_tid)
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
    # 초기 설정 마법사 스냅샷에도 필요에 따라 GStreamer 파이프라인을 적용할 수 있으나, 
    # 안정성을 위해 우선 기존 FFMPEG 방식을 유지합니다.
    cap = cv2.VideoCapture(url.strip())
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
    print(f">> '{title}' 그리기 모드. 점을 찍고 Enter(완료) 또는 ESC(취소). Line 모드는 2점.")
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
        if k==13: break # Enter
        if k==27: pts=[]; break # ESC
        if mode=="line" and len(pts)==2: cv2.waitKey(500); break
    cv2.destroyWindow(wname)
    return pts

def run_wizard_batch_mode(mgr):
    print("\n=== CCTV 일괄 설정 마법사 (Batch Mode) ===")
    selected_indices = []
    total = len(RTSP_LIST)
    for i in range(0, total, BATCH_SIZE):
        batch_urls = RTSP_LIST[i : i + BATCH_SIZE]
        print(f"\n[Batch {i//BATCH_SIZE + 1}] 카메라 {i+1} ~ {min(i+BATCH_SIZE, total)} 로딩 중...")
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
    
    if not selected_indices: print("선택된 카메라가 없습니다."); return

    print(f"\n>> 총 {len(selected_indices)}대 설정 시작")
    for idx in selected_indices:
        url = RTSP_LIST[idx].strip(); ip = extract_ip(url)
        print(f"\n[{ip}] 설정 중..."); frame = capture_snapshot(url)
        if frame is None: print(" -> 실패"); continue
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
        mgr.set_config(ip, {"url": url, "roi_poly": roi_p, "roi_lines": roi_l, "events": events})

def reassign_cameras(dead_gpu_id, active_gpu_ids, cams, detectors):
    if not active_gpu_ids:
        print("🚨 [CRITICAL] 살아남은 GPU가 없습니다. 시스템을 종료해야 합니다.")
        return False
    new_gpu_id = active_gpu_ids[0]
    print(f"⚠️ [Failover] GPU {dead_gpu_id} 사망 -> GPU {new_gpu_id} 로 카메라 이동 중...")
    moved_count = 0
    for c in cams:
        if c.gpu_id == dead_gpu_id:
            c.gpu_id = new_gpu_id
            c.det_h = detectors[new_gpu_id]['h']
            c.det_g = detectors[new_gpu_id]['g']
            moved_count += 1
    print(f"✅ [Failover] 총 {moved_count}대 카메라 재할당 완료.")
    return True

def main():
    cuda.init()
    physical_gpus = cuda.Device.count()
    print(f"✅ [System] Detected {physical_gpus} NVIDIA GPUs.")
    
    use_gpu_count = physical_gpus
    try:
        user_val = input(f">> 사용할 GPU 개수 입력 (기본값: {physical_gpus}, 최대: {physical_gpus}): ")
        if user_val.strip():
            cnt = int(user_val)
            if 1 <= cnt <= physical_gpus:
                use_gpu_count = cnt
    except: pass
    print(f"⚙️ [설정] 총 {use_gpu_count}개의 GPU를 사용합니다.")

    sensitivity = 5
    try:
        val = input(">> 움직임 감지 민감도 설정 (1-10, 엔터시 기본값 5): ")
        if val.strip():
            sensitivity = int(val)
            sensitivity = max(1, min(10, sensitivity))
    except: pass
    print(f"✅ 민감도 설정: {sensitivity}")

    contexts = []
    detectors = {} 
    active_gpu_ids = [] 
    cams = []
    loop_count = 0 
    
    try:
        mgr = ConfigManager(CONFIG_FILE)
        if not os.path.exists(CONFIG_FILE):
            print("설정 파일 없음."); run_wizard_batch_mode(mgr)
        elif input("설정 초기화? (y/n): ").lower() == 'y':
            mgr.clear_all(); run_wizard_batch_mode(mgr)

        if not os.path.exists(MODEL_HELMET_PATH): print("모델 없음"); return

        print(">> Loading Models...")
        for i in range(use_gpu_count):
            try:
                dev = cuda.Device(i)
                ctx = dev.make_context()
                contexts.append(ctx)
                d_h = YoLoTRT(MODEL_HELMET_PATH, ctx)
                d_g = YoLoTRT(MODEL_GENERAL_PATH, ctx)
                detectors[i] = {'h': d_h, 'g': d_g}
                active_gpu_ids.append(i)
                print(f"  -> GPU {i}: Model Loaded Successfully.")
            except Exception as e:
                print(f"❌ GPU {i} 초기화 실패 (Skip): {e}")
                try: ctx.pop() 
                except: pass

        if not active_gpu_ids:
            print("❌ 사용 가능한 GPU가 없습니다. 종료합니다."); return

        load_count = 0
        num_active = len(active_gpu_ids)

        for rtsp in RTSP_LIST:
            ip = extract_ip(rtsp); conf = mgr.get_config(ip)
            if conf and conf.get('events'):
                if 'url' not in conf: conf['url'] = rtsp.strip()
                cam_id = load_count + 1 
                target_gpu_idx = active_gpu_ids[load_count % num_active]
                target_dets = detectors[target_gpu_idx]
                cams.append(Camera(ip, conf, target_dets['h'], target_dets['g'], target_gpu_idx, cam_id, sensitivity))
                print(f"Load [CAM {cam_id}]: {ip} -> GPU {target_gpu_idx}")
                load_count += 1
        
        if not cams: print("카메라 없음"); return
        
        print("\n[INFO] 모니터링 시작 (상시 녹화 모드 + Failover)")

        while True:
            loop_count += 1
            if loop_count % GC_INTERVAL == 0: gc.collect()
            
            raw_data = [c.process_frame() for c in cams]
            processed_results = [None] * len(cams)
            valid_frame_count = 0 
            
            current_active_gpus = list(active_gpu_ids)
            
            for gpu_idx in current_active_gpus:
                cam_indices = [i for i, c in enumerate(cams) if c.gpu_id == gpu_idx]
                if not cam_indices: continue
                
                try:
                    for idx in cam_indices:
                        c = cams[idx]
                        fr, connected = raw_data[idx]
                        if fr is None or not connected:
                            processed_results[idx] = (None, [], [], False)
                            continue
                        valid_frame_count += 1
                        
                        # [수정] 움직임 여부와 상관없이 항상 추론 (AI Always On)
                        run_helmet = any(e in ["no_helmet", "intrusion"] for e in c.events)
                        run_general = any(e in ["illegal_parking", "conveyor_crossing", "signal_vehicle"] for e in c.events)
                        
                        d_h, d_g = [], []
                        if run_helmet: d_h = c.det_h.infer(fr)
                        if run_general: d_g = c.det_g.infer(fr)
                        
                        processed_results[idx] = (fr, d_h, d_g, True)

                except Exception as e:
                    print(f"\n💀 [FATAL ERROR] GPU {gpu_idx} 에서 오류 발생! (GPU 사망 추정)")
                    print(f"   Error: {e}")
                    
                    if gpu_idx in active_gpu_ids:
                        active_gpu_ids.remove(gpu_idx)
                    
                    success = reassign_cameras(gpu_idx, active_gpu_ids, cams, detectors)
                    if not success:
                        print("시스템을 종료합니다.")
                        raise e 
            
            if valid_frame_count == 0: time.sleep(0.01)

            final_imgs = []
            for idx, res in enumerate(processed_results):
                if res is None: 
                    final_imgs.append(cams[idx].last_draw)
                    continue
                fr, d_h, d_g, connected = res
                if not connected:
                    img = cams[idx].draw(None, [], [], {}, connected=False)
                    final_imgs.append(img)
                else:
                    t_h, t_g, alarms = cams[idx].run_logic(fr, d_h, d_g)
                    img = cams[idx].draw(fr, t_h, t_g, alarms, connected=True)
                    final_imgs.append(img)

            valid_imgs = [img for img in final_imgs if img is not None]
            if valid_imgs:
                mosaic = create_mosaic_image(valid_imgs)
                if mosaic is not None: cv2.imshow("Monitor", mosaic)
            
            key = cv2.waitKey(1)
            if key == ord('q'): break
            elif key == ord('e'):
                print("\n" + "="*40); print("       [ 🛠️ 실시간 설정 변경 모드 ]       "); print("="*40)
                try:
                    print("현재 등록된 카메라 목록:")
                    for c in cams: print(f" [CAM {c.cam_id}] {c.ip} (GPU {c.gpu_id}) | Events: {c.events}")
                    
                    val = input("\n>> 수정할 CAM 번호 입력 (취소: Enter): ").strip()
                    if val == "":
                        print(">> 취소되었습니다.")
                        continue
                    target_id = int(val)

                    target_cam = next((c for c in cams if c.cam_id == target_id), None)
                    if target_cam:
                        print(f"\n>> [CAM {target_id}] 현재 이벤트: {target_cam.events}")
                        print("1.침입 2.주정차 3.안전모 4.횡단 5.신호수차량감지")
                        sel = input(">> 활성화할 이벤트 번호 입력 (예: 1,3) / 모두 끄기: 0 : ")
                        new_events = []
                        if '1' in sel: new_events.append("intrusion")
                        if '2' in sel: new_events.append("illegal_parking")
                        if '3' in sel: new_events.append("no_helmet")
                        if '4' in sel: new_events.append("conveyor_crossing")
                        if '5' in sel: new_events.append("signal_vehicle")
                        
                        new_poly = target_cam.roi_poly; new_lines = target_cam.roi_lines
                        need_poly = any(x in new_events for x in ["intrusion", "illegal_parking", "signal_vehicle"])
                        need_line = "conveyor_crossing" in new_events
                        
                        if need_poly or need_line:
                            print(f"\n🎨 [CAM {target_id}] 구역 설정을 위해 화면을 캡처합니다...")
                            
                            snapshot = None
                            if target_cam.reader.frame is not None:
                                snapshot = target_cam.reader.frame.copy()
                            
                            if snapshot is None:
                                print("⚠️ 현재 프레임 없음. 스냅샷 재접속 시도...")
                                snapshot = capture_snapshot(target_cam.reader.url)

                            if snapshot is not None:
                                if need_poly:
                                    print(" -> 구역(Polygon) 그리기 시작")
                                    new_poly = get_roi_points_scaled(snapshot, f"CAM {target_id} Polygon")
                                if need_line:
                                    print(" -> 횡단 라인(Line) 그리기 시작")
                                    new_lines = []
                                    while True:
                                        l = get_roi_points_scaled(snapshot, f"CAM {target_id} Line", mode="line")
                                        if len(l)==2: new_lines.extend(l)
                                        if input("    라인 추가? (y/n): ")!='y': break
                            else: print("❌ 카메라 데이터 없음. 기존 구역 유지.")

                        target_cam.update_config(new_events, new_poly, new_lines)
                        conf_data = mgr.get_config(target_cam.ip)
                        if conf_data:
                            conf_data['events'] = new_events
                            conf_data['roi_poly'] = new_poly
                            conf_data['roi_lines'] = new_lines
                            mgr.set_config(target_cam.ip, conf_data)
                        print(f"✅ [CAM {target_id}] 설정 및 구역 업데이트 완료: {new_events}")
                except Exception as e:
                    print(f"❌ 설정 오류: {e}")

    except Exception: traceback.print_exc()
    finally:
        print("\n[종료 절차 시작] 리소스 정리 중...")
        for c in cams: c.stop()
        for gpu_id, det_set in detectors.items():
            if det_set.get('h'): det_set['h'].destroy()
            if det_set.get('g'): det_set['g'].destroy()
        for ctx in contexts:
            try: ctx.pop()
            except: pass
        cv2.destroyAllWindows()
        print("[종료 완료]")

if __name__ == "__main__":
    main()