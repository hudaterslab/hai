import cv2
import numpy as np
import time
import datetime
import threading
import queue
import os
import logging
import re
from urllib.parse import unquote # 💡 추가: URL 인코딩 해제
from collections import deque, defaultdict
from common import (
    SYS_CFG, EVENT_ROOT_DIR, WATCHDOG_TIMEOUT, STREAM_RECONNECT_DELAY_SEC, 
    denormalize_roi_points, save_event_image_with_mark, ID_H_HELMET, ID_H_NO_HELMET, 
    ID_G_PERSON, ID_G_CAR, ID_G_TRUCK, ID_PERSON_LOW, ID_REFLECTIVE_VEST, calculate_iou
)
from event import MotionDetector, EVENT_REGISTRY
from ai_core import SORTTracker

logger = logging.getLogger("VMS_SYSTEM")

class FrameReader:
    def __init__(self, url, ip):
        # 💡 강력한 URL 멸균: %20 등 인코딩 문자를 실제 문자로 치환 후, 모든 종류의 공백과 보이지 않는 유니코드 문자 파쇄
        raw_url = unquote(url)
        clean_url = re.sub(r'[\s\u200B\u200C\u200D\uFEFF]+', '', raw_url.strip())
        
        self.url = clean_url
        self.ip = ip
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_frame_time = time.time()
        self.is_stuck = False
        
        self.out_w = 640
        self.out_h = 480
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"
        
        while self.running:
            self.connected = False
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                time.sleep(STREAM_RECONNECT_DELAY_SEC)
                continue
                
            self.connected = True
            self.last_frame_time = time.time()
            self.is_stuck = False
            
            while self.running and cap.isOpened():
                if time.time() - self.last_frame_time > WATCHDOG_TIMEOUT:
                    logger.warning(f"[{self.ip}] OpenCV Read Timeout. 멈춤(Deadlock) 감지.")
                    self.is_stuck = True 
                    break
                
                ret, fr = cap.read()
                if not ret: 
                    logger.warning(f"[{self.ip}] 스트림 끊김.")
                    break
                
                if fr is not None:
                    fr = cv2.resize(fr, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)

                with self.lock:
                    self.frame = fr
                    self.fid += 1
                    self.last_frame_time = time.time()
                
                time.sleep(0.005)
            
            self.connected = False
            try: cap.release()
            except: pass
            
            if self.running: time.sleep(STREAM_RECONNECT_DELAY_SEC)

    def read(self):
        with self.lock:
            if self.is_stuck or (time.time() - self.last_frame_time > WATCHDOG_TIMEOUT):
                return None, self.fid, False
            return self.frame, self.fid, self.connected

class VideoRecorder:
    def __init__(self, ip):
        self.ip = ip
        self.buffer = deque(maxlen=SYS_CFG.get("REC_FPS", 15) * 10) 
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
            self.record_end_time = now + 10.0
        else:
            logger.info(f"🎥 [녹화시작] {self.ip} - {event_name}")
            self.recording = True
            self.record_end_time = now + 10.0
            self.current_event = event_name
            temp_buffer = list(self.buffer)
            for f in temp_buffer: self.write_queue.put(f)

    def _writer_loop(self):
        writer = None
        while self.running:
            try: frame = self.write_queue.get(timeout=1.0)
            except queue.Empty: continue

            if frame is None:
                if writer: writer.release(); writer = None
                continue

            if writer is None:
                dpath = os.path.join(EVENT_ROOT_DIR, "events", self.ip, "videos", self.current_event)
                if not os.path.exists(dpath): os.makedirs(dpath, exist_ok=True)
                fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.ip}_{self.current_event}.mp4"
                fpath = os.path.join(dpath, fname)
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(fpath, fourcc, SYS_CFG.get("REC_FPS", 15), (w, h))
            if writer: writer.write(frame)

class Camera:
    def __init__(self, ip, conf, face_engine, cam_id, sensitivity):
        self.ip = ip; self.conf = conf 
        self.reader = FrameReader(conf['url'], ip)
        self.terminal_id = str(conf.get('terminal_id', SYS_CFG.get("terminal_id", "99999")))
        self.cctv_id = int(conf.get('cctv_id', 1))
        self.event_config = conf.get('event_config', {})
        self.roi_poly_norm = conf.get('roi_poly_norm', [])
        self.roi_lines_norm = conf.get('roi_lines_norm', [])
        self.using_normalized_roi = bool(self.roi_poly_norm or self.roi_lines_norm)
        self.roi_poly, self.roi_lines = [], []
        self.events = conf.get('events', [])
        self.face_detector = face_engine
        self.cam_id = cam_id 
        
        main_conf = SYS_CFG.get("model_confidences", {}).get("MAIN", 0.40)
        self.face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
        self.helmet_conf = SYS_CFG.get("model_confidences", {}).get("HELMET", 0.45) 
        
        self.fps = SYS_CFG.get("REC_FPS", 3)
        self.skip = SYS_CFG.get("SKIP_FRAMES", 1)
        target_buffer = max(1, int(SYS_CFG.get("track_buffer_sec", 1.5) * (self.fps / self.skip)))
        
        self.main_tracker = SORTTracker(track_thresh=main_conf, track_buffer=target_buffer, is_helmet=False)
        self.helmet_tracker = SORTTracker(track_thresh=self.helmet_conf, track_buffer=target_buffer, is_helmet=True)
        
        self.alerted, self.last_evt_t, self.visual_alarms, self.face_blur_cache = defaultdict(set), {}, {}, {}
        self.roi_frame_shape = None
        self.config_lock = threading.Lock() 
        self.motion_det = MotionDetector(sensitivity)
        
        self.recorder = VideoRecorder(ip)
        
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

    def _update_runtime_roi(self, frame_shape):
        if not self.using_normalized_roi or self.roi_frame_shape == frame_shape[:2]: return
        h, w = frame_shape[:2]
        self.roi_poly = denormalize_roi_points(self.roi_poly_norm, w, h)
        self.roi_lines = denormalize_roi_points(self.roi_lines_norm, w, h)
        self.roi_frame_shape = frame_shape[:2]
        self.init_handlers()

    def init_handlers(self):
        self.handlers = []
        for evt in self.events:
            if evt in EVENT_REGISTRY:
                self.handlers.append(EVENT_REGISTRY[evt](self.event_config.get(evt, {}), self.roi_poly, self.roi_lines))

    def _apply_face_blur(self, image):
        if self.face_detector is None: return image
        try:
            for fx1, fy1, fx2, fy2, fscore, _ in self.face_detector.infer(image):
                if fscore <= self.face_conf: continue
                fx1, fy1, fx2, fy2 = max(0, int(fx1)), max(0, int(fy1)), int(fx2), int(fy2)
                fh, fw = fy2 - fy1, fx2 - fx1
                if fw > image.shape[1] * 0.8 or fh > image.shape[0] * 0.8 or image[fy1:fy2, fx1:fx2].size == 0: continue
                small = cv2.resize(image[fy1:fy2, fx1:fx2], (fw // 15 + 1, fh // 15 + 1), interpolation=cv2.INTER_LINEAR)
                image[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
        except Exception: pass
        return image

    def _snap_tracks_to_raw(self, tracks, raw_boxes):
        if raw_boxes is None or len(raw_boxes) == 0 or len(tracks) == 0: return tracks
        snapped_tracks = []
        for t in tracks:
            tx1, ty1, tx2, ty2, tid, conf, cls_id = t
            best_iou, best_raw = 0, None
            for rb in raw_boxes:
                if int(rb[5]) == int(cls_id):
                    iou = calculate_iou((tx1, ty1, tx2, ty2), rb[:4])
                    if iou > best_iou: best_iou, best_raw = iou, rb
            if best_iou > 0.05 and best_raw is not None:
                snapped_tracks.append([best_raw[0], best_raw[1], best_raw[2], best_raw[3], tid, best_raw[4], cls_id])
            else: snapped_tracks.append(t)
        return snapped_tracks

    def run_logic(self, frame, frame_id, raw_boxes=None):
        with self.config_lock:
            motion_mask = self.motion_det.apply(frame) 
            
            main_boxes, helmet_boxes = [], []
            if raw_boxes is not None:
                for b in raw_boxes:
                    cls_id = int(b[5])
                    if cls_id in (ID_H_HELMET, ID_H_NO_HELMET): helmet_boxes.append(b)
                    else: main_boxes.append(b)
            
            main_tracks = self.main_tracker.update(np.array(main_boxes)) if len(main_boxes) > 0 else self.main_tracker.predict_only()
            main_tracks = self._snap_tracks_to_raw(main_tracks, main_boxes)
                
            helmet_tracks = []
            if "no_helmet" in self.events:
                helmet_tracks = self.helmet_tracker.update(np.array(helmet_boxes)) if len(helmet_boxes) > 0 else self.helmet_tracker.predict_only()
                helmet_tracks = self._snap_tracks_to_raw(helmet_tracks, helmet_boxes)

            now = time.time()
            current_alarms = {}
            track_map = {int(t[4]): int(t[6]) for t in main_tracks}
            
            for handler in self.handlers:
                for evt in handler.process(main_tracks, track_map, motion_mask, frame, frame_id, helmet_tracks=helmet_tracks, raw_helmet_boxes=helmet_boxes):
                    draw_tid = evt['tid'] 
                    self._trigger_event(frame, frame_id, draw_tid, handler.event_name, main_tracks, now, evt.get('frame'), evt.get('bbox'))
                    current_alarms[draw_tid] = handler.event_name
            
            for tid, ename in current_alarms.items(): self.visual_alarms[tid] = {'evt': ename, 'expire': now + SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0)}
            for tid in list(self.visual_alarms.keys()):
                if now > self.visual_alarms[tid]['expire']: del self.visual_alarms[tid]
                    
            return main_tracks, {tid: info['evt'] for tid, info in self.visual_alarms.items()}

    def _trigger_event(self, frame, frame_id, tid, event_name, tracks, now, event_frame=None, event_bbox=None):
        real_tid = tid
        if event_name in self.alerted[tid] or now - self.last_evt_t.get(event_name, -999999) < self.event_config.get(event_name, {}).get('cooldown_sec', 600): 
            return
            
        bbox = event_bbox if event_bbox is not None else next((t[:4] for t in tracks if int(t[4]) == real_tid), None)
        if bbox is None: return
            
        source_frame = event_frame if event_frame is not None else frame
        
        if event_frame is None:
            if frame_id not in self.face_blur_cache:
                self.face_blur_cache[frame_id] = self._apply_face_blur(source_frame.copy())
                if len(self.face_blur_cache) > 5: self.face_blur_cache.pop(next(iter(self.face_blur_cache)))
            saved_img = self.face_blur_cache[frame_id]
        else: 
            saved_img = self._apply_face_blur(source_frame.copy())
            
        save_event_image_with_mark(saved_img, self.ip, event_name, bbox, real_tid, terminal_id=self.terminal_id, cctv_id=self.cctv_id)
        
        self.recorder.trigger(event_name)
        self.alerted[tid].add(event_name)
        self.last_evt_t[event_name] = now

    def draw(self, frame, tracks, alarms, connected=True):
        if frame is None or not connected:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
            return blank
            
        h_frame, w_frame = frame.shape[:2]
        
        if len(alarms) > 0: cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 0, 255), 4)
        if self.roi_poly: cv2.polylines(frame, [np.array(self.roi_poly, np.int32)], True, (0,255,255), 1)
        if self.roi_lines:
            for i in range(0, len(self.roi_lines) - 1, 2): cv2.line(frame, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0,0,255), 1)
        
        for t in tracks:
            x1, y1, x2, y2, tid, conf, cls_id = map(float, t)
            tid, cls_id = int(tid), int(cls_id)
            
            if cls_id == ID_H_HELMET: color, label = (255, 0, 0), "Helmet"
            elif cls_id == ID_H_NO_HELMET: color, label = (0, 0, 255), "No-Helmet"
            elif cls_id == ID_G_PERSON: color, label = (0, 255, 0), "Person"
            elif cls_id == ID_G_CAR: color, label = (255, 100, 0), "Car"
            elif cls_id == ID_G_TRUCK: color, label = (255, 100, 0), "Truck"
            elif cls_id == ID_PERSON_LOW: color, label = (0, 255, 100), "LowBody"
            elif cls_id == ID_REFLECTIVE_VEST: color, label = (255, 255, 0), "Vest"
            else: color, label = (255, 255, 255), "OBJ"
            
            if tid in alarms: color, label = (0, 0, 255), f"ALARM: {label}"
                
            thickness = 2 if tid in alarms else 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f"{label} [{tid}]", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        cv2.rectangle(frame, (0, 0), (60, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"C{self.cam_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.recorder.recording:
            cv2.circle(frame, (w_frame - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w_frame - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        active_alarms = set(alarms.values())
        for i, handler in enumerate(self.handlers):
            if handler.event_name in active_alarms: color, text = (0, 0, 255), f"[!] {handler.gui_name}"
            else: color, text = (0, 255, 0), f" -  {handler.gui_name}"
            cv2.putText(frame, text, (10, h_frame - 15 - (len(self.handlers)-1-i)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return frame

    def stop(self):
        self.reader.running = False
        self.recorder.running = False
        if self.reader.thread.is_alive(): self.reader.thread.join(timeout=3.0)