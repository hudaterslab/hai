import cv2
import numpy as np
import time
import datetime
import threading
import queue
import os
import logging
import subprocess
import psutil
from collections import deque, defaultdict
from common import (
    SYS_CFG, EVENT_ROOT_DIR, WATCHDOG_TIMEOUT, STREAM_RECONNECT_DELAY_SEC, 
    denormalize_roi_points, save_event_image_with_mark, ID_H_HELMET, ID_H_NO_HELMET, 
    ID_G_PERSON, ID_G_CAR, ID_G_TRUCK, ID_PERSON_LOW, ID_REFLECTIVE_VEST,
    NAS_UPLOADER_POOL, _upload_to_nas_task 
)
from event import MotionDetector, EVENT_REGISTRY
from ai_core import SORTTracker

logger = logging.getLogger("VMS_SYSTEM")

class FrameReader:
    def __init__(self, url, ip):
        self.url = url.replace(" ", "").replace("\n", "").replace("\r", "").strip()
        self.ip = ip
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_frame_time = time.time()
        
        self.out_w = 640
        self.out_h = 480
        self.frame_bytes = self.out_w * self.out_h * 3
        
        self.target_fps = SYS_CFG.get("REC_FPS", 3)
        self.process = None
        
        self.use_gstreamer = True
        self.gst_fail_count = 0
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            self.connected = False
            
            if self.use_gstreamer:
                pipeline = (
                    f"rtspsrc location={self.url} latency=500 ! "
                    f"decodebin ! videoconvert ! videorate ! "
                    f"video/x-raw,format=BGR,width={self.out_w},height={self.out_h},framerate={self.target_fps}/1 ! "
                    f"fdsink fd=1 sync=false"
                )
                
                cmd = ['gst-launch-1.0', '-q'] + pipeline.split()
                
                try:
                    self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
                except Exception:
                    self.gst_fail_count += 1
                    time.sleep(STREAM_RECONNECT_DELAY_SEC)
                    continue
                    
                read_success = False
                while self.running:
                    if psutil.cpu_percent(interval=None) > 95:
                        time.sleep(0.05)
                        
                    raw = b''
                    while len(raw) < self.frame_bytes:
                        if not self.running: 
                            break
                        try:
                            chunk = self.process.stdout.read(self.frame_bytes - len(raw))
                            if not chunk: 
                                break
                            raw += chunk
                        except Exception:
                            break
                            
                    if len(raw) != self.frame_bytes: 
                        break
                        
                    img = np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 3)).copy()
                    read_success = True
                    self.connected = True
                    self.gst_fail_count = 0 
                    
                    with self.lock:
                        self.frame = img
                        self.fid += 1
                        self.last_frame_time = time.time()
                        
                self.connected = False
                if self.process:
                    try: 
                        self.process.kill()
                    except Exception: 
                        pass
                    self.process = None
                
                if not read_success:
                    self.gst_fail_count += 1
                    
                if self.gst_fail_count >= 2:
                    self.use_gstreamer = False
                    
            else:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                if not cap.isOpened():
                    time.sleep(STREAM_RECONNECT_DELAY_SEC)
                    continue
                    
                self.connected = True
                
                last_read_time = time.time()
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    now = time.time()
                    if now - last_read_time >= (1.0 / self.target_fps):
                        frame = cv2.resize(frame, (self.out_w, self.out_h))
                        with self.lock:
                            self.frame = frame
                            self.fid += 1
                            self.last_frame_time = now
                        last_read_time = now
                        
                cap.release()
                self.connected = False
                
            if self.running: 
                time.sleep(STREAM_RECONNECT_DELAY_SEC)

    def read(self):
        with self.lock: 
            return (None, self.fid, False) if time.time() - self.last_frame_time > WATCHDOG_TIMEOUT else (self.frame, self.fid, self.connected)
            
    def stop(self, join_timeout=3.0):
        self.running = False
        if self.process:
            try: 
                self.process.kill()
            except Exception: 
                pass
        if self.thread.is_alive(): 
            self.thread.join(timeout=join_timeout)

class Camera:
    # 💡 헬멧 전용 모델(helmet_engine)을 주입받도록 시그니처 변경
    def __init__(self, ip, conf, face_engine, helmet_engine, cam_id, sensitivity):
        self.ip = ip
        self.conf = conf 
        self.reader = FrameReader(conf['url'], ip)
        self.terminal_id = str(conf.get('terminal_id', SYS_CFG.get("terminal_id", "99999")))
        self.cctv_id = int(conf.get('cctv_id', 1))
        self.event_config = conf.get('event_config', {})
        self.roi_poly_norm = conf.get('roi_poly_norm', [])
        self.roi_lines_norm = conf.get('roi_lines_norm', [])
        self.using_normalized_roi = bool(self.roi_poly_norm or self.roi_lines_norm)
        self.roi_poly = []
        self.roi_lines = []
        self.events = conf.get('events', [])
        self.face_detector = face_engine
        self.helmet_detector = helmet_engine  # 헬멧 전용 모델 추가
        self.cam_id = cam_id 
        self.last_submit_fid = -1
        
        main_conf = SYS_CFG.get("model_confidences", {}).get("MAIN", 0.40)
        self.face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
        self.helmet_conf = SYS_CFG.get("model_confidences", {}).get("HELMET", 0.45) # 헬멧 Thres
        
        self.fps = SYS_CFG.get("REC_FPS", 3)
        self.skip = SYS_CFG.get("SKIP_FRAMES", 1)
        
        track_buffer_sec = SYS_CFG.get("track_buffer_sec", 1.5)
        target_buffer = max(1, int(track_buffer_sec * (self.fps / self.skip)))
        
        # 💡 [핵심 보완] Main 모델용 트래커와 Helmet 모델용 트래커를 완전히 분리
        self.main_tracker = SORTTracker(track_thresh=main_conf, track_buffer=target_buffer, is_helmet=False)
        self.helmet_tracker = SORTTracker(track_thresh=self.helmet_conf, track_buffer=target_buffer, is_helmet=False)
        
        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}
        self.face_blur_cache = {}
        self.roi_frame_shape = None
        self.config_lock = threading.Lock() 
        self.motion_det = MotionDetector(sensitivity)
        
        self.obj_history = {}
        self.delayed_logs = []
        
        self.pre_log_sec = SYS_CFG.get("event_pre_log_sec", 2.0)
        self.post_log_sec = SYS_CFG.get("event_post_log_sec", 2.0)
        
        self.init_handlers()

    def _update_runtime_roi(self, frame_shape):
        if not self.using_normalized_roi or self.roi_frame_shape == frame_shape[:2]: 
            return
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
        if self.face_detector is None: 
            return image
        try:
            for fx1, fy1, fx2, fy2, fscore, _ in self.face_detector.infer(image):
                if fscore <= self.face_conf: 
                    continue
                fx1 = max(0, int(fx1))
                fy1 = max(0, int(fy1))
                fx2 = int(fx2)
                fy2 = int(fy2)
                fh = fy2 - fy1
                fw = fx2 - fx1
                if fw > image.shape[1] * 0.8 or fh > image.shape[0] * 0.8 or image[fy1:fy2, fx1:fx2].size == 0: 
                    continue
                small = cv2.resize(image[fy1:fy2, fx1:fx2], (fw // 15 + 1, fh // 15 + 1), interpolation=cv2.INTER_LINEAR)
                image[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
        except Exception: 
            pass
        return image

    def run_logic(self, frame, frame_id, main_boxes=None, helmet_boxes=None):
        with self.config_lock:
            self._update_runtime_roi(frame.shape)
            motion_mask = self.motion_det.apply(frame) 
            
            current_obj_count = len(main_boxes) if main_boxes is not None else 0
            self.obj_history[frame_id] = current_obj_count
            
            for k in list(self.obj_history.keys()):
                if frame_id - k > self.fps * 10:  
                    del self.obj_history[k]
                    
            remaining_logs = []
            for dlog in self.delayed_logs:
                if frame_id >= dlog['target_fid_to_log']:
                    before_fid = dlog['trigger_fid'] - int(self.pre_log_sec * self.fps)
                    before_count = self.obj_history.get(before_fid, "Unknown")
                    trigger_count = self.obj_history.get(dlog['trigger_fid'], "Unknown")
                    after_count = current_obj_count
                    
                    logger.info(f"🚨 [EVENT] CAM:{self.cam_id} | Type:{dlog['event_name']} | "
                                f"Triggered At:{dlog['time_str']} | "
                                f"Obj Count -> -{self.pre_log_sec}s: {before_count} | 0s: {trigger_count} | +{self.post_log_sec}s: {after_count}")
                else:
                    remaining_logs.append(dlog)
            self.delayed_logs = remaining_logs
            
            # 메인 트래커 업데이트
            if main_boxes is not None and len(main_boxes) > 0:
                main_tracks = self.main_tracker.update(np.array(main_boxes))
            else:
                main_tracks = self.main_tracker.predict_only()
                
            # 헬멧 트래커 업데이트 (별도 운용)
            helmet_tracks = []
            if self.helmet_detector and "no_helmet" in self.events:
                if helmet_boxes is not None and len(helmet_boxes) > 0:
                    helmet_tracks = self.helmet_tracker.update(np.array(helmet_boxes))
                else:
                    helmet_tracks = self.helmet_tracker.predict_only()

            now = time.time()
            current_alarms = {}
            track_map = {int(t[4]): int(t[6]) for t in main_tracks}
            
            for handler in self.handlers:
                # 💡 Detector에 kwargs 형태로 helmet_tracks를 전달
                for evt in handler.process(main_tracks, track_map, motion_mask, frame, frame_id, helmet_tracks=helmet_tracks):
                    draw_tid = evt['tid'] 
                    self._trigger_event(frame, frame_id, draw_tid, handler.event_name, main_tracks, now, event_frame=evt.get('frame'), event_bbox=evt.get('bbox'), event_fid=evt.get('fid'))
                    current_alarms[draw_tid] = handler.event_name
            
            for tid, ename in current_alarms.items(): 
                self.visual_alarms[tid] = {'evt': ename, 'expire': now + SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0)}
                
            for tid in list(self.visual_alarms.keys()):
                if now > self.visual_alarms[tid]['expire']: 
                    del self.visual_alarms[tid]
                    
            return main_tracks, {tid: info['evt'] for tid, info in self.visual_alarms.items()}

    def _trigger_event(self, frame, frame_id, tid, event_name, tracks, now, event_frame=None, event_bbox=None, event_fid=None):
        real_tid = tid
        if event_name in self.alerted[tid] or now - self.last_evt_t.get(event_name, -999999) < self.event_config.get(event_name, {}).get('cooldown_sec', 600): 
            return
            
        bbox = event_bbox if event_bbox is not None else next((t[:4] for t in tracks if int(t[4]) == real_tid), None)
        if bbox is None: 
            return
            
        source_fid = event_fid if event_fid is not None else frame_id
        source_frame = event_frame if event_frame is not None else frame
        
        self.delayed_logs.append({
            'event_name': event_name,
            'trigger_fid': source_fid,
            'target_fid_to_log': source_fid + int(self.post_log_sec * self.fps),
            'time_str': datetime.datetime.now().strftime('%H:%M:%S')
        })
        
        if event_frame is None:
            if frame_id not in self.face_blur_cache:
                self.face_blur_cache[frame_id] = self._apply_face_blur(source_frame.copy())
                if len(self.face_blur_cache) > 5: 
                    self.face_blur_cache.pop(next(iter(self.face_blur_cache)))
            saved_img = self.face_blur_cache[frame_id]
        else: 
            saved_img = self._apply_face_blur(source_frame.copy())
            
        save_event_image_with_mark(saved_img, self.ip, event_name, bbox, real_tid, terminal_id=self.terminal_id, cctv_id=self.cctv_id)
        
        self.alerted[tid].add(event_name)
        self.last_evt_t[event_name] = now

    def draw(self, frame, tracks, alarms, connected=True):
        if frame is None or not connected:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
            return blank
            
        h_frame, w_frame = frame.shape[:2]
        
        if len(alarms) > 0: 
            cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 0, 255), 4)
            
        if self.roi_poly: 
            cv2.polylines(frame, [np.array(self.roi_poly, np.int32)], True, (0,255,255), 1)
            
        if self.roi_lines:
            for i in range(0, len(self.roi_lines) - 1, 2): 
                cv2.line(frame, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0,0,255), 1)
        
        for t in tracks:
            tid = int(t[4])
            cls_id = int(t[6])
            
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
            cv2.rectangle(frame, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(frame, f"{label} [{tid}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        cv2.rectangle(frame, (0, 0), (60, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"C{self.cam_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        active_alarms = set(alarms.values())
        for i, handler in enumerate(self.handlers):
            if handler.event_name in active_alarms:
                color, text = (0, 0, 255), f"[!] {handler.gui_name}"
            else:
                color, text = (0, 255, 0), f" -  {handler.gui_name}"
            cv2.putText(frame, text, (10, h_frame - 15 - (len(self.handlers)-1-i)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return frame

    def stop(self):
        self.reader.stop()