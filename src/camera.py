import cv2
import numpy as np
import time
import datetime
import threading
import subprocess
import queue
import os
import logging
from collections import deque, defaultdict
from common import (
    SYS_CFG, EVENT_ROOT_DIR, WATCHDOG_TIMEOUT, STREAM_RECONNECT_DELAY_SEC, 
    denormalize_roi_points, save_event_image_with_mark, ID_H_HELMET, ID_H_NO_HELMET, 
    ID_G_PERSON, ID_G_CAR, ID_G_BUS, ID_G_TRUCK, NAS_UPLOADER_POOL, _upload_to_nas_task 
)
from event import MotionDetector, EVENT_REGISTRY
from ai_core import ByteTracker

logger = logging.getLogger("VMS_SYSTEM")

def get_stream_codec(url):
    try:
        cmd = ['ffprobe', '-v', 'error', '-rtsp_transport', 'tcp', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', url]
        res = subprocess.check_output(cmd, text=True, timeout=5).strip().lower()
        return "hevc" if "hevc" in res or "265" in res else "h264"
    except Exception: 
        return "h264"

class FrameReader:
    def __init__(self, url, ip):
        self.url = url.replace(" ", "").strip()
        self.ip = ip
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_frame_time = time.time()
        self.out_w = 640
        self.out_h = 360
        self.frame_bytes = self.out_w * self.out_h * 3
        self.process = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            self.connected = False
            if not self.url.startswith("rtsp://"):
                cap = cv2.VideoCapture(self.url)
                self.connected = True
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                delay = 1.0 / fps
                while self.running and cap.isOpened():
                    start_t = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    img = cv2.resize(frame, (self.out_w, self.out_h))
                    with self.lock:
                        self.frame = img
                        self.fid += 1
                        self.last_frame_time = time.time()
                    elapsed = time.time() - start_t
                    if delay > elapsed: 
                        time.sleep(delay - elapsed)
                cap.release()
                self.connected = False
                time.sleep(1)
                continue
                
            codec = get_stream_codec(self.url)
            if codec == "hevc":
                pipeline = f"rtspsrc location={self.url} latency=300 drop-on-latency=true protocols=tcp ! rtph265depay ! h265parse ! decodebin ! videoconvert ! video/x-raw,format=BGR,width={self.out_w},height={self.out_h} ! fdsink fd=1 sync=false"
                cmd = ['gst-launch-1.0', '-q'] + pipeline.split()
            else: 
                cmd = ['ffmpeg', '-nostdin', '-rtsp_transport', 'tcp', '-loglevel', 'error', '-i', self.url, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-an', '-s', f'{self.out_w}x{self.out_h}', '-']
                
            try:
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
                self.connected = True
            except Exception:
                time.sleep(STREAM_RECONNECT_DELAY_SEC)
                continue
                
            while self.running:
                raw = b''
                while len(raw) < self.frame_bytes:
                    if not self.running: 
                        break
                    chunk = self.process.stdout.read(self.frame_bytes - len(raw))
                    if not chunk: 
                        break
                    raw += chunk
                if len(raw) != self.frame_bytes: 
                    break
                img = np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 3)).copy()
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

class VideoRecorder:
    def __init__(self, ip, terminal_id="3"):
        self.ip = ip
        self.terminal_id = terminal_id
        self.recording = False
        self.record_end_time = 0
        self.current_event = "unknown"
        self.running = True
        self.buffer = deque(maxlen=SYS_CFG.get("REC_FPS", 30) * SYS_CFG.get("REC_PRE_SEC", 3))
        self.write_queue = queue.Queue(maxsize=SYS_CFG.get("REC_FPS", 30) * SYS_CFG.get("REC_PRE_SEC", 3) * 2)
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        
    def _queue_frame(self, frame):
        try: 
            self.write_queue.put_nowait(frame)
        except queue.Full:
            try: 
                self.write_queue.get_nowait()
            except Exception: 
                pass
            try: 
                self.write_queue.put_nowait(frame)
            except Exception: 
                pass
                
    def update(self, frame, fid):
        if frame is None: 
            return
        self.buffer.append(frame)
        now = fid / 30.0
        if self.recording:
            if now > self.record_end_time:
                self.recording = False
                self._queue_frame(None)
                logger.info(f"🎬 [녹화종료] {self.ip} - {self.current_event}")
            else: 
                self._queue_frame(frame)
                
    def trigger(self, event_name, fid):
        now = fid / 30.0
        self.record_end_time = now + SYS_CFG.get("REC_POST_SEC", 4)
        if not self.recording:
            logger.info(f"🎥 [녹화시작] {self.ip} - {event_name}")
            self.recording = True
            self.current_event = event_name
            for f in list(self.buffer): 
                self._queue_frame(f)
                
    def _writer_loop(self):
        writer = None
        fpath = None
        while self.running or not self.write_queue.empty():
            try: 
                frame = self.write_queue.get(timeout=1.0)
            except Exception: 
                continue
                
            if frame is None:
                if writer:
                    writer.release()
                    writer = None
                    if fpath and os.path.exists(fpath): 
                        NAS_UPLOADER_POOL.submit(_upload_to_nas_task, fpath, "videos", self.ip, self.current_event, self.terminal_id)
                continue
                
            if writer is None:
                dpath = os.path.join(EVENT_ROOT_DIR, "events", self.ip, "videos", self.current_event)
                os.makedirs(dpath, exist_ok=True)
                fpath = os.path.join(dpath, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.ip}_{self.current_event}.mp4")
                writer = cv2.VideoWriter(fpath, cv2.VideoWriter_fourcc(*'mp4v'), SYS_CFG.get("REC_FPS", 30), (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            
        if writer:
            writer.release()
            if fpath and os.path.exists(fpath): 
                NAS_UPLOADER_POOL.submit(_upload_to_nas_task, fpath, "videos", self.ip, self.current_event, self.terminal_id)
                
    def stop(self):
        self.recording = False
        self.running = False
        self._queue_frame(None)
        if self.thread.is_alive(): 
            self.thread.join(timeout=3.0)

class Camera:
    def __init__(self, ip, conf, face_engine, npu_id, cam_id, sensitivity):
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
        self.npu_id = npu_id
        self.cam_id = cam_id 
        self.latest_npu = {'h': None, 'g': None}
        self.last_submit_fid = -1
        self.helmet_tracker = ByteTracker(is_helmet=True)
        self.general_tracker = ByteTracker(is_helmet=False)
        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}
        self.face_blur_cache = {}
        self.roi_frame_shape = None
        self.config_lock = threading.Lock() 
        self.motion_det = MotionDetector(sensitivity)
        self.recorder = VideoRecorder(ip, self.terminal_id) 
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
                if fscore <= 0.3: 
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

    def run_logic(self, frame, frame_id):
        with self.config_lock:
            self._update_runtime_roi(frame.shape)
            motion_mask = self.motion_det.apply(frame) 
            
            if self.latest_npu['h'] is not None: 
                helmet_tracks = self.helmet_tracker.update(self.latest_npu['h'])
                self.latest_npu['h'] = None
            else: 
                helmet_tracks = self.helmet_tracker.predict_only()
                
            if self.latest_npu['g'] is not None: 
                general_tracks = self.general_tracker.update(self.latest_npu['g'])
                self.latest_npu['g'] = None
            else: 
                general_tracks = self.general_tracker.predict_only()

            now = frame_id / 30.0
            current_alarms = {}
            track_maps = {"helmet": {int(t[4]): int(t[6]) for t in helmet_tracks}, "general": {int(t[4]): int(t[6]) for t in general_tracks}}
            
            for handler in self.handlers:
                for evt in handler.process(helmet_tracks, general_tracks, track_maps, motion_mask, frame, frame_id):
                    draw_tid = evt['tid'] + (0 if handler.required_models[0] == "helmet" else 10000)
                    self._trigger_event(frame, frame_id, draw_tid, handler.event_name, helmet_tracks if handler.required_models[0] == "helmet" else general_tracks, now, event_frame=evt.get('frame'), event_bbox=evt.get('bbox'), event_fid=evt.get('fid'))
                    current_alarms[draw_tid] = handler.event_name
            
            for tid, ename in current_alarms.items(): 
                self.visual_alarms[tid] = {'evt': ename, 'expire': now + SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0)}
                
            for tid in list(self.visual_alarms.keys()):
                if now > self.visual_alarms[tid]['expire']: 
                    del self.visual_alarms[tid]
                    
            return helmet_tracks, general_tracks, {tid: info['evt'] for tid, info in self.visual_alarms.items()}

    def _trigger_event(self, frame, frame_id, tid, event_name, tracks, now, event_frame=None, event_bbox=None, event_fid=None):
        real_tid = tid if tid < 10000 else tid - 10000
        
        # 💡 [핵심 원인 수정] 쿨다운 초기값을 -999999(과거)로 설정하여 첫 번째 이벤트가 차단당하는 논리적 버그 해결
        if event_name in self.alerted[tid] or now - self.last_evt_t.get(event_name, -999999) < self.event_config.get(event_name, {}).get('cooldown_sec', 600): 
            return
            
        bbox = event_bbox if event_bbox is not None else next((t[:4] for t in tracks if int(t[4]) == real_tid), None)
        if bbox is None: 
            return
        
        logger.warning(f"🚨 [CAM {self.cam_id}] {event_name} Detected! ID:{real_tid}")
        source_frame = event_frame if event_frame is not None else frame
        source_fid = event_fid if event_fid is not None else frame_id
        
        if event_frame is None:
            if source_fid not in self.face_blur_cache:
                self.face_blur_cache[source_fid] = self._apply_face_blur(source_frame.copy())
                if len(self.face_blur_cache) > 5: 
                    self.face_blur_cache.pop(next(iter(self.face_blur_cache)))
            saved_img = self.face_blur_cache[source_fid]
        else: 
            saved_img = self._apply_face_blur(source_frame.copy())
            
        save_event_image_with_mark(saved_img, self.ip, event_name, bbox, real_tid, terminal_id=self.terminal_id, cctv_id=self.cctv_id)
        self.recorder.trigger(event_name, source_fid)
        self.alerted[tid].add(event_name)
        self.last_evt_t[event_name] = now

    def draw(self, frame, helmet_tracks, general_tracks, alarms, connected=True):
        if frame is None or not connected:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
            return blank
            
        h_frame, w_frame = frame.shape[:2]
        
        if len(alarms) > 0: 
            cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 0, 255), 4)
            
        if self.roi_poly: 
            cv2.polylines(frame, [np.array(self.roi_poly, np.int32)], True, (0,255,255), 1)
            
        if self.roi_lines:
            for i in range(0, len(self.roi_lines) - 1, 2): 
                cv2.line(frame, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0,0,255), 1)
        
        for t in helmet_tracks:
            tid = int(t[4])
            cls_id = int(t[6])
            if cls_id == ID_H_HELMET: 
                color, label = (255, 0, 0), "Helmet"
            elif cls_id == ID_H_NO_HELMET: 
                color, label = (0, 0, 255), "No-Helmet"
            else: 
                color, label = (0, 255, 0), "Person"
                
            if tid in alarms: 
                color, label = (0, 0, 255), f"ALARM: {label}"
                
            thickness = 2 if tid in alarms else 1
            cv2.rectangle(frame, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(frame, f"{label} [{tid}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        for t in general_tracks:
            tid = int(t[4]) + 10000
            cls_id = int(t[6])
            if cls_id == ID_G_PERSON: 
                label = "Person"
            elif cls_id == ID_G_CAR: 
                label = "Car"
            elif cls_id == ID_G_BUS: 
                label = "Bus"
            elif cls_id == ID_G_TRUCK: 
                label = "Truck"
            else: 
                label = "OBJ"
                
            color = (0, 0, 255) if tid in alarms else (255, 100, 0)
            thickness = 2 if tid in alarms else 1
            cv2.rectangle(frame, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(frame, f"{label} [{tid}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        cv2.rectangle(frame, (0, 0), (60, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"C{self.cam_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.recorder.recording:
            cv2.circle(frame, (w_frame - 20, 20), 5, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w_frame - 55, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
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
        self.recorder.stop()