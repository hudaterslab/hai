import os
import cv2
import numpy as np
import time
import queue
import logging
import threading
import subprocess
from common import clean_overlapping_detections, calculate_iou, SYS_CFG

logger = logging.getLogger("VMS_SYSTEM")

def get_model_confidence(engine_path, default_conf=0.45):
    conf_map = SYS_CFG.get("model_confidences", {})
    ep_lower = engine_path.lower()
    
    if "helmet" in ep_lower: 
        return conf_map.get("HELMET", 0.50)
    elif "face" in ep_lower: 
        return conf_map.get("FACE", 0.35)
    else: 
        return conf_map.get("GENERAL", 0.60)

def resolve_model_path(engine_path, is_gpu=False):
    target_path = engine_path.replace(".dxnn", ".pt") if is_gpu else engine_path.replace(".pt", ".dxnn")
    
    if os.path.exists(target_path): 
        return target_path
    return os.path.join("models", os.path.basename(target_path))

def check_deepx_npu():
    try: 
        return subprocess.run(["dxrt-cli", "-i"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).returncode == 0
    except Exception: 
        return False

USE_NPU = check_deepx_npu()
async_result_queue = queue.Queue(maxsize=2000)

if USE_NPU:
    try:
        from dx_engine import InferenceEngine, InferenceOption
        logger.info("🟢 DeepX NPU 활성화: dx_engine을 가동합니다.")
    except Exception:
        USE_NPU = False
        logger.warning("🟡 NPU 임포트 실패. GPU 모드로 대체합니다.")

if USE_NPU:
    def onInferenceCallbackFunc(outputs, user_arg):
        cam_id, model_type, fid, scale, offset, semaphore, is_yolov7, conf_thres, input_tensor = user_arg
        boxes = []
        try:
            pred = np.array(outputs[0], copy=True)
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]: 
                pred = pred.transpose((0, 2, 1))
            if pred.ndim == 3: 
                pred = pred[0] 
            
            C = pred.shape[1]
            if is_yolov7 or C == 6 or C == 85:
                obj_conf = pred[:, 4]
                if C > 5:
                    scores = obj_conf * np.max(pred[:, 5:], axis=1)
                    class_ids = np.argmax(pred[:, 5:], axis=1)
                else:
                    scores = obj_conf
                    class_ids = np.zeros(len(obj_conf), dtype=int)
            else:
                scores = np.max(pred[:, 4:], axis=1)
                class_ids = np.argmax(pred[:, 4:], axis=1)

            mask = scores > 0.1
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            if len(pred) > 0:
                raw_boxes = pred[:, :4].copy()
                raw_boxes[:, 0] = raw_boxes[:, 0] - raw_boxes[:, 2] / 2
                raw_boxes[:, 1] = raw_boxes[:, 1] - raw_boxes[:, 3] / 2
                raw_boxes[:, 2] = raw_boxes[:, 0] + raw_boxes[:, 2]
                raw_boxes[:, 3] = raw_boxes[:, 1] + raw_boxes[:, 3]
                
                indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), scores.tolist(), 0.1, 0.45)
                if len(indices) > 0:
                    dw, dh = offset
                    for i in indices.flatten():
                        bx1, by1 = (raw_boxes[i][0] - dw) / scale, (raw_boxes[i][1] - dh) / scale
                        bx2, by2 = (raw_boxes[i][2] - dw) / scale, (raw_boxes[i][3] - dh) / scale
                        boxes.append([bx1, by1, bx2, by2, scores[i], class_ids[i]])
                        
        except Exception as e: 
            logger.error(f"⚠️ NPU Async Error: {e}")
        finally: 
            # 💡 [핵심 보완] 에러가 났더라도 메인 루프 락을 풀기 위해 결과를 큐에 밀어넣음
            if not async_result_queue.full(): 
                async_result_queue.put((cam_id, model_type, fid, np.array(boxes)))
            semaphore.release() 
        return 0

    class DeepXModelAsync:
        def __init__(self, engine_path):
            self.engine_path = resolve_model_path(engine_path, is_gpu=False)
            self.is_yolov7 = "v7" in os.path.basename(self.engine_path).lower()
            self.conf_thres = get_model_confidence(self.engine_path)
            
            if not os.path.exists(self.engine_path): 
                self.engine = None
            else:
                self.engine = InferenceEngine(self.engine_path, InferenceOption())
                self.engine.register_callback(onInferenceCallbackFunc)
                
        def letter_box(self, img, new_shape=(640,640)):
            h, w = img.shape[:2]
            scale = min(new_shape[0]/h, new_shape[1]/w)
            nw, nh = int(w*scale), int(h*scale)
            resized = cv2.resize(img, (nw, nh))
            canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
            dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
            canvas[dh:dh+nh, dw:dw+nw] = resized
            return canvas, scale, (dw, dh)
            
        def submit_async(self, img, cam_id, model_type, fid, semaphore):
            if img is None or self.engine is None:
                semaphore.release()
                return
                
            npu_input, scale, offset = self.letter_box(img)
            input_tensor = np.ascontiguousarray(cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB))
            user_arg = (cam_id, model_type, fid, scale, offset, semaphore, self.is_yolov7, self.conf_thres, input_tensor)
            self.engine.run_async([input_tensor], user_arg=user_arg)

    class DeepXModelSync:
        def __init__(self, engine_path):
            self.engine_path = resolve_model_path(engine_path, is_gpu=False)
            self.is_yolov7 = "v7" in os.path.basename(self.engine_path).lower()
            if not os.path.exists(self.engine_path): 
                self.engine = None
            else: 
                self.engine = InferenceEngine(self.engine_path, InferenceOption())
                
        def infer(self, img): 
            return []

    VisionModelAsync = DeepXModelAsync
    VisionModelSync = DeepXModelSync

else:
    logger.info("🔵 NVIDIA GPU/CPU 모드 활성화: PyTorch(Ultralytics) 추론 엔진을 가동합니다.")
    
    class GPUModelAsync:
        def __init__(self, engine_path):
            self.pt_path = resolve_model_path(engine_path, is_gpu=True)
            self.is_yolov7 = "v7" in os.path.basename(self.pt_path).lower()
            self.model = None
            try:
                from ultralytics import YOLO
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if os.path.exists(self.pt_path): 
                    self.model = YOLO(self.pt_path)
            except Exception: 
                pass

        def submit_async(self, img, cam_id, model_type, fid, semaphore):
            if img is None or self.model is None:
                semaphore.release()
                return
                
            def _infer():
                boxes = []
                try:
                    results = self.model(img, verbose=False, conf=0.1)
                    for r in results:
                        if r.boxes is not None and len(r.boxes) > 0:
                            xyxy = r.boxes.xyxy.cpu().numpy()
                            conf = r.boxes.conf.cpu().numpy()
                            cls = r.boxes.cls.cpu().numpy()
                            for i in range(len(xyxy)): 
                                boxes.append([xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], conf[i], int(cls[i])])
                except Exception as e: 
                    logger.error(f"⚠️ PyTorch Inference Error (메모리 초과 예상): {e}")
                finally: 
                    # 💡 [핵심 보완] 에러가 났더라도 메인 루프 락을 풀기 위해 결과를 큐에 밀어넣음
                    if not async_result_queue.full(): 
                        async_result_queue.put((cam_id, model_type, fid, np.array(boxes)))
                    semaphore.release()
                    
            threading.Thread(target=_infer, daemon=True).start()

    class GPUModelSync:
        def __init__(self, engine_path):
            self.pt_path = resolve_model_path(engine_path, is_gpu=True)
            self.model = None
            try:
                from ultralytics import YOLO
                if os.path.exists(self.pt_path): 
                    self.model = YOLO(self.pt_path)
            except Exception: 
                pass
                
        def infer(self, img):
            if img is None or self.model is None: 
                return []
            try:
                results = self.model(img, verbose=False, conf=0.1)
                boxes = []
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        conf = r.boxes.conf.cpu().numpy()
                        cls = r.boxes.cls.cpu().numpy()
                        for i in range(len(xyxy)): 
                            boxes.append([xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], conf[i], int(cls[i])])
                return np.array(boxes)
            except Exception: 
                return []

    VisionModelAsync = GPUModelAsync
    VisionModelSync = GPUModelSync

class KalmanBoxTracker:
    def __init__(self, bbox, cls_id, conf):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.cls, self.conf, self.lost = cls_id, conf, 0
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        self.w, self.h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.statePost = np.array([[cx], [cy], [0.], [0.]], np.float32)
        
    def predict(self):
        if self.lost > 0:
            self.kf.statePost[2, 0] *= 0.5  
            self.kf.statePost[3, 0] *= 0.5  
            
        pred = self.kf.predict()
        cx, cy = pred[0, 0], pred[1, 0]
        return np.array([cx - self.w/2, cy - self.h/2, cx + self.w/2, cy + self.h/2])
        
    def correct(self, bbox, conf):
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        self.w, self.h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.conf, self.lost = conf, 0
        
    def get_state(self):
        cx, cy = self.kf.statePost[0, 0], self.kf.statePost[1, 0]
        return np.array([cx - self.w/2, cy - self.h/2, cx + self.w/2, cy + self.h/2])

class SORTTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.3, is_helmet=True):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.is_helmet = is_helmet
        self.next_id = 1
        self.tracks = {}
        
    def _associate(self, detections, trackers_keys, iou_threshold):
        if len(trackers_keys) == 0 or len(detections) == 0: 
            return [], list(range(len(detections))), trackers_keys
            
        iou_matrix = np.zeros((len(detections), len(trackers_keys)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, tid in enumerate(trackers_keys):
                iou_matrix[d, t] = calculate_iou(det[:4], self.tracks[tid].get_state())
                
        matched_indices, unmatched_dets, unmatched_trks = [], [], list(trackers_keys)
        for d in range(len(detections)):
            best_t_idx, best_iou = -1, iou_threshold
            for t, tid in enumerate(trackers_keys):
                if tid in unmatched_trks and iou_matrix[d, t] > best_iou:
                    best_iou, best_t_idx = iou_matrix[d, t], t
                    
            if best_t_idx != -1:
                matched_indices.append((d, trackers_keys[best_t_idx]))
                unmatched_trks.remove(trackers_keys[best_t_idx])
            else: 
                unmatched_dets.append(d)
                
        return matched_indices, unmatched_dets, unmatched_trks
        
    def update(self, detections):
        if len(detections) > 0: 
            detections = clean_overlapping_detections(detections, self.is_helmet)
            
        for tid in list(self.tracks.keys()): 
            self.tracks[tid].predict()
            
        valid_dets = [d for d in detections if d[4] >= self.track_thresh]
        
        matched, unmatched_dets, unmatched_trks = self._associate(valid_dets, list(self.tracks.keys()), self.match_thresh)
        
        for d_idx, tid in matched: 
            self.tracks[tid].correct(valid_dets[d_idx][:4], valid_dets[d_idx][4])
            
        for tid in unmatched_trks: 
            self.tracks[tid].lost += 1
            
        for d_idx in unmatched_dets:
            det = valid_dets[d_idx]
            self.tracks[self.next_id] = KalmanBoxTracker(det[:4], int(det[5]), det[4])
            self.next_id += 1
            
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.lost <= self.track_buffer}
        return self._get_results()
        
    def predict_only(self):
        for tid, trk in self.tracks.items():
            trk.predict()
            trk.lost += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.lost <= self.track_buffer}
        return self._get_results()
        
    def _get_results(self):
        return np.array([[*t.get_state(), tid, t.conf, t.cls] for tid, t in self.tracks.items() if t.lost <= 10])
