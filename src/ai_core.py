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

# ==========================================
# 모델별 동적 신뢰도(Confidence) 임계값 매핑
# ==========================================
def get_model_confidence(engine_path, default_conf=0.45):
    conf_map = SYS_CFG.get("model_confidences", {})
    ep_lower = engine_path.lower()
    if "helmet" in ep_lower:
        return conf_map.get("HELMET", 0.50)
    elif "face" in ep_lower:
        return conf_map.get("FACE", 0.35)
    else:
        return conf_map.get("GENERAL", 0.60)

# ==========================================
# 💡 [핵심 방어 로직] 모델 경로 & 확장자 자동 보정
# ==========================================
def resolve_model_path(engine_path, is_gpu=False):
    target_path = engine_path
    
    # 1. 하드웨어 환경에 맞춰 확장자 강제 변환
    if is_gpu:
        target_path = target_path.replace(".dxnn", ".pt")
    else:
        target_path = target_path.replace(".pt", ".dxnn")
        
    # 2. JSON에 적힌 경로 그대로 존재하면 사용
    if os.path.exists(target_path):
        return target_path
        
    # 3. 파일이 없다면 구버전 설정으로 간주하고, 강제로 models/ 폴더 하위를 탐색
    alt_path = os.path.join("models", os.path.basename(target_path))
    return alt_path

# ==========================================
# 하드웨어 환경(NPU vs GPU) 동적 감지
# ==========================================
def check_deepx_npu():
    try:
        res = subprocess.run(["dxrt-cli", "-i"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0
    except Exception:
        return False

USE_NPU = check_deepx_npu()
async_result_queue = queue.Queue(maxsize=2000)

if USE_NPU:
    try:
        from dx_engine import InferenceEngine, InferenceOption
        logger.info("🟢 DeepX NPU 활성화: dx_engine을 가동합니다.")
    except ImportError:
        USE_NPU = False
        logger.warning("🟡 NPU가 감지되었으나 dx_engine 임포트에 실패했습니다. GPU 모드로 대체합니다.")

# ==========================================
# 1. DeepX NPU 전용 구현체 (USE_NPU == True 일 때만 평가됨)
# ==========================================
if USE_NPU:
    def onInferenceCallbackFunc(outputs, user_arg):
        # 💡 하드코딩된 변수 대신 conf_thres를 매개변수로 주입받습니다.
        cam_id, model_type, fid, scale, offset, semaphore, is_yolov7, conf_thres, input_tensor = user_arg
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

            # 💡 동적으로 할당된 임계값을 사용하여 노이즈(비닐 등)를 걸러냅니다.
            mask = scores > conf_thres
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            boxes = []
            if len(pred) > 0:
                raw_boxes = pred[:, :4].copy()
                raw_boxes[:, 0] = raw_boxes[:, 0] - raw_boxes[:, 2] / 2
                raw_boxes[:, 1] = raw_boxes[:, 1] - raw_boxes[:, 3] / 2
                raw_boxes[:, 2] = raw_boxes[:, 0] + raw_boxes[:, 2]
                raw_boxes[:, 3] = raw_boxes[:, 1] + raw_boxes[:, 3]
                
                indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), scores.tolist(), conf_thres, 0.45)
                if len(indices) > 0:
                    dw, dh = offset
                    for i in indices.flatten():
                        bx1 = (raw_boxes[i][0] - dw) / scale
                        by1 = (raw_boxes[i][1] - dh) / scale
                        bx2 = (raw_boxes[i][2] - dw) / scale
                        by2 = (raw_boxes[i][3] - dh) / scale
                        boxes.append([bx1, by1, bx2, by2, scores[i], class_ids[i]])
                        
            if not async_result_queue.full():
                async_result_queue.put((cam_id, model_type, fid, np.array(boxes)))
                
        except Exception as e:
            logger.error(f"Async Callback Error: {e}")
        finally:
            semaphore.release() 
        return 0

    class DeepXModelAsync:
        def __init__(self, engine_path):
            self.engine_path = resolve_model_path(engine_path, is_gpu=False)
            self.is_yolov7 = "v7" in os.path.basename(self.engine_path).lower()
            self.conf_thres = get_model_confidence(self.engine_path)
            
            if not os.path.exists(self.engine_path):
                logger.error(f"❌ [NPU 모드] 모델 누락: {self.engine_path} (NPU 환경에서는 .dxnn 모델이 필수입니다)")
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
            self.conf_thres = get_model_confidence(self.engine_path)
            
            if not os.path.exists(self.engine_path):
                logger.error(f"❌ [NPU 모드] 모델 누락: {self.engine_path}")
                self.engine = None
            else:
                self.engine = InferenceEngine(self.engine_path, InferenceOption())

        def letter_box(self, img, new_shape=(640,640)):
            h, w = img.shape[:2]
            scale = min(new_shape[0]/h, new_shape[1]/w)
            nw, nh = int(w*scale), int(h*scale)
            resized = cv2.resize(img, (nw, nh))
            canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
            dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
            canvas[dh:dh+nh, dw:dw+nw] = resized
            return canvas, scale, (dw, dh)

        def infer(self, img):
            if img is None or self.engine is None: return []
            npu_input, scale, offset = self.letter_box(img)
            try:
                out = self.engine.run([np.ascontiguousarray(cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB))])
                pred = np.array(out[0])
                
                if pred.ndim == 3 and pred.shape[1] < pred.shape[2]: pred = pred.transpose((0, 2, 1))
                if pred.ndim == 3: pred = pred[0] 
                
                C = pred.shape[1]
                if self.is_yolov7 or C == 6 or C == 85:
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

                mask = scores > self.conf_thres
                pred = pred[mask]; scores = scores[mask]; class_ids = class_ids[mask]
                
                if len(pred) == 0: return []
                boxes = pred[:, :4].copy()
                boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2; boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]; boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_thres, 0.45)
                if len(indices) == 0: return []
                    
                res = []
                dw, dh = offset; h_orig, w_orig = img.shape[:2]
                for i in indices.flatten():
                    x1 = np.clip((boxes[i][0] - dw) / scale, 0, w_orig)
                    y1 = np.clip((boxes[i][1] - dh) / scale, 0, h_orig)
                    x2 = np.clip((boxes[i][2] - dw) / scale, 0, w_orig)
                    y2 = np.clip((boxes[i][3] - dh) / scale, 0, h_orig)
                    res.append([x1, y1, x2, y2, scores[i], class_ids[i]])
                return np.array(res)
            except Exception:
                return []

    VisionModelAsync = DeepXModelAsync
    VisionModelSync = DeepXModelSync

# ==========================================
# 2. NVIDIA GPU 전용 구현체 (USE_NPU == False 일 때만 평가됨)
# ==========================================
else:
    logger.info("🔵 NVIDIA GPU 모드 활성화: PyTorch(Ultralytics) 추론 엔진을 가동합니다.")
    
    class GPUModelAsync:
        def __init__(self, engine_path):
            self.pt_path = resolve_model_path(engine_path, is_gpu=True)
            self.is_yolov7 = "v7" in os.path.basename(self.pt_path).lower()
            self.conf_thres = get_model_confidence(self.pt_path)
            self.model = None
            
            try:
                from ultralytics import YOLO
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if not os.path.exists(self.pt_path):
                    logger.error(f"❌ [GPU 모드] 모델 누락: {self.pt_path} (GPU 환경에서는 .pt 모델이 필수입니다)")
                else:
                    logger.info(f"🚀 GPU 모델 로딩 중 ({self.device}): {self.pt_path}")
                    self.model = YOLO(self.pt_path)
            except ImportError:
                logger.error("❌ GPU 모드를 사용하려면 'ultralytics' 패키지가 필요합니다. (pip install ultralytics torch)")
            except Exception as e:
                logger.error(f"❌ GPU 모델 로드 실패: {e}")

        def submit_async(self, img, cam_id, model_type, fid, semaphore):
            if img is None or self.model is None:
                semaphore.release()
                return
            
            def _infer():
                try:
                    results = self.model(img, verbose=False, conf=self.conf_thres)
                    boxes = []
                    for r in results:
                        if r.boxes is not None and len(r.boxes) > 0:
                            xyxy = r.boxes.xyxy.cpu().numpy()
                            conf = r.boxes.conf.cpu().numpy()
                            cls = r.boxes.cls.cpu().numpy()
                            for i in range(len(xyxy)):
                                boxes.append([xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], conf[i], int(cls[i])])
                    
                    if not async_result_queue.full():
                        async_result_queue.put((cam_id, model_type, fid, np.array(boxes)))
                except Exception as e:
                    logger.error(f"GPU Async Error: {e}")
                finally:
                    semaphore.release()

            threading.Thread(target=_infer, daemon=True).start()

    class GPUModelSync:
        def __init__(self, engine_path):
            self.pt_path = resolve_model_path(engine_path, is_gpu=True)
            self.is_yolov7 = "v7" in os.path.basename(self.pt_path).lower()
            self.conf_thres = get_model_confidence(self.pt_path)
            self.model = None
            
            try:
                from ultralytics import YOLO
                if os.path.exists(self.pt_path):
                    self.model = YOLO(self.pt_path)
                else:
                    logger.error(f"❌ [GPU 모드] 모델 누락: {self.pt_path}")
            except Exception as e:
                logger.error(f"❌ GPU 모델 로드 실패: {e}")

        def infer(self, img):
            if img is None or self.model is None: return []
            try:
                results = self.model(img, verbose=False, conf=self.conf_thres)
                boxes = []
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        conf = r.boxes.conf.cpu().numpy()
                        cls = r.boxes.cls.cpu().numpy()
                        for i in range(len(xyxy)):
                            boxes.append([xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], conf[i], int(cls[i])])
                return np.array(boxes)
            except Exception as e:
                logger.error(f"GPU Sync Error: {e}")
                return []

    VisionModelAsync = GPUModelAsync
    VisionModelSync = GPUModelSync

# ==========================================
# 칼만 필터를 탑재한 SORT Tracker
# ==========================================
class KalmanBoxTracker:
    def __init__(self, bbox, cls_id):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.cls = cls_id
        self.lost = 0
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
        
        self.kf.statePost = np.array([[cx], [cy], [0.], [0.]], np.float32)

    def predict(self):
        pred = self.kf.predict()
        cx, cy = pred[0, 0], pred[1, 0]
        return np.array([cx - self.w/2, cy - self.h/2, cx + self.w/2, cy + self.h/2])

    def correct(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.lost = 0
        
    def get_state(self):
        cx, cy = self.kf.statePost[0, 0], self.kf.statePost[1, 0]
        return np.array([cx - self.w/2, cy - self.h/2, cx + self.w/2, cy + self.h/2])

class SortTracker:
    def __init__(self, max_lost=50, is_helmet=True):
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost
        self.is_helmet = is_helmet

    def update(self, detections):
        detections = clean_overlapping_detections(detections, self.is_helmet)
        used_dets = set()
        
        for tid in list(self.tracks.keys()):
            self.tracks[tid].predict()

        for tid, trk in self.tracks.items():
            best_iou = 0
            best_idx = -1
            pred_bbox = trk.get_state()
            
            for i, det in enumerate(detections):
                if i in used_dets or int(det[5]) != trk.cls:
                    continue
                iou = calculate_iou(pred_bbox, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
                    
            if best_iou > 0.2:
                trk.correct(detections[best_idx][:4])
                used_dets.add(best_idx)
            else:
                trk.lost += 1
                
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.lost <= self.max_lost}
        
        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks[self.next_id] = KalmanBoxTracker(det[:4], int(det[5]))
                self.next_id += 1
                
        return self._get_results()

    def predict_only(self):
        for tid, trk in self.tracks.items():
            trk.predict()
            trk.lost += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.lost <= self.max_lost}
        return self._get_results()

    def _get_results(self):
        return np.array([[*t.get_state(), tid, 1.0, t.cls] for tid, t in self.tracks.items() if t.lost <= 10])