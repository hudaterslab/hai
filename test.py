import os
import cv2
import json
import glob
import time
import math
import numpy as np
import subprocess
from collections import deque, defaultdict

# ==========================================
# [1] multi_event.py 컴포넌트 임포트 (재사용)
# ==========================================
try:
    import multi_event as me
    from multi_event import (
        SYS_CFG, EVENT_REGISTRY, SimpleTracker,
        denormalize_roi_points, extract_ip, run_wizard_batch_mode,
        ID_H_HELMET, ID_H_NO_HELMET, ID_G_PERSON, ID_PERSON_LOW,
        ID_REFLECTIVE_VEST, TARGET_VEHICLES,
        Camera, MotionDetector,
        split_unified_event_detections, detection_array,
        resolve_model_path, ROI_CHANGE_EVENT
    )
except ImportError as e:
    print(f"[오류] multi_event.py 파일을 찾을 수 없거나 임포트 에러가 발생했습니다: {e}")
    exit(1)

# ==========================================
# [2] 환경 설정
# ==========================================
TEST_DIR = "./test_videos"
TEST_RESULT_DIR = "./test_results"
TEST_CONFIG_FILE = os.path.join(TEST_DIR, "test_cameras.json")
TARGET_FPS = SYS_CFG.get("REC_FPS", 3.0)

# 테스트 재현용에서는 실제 관제 API 전송을 막습니다.
# 이벤트 판정/저장 로직은 multi_event.py 경로를 타되, 외부 POST만 차단합니다.
def _test_no_api_send(*args, **kwargs):
    return None
me.send_event_image_to_receiver = _test_no_api_send

# ==========================================
# [3] 모의 프레임 리더 (단말 속도 모사)
# ==========================================
class VideoMockReader:
    def __init__(self, video_path, target_fps=3.0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")

        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.orig_fps <= 0 or math.isnan(self.orig_fps):
            self.orig_fps = 15.0

        self.frame_skip = max(1, int(round(self.orig_fps / target_fps)))
        self.fid = 0

    def read(self):
        # 타겟 FPS에 맞추기 위해 잉여 프레임은 버림(Grab)
        for _ in range(self.frame_skip - 1):
            self.cap.grab()

        ret, frame = self.cap.read()
        if ret:
            self.fid += 1
        return ret, frame, self.fid

    def release(self):
        self.cap.release()

# ==========================================
# [4] 크로스 플랫폼 모델 래퍼 (서버/PC 테스트용)
# ==========================================
class DualModelWrapper:
    def __init__(self, model_name_from_cfg):
        self.base_name = model_name_from_cfg.rsplit('.', 1)[0]

        try:
            import dx_engine
            from multi_event import YoLoDeepX
            self.ext = 'dxnn'
            self.model_path = f"{self.base_name}.dxnn"
            self.model = YoLoDeepX(self.model_path)
            print(f"? [Model] DeepX NPU 로드 완료: {self.model_path}")
            return
        except ImportError:
            pass

        self.ext = 'pt'
        self.model_path = f"{self.base_name}.pt"
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"? [Model] 서버/PC 환경 감지 - PyTorch 로드 완료: {self.model_path}")
        except ImportError:
            raise ImportError("PyTorch(.pt) 모델을 사용하려면 'pip install ultralytics'가 필요합니다.")

    def infer(self, img, conf_override=0.40):
        if img is None:
            return np.empty((0, 6))

        if self.ext == 'pt':
            results = self.model(img, verbose=False, conf=conf_override)
            res = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    c = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    res.append([x1, y1, x2, y2, c, cls_id])
            return np.array(res) if res else np.empty((0, 6))
        else:
            return self.model.infer(img, conf_override)

# ==========================================
# [4-1] 테스트용 Camera 래퍼
#      - RTSP FrameReader만 빼고, run_logic/draw/privacy_blur는 multi_event.py 그대로 사용
# ==========================================
class DummyRecorder:
    def trigger(self, event_name, objects_meta=None, event_meta=None):
        # test.py의 결과 영상 저장은 아래 VideoWriter가 담당합니다.
        return None

    def update(self, frame, infer_meta=None):
        return None

class ReplayCamera(Camera):
    def __init__(self, ip, conf, det_main, det_helmet, det_face, det_signalman, det_plate,
                 cam_id, event_inference_mode="separate"):
        # Camera.__init__을 호출하면 RTSP FrameReader 스레드가 뜨므로 호출하지 않습니다.
        # 대신 multi_event.py의 Camera 필드 구조만 동일하게 구성합니다.
        self.ip = ip
        self.conf = conf
        self.cam_id = cam_id
        self.event_inference_mode = event_inference_mode
        self.events = conf.get('events', [])

        self.det_main = det_main
        self.det_helmet = det_helmet
        self.det_face = det_face
        self.det_signalman = det_signalman
        self.det_plate = det_plate

        self.trk_main = SimpleTracker()
        self.trk_helmet = SimpleTracker()
        self.trk_signalman = SimpleTracker()

        self.reader = None
        self.recorder = DummyRecorder()
        self.motion_det = MotionDetector()

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
        self._rebuild_handlers()

# ==========================================
# [5] 설정 관리 및 실행
# ==========================================
def load_or_run_wizard(video_files):
    configs = {}
    if os.path.exists(TEST_CONFIG_FILE):
        with open(TEST_CONFIG_FILE, 'r', encoding='utf-8') as f:
            configs = json.load(f)

    missing_videos = []
    for v in video_files:
        v_key = extract_ip(v)
        if v_key not in configs or not configs[v_key].get('events'):
            missing_videos.append(v)

    if missing_videos:
        print(f"\n[알림] 설정이 없는 테스트 영상 {len(missing_videos)}건이 발견되었습니다. 설정 마법사를 실행합니다.")
        new_configs = run_wizard_batch_mode(missing_videos, configs)
        try:
            with open(TEST_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(new_configs, f, indent=4)
            configs = new_configs
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")

    return configs

def _load_optional_model(model_name, label):
    try:
        return DualModelWrapper(model_name)
    except Exception as e:
        print(f"[Model Warning] {label} 모델 로드 실패. 해당 기능은 비활성화됩니다: {e}")
        return None

def main():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"[{TEST_DIR}] 폴더를 생성했습니다. 테스트할 영상을 넣고 다시 실행하십시오.")
        return

    if not os.path.exists(TEST_RESULT_DIR):
        os.makedirs(TEST_RESULT_DIR)

    video_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.mp4")) + glob.glob(os.path.join(TEST_DIR, "*.avi")))
    if not video_files:
        print(f"[{TEST_DIR}] 폴더 내에 영상 파일이 없습니다.")
        return

    configs = load_or_run_wizard(video_files)

    # 실제 multi_event.py의 운영 추론 경로를 그대로 따릅니다.
    # multi_event.py는 INFERENCE_MODE 값과 무관하게 항상 MAIN 모델 한 번의 추론 결과를
    # split_unified_event_detections()로 클래스별로 나눠서 사용합니다(event_inference_mode="main").
    # 따라서 test.py도 동일하게 MAIN 모델만 사용하고, 신호수는 MAIN 출력 split에서 가져옵니다.
    try:
        models_cfg = SYS_CFG.get("models", {})
        event_inference_mode = "main"

        model_main = DualModelWrapper(models_cfg.get("MAIN", "hanjin_cctv.dxnn"))
        model_signalman = None  # 신호수는 MAIN 모델 출력 split(d_signalman_res)에서 가져옵니다.

        model_helmet = DualModelWrapper(models_cfg.get("HELMET", "helmet_3cls_v8.dxnn"))

        # test.py 결과 영상에도 개인정보 블러를 적용하기 위해 FACE/PLATE 모델을 로드합니다.
        # 모델이 없으면 테스트 자체는 계속 진행합니다.
        model_face = _load_optional_model(models_cfg.get("FACE", "yolov8m-face.dxnn"), "FACE")
        model_plate = _load_optional_model(models_cfg.get("PLATE", "license_plate_detector.dxnn"), "PLATE")

        main_conf = SYS_CFG["model_confidences"]["MAIN"]
        helmet_conf = SYS_CFG["model_confidences"]["HELMET"]
        person_conf = SYS_CFG.get("model_confidences", {}).get("PERSON", 0.35)
        signalman_conf = SYS_CFG.get("model_confidences", {}).get("SIGNALMAN", person_conf)

    except Exception as e:
        print(f"[Model Load Error] {e}")
        return

    print("\n=====================================")
    print(f"?? 테스트 분석 시작 (목표 프레임: {TARGET_FPS} FPS / event inference: {event_inference_mode})")
    print(f"저장 경로: {os.path.abspath(TEST_RESULT_DIR)}")
    print("=====================================")

    for v_idx, video_path in enumerate(video_files):
        v_key = extract_ip(video_path)
        conf = configs.get(v_key, {})
        events = conf.get('events', [])

        if not events:
            continue

        video_filename = os.path.basename(video_path)
        name_only, ext_only = os.path.splitext(video_filename)
        result_video_path = os.path.join(TEST_RESULT_DIR, f"{name_only}_result.mp4")

        print(f"\n▶ [{v_idx+1}/{len(video_files)}] 재생 및 녹화 중: {video_filename} | 적용 이벤트: {events}")

        reader = VideoMockReader(video_path, target_fps=TARGET_FPS)
        video_writer = None

        replay_cam = ReplayCamera(
            ip=v_key,
            conf=conf,
            det_main=model_main,
            det_helmet=model_helmet,
            det_face=model_face,
            det_signalman=model_signalman,
            det_plate=model_plate,
            cam_id=v_idx + 1,
            event_inference_mode=event_inference_mode
        )

        while True:
            ret, frame, fid = reader.read()
            if not ret:
                break

            # ---------------------------------------------------------
            # [실제 multi_event.py의 run_camera_inference()와 동일한 추론 입력 구성]
            # - MAIN 모델 한 번 추론 후 split_unified_event_detections()로 클래스 분리
            # - 신호수(signalman)는 MAIN 출력 split 결과(d_signalman_res) 사용
            # - 헬멧은 "no_helmet" 이벤트일 때만 helmet 모델 추론
            # ---------------------------------------------------------
            active_detection_events = [evt for evt in replay_cam.events if evt != ROI_CHANGE_EVENT]
            base_conf = min(main_conf, person_conf, signalman_conf)
            raw_dets = replay_cam.det_main.infer(frame, conf_override=base_conf)
            t_main_input, _, d_signalman_res = split_unified_event_detections(
                raw_dets,
                active_detection_events,
                main_conf=main_conf,
                person_conf=person_conf,
                helmet_conf=helmet_conf,
                signalman_conf=signalman_conf
            )

            d_helmet_res = np.empty((0, 6))
            if "no_helmet" in replay_cam.events:
                d_helmet_res = replay_cam.det_helmet.infer(frame, conf_override=helmet_conf)

            # ---------------------------------------------------------
            # [핵심] 이벤트 재현은 multi_event.py의 Camera.run_logic() 그대로 사용
            # - MotionDetector
            # - SimpleTracker 3종(main/helmet/signalman)
            # - 이벤트 핸들러 kwargs
            # - cooldown / visual alarm / decision_trace
            # ---------------------------------------------------------
            t_main, t_helmet, t_signalman, alarms, new_events = replay_cam.run_logic(
                frame,
                fid,
                t_main_input,
                d_helmet_res,
                d_signalman_res
            )

            for ev_data in new_events:
                tids = []
                for obj in ev_data.get('objects', []):
                    if obj.get('tid') is not None:
                        tids.append(str(obj.get('tid')))
                tid_text = ",".join(tids) if tids else "-"
                print(f"?? [{ev_data.get('event_name', '').upper()} 알람 발생!] FID:{fid} | TID:{tid_text}")

            # ---------------------------------------------------------
            # [추가] test.py 결과 영상에도 얼굴/번호판 모자이크 적용
            # - 실제 multi_event.py의 apply_privacy_blur() 재사용
            # - 그 위에 draw()로 ROI/BBox/알람 UI 렌더링
            # ---------------------------------------------------------
            render_base = frame.copy()
            if replay_cam.det_face is not None or replay_cam.det_plate is not None:
                render_base, _privacy_meta = replay_cam.apply_privacy_blur(
                    render_base,
                    t_main,
                    blur_face=True,
                    blur_plate=True
                )

            render_frame = replay_cam.draw(render_base, t_main, t_helmet, t_signalman, alarms, connected=True)
            cv2.putText(render_frame, f"TEST MODE | {TARGET_FPS} FPS | FID: {fid}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 렌더링된 프레임을 영상 파일로 기록
            if video_writer is None:
                h_out, w_out = render_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(result_video_path, fourcc, TARGET_FPS, (w_out, h_out))

            video_writer.write(render_frame)

            cv2.imshow("Video Test", render_frame)
            if cv2.waitKey(1) == ord('q'):
                print("테스트를 강제로 다음 영상으로 넘깁니다.")
                break

        if video_writer is not None:
            video_writer.release()
        reader.release()

    cv2.destroyAllWindows()
    print("\n? 모든 비디오 테스트 및 결과 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()
