import os
import cv2
import json
import glob
import math
import numpy as np
from collections import deque, defaultdict
#test 주석

# ==========================================
# [1] multi_event.py 컴포넌트 임포트 (재사용)
# ==========================================
try:
    import multi_event as me
    from multi_event import (
        SYS_CFG, SimpleTracker,
        extract_ip,
        Camera, MotionDetector,
        split_unified_event_detections,
        ROI_CHANGE_EVENT,
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

# 화각 변경(roi_change) 검사 주기를 운영(ALIGN_INTERVAL_SEC=300s, 5분) 대신
# 테스트 재현용 3초로 단축합니다.
# _update_alignment()가 모듈 전역 ALIGN_INTERVAL_SEC을 런타임에 참조하므로
# me.ALIGN_INTERVAL_SEC 만 덮어쓰면 그대로 반영됩니다.
me.ALIGN_INTERVAL_SEC = 3.0

# 테스트 영상은 RTSP 연결 직후의 무효 프레임 안정화 대기가 사실상 불필요하므로
# 앵커 등록 전 대기(ANCHOR_STARTUP_DELAY_SEC, 운영 10초)를 1초로 단축합니다.
# 이 값도 _update_alignment()가 런타임에 모듈 전역을 참조하므로 덮어쓰기만 하면 반영됩니다.
me.ANCHOR_STARTUP_DELAY_SEC = 1.0

# 신호수(signal_vehicle) 이벤트: 라인트럭이 '주정차'로 인정되는 정지 지속 시간을
# 운영 60초(parked_threshold_sec) 대신 테스트용 1초로 단축합니다.
# SignalVehicleDetector.__init__이 config.get("parked_threshold_sec", 60.0)으로 읽고,
# 핸들러는 _rebuild_handlers()가 SYS_CFG["event_config"]["signal_vehicle"]로 생성하므로,
# (ReplayCamera 생성 전에) 이 dict를 덮어쓰면 그대로 5초가 반영됩니다.
me.SYS_CFG.setdefault("event_config", {}).setdefault("signal_vehicle", {})["parked_threshold_sec"] = 1.0

# roi_change 판정 CSV(decision.csv)를 운영 기본 경로(PROJECT_ROOT/logs/roi_align/...) 대신
# test.py 결과 폴더(TEST_RESULT_DIR)에 저장합니다.
# append_csv_log(self, row, path=ROI_ALIGN_CSV_LOG_FILE)의 path 기본값은 함수 정의 시점에
# 바인딩되므로, me.ROI_ALIGN_CSV_LOG_FILE만 바꿔서는 반영되지 않습니다.
# 따라서 싱글톤(ROI_ALIGN_LEARNING_STORE)의 append_csv_log를 래핑해 강제로 경로를 지정합니다.
TEST_CSV_PATH = os.path.abspath(os.path.join(TEST_RESULT_DIR, "roi_align_decisions.csv"))
me.ROI_ALIGN_CSV_LOG_FILE = TEST_CSV_PATH  # 시작 시 경로 출력 등 참조용

_orig_append_csv_log = me.ROI_ALIGN_LEARNING_STORE.append_csv_log
def _append_csv_log_to_test_dir(row, path=None):
    # path 인자는 무시하고 항상 test_results 경로에 기록
    return _orig_append_csv_log(row, path=TEST_CSV_PATH)
me.ROI_ALIGN_LEARNING_STORE.append_csv_log = _append_csv_log_to_test_dir

# test.py의 모든 결과물은 test_results 폴더 안에만 있어야 합니다.
# 운영의 save_event_image_with_mark()는 이벤트 이미지를 CCTV_EVENT_ALERT 폴더에 저장하고
# API 전송 큐에 등록하므로, 이를 막습니다(반환 None → run_logic은 evidence_paths 없이 진행).
# 대신 test.py 메인 루프에서 '모든 객체 class/conf가 그려진' render_frame을 test_results에 직접 저장합니다.
def _test_no_save_event_image(*args, **kwargs):
    return None
me.save_event_image_with_mark = _test_no_save_event_image
# 혹시 다른 경로에서 EVENT_ROOT_DIR을 참조하더라도 test_results 밖으로 새지 않도록 안전망.
me.EVENT_ROOT_DIR = os.path.abspath(TEST_RESULT_DIR)

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
    def __init__(self, model_name_from_cfg, model_key=None):
        # model_key(MAIN/HELMET/FACE/PLATE)별로 운영(multi_event.py)과 동일한
        # output_format / pool_size 를 적용합니다.
        # 특히 MAIN(hanjin_cctv_v2.dxnn)은 PPU 디코드가 필수라, output_format을 넘기지 않으면
        # _resolve_output_format("auto") 가 "yolo"로 떨어져 잘못된 postprocess를 타고
        # "zero-size array to reduction operation maximum" 에러가 발생합니다.
        self.base_name = model_name_from_cfg.rsplit('.', 1)[0]

        try:
            import dx_engine
            from multi_event import YoLoDeepX
            self.ext = 'dxnn'
            self.model_path = me.resolve_model_path(f"{self.base_name}.dxnn")

            # 운영(multi_event.py)의 모델 로드와 동일하게 output_format / pool_size 결정
            if model_key == "MAIN":
                out_fmt = me.get_main_model_output_format(self.model_path)
                pool = max(3, me.get_model_engine_pool_size("MAIN"))
            elif model_key:
                out_fmt = me.get_model_output_format(model_key)
                pool = me.get_model_engine_pool_size(model_key)
            else:
                out_fmt = "auto"
                pool = 1

            self.model = YoLoDeepX(self.model_path, output_format=out_fmt, pool_size=pool)
            print(f"✅ [Model] DeepX NPU 로드 완료: {self.model_path} (format={out_fmt}, pool={pool})")
            return
        except ImportError:
            pass

        self.ext = 'pt'
        self.model_path = me.resolve_model_path(f"{self.base_name}.pt")
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"✅ [Model] 서버/PC 환경 감지 - PyTorch 로드 완료: {self.model_path}")
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
#      - RTSP FrameReader만 빼고, run_logic/draw/privacy_blur/_update_alignment 등은
#        multi_event.py의 Camera 메서드를 그대로 상속해서 사용합니다.
# ==========================================
class DummyRecorder:
    # multi_event.py의 run_logic은 recorder.trigger(..., current_fps=...) 형태로 호출하므로
    # 시그니처에 의존하지 않도록 *args/**kwargs로 받습니다.
    def trigger(self, *args, **kwargs):
        # test.py의 결과 영상 저장은 아래 VideoWriter가 담당합니다.
        return None

    def update(self, *args, **kwargs):
        return None

class ReplayCamera(Camera):
    def __init__(self, ip, conf, det_main, det_helmet, det_face, det_signalman, det_plate,
                 cam_id, event_inference_mode="main"):
        # Camera.__init__을 호출하면 RTSP FrameReader 스레드와 VideoRecorder가 뜨므로 호출하지 않습니다.
        # 대신 multi_event.py의 Camera.__init__과 동일한 필드를 그대로 구성합니다.
        # (reader/recorder만 테스트용으로 대체)
        self.ip = ip
        self.camera_key = ip
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

        self.roi_frame_shape = None  # 해상도 변경 감지용
        self.status_history = deque(maxlen=10)
        # ROI 자동 보정(화각 변경) 상태 초기화: self.aligner / anchor_set / last_align_time 등 생성
        self._reset_alignment_state("ALIGN INIT")
        self._rebuild_handlers()

    def draw(self, fr, t_main, t_helmet, t_signalman, alarms, connected=True):
        # 운영 draw()를 그대로 사용(ROI/알람/이벤트 메뉴 등 동일). 그 위에 테스트 전용 표시를 덧그립니다.
        out = super().draw(fr, t_main, t_helmet, t_signalman, alarms, connected)
        if out is None or not connected:
            return out

        # [테스트 전용] 운영 draw()는 활성 이벤트와 무관한 클래스의 박스를 숨기지만(allowed_classes 필터),
        # 테스트에서는 '탐지된 모든 객체'를 보고 싶으므로 숨겨진 t_main 객체도 추가로 그립니다.
        # 트래커/핸들러에 들어가는 입력은 그대로이므로 이벤트 발생 조건에는 전혀 영향이 없습니다(순수 시각화).
        allowed = set()
        if "signal_vehicle" in self.events:
            allowed.add(me.ID_G_TRUCK)
        if "no_helmet" in self.events or "conveyor_crossing" in self.events or "intrusion" in self.events:
            allowed.update([me.ID_G_PERSON, me.ID_PERSON_LOW])
        if "illegal_parking" in self.events or "intrusion" in self.events:
            allowed.update(me.TARGET_VEHICLES)

        names = {
            me.ID_G_PERSON: "Person", me.ID_G_CAR: "Car", me.ID_PERSON_LOW: "LowBody",
            me.ID_REFLECTIVE_VEST: "Vest", me.ID_G_TRUCK: "Truck",
        }
        # 라벨은 2줄로 표시합니다(한 줄에 class+conf를 합치면 화면 가장자리에서 자주 짤림):
        #   1번째 줄(박스 위)  : 클래스명[tid]
        #   2번째 줄(박스 안 상단): conf 0.xx
        for t in t_main:
            tid, cls_id = int(t[4]), int(t[6])
            conf = float(t[5])
            x1, y1, x2, y2 = map(int, t[:4])
            hidden = (tid not in alarms) and (cls_id not in allowed)

            if hidden:
                # 운영 draw()가 숨긴 객체: 회색 박스 + 클래스명(1번째 줄)을 직접 그림
                cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(out, f"{names.get(cls_id, f'cls{cls_id}')}[{tid}]",
                            (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                conf_color = (200, 200, 200)
            else:
                # 알람/감시대상 클래스는 super().draw()가 이미 박스+클래스명(1번째 줄)을 그림 → conf만 추가
                conf_color = (0, 0, 255) if tid in alarms else (0, 255, 255)

            # 2번째 줄: confidence (박스 안 상단)
            cv2.putText(out, f"conf {conf:.2f}", (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)
        return out

# ==========================================
# [5] 설정 마법사 (test.py 전용)
#     - multi_event.run_wizard_batch_mode 를 그대로 재현하되 '6.roi화각변경'을 추가합니다.
#     - roi_change 는 events 에 "roi_change" 를 추가하기만 하면 됩니다(별도 ROI 영역 입력 없음).
#     - 운영(multi_event.py)의 wizard는 건드리지 않습니다.
# ==========================================
def run_test_wizard_batch_mode(video_files, existing_configs=None):
    print("=== 테스트 설정 마법사 시작 ===")
    configs = existing_configs.copy() if existing_configs else {}

    batch_size = getattr(me, "BATCH_SIZE", 9)
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        frames = [me.capture_snapshot(v) for v in batch]

        display = []
        for frm in frames:
            if frm is None:
                blk = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blk, "Conn Fail", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                display.append(blk)
            else:
                display.append(frm)

        mosaic = me.create_mosaic_image(display)
        cols = max(1, math.ceil(math.sqrt(len(display))))
        rows = max(1, math.ceil(len(display) / cols))
        cw = me.SCREEN_WIDTH // cols
        ch = me.SCREEN_HEIGHT // rows

        for idx in range(len(display)):
            r, c = divmod(idx, cols)
            cx, cy = c * cw, r * ch
            cv2.rectangle(mosaic, (cx, cy), (cx + 50, cy + 50), (255, 255, 255), -1)
            cv2.putText(mosaic, str(idx + 1), (cx + 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        cv2.imshow("Select Cameras", mosaic)
        cv2.waitKey(1)

        sel = me.guarded_input(
            f">> [Batch {i // batch_size + 1}] 설정할 영상 번호 (예: 1,3,5 / 건너뛰기: 엔터): "
        ).strip()
        if not sel:
            continue

        try:
            nums = [int(s.strip()) for s in sel.split(',')]
            for n in nums:
                if 1 <= n <= len(batch) and frames[n - 1] is not None:
                    url = batch[n - 1]
                    ip = extract_ip(url)

                    print(f"[{ip}] 1.침입 2.주정차 3.안전모 4.횡단 5.신호수차량 6.roi화각변경")
                    evts = me.guarded_input(f"[{ip}] 이벤트 선택 (예: 1,4,6): ")
                    events = []

                    if '1' in evts: events.append("intrusion")
                    if '2' in evts: events.append("illegal_parking")
                    if '3' in evts: events.append("no_helmet")
                    if '4' in evts: events.append("conveyor_crossing")
                    if '5' in evts: events.append("signal_vehicle")
                    if '6' in evts: events.append(ROI_CHANGE_EVENT)

                    roi_p = []
                    roi_l = []

                    if any(e in events for e in ["intrusion", "illegal_parking", "no_helmet", "signal_vehicle"]):
                        roi_p = me.get_roi_points_scaled(frames[n - 1], f"Polygon - CAM: {ip}")

                    if "conveyor_crossing" in events:
                        while True:
                            l = me.get_roi_points_scaled(frames[n - 1], f"Line - CAM: {ip}", mode="line")
                            if len(l) == 2:
                                roi_l.extend(l)
                            if me.guarded_input("횡단 라인을 추가하시겠습니까? (y/n): ") != 'y':
                                break

                    # 6.roi화각변경(roi_change): events 에 추가만 하면 동작(별도 ROI 영역 입력 없음)
                    if ROI_CHANGE_EVENT in events:
                        print(f"[{ip}] roi_change 활성화 (감지 영역 지정 불필요)")

                    configs[ip] = {
                        "url": url,
                        "events": events,
                        "roi_poly_norm": roi_p,
                        "roi_lines_norm": roi_l,
                    }
        except Exception as e:
            print(f"마법사 설정 중 오류 발생: {e}")

    try:
        cv2.destroyWindow("Select Cameras")
    except Exception:
        pass
    return configs

# ==========================================
# [6] 설정 관리 및 실행
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
        new_configs = run_test_wizard_batch_mode(missing_videos, configs)
        try:
            with open(TEST_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(new_configs, f, indent=4)
            configs = new_configs
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")

    return configs

def _load_optional_model(model_name, label, model_key=None):
    try:
        return DualModelWrapper(model_name, model_key=model_key)
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

    # ---------------------------------------------------------
    # 실제 multi_event.py의 운영 추론 경로를 그대로 따릅니다.
    # multi_event.py는 INFERENCE_MODE 값과 무관하게 항상 MAIN 모델 한 번의 추론 결과를
    # split_unified_event_detections()로 클래스별로 나눠서 사용합니다(event_inference_mode="main").
    # 신호수(signalman)는 별도 모델이 아니라 MAIN 출력 split(d_signalman_res)에서 가져옵니다.
    # ---------------------------------------------------------
    try:
        models_cfg = SYS_CFG.get("models", {})
        event_inference_mode = "main"

        model_main = DualModelWrapper(models_cfg.get("MAIN", "hanjin_cctv.dxnn"), model_key="MAIN")
        model_signalman = None

        model_helmet = DualModelWrapper(models_cfg.get("HELMET", "helmet_3cls_v8.dxnn"), model_key="HELMET")

        # test.py 결과 영상에도 개인정보 블러를 적용하기 위해 FACE/PLATE 모델을 로드합니다.
        # 모델이 없으면 테스트 자체는 계속 진행합니다.
        model_face = _load_optional_model(models_cfg.get("FACE", "yolov8m-face.dxnn"), "FACE", model_key="FACE")
        model_plate = _load_optional_model(models_cfg.get("PLATE", "license_plate_detector.dxnn"), "PLATE", model_key="PLATE")

        main_conf = SYS_CFG["model_confidences"]["MAIN"]
        helmet_conf = SYS_CFG["model_confidences"]["HELMET"]
        person_conf = SYS_CFG.get("model_confidences", {}).get("PERSON", 0.35)
        signalman_conf = SYS_CFG.get("model_confidences", {}).get("SIGNALMAN", person_conf)

    except Exception as e:
        print(f"[Model Load Error] {e}")
        return

    print("\n=====================================")
    print(f"🚀 테스트 분석 시작 (목표 프레임: {TARGET_FPS} FPS / event inference: {event_inference_mode})")
    print(f"📐 화각 변경(roi_change) 방식: 전체 화면 3×3 격자(ROI 영역 미사용, full-frame phaseCorrelate)")
    print(f"📐 화각 변경(roi_change) 검사 주기: {me.ALIGN_INTERVAL_SEC:.0f}초")
    print(f"🗂  ROI 판정 CSV 로그: {me.ROI_ALIGN_CSV_LOG_FILE}")
    print(f"저장 경로: {os.path.abspath(TEST_RESULT_DIR)}")
    print("=====================================")

    for v_idx, video_path in enumerate(video_files):
        v_key = extract_ip(video_path)
        conf = configs.get(v_key, {})
        events = conf.get('events', [])

        if not events:
            continue

        video_filename = os.path.basename(video_path)
        name_only, _ = os.path.splitext(video_filename)
        result_video_path = os.path.join(TEST_RESULT_DIR, f"{name_only}_result.mp4")

        print(f"\n▶ [{v_idx+1}/{len(video_files)}] 재생 및 녹화 중: {video_filename} | 적용 이벤트: {events}")

        reader = VideoMockReader(video_path, target_fps=TARGET_FPS)
        video_writer = None
        prev_align_status = ""       # 정렬 상태 문자열 변화 감지(중복 출력 방지)용

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
            # - _update_alignment()  ← 화각 변경(roi_change) 검출이 여기서 수행됩니다.
            # - 이벤트 핸들러 kwargs / cooldown / visual alarm / decision_trace
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
                print(f"🚨 [{ev_data.get('event_name', '').upper()} 알람 발생!] FID:{fid} | TID:{tid_text}")

            # ---------------------------------------------------------
            # [화각 변경] roi_change는 핸들러 이벤트가 아니라 _update_alignment()가
            # self.align_shifted / self.align_status_text 로 신호합니다(new_events에 안 들어옴).
            # 재현 시 "왜 화각변경으로 판단했는지" 확인할 수 있도록 상태 전이를 출력합니다.
            # ---------------------------------------------------------
            if ROI_CHANGE_EVENT in replay_cam.events:
                cur_status = str(getattr(replay_cam, "align_status_text", ""))
                # 매 검사 결과는 "GRID <decision> ..."(normal/suspect/confirm), 헬스체크 발사 시점은
                # "ROI SETUP REQUIRED confirm ..." 로 나옵니다. ANCHOR SET 시점도 한 번 출력합니다.
                # (ANCHOR WAIT 카운트다운은 매 프레임 바뀌므로 스팸 방지를 위해 제외)
                is_decision = cur_status.startswith("GRID ") or cur_status.startswith("ROI SETUP REQUIRED")
                is_anchor_set = cur_status.startswith("ANCHOR SET")
                if (is_decision and cur_status != prev_align_status) or \
                   (is_anchor_set and not prev_align_status.startswith("ANCHOR SET")):
                    if replay_cam.align_shifted:
                        tag = "🔴 CONFIRM (틀어짐 확정→API)"
                    elif is_decision:
                        tag = "🟡 GRID (normal/suspect)"
                    else:
                        tag = "🟢 ANCHOR SET(기준 등록)"
                    print(f"📐 [{tag}] FID:{fid} | {cur_status}")

                    # 격자 9칸을 칸별로 출력합니다.
                    #   미측정칸="x" / 측정칸=이동량px, (s..)=그 칸 std(텍스처). std임계 이상이면 측정칸.
                    #   움직인 칸(>임계)은 방향 판정도 함께: ✓=대표 방향과 같음(cos>=임계) / ✗=다른 방향, cos값 표시.
                    # 측정 성공한 칸이 모두 10px 이상 움직이고, 그중 같은 방향(✓) 칸이 동적 방향 정족수 이상이면 '틀어짐'.
                    grid_res = getattr(getattr(replay_cam, "aligner", None), "last_grid_result", None)
                    if is_decision and grid_res and grid_res.get("cells"):
                        cell_strs = []
                        for ci, c in enumerate(grid_res["cells"]):
                            mv = me._format_grid_cell_diag(c)            # 이동 px 또는 "x"
                            s = me._format_grid_cell_std(c)              # 그 칸 std
                            if c.get("moving"):
                                mark = "✓" if c.get("consistent") else "✗"
                                cell_strs.append(f"#{ci}:{mv}px{mark}cos{c.get('cos', 0.0):.2f}(s{s})")
                            else:
                                cell_strs.append(f"#{ci}:{mv}(s{s})")
                        cell_diag = " ".join(cell_strs)
                        print(
                            f"    └ cells(전체측정칸 moving={int(bool(grid_res.get('all_measured_moving')))} / "
                            f"같은방향✓ {grid_res.get('consistent_quorum', 0)}칸↑=틀어짐 / "
                            f"cos>={me.GRID_DIRECTION_COS_MIN} / std임계 {me.GRID_CELL_MIN_STD:.0f}) "
                            f"meas={grid_res.get('n_measurable')}/q={grid_res.get('quorum')} "
                            f"moving={grid_res.get('n_moving')} consistent={grid_res.get('consistent')} "
                            f"frame_std={grid_res.get('frame_std', 0.0):.1f}: {cell_diag}"
                        )
                prev_align_status = cur_status

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

            # 화각 변경 상태를 화면에도 표기 (draw()는 align 상태를 그리지 않으므로 test.py에서 보조 표기)
            # 3×3 격자 ALIGN 표시 문자열이
            # 길어서 잘림 길이를 넉넉히 둡니다.
            if ROI_CHANGE_EVENT in replay_cam.events:
                status_text = str(getattr(replay_cam, "align_status_text", ""))
                status_color = (0, 0, 255) if replay_cam.align_shifted else (0, 255, 0)
                cv2.putText(render_frame, f"ALIGN: {status_text[:95]}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

            # ---------------------------------------------------------
            # [이벤트 이미지 저장] 운영처럼 CCTV_EVENT_ALERT가 아니라 test_results에 저장합니다.
            # render_frame에는 모든 탐지 객체의 class명/conf가 그려져 있어(ReplayCamera.draw 오버라이드),
            # 관제 전송 이미지보다 더 많은 정보를 담습니다. (개인정보 블러도 위에서 이미 적용됨)
            # ---------------------------------------------------------
            for ev_data in new_events:
                ev_name = str(ev_data.get('event_name', 'event'))
                ev_tids = "_".join(
                    str(o.get('tid')) for o in ev_data.get('objects', []) if o.get('tid') is not None
                ) or "x"
                snap_path = os.path.join(TEST_RESULT_DIR, f"{name_only}_event_FID{fid}_{ev_name}_tid{ev_tids}.jpg")
                cv2.imwrite(snap_path, render_frame)
                print(f"💾 이벤트 이미지 저장: {snap_path}")

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
    print("\n✅ 모든 비디오 테스트 및 결과 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()
