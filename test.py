import os
import cv2
import glob
import json
import numpy as np
import time
import math

# ==========================================
# [1] multi_event.py 컴포넌트 임포트 (재사용)
# ==========================================
try:
    import multi_event
    from multi_event import (
        SYS_CFG, EVENT_REGISTRY, SimpleTracker, 
        denormalize_roi_points, extract_ip, run_wizard_batch_mode,
        ID_H_HELMET, ID_H_NO_HELMET, ID_G_PERSON, ID_PERSON_LOW, 
        ID_REFLECTIVE_VEST, TARGET_VEHICLES
    )
except ImportError as e:
    print(f"[오류] multi_event.py 파일을 찾을 수 없거나 임포트 에러가 발생했습니다: {e}")
    exit(1)

# ==========================================
# [2] 상수 및 테스트 환경 설정
# ==========================================
TEST_DIR = "./test"
TEST_CONFIG_FILE = os.path.join(TEST_DIR, "test_cameras.json")
TARGET_FPS = SYS_CFG.get("REC_FPS", 3.0)

# ==========================================
# [3] 모의 RTSP 리더 (단말 속도 모사)
# ==========================================
class RTSPMockReader:
    def __init__(self, video_path, target_fps=3.0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")
            
        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.orig_fps <= 0 or math.isnan(self.orig_fps):
            self.orig_fps = 30.0
            
        self.frame_skip = max(1, int(round(self.orig_fps / target_fps)))
        self.fid = 0

    def read(self):
        for _ in range(self.frame_skip - 1):
            self.cap.grab()
            
        ret, frame = self.cap.read()
        if ret:
            self.fid += 1
        return ret, frame, self.fid

    def release(self):
        self.cap.release()

# ==========================================
# [4] 크로스 플랫폼 모델 래퍼 (환경 지능형 스위칭)
# ==========================================
import subprocess

class DualModelWrapper:
    def __init__(self, model_name_from_cfg):
        self.base_name = model_name_from_cfg.rsplit('.', 1)[0]
        
        # 1. 단말 환경 (DeepX NPU) 감지 시도
        try:
            import dx_engine
            from multi_event import YoLoDeepX
            self.ext = 'dxnn'
            self.model_path = f"{self.base_name}.dxnn"
            self.model = YoLoDeepX(self.model_path)
            print(f"✅ [Model] 단말 환경 감지 - DeepX NPU 로드 완료: {self.model_path}")
            return
        except ImportError:
            pass # NPU 환경이 아니므로 다음 단계로 진행

        # 2. 서버 및 일반 PC 환경 분기 (nvidia-smi 체크)
        is_server_gpu = False
        try:
            # OS 레벨에서 nvidia-smi 명령어 실행 가능 여부 확인
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                is_server_gpu = True
        except FileNotFoundError:
            pass
            
        self.ext = 'pt'
        self.model_path = f"{self.base_name}.pt"
        
        try:
            from ultralytics import YOLO
            import torch
            
            self.model = YOLO(self.model_path)
            
            if is_server_gpu:
                # nvidia-smi는 확인되었으나, PyTorch가 CUDA를 지원하는지 교차 검증
                if torch.cuda.is_available():
                    print(f"✅ [Model] 서버 환경(NVIDIA GPU) 감지 - PyTorch(CUDA) 로드 완료: {self.model_path}")
                else:
                    print(f"⚠️ [Model] 서버 환경(NVIDIA GPU) 감지됨, 하지만 PyTorch가 CPU 버전입니다. 추론 속도 저하가 발생할 수 있습니다: {self.model_path}")
            else:
                print(f"ℹ️ [Model] 일반 환경 감지 - NPU/GPU 없음. CPU 기반 PyTorch를 로드합니다: {self.model_path}")
                
        except ImportError:
            raise ImportError("PyTorch(.pt) 모델을 사용하려면 'pip install ultralytics'가 필요합니다.")

    def infer(self, img, conf_override=0.40):
        if img is None: 
            return np.empty((0,6))
            
        if self.ext == 'pt':
            results = self.model(img, verbose=False, conf=conf_override)
            res = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # 결과를 CPU로 내리고 numpy 배열로 변환
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    c = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    res.append([x1, y1, x2, y2, c, cls_id])
            return np.array(res) if res else np.empty((0,6))
        else:
            return self.model.infer(img, conf_override)

# ==========================================
# [5] 테스트 설정 관리 (마법사 자동 연동)
# ==========================================
def load_or_create_test_configs(video_files):
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
        print(f"\n[알림] 설정이 없는 테스트 영상 {len(missing_videos)}건이 발견되었습니다. multi_event 마법사를 실행합니다.")
        new_configs = run_wizard_batch_mode(missing_videos, configs)
        try:
            with open(TEST_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(new_configs, f, indent=4)
            configs = new_configs
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")
            
    return configs

# ==========================================
# [6] 메인 실행 루프
# ==========================================
def main():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"[{TEST_DIR}] 폴더를 생성했습니다. 테스트 영상을 넣고 다시 실행하십시오.")
        return

    video_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.mp4")) + glob.glob(os.path.join(TEST_DIR, "*.avi")))
    if not video_files:
        print(f"[{TEST_DIR}] 폴더 내에 영상 파일이 없습니다.")
        return

    configs = load_or_create_test_configs(video_files)

    print("\n=====================================")
    print(f"테스트 비디오 로드 완료: 총 {len(video_files)}건")
    print(f"목표 단말 모사 FPS: {TARGET_FPS} FPS")
    print("=====================================")

    try:
        model_main = DualModelWrapper(SYS_CFG["models"]["MAIN"])
        model_helmet = DualModelWrapper(SYS_CFG["models"]["HELMET"])
        main_conf = SYS_CFG["model_confidences"]["MAIN"]
        helmet_conf = SYS_CFG["model_confidences"]["HELMET"]
    except Exception as e:
        print(f"[Model Load Error] {e}")
        return

    for v_idx, video_path in enumerate(video_files):
        print(f"\n▶ [{v_idx+1}/{len(video_files)}] 분석 시작: {os.path.basename(video_path)}")
        v_key = extract_ip(video_path)
        conf = configs.get(v_key, {})
        events = conf.get('events', [])
        
        if not events:
            print("설정된 이벤트가 없어 건너뜁니다.")
            continue
            
        reader = RTSPMockReader(video_path, target_fps=TARGET_FPS)
        trk_main = SimpleTracker()
        trk_helmet = SimpleTracker()
        
        roi_poly_norm = conf.get('roi_poly_norm', [])
        roi_lines_norm = conf.get('roi_lines_norm', [])
        roi_frame_shape = None
        handlers = {}
        
        alarms_display = set()
        alarm_cooldown = 0
        
        while True:
            ret, frame, fid = reader.read()
            if not ret:
                break
                
            # 프레임 해상도에 맞춘 ROI 역정규화 및 핸들러 초기화
            if roi_frame_shape != frame.shape[:2]:
                h, w = frame.shape[:2]
                roi_poly = denormalize_roi_points(roi_poly_norm, w, h)
                roi_lines = denormalize_roi_points(roi_lines_norm, w, h)
                
                for ename in events:
                    if ename in EVENT_REGISTRY:
                        event_cfg = SYS_CFG.get("event_config", {}).get(ename, {})
                        handlers[ename] = EVENT_REGISTRY[ename](event_cfg, roi_poly, roi_lines)
                roi_frame_shape = frame.shape[:2]

            # 추론 및 트래킹
            d_main_res = model_main.infer(frame, conf_override=main_conf)
            d_helmet_res = model_helmet.infer(frame, conf_override=helmet_conf)
            
            d_main_filtered = [d for d in d_main_res if int(d[5]) not in [ID_H_HELMET, ID_H_NO_HELMET]]
            t_main = trk_main.update(d_main_filtered)
            
            d_helmet_filtered = [d for d in d_helmet_res if int(d[5]) == ID_H_NO_HELMET]
            t_helmet = trk_helmet.update(d_helmet_filtered)
            
            track_map = {int(t[4]): int(t[6]) for t in t_main}
            
            # 이벤트 판별 (multi_event의 로직 재사용)
            for ename, handler in handlers.items():
                kwargs = {'helmet_tracks': t_helmet} if ename == "no_helmet" else {}
                triggered = handler.process(t_main, track_map, None, frame, fid, **kwargs)
                
                for ev in triggered:
                    print(f"🚨 [{ename.upper()} 알람] FID:{fid} | TID:{ev['tid']} | BBOX:{tuple(map(int, ev['bbox']))}")
                    alarms_display.add(ev['tid'])
                    alarm_cooldown = max(1, int(TARGET_FPS * 5)) # 5초 유지

            # 렌더링 파트
            render_frame = frame.copy()
            
            # ROI 마킹
            if roi_poly:
                cv2.polylines(render_frame, [np.array(roi_poly, np.int32)], True, (0, 255, 255), 2)
            if roi_lines:
                for i in range(0, len(roi_lines), 2):
                    if i + 1 < len(roi_lines): 
                        cv2.line(render_frame, tuple(roi_lines[i]), tuple(roi_lines[i+1]), (0, 0, 255), 2)

            for t in t_main:
                if int(t[6]) != ID_G_PERSON: continue
                tid = int(t[4])
                color = (0, 0, 255) if tid in alarms_display else (0, 255, 0)
                cv2.rectangle(render_frame, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, 2)
                cv2.putText(render_frame, f"P[{tid}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            for t in t_helmet:
                cv2.rectangle(render_frame, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (0, 165, 255), 2)
                cv2.putText(render_frame, f"No-Hel[{int(t[4])}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            if alarm_cooldown > 0:
                cv2.rectangle(render_frame, (0, 0), (render_frame.shape[1], render_frame.shape[0]), (0, 0, 255), 10)
                alarm_cooldown -= 1
            else:
                alarms_display.clear()

            cv2.putText(render_frame, f"SIMUL: {TARGET_FPS} FPS | FID: {fid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Test Simulation", render_frame)
            
            if cv2.waitKey(1) == ord('q'):
                reader.release()
                cv2.destroyAllWindows()
                print("테스트가 강제 종료되었습니다.")
                return

        reader.release()
        
    cv2.destroyAllWindows()
    print("\n✅ 모든 비디오 파일의 테스트가 완료되었습니다.")

if __name__ == "__main__":
    main()
