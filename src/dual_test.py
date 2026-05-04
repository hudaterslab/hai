import os
import sys
import cv2
import json
import numpy as np
import time
from collections import defaultdict

from common import (
    PROJECT_ROOT,
    ID_G_PERSON, ID_H_HELMET, ID_PERSON_LOW, ID_REFLECTIVE_VEST,
    ID_G_TRUCK, ID_H_NO_HELMET, ID_G_CAR, TARGET_VEHICLES,
    SCREEN_WIDTH, SCREEN_HEIGHT, SYS_CFG, get_center_point, get_distance
)
import event
from ai_core import VisionModelSync, SORTTracker

TEST_JSON_PATH = os.path.join(os.path.dirname(__file__), "test.json")
TEST_VIDEO_DIR = "test"
DEBUG_LOG_DIR = os.path.join(PROJECT_ROOT, "debug_logs")
RESULT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "result_videos") # 💡 녹화 영상 저장 폴더 추가

GWKIM_TO_COMMON_MAP = {
    0: ID_H_HELMET,
    1: ID_H_NO_HELMET,
    2: ID_G_PERSON,
    3: ID_G_CAR,
    4: ID_PERSON_LOW,
    5: ID_REFLECTIVE_VEST,
    6: ID_G_TRUCK
}

def load_test_config():
    if os.path.exists(TEST_JSON_PATH):
        with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_test_config(config):
    with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def apply_face_blur(image, face_detector, face_conf):
    if face_detector is None or image is None: 
        return image
    res = image.copy()
    try:
        for fx1, fy1, fx2, fy2, fscore, _ in face_detector.infer(image):
            if fscore <= face_conf: 
                continue
            fx1, fy1 = max(0, int(fx1)), max(0, int(fy1))
            fx2, fy2 = int(fx2), int(fy2)
            fh, fw = fy2 - fy1, fx2 - fx1
            if fw > image.shape[1] * 0.8 or fh > image.shape[0] * 0.8 or fw <= 0 or fh <= 0: 
                continue
            small = cv2.resize(res[fy1:fy2, fx1:fx2], (fw // 15 + 1, fh // 15 + 1), interpolation=cv2.INTER_LINEAR)
            res[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
    except Exception: 
        pass
    return res

def main():
    if not os.path.exists(TEST_VIDEO_DIR):
        print(f"⚠️ [에러] 테스트 폴더를 찾을 수 없습니다: {TEST_VIDEO_DIR}")
        return

    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_VIDEO_DIR, exist_ok=True) # 💡 녹화 폴더 생성
    
    video_files = sorted([f for f in os.listdir(TEST_VIDEO_DIR) if f.lower().endswith(('.avi', '.mp4', '.mkv'))])
    
    if not video_files:
        print(f"⚠️ [에러] '{TEST_VIDEO_DIR}' 폴더 내에 영상 파일이 존재하지 않습니다.")
        return

    all_configs = load_test_config()

    def get_abs_path(cfg_path):
        if os.path.isabs(cfg_path): return cfg_path
        return os.path.join(PROJECT_ROOT, cfg_path)

    main_path = get_abs_path("models/hanjin_cctv.pt")
    gwkim_path = get_abs_path("models/hanjin_cctv_gwkim.pt") # 확장자 주의 (.pt로 변경 권장)
    face_path = get_abs_path("models/yolov8m-face.pt")
    helmet_path = get_abs_path("models/helmet_3cls_v8.pt")

    print("\n⏳ 두 개의 모델(ORG, GWKIM) 및 헬멧/얼굴 모델을 메모리에 로드 중입니다...")
    
    engine_org = VisionModelSync(main_path)
    engine_gwkim = VisionModelSync(gwkim_path)
    engine_face = VisionModelSync(face_path)
    engine_helmet = VisionModelSync(helmet_path)

    main_conf = 0.40
    face_conf = 0.35
    helmet_conf = 0.45

    MAX_SNAPSHOTS = 4  
    
    VIDEO_W_DISP = 640
    VIDEO_H_DISP = 480
    CANVAS_WIDTH = 320 + VIDEO_W_DISP + VIDEO_W_DISP + 320 
    CANVAS_HEIGHT = max(VIDEO_H_DISP, 180 * MAX_SNAPSHOTS)

    target_fps = 5 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 💡 비디오 코덱 설정 (mp4)

    print("=" * 60)
    print("🚀 [배치 프로세싱 시작] 모든 영상을 순차적으로 테스트하고 녹화합니다.")
    print("=" * 60)

    for video_filename in video_files:
        video_path = os.path.join(TEST_VIDEO_DIR, video_filename)
        video_config = all_configs.get(video_filename)

        # 💡 자동화 테스트를 위해 설정이 없으면 사용자 입력을 기다리지 않고 기본값 자동 할당
        if not video_config:
            video_config = {"roi_poly": [], "roi_lines": [], "events": ["intrusion", "no_helmet"]}
            all_configs[video_filename] = video_config
            save_test_config(all_configs)

        roi_poly = video_config.get("roi_poly", [])
        roi_lines = video_config.get("roi_lines", [])
        active_events = video_config.get("events", [])

        if not active_events: 
            print(f"⏩ [{video_filename}] 활성화된 이벤트가 없어 건너뜁니다.")
            continue

        cap = cv2.VideoCapture(video_path)
        
        # 💡 VideoWriter 객체 초기화 (영상의 원래 배속을 유지하기 위해 target_fps 기준으로 저장)
        output_video_path = os.path.join(RESULT_VIDEO_DIR, f"result_{video_filename}.mp4")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, (CANVAS_WIDTH, CANVAS_HEIGHT))
        
        print(f"\n▶️ [{video_filename}] 테스트 및 녹화 시작 -> 저장 경로: {output_video_path}")

        motion_detector = event.MotionDetector(sensitivity=5)
        
        tracker_org = SORTTracker(track_thresh=main_conf, track_buffer=1.5*target_fps, is_helmet=False)
        tracker_gwkim = SORTTracker(track_thresh=main_conf, track_buffer=1.5*target_fps, is_helmet=False)
        tracker_helmet = SORTTracker(track_thresh=helmet_conf, track_buffer=1.5*target_fps, is_helmet=False)
        
        detectors_org = [event.EVENT_REGISTRY[evt](SYS_CFG.get("event_config", {}).get(evt, {}), roi_poly=roi_poly, roi_lines=roi_lines) for evt in active_events if evt in event.EVENT_REGISTRY]
        detectors_gwkim = [event.EVENT_REGISTRY[evt](SYS_CFG.get("event_config", {}).get(evt, {}), roi_poly=roi_poly, roi_lines=roi_lines) for evt in active_events if evt in event.EVENT_REGISTRY]
        
        snapshot_queue_org = []
        snapshot_queue_gwkim = []
        
        persistent_alarms_org = {} 
        persistent_alarms_gwkim = {} 

        simulated_fid = 0
        
        # 배치 모드이므로 일시정지는 기본적으로 해제
        is_paused = False
        advance_one_frame = False
        force_quit_all = False

        while True:
            if not is_paused or advance_one_frame:
                ret, frame = cap.read()
                advance_one_frame = False
                
                if not ret: 
                    break # 영상이 끝나면 자연스럽게 다음 영상으로 넘어감
                    
                simulated_fid += 1
                
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
                disp_org = frame.copy()
                disp_gwkim = frame.copy()
                motion_mask = motion_detector.apply(frame)

                # --- 추론 ---
                main_boxes_org = engine_org.infer(frame)
                raw_boxes_gwkim = engine_gwkim.infer(frame)
                helmet_boxes = engine_helmet.infer(frame) if "no_helmet" in active_events else []
                
                main_boxes_gwkim = []
                for b in raw_boxes_gwkim:
                    b = list(b)
                    cls_orig = int(b[5])
                    if cls_orig in GWKIM_TO_COMMON_MAP:
                        b[5] = GWKIM_TO_COMMON_MAP[cls_orig]
                        main_boxes_gwkim.append(b)
                
                # --- 트래킹 ---
                tracks_org = tracker_org.update(np.array(main_boxes_org)) if len(main_boxes_org) > 0 else tracker_org.predict_only()
                tracks_gwkim = tracker_gwkim.update(np.array(main_boxes_gwkim)) if len(main_boxes_gwkim) > 0 else tracker_gwkim.predict_only()
                tracks_helmet = tracker_helmet.update(np.array(helmet_boxes)) if len(helmet_boxes) > 0 else tracker_helmet.predict_only()
                    
                map_org = {int(t[4]): int(t[6]) for t in tracks_org}
                map_gwkim = {int(t[4]): int(t[6]) for t in tracks_gwkim}

                # --- 이벤트 판독 (ORG) ---
                for detector in detectors_org:
                    triggered = detector.process(tracks=tracks_org, track_map=map_org, motion_mask=motion_mask, frame=frame, fid=simulated_fid, helmet_tracks=tracks_helmet)
                    for evt in triggered:
                        tid = evt['tid']
                        persistent_alarms_org[tid] = detector.event_name
                        
                        x1, y1, x2, y2 = map(int, evt['bbox'])
                        snap = apply_face_blur(frame.copy(), engine_face, face_conf)
                        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(snap, f"ORG: {detector.event_name}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        snapshot_queue_org.insert(0, cv2.resize(snap, (320, 180)))
                        if len(snapshot_queue_org) > MAX_SNAPSHOTS: snapshot_queue_org.pop()

                for t in tracks_org:
                    x1, y1, x2, y2, tid, conf, cls_id = map(float, t)
                    tid, cls_id = int(tid), int(cls_id)
                    color = (0, 0, 255) if tid in persistent_alarms_org else (0, 255, 0)
                    cv2.rectangle(disp_org, (int(x1), int(y1)), (int(x2), int(y2)), color, 2 if tid in persistent_alarms_org else 1)
                    cv2.putText(disp_org, f"ID:{tid} Cls:{cls_id}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # --- 이벤트 판독 (GWKIM) ---
                for detector in detectors_gwkim:
                    triggered = detector.process(tracks=tracks_gwkim, track_map=map_gwkim, motion_mask=motion_mask, frame=frame, fid=simulated_fid, helmet_tracks=tracks_gwkim)
                    for evt in triggered:
                        tid = evt['tid']
                        persistent_alarms_gwkim[tid] = detector.event_name
                        
                        x1, y1, x2, y2 = map(int, evt['bbox'])
                        snap = apply_face_blur(frame.copy(), engine_face, face_conf)
                        cv2.rectangle(snap, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(snap, f"GWKIM: {detector.event_name}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        
                        snapshot_queue_gwkim.insert(0, cv2.resize(snap, (320, 180)))
                        if len(snapshot_queue_gwkim) > MAX_SNAPSHOTS: snapshot_queue_gwkim.pop()

                for t in tracks_gwkim:
                    x1, y1, x2, y2, tid, conf, cls_id = map(float, t)
                    tid, cls_id = int(tid), int(cls_id)
                    color = (255, 0, 255) if tid in persistent_alarms_gwkim else (255, 255, 0)
                    cv2.rectangle(disp_gwkim, (int(x1), int(y1)), (int(x2), int(y2)), color, 2 if tid in persistent_alarms_gwkim else 1)
                    cv2.putText(disp_gwkim, f"ID:{tid} Cls:{cls_id}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # --- 캔버스 렌더링 ---
                canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                
                disp_org_small = cv2.resize(disp_org, (VIDEO_W_DISP, VIDEO_H_DISP))
                disp_gwkim_small = cv2.resize(disp_gwkim, (VIDEO_W_DISP, VIDEO_H_DISP))
                
                canvas[0:VIDEO_H_DISP, 320:320+VIDEO_W_DISP] = disp_org_small
                canvas[0:VIDEO_H_DISP, 320+VIDEO_W_DISP:320+VIDEO_W_DISP*2] = disp_gwkim_small
                
                cv2.putText(canvas, "[ ORIGINAL MODEL ]", (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(canvas, "[ GWKIM MODEL ]", (330+VIDEO_W_DISP, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                for idx in range(MAX_SNAPSHOTS):
                    y_offset = idx * 180
                    
                    if idx < len(snapshot_queue_org):
                        canvas[y_offset:y_offset+180, 0:320] = snapshot_queue_org[idx]
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (40, 40, 40), -1)
                    
                    if idx < len(snapshot_queue_gwkim):
                        canvas[y_offset:y_offset+180, CANVAS_WIDTH-320:CANVAS_WIDTH] = snapshot_queue_gwkim[idx]
                        cv2.rectangle(canvas, (CANVAS_WIDTH-320, y_offset), (CANVAS_WIDTH, y_offset+180), (255, 0, 255), 2)
                    else:
                        cv2.rectangle(canvas, (CANVAS_WIDTH-320, y_offset), (CANVAS_WIDTH, y_offset+180), (40, 40, 40), -1)

                # 💡 완성된 프레임을 VideoWriter를 통해 녹화
                video_writer.write(canvas)

            cv2.imshow("Dual Batch & Record Mode", canvas)
            
            # Batch Mode이므로 대기 시간을 1ms로 최소화하여 빠르게 렌더링
            key = cv2.waitKey(1 if not is_paused else 0) & 0xFF
            if key == ord(' '): is_paused = not is_paused
            elif key == ord('f') and is_paused: advance_one_frame = True
            elif key == ord('q'): force_quit_all = True; break
            elif key == ord('n'): break # 현재 영상 녹화를 중단하고 다음 영상으로 스킵

        # 💡 한 영상 처리가 끝나면 반드시 writer와 cap을 해제
        video_writer.release()
        cap.release()
        print(f"✅ [{video_filename}] 처리 완료 및 녹화본 저장 완료.")
        
        if force_quit_all: 
            print("🛑 사용자에 의해 일괄 테스트가 강제 종료되었습니다.")
            break

    cv2.destroyAllWindows()
    print("\n🎉 모든 영상의 배치 프로세싱 및 녹화가 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()