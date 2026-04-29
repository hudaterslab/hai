import os
import sys
import cv2
import json
import numpy as np
import time
from collections import defaultdict

# 💡 단말기 코어를 그대로 사용 (PROJECT_ROOT 추가)
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

def load_test_config():
    if os.path.exists(TEST_JSON_PATH):
        with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_test_config(config):
    with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def apply_face_blur(image, face_detector, face_conf):
    """단말기의 얼굴 모자이크 로직 시뮬레이션"""
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

class ROIPicker:
    def __init__(self, frame, selected_events):
        self.frame = frame
        self.clone = frame.copy()
        self.poly_points = []
        self.line_points = []
        self.selected_events = selected_events
        
        requires_poly = any(e in ['intrusion', 'illegal_parking', 'signal_vehicle'] for e in selected_events)
        requires_line = 'conveyor_crossing' in selected_events
        
        self.mode = 'L' if requires_line and not requires_poly else 'P'
        self.window_name = "ROI Setup (P: Poly, L: Line, RightClick: Undo, S: Save, C: Clear)"
        
    def mouse_callback(self, cv_event, x, y, flags, param):
        if cv_event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'P': self.poly_points.append([x, y])
            elif self.mode == 'L': self.line_points.append([x, y])
        elif cv_event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == 'P' and self.poly_points: self.poly_points.pop()
            elif self.mode == 'L' and self.line_points: self.line_points.pop()

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            temp_frame = self.clone.copy()
            if len(self.poly_points) > 0:
                pts = np.array(self.poly_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
                for pt in self.poly_points: cv2.circle(temp_frame, tuple(pt), 4, (0, 0, 255), -1)
                    
            if len(self.line_points) > 0:
                pts = np.array(self.line_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                for pt in self.line_points: cv2.circle(temp_frame, tuple(pt), 4, (0, 255, 0), -1)

            mode_text = "Mode: POLYGON" if self.mode == 'P' else "Mode: LINE"
            cv2.putText(temp_frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(self.window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'): self.mode = 'P'
            elif key == ord('l'): self.mode = 'L'
            elif key == ord('c'): self.poly_points, self.line_points = [], []
            elif key == ord('s'):
                print("✅ ROI 설정이 저장되었습니다.")
                break
            elif key == 27: 
                print("❌ ROI 설정을 취소합니다.")
                sys.exit(0)
                
        cv2.destroyWindow(self.window_name)
        return self.poly_points, self.line_points

def main():
    if not os.path.exists(TEST_VIDEO_DIR):
        print(f"⚠️ [에러] 테스트 폴더를 찾을 수 없습니다: {TEST_VIDEO_DIR}")
        return

    video_files = sorted([f for f in os.listdir(TEST_VIDEO_DIR) if f.lower().endswith(('.avi', '.mp4', '.mkv'))])
    
    if not video_files:
        print(f"⚠️ [에러] '{TEST_VIDEO_DIR}' 폴더 내에 영상 파일이 존재하지 않습니다.")
        return

    all_configs = load_test_config()
    
    print("=" * 60)
    auto_next_input = input(">> 테스트 종료 후 다음 영상으로 자동 진행하시겠습니까? (y/N): ").strip().lower()
    auto_next = True if auto_next_input == 'y' else False
    print("=" * 60)

    # 💡 [핵심 보완] 실행 위치와 상관없이 무조건 올바른 절대 경로를 매핑
    def get_abs_path(cfg_path):
        if os.path.isabs(cfg_path): return cfg_path
        return os.path.join(PROJECT_ROOT, cfg_path)

    main_path = get_abs_path(SYS_CFG.get("models", {}).get("MAIN", "models/hanjin_cctv.pt"))
    face_path = get_abs_path(SYS_CFG.get("models", {}).get("FACE", "models/yolov8m-face.pt"))
    helmet_path = get_abs_path(SYS_CFG.get("models", {}).get("HELMET", "models/helmet_3cls_v8.dxnn"))

    print("\n⏳ AI 통합 모델 및 헬멧/얼굴 모델을 메모리에 로드 중입니다...")
    print(f" - MAIN 타겟: {main_path}")
    print(f" - FACE 타겟: {face_path}")
    print(f" - HELMET 타겟: {helmet_path}")
    
    engine_main = VisionModelSync(main_path)
    engine_face = VisionModelSync(face_path)
    engine_helmet = VisionModelSync(helmet_path)

    # 💡 [상용화 디버깅] 메모리에 실제 모델이 로드되었는지 강제 검증 (사일런트 페일 방지)
    def check_engine(engine, name):
        loaded = False
        if hasattr(engine, 'model') and engine.model is not None: loaded = True
        if hasattr(engine, 'engine') and engine.engine is not None: loaded = True
        if not loaded:
            print(f"❌ [치명적 오류] {name} 모델 로드 실패! (경로 미스매치 또는 패키지 오류)")
        return loaded
        
    main_ok = check_engine(engine_main, "MAIN (hanjin_cctv)")
    helmet_ok = check_engine(engine_helmet, "HELMET (helmet_3cls_v8)")
    check_engine(engine_face, "FACE")
    
    if not main_ok:
        print("⚠️ 주의: BBOX가 하나도 렌더링되지 않는 원인입니다. 코덱, 경로, 또는 ultralytics 설치 상태를 점검하십시오.")

    main_conf = SYS_CFG.get("model_confidences", {}).get("MAIN", 0.40)
    face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
    helmet_conf = SYS_CFG.get("model_confidences", {}).get("HELMET", 0.45)

    snapshot_queue = []
    MAX_SNAPSHOTS = 4  
    
    CANVAS_WIDTH = SCREEN_WIDTH + 640 
    CANVAS_HEIGHT = SCREEN_HEIGHT

    base_skip_frames = SYS_CFG.get("SKIP_FRAMES", 1)
    target_fps = SYS_CFG.get("REC_FPS", 3)

    force_quit_all = False

    for video_filename in video_files:
        if force_quit_all:
            break
            
        video_path = os.path.join(TEST_VIDEO_DIR, video_filename)

        while True:
            video_config = all_configs.get(video_filename)

            if not video_config:
                print(f"\n========================================================")
                print(f"🛠️ [{video_filename}] 설정 마법사 실행")
                print(f"========================================================")
                cap = cv2.VideoCapture(video_path)
                ret, first_frame = cap.read()
                cap.release()
                
                if not ret: 
                    print(f"⚠️ 영상을 읽을 수 없습니다. 건너뜁니다.")
                    break
                    
                first_frame = cv2.resize(first_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
                preview_window = f"Preview: {video_filename}"
                cv2.imshow(preview_window, first_frame)
                cv2.waitKey(1) 
                
                print("\n테스트할 이벤트 번호를 콤마(,)로 구분하여 입력하세요.")
                print("1: 침입(Intrusion), 2: 주정차(Parking), 3: 횡단(Crossing), 4: 안전모(Helmet), 5: 신호수차량(SignalVehicle)")
                event_input = input("입력 (건너뛰려면 엔터): ")
                cv2.destroyWindow(preview_window)
                
                if not event_input.strip(): 
                    break 
                    
                event_map = {"1": "intrusion", "2": "illegal_parking", "3": "conveyor_crossing", "4": "no_helmet", "5": "signal_vehicle"}
                selected_events = [event_map[v.strip()] for v in event_input.split(',') if v.strip() in event_map]
                
                picker = ROIPicker(first_frame, selected_events)
                poly_pts, line_pts = picker.run()
                
                video_config = {"roi_poly": poly_pts, "roi_lines": line_pts, "events": selected_events}
                all_configs[video_filename] = video_config
                save_test_config(all_configs)

            roi_poly = video_config.get("roi_poly", [])
            roi_lines = video_config.get("roi_lines", [])
            active_events = video_config.get("events", [])

            if not active_events: 
                break

            cap = cv2.VideoCapture(video_path)
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(round(native_fps / target_fps))) if native_fps > 0 else 1
            
            print(f"\n▶️ [{video_filename}] 재생 시작. (n: 다음, r: 재시작, c: 설정 초기화(재설정), q: 종료, 1/2/3: 배속)")
            print(f"⚙️ 원본 FPS: {native_fps:.1f} -> 설정 FPS: {target_fps} (Frame Interval: {frame_interval})")

            active_skip_frames = base_skip_frames
            motion_detector = event.MotionDetector(sensitivity=5)
            trajectory_tracker = event.TrajectoryTracker(max_len=30)
            
            target_buffer = max(1, int(SYS_CFG.get("track_buffer_sec", 1.5) * (target_fps / max(1, active_skip_frames))))
            
            tracker_main = SORTTracker(track_thresh=main_conf, track_buffer=target_buffer, is_helmet=False)
            tracker_helmet = SORTTracker(track_thresh=helmet_conf, track_buffer=target_buffer, is_helmet=False)
            
            detectors = []
            for evt_name in active_events:
                if evt_name in event.EVENT_REGISTRY:
                    detectors.append(event.EVENT_REGISTRY[evt_name](SYS_CFG.get("event_config", {}).get(evt_name, {}), roi_poly=roi_poly, roi_lines=roi_lines))
            
            raw_fid = 0
            simulated_fid = 0
            snapshot_cooldowns = {}
            last_canvas = None
            play_delay = 30 
            speed_text = "1x"
            
            action = "next" 

            while True:
                ret, frame = cap.read()
                if not ret:
                    if last_canvas is not None:
                        overlay = last_canvas.copy()
                        cv2.rectangle(overlay, (320, 0), (SCREEN_WIDTH + 320, CANVAS_HEIGHT), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, last_canvas, 0.4, 0, last_canvas)
                        cv2.putText(last_canvas, "VIDEO ENDED", (320 + SCREEN_WIDTH//2 - 150, CANVAS_HEIGHT//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        cv2.putText(last_canvas, "Press 'n': Next | 'r': Replay | 'c': Reconfig | 'q': Quit", (320 + SCREEN_WIDTH//2 - 280, CANVAS_HEIGHT//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow("CCTV Event Test Runner", last_canvas)
                        
                        if auto_next:
                            cv2.waitKey(1000)
                            action = "next"
                        else:
                            while True:
                                k = cv2.waitKey(0) & 0xFF
                                if k == ord('q'): action = "quit"; break
                                elif k == ord('n') or k == 27: action = "next"; break
                                elif k == ord('r'): action = "replay"; break
                                elif k == ord('c'): action = "reconfig"; break
                    break
                    
                raw_fid += 1
                
                if raw_fid % frame_interval != 0:
                    continue

                simulated_fid += 1
                
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
                display_frame = frame.copy()
                motion_mask = motion_detector.apply(frame)

                main_boxes = []
                helmet_boxes = []
                if active_skip_frames == 0 or (simulated_fid - 1) % (active_skip_frames + 1) == 0:
                    main_boxes = engine_main.infer(frame)
                    if "no_helmet" in active_events:
                        helmet_boxes = engine_helmet.infer(frame)
                
                main_tracks = tracker_main.update(np.array(main_boxes)) if len(main_boxes) > 0 else tracker_main.predict_only()
                helmet_tracks = tracker_helmet.update(np.array(helmet_boxes)) if len(helmet_boxes) > 0 else tracker_helmet.predict_only()
                    
                track_map = {int(t[4]): int(t[6]) for t in main_tracks}
                trajectory_tracker.update_and_draw(display_frame, main_tracks)

                for detector in detectors:
                    triggered_events = detector.process(
                        tracks=main_tracks, track_map=track_map, motion_mask=motion_mask,
                        frame=frame, fid=simulated_fid, helmet_tracks=helmet_tracks
                    )
                    
                    for evt in triggered_events:
                        tid = evt['tid']
                        bbox = evt['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(display_frame, f"EVENT: {detector.event_name}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                        evt_key = (detector.event_name, tid)
                        current_time = time.time()
                        
                        if evt_key not in snapshot_cooldowns or (current_time - snapshot_cooldowns[evt_key] > 5.0):
                            snapshot_cooldowns[evt_key] = current_time
                            print(f"[🔥 새 이벤트 발생] {detector.event_name} | ID: {tid} | Frame: {simulated_fid} (Raw: {raw_fid})")
                            
                            base_snap_frame = evt.get('frame')
                            if base_snap_frame is None: base_snap_frame = frame.copy()
                            else: base_snap_frame = base_snap_frame.copy()
                                
                            blurred_snap = apply_face_blur(base_snap_frame, engine_face, face_conf)
                            
                            cv2.rectangle(blurred_snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            now_str = time.strftime('%H:%M:%S')
                            cv2.putText(blurred_snap, f"{detector.event_name} ID:{tid}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            new_snap = cv2.resize(blurred_snap, (320, 180))
                            snapshot_queue.insert(0, new_snap)
                            if len(snapshot_queue) > MAX_SNAPSHOTS:
                                snapshot_queue.pop()

                for t in main_tracks:
                    x1, y1, x2, y2, tid, conf, cls_id = map(float, t)
                    cls_id = int(cls_id)
                    
                    if cls_id in [ID_H_HELMET, ID_H_NO_HELMET]:
                        continue 
                        
                    color, label = (255, 255, 255), f"OBJ:{int(tid)}"
                    if cls_id == ID_G_PERSON: color, label = (0, 255, 0), f"Person:{int(tid)}"
                    elif cls_id == ID_G_CAR: color, label = (255, 100, 0), f"Car:{int(tid)}"
                    elif cls_id == ID_G_TRUCK: color, label = (255, 100, 0), f"Truck:{int(tid)}"
                    elif cls_id == ID_PERSON_LOW: color, label = (0, 255, 100), f"LowBody:{int(tid)}"
                    elif cls_id == ID_REFLECTIVE_VEST: color, label = (255, 255, 0), f"Vest:{int(tid)}"
                        
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(display_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if "no_helmet" in active_events:
                    for t in helmet_tracks:
                        x1, y1, x2, y2, tid, conf, cls_id = map(float, t)
                        cls_id = int(cls_id)
                        
                        if cls_id == 0: color, label = (255, 200, 0), f"Helmet(H):{int(tid)}"
                        elif cls_id == 1: color, label = (0, 0, 255), f"Head(H):{int(tid)}"
                        else: continue
                            
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(display_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if len(roi_poly) >= 3:
                    pts = np.array(roi_poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
                    
                if len(roi_lines) >= 2:
                    pts = np.array(roi_lines, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                            
                ui_text = f"Speed: {speed_text} | FPS Sim: {target_fps} | Skip: {active_skip_frames} | Press 'c' to Reconfig"
                cv2.rectangle(display_frame, (10, 10), (700, 45), (0, 0, 0), -1)
                cv2.putText(display_frame, ui_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                canvas[0:SCREEN_HEIGHT, 320:SCREEN_WIDTH + 320] = display_frame

                if motion_mask is not None:
                    mask_resized = cv2.resize(cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR), (320, 180))
                    canvas[0:180, SCREEN_WIDTH + 320:CANVAS_WIDTH] = mask_resized
                    cv2.rectangle(canvas, (SCREEN_WIDTH + 320, 0), (CANVAS_WIDTH, 180), (255, 0, 0), 2)
                    cv2.putText(canvas, "[ MOTION MASK ]", (SCREEN_WIDTH + 330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                for idx in range(MAX_SNAPSHOTS):
                    y_offset = idx * 180
                    if idx < len(snapshot_queue):
                        snap = snapshot_queue[idx]
                        canvas[y_offset:y_offset+180, 0:320] = snap
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (0, 255, 255), 2)
                        cv2.putText(canvas, f"[ API BLUR SNAP {idx+1} ]", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (40, 40, 40), -1)
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (100, 100, 100), 2)
                        cv2.putText(canvas, f"EMPTY SLOT {idx+1}", (100, y_offset + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                last_canvas = canvas.copy()
                cv2.imshow("CCTV Event Test Runner", canvas)
                
                key = cv2.waitKey(play_delay) & 0xFF
                if key == ord('q'): action = "quit"; break
                elif key == ord('n'): action = "next"; break
                elif key == ord('r'): action = "replay"; break
                elif key == ord('c'): action = "reconfig"; break
                elif key == ord('1'): play_delay, speed_text = 30, "1x"
                elif key == ord('2'): play_delay, speed_text = 5, "FAST"
                elif key == ord('3'): play_delay, speed_text = 100, "SLOW"

            cap.release()
            
            if action == "quit": 
                force_quit_all = True
                break
            elif action == "next":
                break 
            elif action == "replay":
                continue 
            elif action == "reconfig":
                print(f"🔄 [{video_filename}] 설정을 초기화하고 마법사를 다시 시작합니다.")
                if video_filename in all_configs:
                    del all_configs[video_filename]
                    save_test_config(all_configs)
                continue 

    cv2.destroyAllWindows()
    print("\n✅ 모든 영상의 테스트가 완료되었습니다.")

if __name__ == "__main__":
    main()