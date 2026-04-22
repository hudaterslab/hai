import os
import sys
import cv2
import json
import numpy as np
import time
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ [오류] ultralytics 패키지가 설치되지 않았습니다. 'pip install ultralytics'를 실행하십시오.")
    sys.exit(1)

# 로컬 임포트 
from common import (
    ID_G_PERSON, ID_G_CAR, ID_H_PERSON, ID_H_NO_HELMET, ID_H_HELMET, TARGET_VEHICLES,
    SCREEN_WIDTH, SCREEN_HEIGHT, SYS_CFG, get_center_point, get_distance
)
import event

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

# 💡 [핵심 추가] Tracking 단절 시 ID를 강제로 복구하는 Re-ID 클래스
class TrackReID:
    def __init__(self, max_lost=15, max_dist=100):
        self.last_seen = {}
        self.id_map = {}
        self.max_lost = max_lost
        self.max_dist = max_dist

    def process(self, tracks, fid):
        mapped_tracks = []
        for t in tracks:
            x1, y1, x2, y2, tid, cls_id, conf = t
            tid = int(tid)
            
            # 기존에 맵핑된 이력이 있다면 기존 ID를 계승
            while tid in self.id_map:
                tid = self.id_map[tid]
                
            center = get_center_point(x1, y1, x2, y2)
            
            # 처음 보는 ID라면, 최근에 근처에서 잃어버린 ID가 있는지 탐색
            if tid not in self.last_seen:
                best_match = None
                min_dist = self.max_dist
                
                for lost_id, data in self.last_seen.items():
                    frames_lost = fid - data['fid']
                    if 0 < frames_lost <= self.max_lost and data['cls'] == cls_id:
                        l_center = get_center_point(*data['bbox'])
                        dist = get_distance(center, l_center)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = lost_id
                            
                if best_match is not None:
                    self.id_map[tid] = best_match
                    tid = best_match
            
            self.last_seen[tid] = {'bbox': (x1, y1, x2, y2), 'fid': fid, 'cls': cls_id}
            mapped_tracks.append([x1, y1, x2, y2, float(tid), cls_id, conf])
            
        return mapped_tracks

class ROIPicker:
    def __init__(self, frame, selected_events):
        self.frame = frame
        self.clone = frame.copy()
        self.poly_points = []
        self.line_points = []
        self.selected_events = selected_events
        
        requires_poly = any(e in ['intrusion', 'illegal_parking', 'signal_vehicle'] for e in selected_events)
        requires_line = 'conveyor_crossing' in selected_events
        
        if requires_line and not requires_poly:
            self.mode = 'L'
        else:
            self.mode = 'P'
            
        self.window_name = "ROI Setup (P: Poly, L: Line, RightClick: Undo, S: Save, C: Clear)"
        
    def mouse_callback(self, cv_event, x, y, flags, param):
        if cv_event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'P':
                self.poly_points.append([x, y])
            elif self.mode == 'L':
                self.line_points.append([x, y])
        elif cv_event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == 'P' and self.poly_points:
                self.poly_points.pop()
            elif self.mode == 'L' and self.line_points:
                self.line_points.pop()

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            temp_frame = self.clone.copy()
            
            if len(self.poly_points) > 0:
                pts = np.array(self.poly_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
                for pt in self.poly_points:
                    cv2.circle(temp_frame, tuple(pt), 4, (0, 0, 255), -1)
                    
            if len(self.line_points) > 0:
                pts = np.array(self.line_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                for pt in self.line_points:
                    cv2.circle(temp_frame, tuple(pt), 4, (0, 255, 0), -1)

            mode_text = "Mode: POLYGON" if self.mode == 'P' else "Mode: LINE"
            cv2.putText(temp_frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(self.window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):
                self.mode = 'P'
            elif key == ord('l'):
                self.mode = 'L'
            elif key == ord('c'):
                self.poly_points = []
                self.line_points = []
            elif key == ord('s'):
                requires_poly = any(e in ['intrusion', 'illegal_parking', 'signal_vehicle'] for e in self.selected_events)
                requires_line = 'conveyor_crossing' in self.selected_events
                
                if requires_poly and len(self.poly_points) < 3:
                    print("⚠️ [경고] 폴리곤 기반 이벤트가 선택되었으나 폴리곤(점 3개 이상)이 명확히 그려지지 않았습니다.")
                    continue
                if requires_line and len(self.line_points) < 2:
                    print("⚠️ [경고] 횡단 이벤트가 선택되었으나 라인(점 2개 이상)이 그려지지 않았습니다.")
                    continue
                    
                print("✅ ROI 설정이 저장되었습니다.")
                break
            elif key == 27: 
                print("❌ ROI 설정을 취소합니다.")
                sys.exit(0)
                
        cv2.destroyWindow(self.window_name)
        return self.poly_points, self.line_points

def parse_yolo_tracks(results):
    tracks = []
    if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for box, tid, cls, conf in zip(boxes, ids, classes, confs):
            tracks.append([box[0], box[1], box[2], box[3], int(tid), int(cls), float(conf)])
    return tracks

def main():
    general_model_path = os.path.join("..", "models", "YOLOV8M-1.pt")
    helmet_model_path = os.path.join("..", "models", "helmet_3cls_v8.pt")

    if not os.path.exists(TEST_VIDEO_DIR):
        print(f"⚠️ [에러] 테스트 폴더를 찾을 수 없습니다: {TEST_VIDEO_DIR}")
        return

    video_files = [f for f in os.listdir(TEST_VIDEO_DIR) if f.lower().endswith(('.avi', '.mp4', '.mkv'))]
    
    if not video_files:
        print(f"⚠️ [에러] '{TEST_VIDEO_DIR}' 폴더 내에 영상 파일이 존재하지 않습니다.")
        return

    all_configs = load_test_config()

    print("⏳ AI 모델을 GPU 메모리에 로드 중입니다...")
    model_general = YOLO(general_model_path)
    model_helmet = YOLO(helmet_model_path) if os.path.exists(helmet_model_path) else None

    snapshot_queue = []
    MAX_SNAPSHOTS = 4  
    
    CANVAS_WIDTH = SCREEN_WIDTH + 640 
    CANVAS_HEIGHT = SCREEN_HEIGHT

    base_skip_frames = SYS_CFG.get("SKIP_FRAMES", 4)

    for video_filename in video_files:
        print(f"\n========================================================")
        print(f"🎬 현재 테스트 중인 영상: {video_filename}")
        print(f"========================================================")
        
        video_path = os.path.join(TEST_VIDEO_DIR, video_filename)
        video_config = all_configs.get(video_filename)

        if not video_config:
            print(f"\n[{video_filename}] 에 대한 설정이 test.json에 없습니다. 최초 설정을 진행합니다.")
            
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret:
                print("⚠️ [에러] 영상의 첫 프레임을 읽을 수 없습니다.")
                continue
                
            first_frame = cv2.resize(first_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            preview_window = f"Preview: {video_filename} (Check Terminal)"
            cv2.imshow(preview_window, first_frame)
            cv2.waitKey(1) 
            
            print("\n테스트할 이벤트 번호를 콤마(,)로 구분하여 입력하세요.")
            print("1: 침입(Intrusion), 2: 주정차(Parking), 3: 횡단(Crossing), 4: 안전모(Helmet), 5: 신호수차량(SignalVehicle)")
            event_input = input("입력 (건너뛰려면 엔터): ")
            
            cv2.destroyWindow(preview_window)
            
            if not event_input.strip():
                print("설정을 건너뜁니다.")
                continue
                
            event_map = {
                "1": "intrusion", "2": "illegal_parking", "3": "conveyor_crossing", 
                "4": "no_helmet", "5": "signal_vehicle"
            }
            selected_events = [event_map[v.strip()] for v in event_input.split(',') if v.strip() in event_map]
            
            print("\n--- ROI 그리기 도구 ---")
            print("P 키: 폴리곤 모드 / L 키: 라인 모드")
            print("마우스 좌클릭: 점 추가 / 우클릭: 취소")
            print("S 키: 저장 및 종료 / C 키: 초기화")
            
            picker = ROIPicker(first_frame, selected_events)
            poly_pts, line_pts = picker.run()
            
            video_config = {
                "roi_poly": poly_pts,
                "roi_lines": line_pts,
                "events": selected_events
            }
            all_configs[video_filename] = video_config
            save_test_config(all_configs)
            print(f"✅ {TEST_JSON_PATH} 에 설정을 저장했습니다.\n")

        roi_poly = video_config.get("roi_poly", [])
        roi_lines = video_config.get("roi_lines", [])
        active_events = video_config.get("events", [])

        if not active_events:
            print("활성화된 이벤트가 없어 다음 영상으로 넘어갑니다.")
            continue

        force_quit_all = False

        while True:
            cap = cv2.VideoCapture(video_path)
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            active_skip_frames = base_skip_frames
            
            if video_fps > 0 and video_fps < 15:
                print(f"⚠️ [알림] 영상 FPS가 {video_fps:.1f}로 매우 낮습니다. (이미 프레임 드롭된 영상)")
                print("⚠️ 이중 드롭(Double Drop)을 방지하기 위해 프레임 스킵을 자동으로 끕니다.")
                active_skip_frames = 0

            motion_detector = event.MotionDetector(sensitivity=5)
            trajectory_tracker = event.TrajectoryTracker(max_len=30)
            
            # 💡 [핵심 연동] Re-ID 모듈 초기화
            reid_general = TrackReID(max_lost=15, max_dist=100)
            reid_helmet = TrackReID(max_lost=15, max_dist=100)
            
            config_mock = {
                "snapshot_mode": "crossing_moment", "distance_ratio": 0.3,
                "min_distance_px": 15, "candidate_ttl_sec": 5.0,
                "direction_check": True, "max_crossing_angle": 45.0
            }
            
            detectors = []
            if "intrusion" in active_events: detectors.append(event.IntrusionDetector(config_mock, roi_poly=roi_poly))
            if "illegal_parking" in active_events: detectors.append(event.ParkingDetector(config_mock, roi_poly=roi_poly))
            if "conveyor_crossing" in active_events: detectors.append(event.CrossingDetector(config_mock, roi_lines=roi_lines))
            if "no_helmet" in active_events: detectors.append(event.HelmetDetector(config_mock))
            if "signal_vehicle" in active_events: detectors.append(event.SignalVehicleDetector(config_mock, roi_poly=roi_poly))

            need_general = any("general" in getattr(d, 'required_models', []) for d in detectors)
            need_helmet = any("helmet" in getattr(d, 'required_models', []) for d in detectors)
            
            fid = 0
            snapshot_cooldowns = {}
            last_canvas = None
            
            skip_next = False
            replay_video = False
            
            play_delay = 30 
            speed_text = "1x"
            
            full_track_history = defaultdict(list)
            triggered_events_log = []
            
            print(f"▶️ [{video_filename}] 재생. (n: 다음, r: 재시작, q: 종료, s: 스킵 토글, 1/2/3: 배속)")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"영상이 종료되었습니다. 스냅샷 리뷰를 위해 일시정지합니다.")
                    
                    log_data = {
                        "video_filename": video_filename,
                        "roi_poly": roi_poly,
                        "roi_lines": roi_lines,
                        "active_events": active_events,
                        "events_triggered": triggered_events_log,
                        "tracks": dict(full_track_history)
                    }
                    log_path = os.path.join(TEST_VIDEO_DIR, f"{os.path.splitext(video_filename)[0]}_log.json")
                    with open(log_path, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=4, ensure_ascii=False)
                    
                    if last_canvas is not None:
                        overlay = last_canvas.copy()
                        cv2.rectangle(overlay, (320, 0), (SCREEN_WIDTH + 320, CANVAS_HEIGHT), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, last_canvas, 0.4, 0, last_canvas)
                        
                        cv2.putText(last_canvas, "VIDEO ENDED (LOG EXPORTED)", (320 + SCREEN_WIDTH//2 - 250, CANVAS_HEIGHT//2 - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        
                        cv2.putText(last_canvas, "Press 'n': Next | 'r': Replay | 'q': Quit", (320 + SCREEN_WIDTH//2 - 250, CANVAS_HEIGHT//2 + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        cv2.imshow("CCTV Event Test Runner", last_canvas)
                        
                        while True:
                            k = cv2.waitKey(0) & 0xFF
                            if k == ord('q'):
                                force_quit_all = True
                                break
                            elif k == ord('n') or k == 27:
                                skip_next = True
                                break
                            elif k == ord('r'):
                                replay_video = True
                                break
                    break
                    
                fid += 1
                
                if active_skip_frames > 0 and (fid - 1) % (active_skip_frames + 1) != 0:
                    continue

                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
                display_frame = frame.copy()

                motion_mask = motion_detector.apply(frame)

                general_tracks = []
                if model_general and need_general:
                    g_results = model_general.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.15)
                    raw_general_tracks = parse_yolo_tracks(g_results)
                    
                    # 💡 [핵심 연동] Re-ID 필터를 거쳐 끊어진 ID를 복구합니다.
                    general_tracks = reid_general.process(raw_general_tracks, fid)
                    
                    for t in general_tracks:
                        x1, y1, x2, y2, tid, cls_id, _ = t
                        full_track_history[f"G_{int(tid)}"].append({
                            "frame": fid,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "class": int(cls_id)
                        })
                
                helmet_tracks = []
                if model_helmet and need_helmet:
                    h_results = model_helmet.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.20)
                    raw_helmet_tracks = parse_yolo_tracks(h_results)
                    
                    # 💡 [핵심 연동] 헬멧 모델에도 Re-ID 적용
                    helmet_tracks = reid_helmet.process(raw_helmet_tracks, fid)
                    
                    for t in helmet_tracks:
                        x1, y1, x2, y2, tid, cls_id, _ = t
                        full_track_history[f"H_{int(tid)}"].append({
                            "frame": fid,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "class": int(cls_id)
                        })

                track_maps = {
                    "helmet": {int(t[4]): int(t[5]) for t in helmet_tracks},
                    "general": {int(t[4]): int(t[5]) for t in general_tracks}
                }

                trajectory_tracker.update_and_draw(display_frame, general_tracks)

                for detector in detectors:
                    triggered_events = detector.process(
                        helmet_tracks=helmet_tracks, general_tracks=general_tracks,
                        track_maps=track_maps, motion_mask=motion_mask,
                        frame=frame, fid=fid
                    )
                    
                    for evt in triggered_events:
                        tid = evt['tid']
                        bbox = evt['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(display_frame, f"EVENT: {detector.event_name}", (x1, max(20, y1-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                        triggered_events_log.append({
                            "frame": fid,
                            "event": detector.event_name,
                            "tid": int(tid),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })

                        evt_key = (detector.event_name, tid)
                        current_time = time.time()
                        
                        if evt_key not in snapshot_cooldowns or (current_time - snapshot_cooldowns[evt_key] > 5.0):
                            snapshot_cooldowns[evt_key] = current_time
                            print(f"[🔥 새 이벤트 발생] {detector.event_name} | ID: {tid} | Frame: {fid}")
                            
                            base_snap_frame = evt.get('frame')
                            if base_snap_frame is None:
                                base_snap_frame = frame.copy()
                            else:
                                base_snap_frame = base_snap_frame.copy()
                                
                            cv2.rectangle(base_snap_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            now_str = time.strftime('%H:%M:%S')
                            cv2.putText(base_snap_frame, f"{detector.event_name} ID:{tid} {now_str}", 
                                        (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            new_snap = cv2.resize(base_snap_frame, (320, 180))
                            snapshot_queue.insert(0, new_snap)
                            if len(snapshot_queue) > MAX_SNAPSHOTS:
                                snapshot_queue.pop()

                for t in general_tracks:
                    x1, y1, x2, y2, tid, cls_id, conf = map(float, t)
                    cls_id = int(cls_id)
                    if cls_id == ID_G_PERSON:
                        color = (0, 255, 0)
                        label = f"P:{int(tid)} {conf:.2f}"
                    elif cls_id in TARGET_VEHICLES:
                        color = (255, 150, 0)
                        label = f"V:{int(tid)} {conf:.2f}"
                    else:
                        color = (150, 150, 150)
                        label = f"G_{cls_id}:{int(tid)} {conf:.2f}"
                        
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    cv2.putText(display_frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                for t in helmet_tracks:
                    x1, y1, x2, y2, tid, cls_id, conf = map(float, t)
                    cls_id = int(cls_id)
                    if cls_id == ID_H_HELMET:
                        color = (255, 255, 0)
                        label = f"H_ON:{int(tid)} {conf:.2f}"
                    elif cls_id == ID_H_NO_HELMET:
                        color = (0, 0, 255)
                        label = f"H_OFF:{int(tid)} {conf:.2f}"
                    elif cls_id == ID_H_PERSON:
                        color = (255, 0, 255)
                        label = f"HP:{int(tid)} {conf:.2f}"
                    else:
                        color = (200, 200, 200)
                        label = f"H_{cls_id}:{int(tid)} {conf:.2f}"

                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(display_frame, label, (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if len(roi_poly) >= 3:
                    pts = np.array(roi_poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
                    
                if len(roi_lines) >= 2:
                    pts = np.array(roi_lines, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                            
                cv2.putText(display_frame, f"Speed: {speed_text} | Skip: {active_skip_frames} (Raw FPS: {video_fps:.1f})", (20, SCREEN_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

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
                        cv2.putText(canvas, f"[ API SNAPSHOT {idx+1} ]", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (40, 40, 40), -1)
                        cv2.rectangle(canvas, (0, y_offset), (320, y_offset+180), (100, 100, 100), 2)
                        cv2.putText(canvas, f"EMPTY SLOT {idx+1}", (100, y_offset + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                last_canvas = canvas.copy()
                cv2.imshow("CCTV Event Test Runner", canvas)
                
                key = cv2.waitKey(play_delay) & 0xFF
                if key == ord('q'):
                    force_quit_all = True
                    break
                elif key == ord('n'):
                    skip_next = True
                    break
                elif key == ord('r'):
                    replay_video = True
                    break
                elif key == ord('s'):
                    if active_skip_frames > 0:
                        active_skip_frames = 0
                        print("🔄 프레임 스킵을 껐습니다. (모든 프레임 분석)")
                    else:
                        active_skip_frames = base_skip_frames
                        print(f"🔄 프레임 스킵을 켰습니다. (Skip: {active_skip_frames})")
                elif key == ord('1'):
                    play_delay = 30
                    speed_text = "1x"
                elif key == ord('2'):
                    play_delay = 5
                    speed_text = "FAST"
                elif key == ord('3'):
                    play_delay = 100
                    speed_text = "SLOW"

            cap.release()
            
            if force_quit_all:
                break
            if skip_next:
                break
            if replay_video:
                continue 

        if force_quit_all:
            break
            
    cv2.destroyAllWindows()
    print("\n✅ 모든 영상의 테스트가 완료되었습니다.")

if __name__ == "__main__":
    main()