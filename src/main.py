import os
import sys
import time
import gc
import cv2
import numpy as np
import psutil
import threading
import queue
import csv
import concurrent.futures
import logging
import traceback
import math

from common import (SYS_CFG, CAMERA_LIST_FILE, CONFIG_COMMON_FILE, CONFIG_CAMERAS_FILE, 
                    ConfigManager, create_mosaic_image, extract_ip, normalize_roi_points, 
                    BATCH_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, setup_logging, load_rtsp_list_from_csv)
from event import EVENT_REGISTRY

from ai_core import VisionModelAsync, VisionModelSync, async_result_queue
from camera import Camera

logger = logging.getLogger("VMS_SYSTEM")

def check_rtsp_stream(url):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
    ret = cap.isOpened()
    cap.release()
    return url if ret else None

def auto_discover_cameras():
    logger.info("📡 네트워크 카메라 자동 스캔 중... (192.168.1.170:9001~9040)")
    targets = [f"rtsp://192.168.1.170:{port}/S.mp4" for port in range(9001, 9041)]
    valid_urls = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for future in concurrent.futures.as_completed([executor.submit(check_rtsp_stream, u) for u in targets]):
            if future.result():
                valid_urls.append(future.result())
                
    if valid_urls:
        with open(CAMERA_LIST_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows([["rtsp_url"]] + [[u] for u in valid_urls])
        logger.info(f"✅ 총 {len(valid_urls)}대의 카메라가 발견되어 cameras.csv를 생성했습니다.")
        
    return valid_urls

def capture_snapshot_clean(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
        
    best_frame = None
    for _ in range(20):
        ret, frame = cap.read()
        if ret and frame is not None:
            best_frame = frame
            
    cap.release()
    return best_frame

def get_roi_points_scaled(frame, title, mode="poly"):
    pts = []
    orig_h, orig_w = frame.shape[:2]
    disp_w = 960
    scale = disp_w / orig_w
    disp_h = int(orig_h * scale)
    disp_frame = cv2.resize(frame, (disp_w, disp_h))
    
    wname = "ROI Setup Window"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, disp_w, disp_h)
    
    def mouse_cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            if mode == "line" and len(pts) >= 2:
                return
            pts.append([int(x / scale), int(y / scale)])
            
    cv2.setMouseCallback(wname, mouse_cb)
    logger.info(f"[{title}] 그리기 모드. 점을 찍고 Enter(완료) 또는 ESC(취소). Line 모드는 2점.")
    
    while True:
        temp = disp_frame.copy()
        dp = [[int(p[0] * scale), int(p[1] * scale)] for p in pts]
        
        cv2.putText(temp, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if mode == "line":
            if len(dp) == 1:
                cv2.circle(temp, tuple(dp[0]), 5, (0, 0, 255), -1)
            elif len(dp) == 2:
                cv2.line(temp, tuple(dp[0]), tuple(dp[1]), (0, 0, 255), 2)
        else:
            if len(dp) > 0:
                cv2.polylines(temp, [np.array(dp, np.int32)], True, (0, 255, 0), 2)
                
        cv2.imshow(wname, temp)
        k = cv2.waitKey(1)
        
        if k == 13:
            break 
        if k == 27:
            pts = []
            break 
        if mode == "line" and len(pts) == 2:
            cv2.waitKey(500)
            break
            
    cv2.destroyWindow(wname)
    return normalize_roi_points(pts, orig_w, orig_h)

def run_wizard_batch_mode(config_manager, rtsp_list):
    total = len(rtsp_list)
    if total == 0:
        return logger.warning("설정할 카메라가 없습니다.")
        
    available_events = list(EVENT_REGISTRY.values())
    menu_str = " ".join([f"{i+1}.{evt.menu_name}" for i, evt in enumerate(available_events)])
    
    for i in range(0, total, BATCH_SIZE):
        batch_urls = rtsp_list[i : i + BATCH_SIZE]
        frames = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            frames = list(executor.map(capture_snapshot_clean, batch_urls))
            
        display_frames = []
        for idx, frm in enumerate(frames):
            if frm is None:
                blk = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blk, "Conn Fail", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                display_frames.append(blk)
            else:
                display_frames.append(frm)
                
        mosaic = create_mosaic_image(display_frames)
        cols = max(1, math.ceil(math.sqrt(len(display_frames))))
        cw = SCREEN_WIDTH // cols
        ch = SCREEN_HEIGHT // max(1, math.ceil(len(display_frames) / cols))
        
        for idx in range(len(display_frames)):
            r, c = divmod(idx, cols)
            cx, cy = c * cw, r * ch
            cv2.rectangle(mosaic, (cx, cy), (cx + 50, cy + 50), (255, 255, 255), -1)
            cv2.putText(mosaic, str(idx + 1), (cx + 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
        cv2.namedWindow("Select Cameras", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Cameras", 1280, 720)
        cv2.imshow("Select Cameras", mosaic)
        cv2.waitKey(1)
        
        sel = input(">> 선택 (예: 1,3,5): ").strip()
        selected_indices = []
        if sel:
            for n in [int(s.strip()) for s in sel.split(',') if s.strip().isdigit()]:
                if 1 <= n <= len(batch_urls):
                    selected_indices.append(i + (n - 1))
        cv2.destroyWindow("Select Cameras")
        
        for idx in selected_indices:
            url = rtsp_list[idx].strip()
            ip = extract_ip(url)
            frame = capture_snapshot_clean(url)
            
            if frame is None:
                continue
                
            height, width = frame.shape[:2]
            ratio = 960 / width
            preview = cv2.resize(frame, (960, int(height * ratio)))
            
            win_name = "Camera Check"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 960, int(height * ratio))
            cv2.imshow(win_name, preview)
            cv2.moveWindow(win_name, 100, 100)
            cv2.waitKey(1)
            
            print(f"[{ip}]")
            print(menu_str)
            sel = input(">> 이벤트 선택 (예: 1,4,5): ")
            cv2.destroyWindow(win_name)
            
            events = []
            needs_poly = False
            needs_line = False
            
            for evt_idx, evt_class in enumerate(available_events):
                if str(evt_idx + 1) in sel:
                    events.append(evt_class.event_name)
                    if evt_class.roi_type == "polygon": needs_poly = True
                    if evt_class.roi_type == "line": needs_line = True
            
            roi_p = []
            roi_l = []
            
            if needs_poly:
                roi_p = get_roi_points_scaled(frame, "Polygon")
                
            if needs_line:
                while True:
                    l = get_roi_points_scaled(frame, "Line", mode="line")
                    if len(l) == 2:
                        roi_l.extend(l)
                    if input("    라인 추가? (y/n): ") != 'y':
                        break
            
            config_manager.set_config(ip, {
                "url": url, 
                "roi_poly_norm": roi_p, 
                "roi_lines_norm": roi_l, 
                "events": events
            })

def prompt_runtime_options():
    sensitivity = 5
    try:
        val = input(">> 움직임 감지 민감도 설정 (1-10, 엔터시 기본값 5): ")
        if val.strip():
            sensitivity = max(1, min(10, int(val)))
    except Exception:
        pass
        
    val_disp = input(">> 모니터링 화면(GUI)을 출력하시겠습니까? (y/N, 기본값 y): ").strip().lower()
    use_display = False if val_disp == 'n' else True
    
    use_drawing = True
    if use_display:
        val_draw = input(">> 화면에 박스 및 텍스트(시각화)를 그리시겠습니까? (y/N, 기본값 y): ").strip().lower()
        use_drawing = False if val_draw == 'n' else True
    else:
        use_drawing = False
        
    return sensitivity, use_display, use_drawing

def prepare_config_manager(rtsp_list):
    config_manager = ConfigManager(CONFIG_COMMON_FILE, CONFIG_CAMERAS_FILE)
    
    added_new = False
    for url in rtsp_list:
        ip = extract_ip(url)
        if ip not in config_manager.camera_configs:
            config_manager.camera_configs[ip] = {
                "url": url,
                "events": [],
                "roi_poly_norm": [],
                "roi_lines_norm": []
            }
            added_new = True

    for idx, ip in enumerate(config_manager.camera_configs.keys(), start=1):
        config_manager.camera_configs[ip]["cctv_id"] = idx
        
    if added_new:
        config_manager.save()
        config_manager.config = config_manager.build_runtime_config()
        logger.info("✅ 새로운 RTSP 주소가 감지되어 자동으로 cameras.json에 등록 및 ID 번호 부여가 완료되었습니다.")

    val_setup = input(">> 특정 카메라의 이벤트/ROI 설정 마법사를 실행하시겠습니까? (y/N, 기본값 N): ").strip().lower()
    if val_setup == 'y':
        run_wizard_batch_mode(config_manager, rtsp_list)
        for idx, ip in enumerate(config_manager.camera_configs.keys(), start=1):
            config_manager.camera_configs[ip]["cctv_id"] = idx
        config_manager.save()
        config_manager.config = config_manager.build_runtime_config()
            
    return config_manager

def main():
    setup_logging(SYS_CFG)
    logger.info("="*60)
    logger.info("🚀 [VMS 시스템] 모듈형 Async 프로덕션 부팅")
    logger.info("="*60)
    
    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    if not rtsp_list:
        rtsp_list = auto_discover_cameras()
        if not rtsp_list:
            return logger.error("네트워크에 카메라가 없습니다.")

    cams = [] 
    
    try:
        sensitivity, use_display, use_drawing = prompt_runtime_options()
        config_manager = prepare_config_manager(rtsp_list)

        engines_h = [VisionModelAsync(SYS_CFG.get("models", {}).get("HELMET", "models/helmet_3cls_v8.dxnn")) for _ in range(3)]
        engines_g = [VisionModelAsync(SYS_CFG.get("models", {}).get("GENERAL", "models/YOLOV8M-1.dxnn")) for _ in range(3)]
        face_engine = VisionModelSync(SYS_CFG.get("models", {}).get("FACE", "models/yolov8m-face.dxnn")) 
        
        npu_semaphore = threading.Semaphore(18) 

        for i, rtsp in enumerate(rtsp_list):
            ip = extract_ip(rtsp)
            conf = config_manager.get_config(ip)
            if conf and conf.get('events'):
                cams.append(Camera(ip, conf, face_engine, i % 3, len(cams) + 1, sensitivity))
        
        if not cams:
            return logger.warning("이벤트가 설정되어 활성화된 카메라가 없습니다.")

        target_fps = SYS_CFG.get("REC_FPS", 30)
        dynamic_delay = 1.0 / target_fps
        loop_count = 0 
        
        # 💡 [핵심] Queue 병목 방지를 위한 상태 딕셔너리 (Lag 0초 보장)
        pending_frames = {i: {'h': False, 'g': False} for i in range(len(cams))}
        latest_applied_fid = {i: {'h': -1, 'g': -1} for i in range(len(cams))}
        
        logger.info("모니터링 시작 (종료: Ctrl+C 또는 'q')")
        
        while True:
            start_time = time.time()
            cpu_usage = psutil.cpu_percent(interval=None)
            
            if cpu_usage > 85:
                target_fps = max(5, target_fps - 2)
            elif cpu_usage < 60:
                target_fps = min(SYS_CFG.get("REC_FPS", 30), target_fps + 1)
                
            dynamic_delay = 1.0 / target_fps

            if loop_count % (target_fps * 10) == 0:
                logger.info(f"[Heartbeat] 카메라 {sum(1 for c in cams if c.reader.connected)}/{len(cams)} | CPU {cpu_usage}%")
                
            loop_count += 1
            if loop_count % 300 == 0:
                gc.collect()
            
            for i, c in enumerate(cams):
                fr, fid, connected = c.reader.read()
                if fr is not None and connected and fid > c.last_submit_fid:
                    if fid % SYS_CFG.get("SKIP_FRAMES", 4) == 0:
                        engine_idx = i % 3
                        
                        run_helmet = any("helmet" in h.required_models for h in c.handlers)
                        run_general = any("general" in h.required_models for h in c.handlers)
                        
                        # 💡 [핵심] 이전 결과가 리턴되지 않았다면 쿨하게 스킵 (지연시간 완전 차단)
                        if run_helmet and not pending_frames[i]['h'] and npu_semaphore.acquire(blocking=False):
                            pending_frames[i]['h'] = True
                            engines_h[engine_idx].submit_async(fr, i, "h", fid, npu_semaphore)
                            
                        if run_general and not pending_frames[i]['g'] and npu_semaphore.acquire(blocking=False):
                            pending_frames[i]['g'] = True
                            engines_g[engine_idx].submit_async(fr, i, "g", fid, npu_semaphore)
                            
                    c.last_submit_fid = fid

            while not async_result_queue.empty():
                try:
                    c_id, model_type, res_fid, boxes = async_result_queue.get_nowait()
                    # 결과 도착 시 락 해제
                    pending_frames[c_id][model_type] = False
                    
                    if res_fid > latest_applied_fid[c_id][model_type]:
                        latest_applied_fid[c_id][model_type] = res_fid
                        cams[c_id].latest_npu[model_type] = boxes
                except queue.Empty:
                    break

            final_imgs = []
            for idx, c in enumerate(cams):
                frame, fid, connected = c.reader.read()
                if frame is None or not connected: 
                    if use_display and use_drawing:
                        final_imgs.append(c.draw(None, [], [], {}, connected=False))
                else:
                    c.recorder.update(frame, fid)
                    t_h, t_g, alarms = c.run_logic(frame, fid)
                    
                    if use_display:
                        if use_drawing:
                            final_imgs.append(cv2.resize(c.draw(frame, t_h, t_g, alarms, connected=True), (640, 360)))
                        else:
                            final_imgs.append(cv2.resize(frame, (640, 360)))

            if use_display and final_imgs:
                mosaic = create_mosaic_image(final_imgs)
                if mosaic is not None:
                    cv2.imshow("VMS Monitor", mosaic)
                if cv2.waitKey(1) == ord('q'):
                    break

            sleep_time = dynamic_delay - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("모니터링 중단 (사용자 요청).")
    except Exception as e:
        logger.error(f"예외 발생: {e}")
        traceback.print_exc()
    finally:
        logger.info("시스템 자원을 정리하고 안전하게 종료합니다...")
        for c in cams:
            c.stop()
        cv2.destroyAllWindows()
        # 💡 [핵심] 스레드와 PyTorch 텐서가 엉키면서 발생하는 134 에러를 방지하는 명시적 강제 종료
        os._exit(0)

if __name__ == "__main__":
    main()