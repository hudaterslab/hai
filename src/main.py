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
import shutil

from common import (SYS_CFG, CAMERA_LIST_FILE, CONFIG_COMMON_FILE, CONFIG_CAMERAS_FILE, 
                    ConfigManager, create_mosaic_image, extract_ip, normalize_roi_points, 
                    BATCH_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, setup_logging, load_rtsp_list_from_csv,
                    save_json_file)
from event import EVENT_REGISTRY
from ai_core import VisionModelSync
from camera import Camera
from camera import FrameReader

logger = logging.getLogger("VMS_SYSTEM")

def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=None)
    cpu_temp = "N/A"
    chip_temp = "N/A"
    
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            for name in ['cpu_thermal', 'cpu-thermal', 'coretemp', 'k10temp', 'soc_therm']:
                if name in temps and temps[name]:
                    cpu_temp = f"{temps[name][0].current:.1f}°C"
                    break
    except Exception: pass
    
    if cpu_temp == "N/A" and os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                cpu_temp = f"{int(f.read().strip()) / 1000.0:.1f}°C"
        except Exception: pass

    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], stderr=subprocess.DEVNULL, text=True)
            chip_temp = f"{out.strip()}°C (NV-GPU)"
        elif shutil.which("vcgencmd"):
            out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, text=True)
            temp_val = out.replace("temp=", "").replace("'C", "").strip()
            chip_temp = f"{temp_val}°C (RPI-GPU)"
    except Exception: pass
    
    return cpu_usage, cpu_temp, chip_temp

def capture_snapshot_clean(url):
    temp_reader = FrameReader(url, ip="snapshot_test")
    start_time = time.time()
    valid_frame = None
    
    # 💡 I-frame 수신까지 충분히 기다림 (최대 20초)
    while time.time() - start_time < 20.0:
        frame, _, connected = temp_reader.read()
        if connected and frame is not None:
            mean_val = np.mean(frame)
            std_val = np.std(frame)
            
            # 💡 [핵심 검증] 픽셀 분산(std)이 15 이상인 진짜 화면일 때만 통과
            is_corrupted = (std_val < 15.0 and 100 < mean_val < 150) or (mean_val <= 1.0)
            
            if not is_corrupted:
                valid_frame = frame
                break
        time.sleep(0.5)
        
    temp_reader.stop()
    return valid_frame

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
            if mode == "line" and len(pts) >= 2: return
            pts.append([int(x / scale), int(y / scale)])
            
    cv2.setMouseCallback(wname, mouse_cb)
    print(f"[{title}] 그리기 모드. 점을 찍고 Enter(완료) 또는 ESC(취소). Line 모드는 2점.")
    
    while True:
        temp = disp_frame.copy()
        dp = [[int(p[0] * scale), int(p[1] * scale)] for p in pts]
        cv2.putText(temp, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if mode == "line":
            if len(dp) == 1: cv2.circle(temp, tuple(dp[0]), 5, (0, 0, 255), -1)
            elif len(dp) == 2: cv2.line(temp, tuple(dp[0]), tuple(dp[1]), (0, 0, 255), 2)
        else:
            if len(dp) > 0: cv2.polylines(temp, [np.array(dp, np.int32)], True, (0, 255, 0), 2)
                
        cv2.imshow(wname, temp)
        k = cv2.waitKey(1)
        if k == 13: break 
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
    if total == 0: return logger.warning("설정할 카메라가 없습니다.")
        
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
                if 1 <= n <= len(batch_urls): selected_indices.append(i + (n - 1))
        cv2.destroyWindow("Select Cameras")
        
        for idx in selected_indices:
            url = rtsp_list[idx].strip()
            ip = extract_ip(url)
            frame = capture_snapshot_clean(url)
            if frame is None: continue
                
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
            if needs_poly: roi_p = get_roi_points_scaled(frame, "Polygon")
            if needs_line:
                while True:
                    l = get_roi_points_scaled(frame, "Line", mode="line")
                    if len(l) == 2: roi_l.extend(l)
                    if input("    라인 추가? (y/n): ") != 'y': break
            
            config_manager.set_config(ip, {"url": url, "roi_poly_norm": roi_p, "roi_lines_norm": roi_l, "events": events})

def prompt_runtime_options():
    current_terminal_id = str(SYS_CFG.get("terminal_id", "99999"))
    if current_terminal_id == "99999":
        print("\n⚠️ [경고] 현재 단말기 ID(terminal_id)가 초기값(99999)으로 설정되어 있습니다.")
        val_tid = input(">> 배정받은 실제 단말기 ID를 입력해주세요 (예: 3): ").strip()
        if val_tid:
            SYS_CFG["terminal_id"] = val_tid
            save_json_file(CONFIG_COMMON_FILE, SYS_CFG)
            print(f"✅ 단말기 ID가 '{val_tid}'(으)로 업데이트 되었습니다.")
            
    sensitivity = 5
    try:
        val = input("\n>> 움직임 감지 민감도 설정 (1-10, 엔터시 기본값 5): ")
        if val.strip(): sensitivity = max(1, min(10, int(val)))
    except Exception: pass
        
    val_disp = input(">> 모니터링 화면(GUI)을 출력하시겠습니까? (y/N, 기본값 y): ").strip().lower()
    use_display = False if val_disp == 'n' else True
    
    use_drawing = True
    if use_display:
        val_draw = input(">> 화면에 박스 및 텍스트(시각화)를 그리시겠습니까? (y/N, 기본값 y): ").strip().lower()
        use_drawing = False if val_draw == 'n' else True
    else: use_drawing = False
        
    return sensitivity, use_display, use_drawing

def prepare_config_manager(rtsp_list):
    config_manager = ConfigManager(CONFIG_COMMON_FILE, CONFIG_CAMERAS_FILE)
    
    added_new = False
    for url in rtsp_list:
        ip = extract_ip(url)
        if ip not in config_manager.camera_configs:
            config_manager.camera_configs[ip] = {"url": url, "events": [], "roi_poly_norm": [], "roi_lines_norm": []}
            added_new = True

    for idx, ip in enumerate(config_manager.camera_configs.keys(), start=1):
        config_manager.camera_configs[ip]["cctv_id"] = idx
        
    if added_new:
        config_manager.save()
        config_manager.config = config_manager.build_runtime_config()

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
    logger.info("🚀 [VMS 시스템] 모듈형 Sync(통합모델) 프로덕션 부팅")
    logger.info("="*60)
    
    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    
    if not rtsp_list:
        logger.error(f"❌ 설정된 카메라가 없습니다. '{CAMERA_LIST_FILE}' 파일에 RTSP 주소가 입력되어 있는지 확인하십시오.")
        print(f"\n[오류] '{CAMERA_LIST_FILE}' 파일에 카메라 RTSP 주소가 없습니다.")
        print("프로그램을 종료합니다.")
        sys.exit(1)

    cams = [] 
    
    try:
        sensitivity, use_display, use_drawing = prompt_runtime_options()
        config_manager = prepare_config_manager(rtsp_list)

        engine_main = VisionModelSync(SYS_CFG.get("models", {}).get("MAIN", "models/hanjin_cctv.pt"))
        face_engine = VisionModelSync(SYS_CFG.get("models", {}).get("FACE", "models/yolov8m-face.pt")) 
        engine_helmet = VisionModelSync(SYS_CFG.get("models", {}).get("HELMET", "models/helmet_3cls_v8.dxnn"))
        
        for i, rtsp in enumerate(rtsp_list):
            ip = extract_ip(rtsp)
            conf = config_manager.get_config(ip)
            if conf and conf.get('events'):
                cams.append(Camera(ip, conf, face_engine, engine_helmet, len(cams) + 1, sensitivity))
        
        if not cams: return logger.warning("이벤트가 설정되어 활성화된 카메라가 없습니다.")

        target_fps = SYS_CFG.get("REC_FPS", 30)
        dynamic_delay = 1.0 / target_fps
        loop_count = 0 
        
        logger.info("모니터링 시작 (종료: Ctrl+C 또는 'q')")
        
        while True:
            start_time = time.time()
            cpu_usage, cpu_temp, chip_temp = get_system_metrics()
            
            if cpu_usage > 85:
                target_fps = max(5, target_fps - 2)
            elif cpu_usage < 60:
                target_fps = min(SYS_CFG.get("REC_FPS", 30), target_fps + 1)
                
            dynamic_delay = 1.0 / target_fps

            if loop_count % (target_fps * 10) == 0:
                active_cams = sum(1 for c in cams if c.reader.connected)
                logger.info(f"💓 [STATUS] Active Cams: {active_cams}/{len(cams)} | CPU: {cpu_usage}% ({cpu_temp}) | CHIP: {chip_temp}")
                
            loop_count += 1
            if loop_count % 300 == 0: gc.collect()
            
            final_imgs = []
            for idx, c in enumerate(cams):
                frame, fid, connected = c.reader.read()
                if frame is None or not connected: 
                    if use_display and use_drawing:
                        final_imgs.append(c.draw(None, [], [], {}, connected=False))
                    continue
                
                main_boxes = None
                helmet_boxes = None
                
                if fid > c.last_submit_fid:
                    if fid % SYS_CFG.get("SKIP_FRAMES", 1) == 0:
                        main_boxes = engine_main.infer(frame)
                        if "no_helmet" in c.events and c.helmet_detector:
                            helmet_boxes = c.helmet_detector.infer(frame)
                            
                    c.last_submit_fid = fid
                
                t_main, alarms = c.run_logic(frame, fid, main_boxes, helmet_boxes)
                
                if use_display:
                    if use_drawing: final_imgs.append(cv2.resize(c.draw(frame, t_main, alarms, connected=True), (640, 360)))
                    else: final_imgs.append(cv2.resize(frame, (640, 360)))

            if use_display and final_imgs:
                mosaic = create_mosaic_image(final_imgs)
                if mosaic is not None: cv2.imshow("VMS Monitor", mosaic)
                if cv2.waitKey(1) == ord('q'): break

            sleep_time = dynamic_delay - (time.time() - start_time)
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("모니터링 중단 (사용자 요청).")
    except Exception as e:
        logger.error(f"예외 발생: {e}")
        traceback.print_exc()
    finally:
        logger.info("시스템 자원을 정리하고 안전하게 종료합니다...")
        for c in cams: c.stop()
        cv2.destroyAllWindows()
        os._exit(0)

if __name__ == "__main__":
    main()