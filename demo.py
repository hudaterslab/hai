# demo.py
# ============================================================
# ROI alignment VNC 데모용 실행기 (GUI 모드 강제 활성화)
# ============================================================

import os
import sys
import time
import cv2
import numpy as np
import threading

# ❗ 가장 중요한 핵심: multi_event.py가 화면을 그리도록 강제로 --gui 플래그 주입
if "--gui" not in sys.argv:
    sys.argv.append("--gui")

import multi_event

from multi_event2 import CameraV2

# ============================================================
# 1. 데모 설정
# ============================================================

DEMO_MODE = True
DEMO_TARGET_CAM_IDS = {"6", "206_stream2"}

# 1. 300프레임에서 틀어짐 반영
DEMO_TILT_START_FID = 300 

# 2. 위로/오른쪽으로 10도 꺾임 효과 (회전 및 상하좌우 픽셀 이동)
DEMO_TILT_ANGLE_DEG = 10.0  # 화면 회전 각도 10도
DEMO_SHIFT_X = -50          # 카메라가 오른쪽을 보면 화면은 왼쪽(-)으로 밀림
DEMO_SHIFT_Y = 50           # 카메라가 위를 보면 화면은 아래(+)로 밀림
DEMO_SCALE = 1.0

DEMO_AUTO_EXIT_SEC = 40.0
DEMO_FORCE_FAST_ALIGNMENT = True

if DEMO_FORCE_FAST_ALIGNMENT:
    import multi_event2
    multi_event2.ALIGN_INTERVAL_SEC = 5.0


# ============================================================
# 2. VNC 화면 녹화용 데모 카메라 클래스
# ============================================================

class DemoCamera(CameraV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_tilt_matrix = None
        self.demo_last_frame = None
        self.demo_tilt_active = False

    def _is_demo_target(self):
        if not DEMO_MODE:
            return False
        if not DEMO_TARGET_CAM_IDS:
            return True
        return str(self.cam_id) in DEMO_TARGET_CAM_IDS

    def _build_demo_tilt_matrix(self, frame):
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        m_affine = cv2.getRotationMatrix2D(center, DEMO_TILT_ANGLE_DEG, DEMO_SCALE)
        m_affine[0, 2] += DEMO_SHIFT_X
        m_affine[1, 2] += DEMO_SHIFT_Y

        return np.vstack([m_affine, [0, 0, 1]]).astype(np.float32)

    def _apply_demo_tilt_if_needed(self, frame, fid):
        if frame is None or not self._is_demo_target():
            return frame

        if fid < DEMO_TILT_START_FID:
            self.demo_tilt_active = False
            return frame

        if self.demo_tilt_matrix is None:
            self.demo_tilt_matrix = self._build_demo_tilt_matrix(frame)
            print(f"[DEMO] ⚠️ CAM {self.cam_id}: 화면 틀어짐 발생! (보정 시작 대기)")

        self.demo_tilt_active = True

        warped = cv2.warpPerspective(
            frame,
            self.demo_tilt_matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return warped

    def run_logic(self, fr, fid, d_main_res, d_helmet_res):
        if fr is None:
            return [], [], {}

        demo_frame = self._apply_demo_tilt_if_needed(fr, fid)
        self.demo_last_frame = demo_frame

        return super().run_logic(demo_frame, fid, d_main_res, d_helmet_res)

    def draw(self, frame, tracks_main, tracks_helmet, alarms, connected):
        display_frame = self.demo_last_frame if self.demo_last_frame is not None else frame
        render_frame = super().draw(display_frame, tracks_main, tracks_helmet, alarms, connected)

        if render_frame is not None:
            label = "DEMO: Shifted Camera" if self.demo_tilt_active else "DEMO: Normal"
            color = (0, 0, 255) if self.demo_tilt_active else (0, 200, 0)

            cv2.putText(render_frame, label, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(render_frame, f"CAM {self.cam_id}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return render_frame


# ============================================================
# 3. 타이머 패치 
# ============================================================

demo_timer_started = False
ORIGINAL_CREATE_MOSAIC_IMAGE = multi_event.create_mosaic_image

def force_exit_after_timeout(timeout_sec):
    time.sleep(timeout_sec)
    print(f"\n[DEMO] ⏱️ {timeout_sec}초 경과! 데모 녹화가 완료되어 시스템을 종료합니다.")
    os._exit(0)

def create_mosaic_image_with_timer(images, screen_w=multi_event.SCREEN_WIDTH, screen_h=multi_event.SCREEN_HEIGHT):
    global demo_timer_started
    
    # 원본 multi_event.py가 화면을 제대로 그리도록 놔둡니다.
    mosaic = ORIGINAL_CREATE_MOSAIC_IMAGE(images, screen_w, screen_h)
    
    if mosaic is not None and not demo_timer_started:
        print(f"\n[DEMO] 🎥 첫 화면 출력 감지! 지금부터 {DEMO_AUTO_EXIT_SEC}초 카운트다운을 시작합니다.")
        threading.Thread(target=force_exit_after_timeout, args=(DEMO_AUTO_EXIT_SEC,), daemon=True).start()
        demo_timer_started = True
        
    return mosaic


# ============================================================
# 4. 실행
# ============================================================

if __name__ == "__main__":
    print(f"[DEMO] 🎬 VNC 화면 녹화용 데모 스크립트 실행 (--gui 강제 적용됨)")
    print(f"[DEMO] 터미널에 y/n 등을 입력하고 기다리시면 'Monitor' 창이 뜹니다.")

    multi_event.Camera = DemoCamera
    multi_event.create_mosaic_image = create_mosaic_image_with_timer
    
    try:
        multi_event.main()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
