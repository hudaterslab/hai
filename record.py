#!/usr/bin/env python3
"""현장 녹화: 데모 카메라 원본 영상(1080p H.265 15fps)을 다시 압축하지 않고 그대로 파일로 저장하면서 카메라 화면을 띄웁니다.

실행: python3 record.py              -> 카메라 화면이 뜨고 바로 녹화 시작
      python3 record.py --minutes 30   -> 30분 뒤 자동으로 멈춤
      python3 record.py --no-view      -> 화면 없이 녹화만
멈춤: 카메라 화면에서 q 또는 ESC, 화면 창 닫기, 터미널에서 Ctrl+C

- 파일: ~/Desktop/fieldrec/rec_날짜_시각.mkv (단말기 바탕화면 fieldrec 폴더), 10분마다 새 파일
- 카메라 연결은 하나만 씀: ffmpeg 가 원본은 그대로 파일로 쓰고, 같은 영상을 작게(960x540) 풀어서 화면용으로 넘김
- 영상 트랙만 받습니다 (카메라의 메타데이터 트랙 때문에 연결이 끊기는 문제를 피함, 데모와 같은 방식)
- 카메라 연결이 끊기면 3초 뒤 자동으로 다시 연결해 새 파일로 이어서 녹화합니다
- 단말기 디스크 남은 공간이 1.5GB 아래로 내려가면 녹화를 멈춥니다 (단말기 공간 보호)
"""
import argparse
import datetime
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

URL = "rtsp://192.168.50.100:554/media/live/1/1"
OUT_DIR = Path.home() / "Desktop" / "fieldrec"
SEGMENT_SEC = 600      # 파일 하나 길이 (초)
MIN_FREE_GB = 1.5      # 남은 공간이 이보다 적으면 녹화 중지
RETRY_SEC = 3          # 연결이 끊기면 다시 연결하기까지 기다리는 시간
STATUS_SEC = 60        # 상태를 터미널에 찍는 간격
VIEW_W, VIEW_H = 960, 540   # 카메라 화면 크기
WINDOW = "REC - q: stop"

stop_reason = None


def log(msg, out_dir):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(out_dir / "record.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def recorded(out_dir, since):
    files = [f for f in out_dir.glob("rec_*.mkv") if f.stat().st_mtime >= since]
    return len(files), sum(f.stat().st_size for f in files) / 1e9


def ffmpeg_cmd(url, out_dir, segment_sec, view):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-rtsp_transport", "tcp", "-allowed_media_types", "video", "-stimeout", "3000000",
           "-fflags", "+genpts",   # 카메라가 첫 프레임에 시간값을 안 붙여 보내는 경우 채움
           "-i", url,
           # 1) 원본 그대로 파일로
           "-map", "0:v:0", "-c", "copy",
           "-f", "segment", "-segment_time", str(segment_sec), "-segment_format", "matroska",
           "-reset_timestamps", "1", "-strftime", "1",
           str(out_dir / "rec_%Y%m%d_%H%M%S.mkv")]
    if view:
        # 2) 화면용: 작게 풀어서 파이프로
        cmd += ["-map", "0:v:0", "-vf", f"scale={VIEW_W}:{VIEW_H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd


def stop_ffmpeg(proc):
    """ffmpeg 에 Ctrl+C 를 보내 지금 파일을 마무리하게 한 뒤 끝날 때까지 기다림"""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class FrameReader(threading.Thread):
    """ffmpeg 화면용 출력을 계속 읽어 가장 최근 프레임만 들고 있음 (안 읽으면 ffmpeg 가 멈춰 녹화도 멈추므로 늘 읽음)"""

    def __init__(self, stream):
        super().__init__(daemon=True)
        self.stream = stream
        self.frame = None
        self.t = 0.0

    def run(self):
        import numpy as np
        size = VIEW_W * VIEW_H * 3
        while True:
            buf = self.stream.read(size)
            if not buf or len(buf) < size:
                return
            self.frame = np.frombuffer(buf, np.uint8).reshape(VIEW_H, VIEW_W, 3)
            self.t = time.time()


def main():
    global stop_reason
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="데모 카메라 원본 녹화 (실행하면 바로 시작)")
    ap.add_argument("--minutes", type=float, default=0, help="이 시간(분)이 지나면 자동으로 멈춤 (0 = 계속)")
    ap.add_argument("--out", default=str(OUT_DIR), help="저장 폴더")
    ap.add_argument("--url", default=URL, help="카메라 RTSP 주소")
    ap.add_argument("--no-view", action="store_true", help="카메라 화면 없이 녹화만")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg 가 없습니다.")

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    cv2 = None
    if view:
        try:
            import cv2
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, VIEW_W, VIEW_H)
        except Exception as e:
            print(f"카메라 화면을 띄울 수 없어 녹화만 합니다: {e}", flush=True)
            view = False

    def on_signal(signum, frame):
        global stop_reason
        stop_reason = "사용자가 멈춤"
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    started = time.time()
    end_at = started + args.minutes * 60 if args.minutes > 0 else None
    log(f"녹화 시작 -> {out_dir} (파일 {SEGMENT_SEC // 60}분 단위, 남은 공간 {free_gb(out_dir):.1f}GB, "
        f"멈춤: {'화면에서 q / ' if view else ''}Ctrl+C)", out_dir)

    last_status = started
    blank = None
    while stop_reason is None:
        if free_gb(out_dir) < MIN_FREE_GB:
            stop_reason = f"남은 공간 부족 ({free_gb(out_dir):.1f}GB < {MIN_FREE_GB}GB)"
            break
        # ffmpeg 는 따로 프로세스 그룹으로 실행: Ctrl+C 는 이 프로그램만 받고, 여기서 ffmpeg 를 깔끔하게 멈춤
        proc = subprocess.Popen(ffmpeg_cmd(args.url, out_dir, SEGMENT_SEC, view), start_new_session=True,
                                stdout=subprocess.PIPE if view else subprocess.DEVNULL)
        reader = None
        if view:
            reader = FrameReader(proc.stdout)
            reader.start()
        while proc.poll() is None:
            now = time.time()
            if view:
                frame = reader.frame
                if frame is None or now - reader.t > 3:
                    if blank is None:
                        import numpy as np
                        blank = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
                    shown = (frame if frame is not None else blank).copy() // 3
                    cv2.putText(shown, "CONNECTING...", (30, VIEW_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                else:
                    shown = frame.copy()
                el = int(now - started)
                n, gb = recorded(out_dir, started)
                cv2.rectangle(shown, (0, 0), (VIEW_W, 34), (0, 0, 0), -1)
                if el % 2 == 0:
                    cv2.circle(shown, (18, 17), 8, (0, 0, 255), -1)
                cv2.putText(shown, f"REC {el // 3600:02d}:{el % 3600 // 60:02d}:{el % 60:02d}  files {n}  {gb:.2f}GB  "
                                   f"free {free_gb(out_dir):.1f}GB   q: stop", (34, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    stop_reason = "화면에서 q"
                elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "화면 창을 닫음"
            else:
                time.sleep(1)
            if stop_reason is None and end_at is not None and now >= end_at:
                stop_reason = f"정한 시간({args.minutes:g}분)이 지남"
            if stop_reason is None and free_gb(out_dir) < MIN_FREE_GB:
                stop_reason = f"남은 공간 부족 ({free_gb(out_dir):.1f}GB < {MIN_FREE_GB}GB)"
            if stop_reason is not None:
                stop_ffmpeg(proc)
                break
            if now - last_status >= STATUS_SEC:
                last_status = now
                n, gb = recorded(out_dir, started)
                mins = (now - started) / 60
                rate = gb / max(mins / 60, 1e-6)
                left_h = max(free_gb(out_dir) - MIN_FREE_GB, 0) / rate if rate > 0 else float("inf")
                log(f"녹화 중 {mins:.0f}분 | 파일 {n}개 {gb:.2f}GB | 남은 공간 {free_gb(out_dir):.1f}GB"
                    f" (이 속도면 약 {left_h:.1f}시간 더 가능)", out_dir)
        if stop_reason is None:
            log(f"카메라 연결이 끊김 (ffmpeg 종료 코드 {proc.returncode}) -> {RETRY_SEC}초 뒤 다시 연결", out_dir)
            time.sleep(RETRY_SEC)

    if view:
        cv2.destroyAllWindows()
    n, gb = recorded(out_dir, started)
    log(f"녹화 끝 ({stop_reason}) | {(time.time() - started) / 60:.1f}분, 파일 {n}개 {gb:.2f}GB | 남은 공간 {free_gb(out_dir):.1f}GB", out_dir)


if __name__ == "__main__":
    main()
