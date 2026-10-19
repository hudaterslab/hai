#!/usr/bin/env python3
"""방망이 모델만 돌려서 카메라 화면에 박스를 보여 줍니다 (오탐·정탐 확인용, 판정 로직 없음).

실행: python3 test_bat.py                 -> bat_v5.dxnn, conf 0.3, 데모 카메라
      python3 test_bat.py --conf 0.5
      python3 test_bat.py --model bat_v2.dxnn
      python3 test_bat.py --src ~/Desktop/fieldrec/rec_20261016_131738.mkv   (녹화 영상)
끝: q 또는 ESC
- 데모가 켜져 있으면 실행하지 않습니다 (NPU·카메라 충돌). 처리는 데모와 같음: 720x404 -> 640 레터박스 -> RGB -> NPU
- 1클래스 모델(bat_v3/v4)은 0번, 80클래스 모델(bat_v1/v2)은 데모처럼 가장 높은 클래스가 34번(방망이)인 박스만 씀
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

URL = "rtsp://192.168.50.100:554/media/live/1/1"
W, H = 720, 404
WINDOW = "bat test - q: quit"


class Reader(threading.Thread):
    """ffmpeg 출력에서 가장 최근 프레임만 들고 있음"""

    def __init__(self, src):
        super().__init__(daemon=True)
        live = not os.path.exists(src)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        cmd += ["-rtsp_transport", "tcp", "-allowed_media_types", "video", "-fflags", "+genpts"] if live else ["-re"]
        cmd += ["-i", src, "-map", "0:v:0", "-vf", f"scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, start_new_session=True)
        self.frame, self.count, self.ended = None, 0, False

    def run(self):
        while True:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                self.ended = True
                return
            self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.count += 1


def detect(engine, frame, conf):
    """[(x1, y1, x2, y2, 점수)] 방망이만, NMS 0.45 (데모와 같은 레터박스·후처리)"""
    h, w = frame.shape[:2]
    s = min(640 / h, 640 / w)
    nw, nh = int(w * s), int(h * s)
    dw, dh = (640 - nw) // 2, (640 - nh) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(frame, (nw, nh))
    pred = np.array(engine.run([np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))])[0])
    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(0, 2, 1)
    pred = pred[0] if pred.ndim == 3 else pred
    cls_scores = pred[:, 4:]
    if cls_scores.shape[1] == 1:
        scores = cls_scores[:, 0]   # 1클래스 모델 (bat_v3/v4): 0번 = 방망이
    else:
        # 데모와 같이 가장 높은 클래스가 방망이인 박스만 (80클래스 모델 bat_v1/v2: 34번, 그 밖: 0번)
        bat = 34 if cls_scores.shape[1] == 80 else 0
        scores = np.where(cls_scores.argmax(axis=1) == bat, cls_scores[:, bat], 0.0)
    pred, scores = pred[scores >= conf], scores[scores >= conf]
    if len(pred) == 0:
        return []
    xywh = pred[:, :4].astype(np.float32).copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.astype(float).tolist(), float(conf), 0.45)).reshape(-1)
    out = []
    for i in keep:
        x, y, bw, bh = xywh[i]
        out.append((max(0.0, (x - dw) / s), max(0.0, (y - dh) / s), min(w, (x + bw - dw) / s), min(h, (y + bh - dh) / s), float(scores[i])))
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="방망이 모델만 돌려서 화면에 보여 줍니다")
    ap.add_argument("--model", default="bat_v5.dxnn", help="모델 파일 (indemo 폴더 기준 또는 경로)")
    ap.add_argument("--conf", type=float, default=0.3, help="이 점수 이상만 그림 (데모 BAT 기준 0.3)")
    ap.add_argument("--src", default=URL, help="카메라 RTSP 주소 또는 녹화 영상 파일")
    args = ap.parse_args()

    if subprocess.run(["pgrep", "-f", "multi_event_demo.py"], capture_output=True).returncode == 0:
        sys.exit("데모(multi_event_demo.py)가 켜져 있습니다. 데모를 끄고 실행하세요.")
    model = Path(os.path.expanduser(args.model))
    if not model.is_absolute() and not model.exists():
        model = here / model
    if not model.exists():
        sys.exit(f"모델 파일이 없습니다: {model}")
    from dx_engine import InferenceEngine, InferenceOption
    engine = InferenceEngine(str(model), InferenceOption())

    reader = Reader(os.path.expanduser(args.src))
    reader.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    last, fps, t0 = -1, 0.0, time.time()
    while not reader.ended:
        if reader.frame is None or reader.count == last:
            if (cv2.waitKey(5) & 0xFF) in (ord("q"), 27):
                break
            continue
        last = reader.count
        frame = reader.frame.copy()
        dets = detect(engine, frame, args.conf)
        now = time.time()
        fps, t0 = 0.9 * fps + 0.1 / max(1e-3, now - t0), now
        view = cv2.resize(frame, (1280, 720))
        sx, sy = 1280 / W, 720 / H
        for x1, y1, x2, y2, sc in dets:
            p1 = (int(x1 * sx), int(y1 * sy))
            cv2.rectangle(view, p1, (int(x2 * sx), int(y2 * sy)), (0, 0, 255), 2)
            cv2.putText(view, f"bat {sc:.2f}", (p1[0], max(20, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(view, f"{model.name}  conf {args.conf:.2f}  bats {len(dets)}  FPS {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    reader.proc.kill()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
