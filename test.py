#!/usr/bin/env python3
"""모자 모델 확인: 모자 모델 하나만 불러서 카메라 화면에 결과를 보여 줍니다.
기본은 hat_v1.dxnn (모자만 학습한 1클래스: 0 모자). 클래스 수로 알아서 구분:
  1클래스 hat_v1 (0 모자) / 2클래스 headhat = 지금 helmet_v3.dxnn (0 머리 / 1 모자) / 예전 3클래스 (0 안전모 / 1 머리 / 2 모자)

실행: python3 test.py                     -> hat_v1.dxnn, 데모 카메라 실시간
      python3 test.py --model helmet_v3.dxnn   -> 데모 HELMET2 에 들어 있는 headhat 모델
      python3 test.py --src ~/Desktop/fieldrec/rec_20261016_131738.mkv   -> 녹화 영상 (비니·버킷햇 장면 있음)
      python3 test.py --conf 0.4          -> 굵은 박스 기준 (기본 0.55, 데모 HELMET2 는 지금 0.8)
키: q 또는 ESC 끝, + / - 기준 0.05씩 올림/내림, 스페이스 멈춤, s 화면 저장 (/tmp)

- 데모(multi_event_demo.py)가 켜져 있으면 실행하지 않습니다 (NPU·카메라를 같이 쓰면 안 됨). --force 로 무시
- 데모와 같은 처리: 영상을 720x404 로 줄임 -> 640 레터박스(회색 114, 가운데) -> RGB -> NPU -> 클래스별 NMS
- 박스 색: 안전모 초록 / 머리 빨강 / 모자 보라. 기준(--conf) 이상은 굵게, --show(기본 0.25) ~ 기준은 가늘게
- 라벨에 그 박스의 모자 점수도 같이 표시: 예) helmet 0.85 (hat 0.01) -> 모자가 hat 에서 얼마나 먼지 보임
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
W, H = 720, 404                 # 데모와 같은 디코딩 크기
VIEW_W, VIEW_H = 1280, 720      # 화면 크기
NAMES_BY_NC = {3: ["helmet", "head", "hat"],   # 예전 helmet_v3: 0 안전모 / 1 머리 / 2 모자
               2: ["head", "hat"],             # headhat 모델 (지금 helmet_v3.dxnn): 0 머리 / 1 모자
               1: ["hat"]}                     # hat_v1.dxnn (모자만 학습): 0 모자
COLOR_OF = {"helmet": (0, 200, 0), "head": (0, 0, 255), "hat": (255, 0, 255)}   # 안전모 초록 / 머리 빨강 / 모자 보라 (BGR)
WINDOW = "hat model test - q: quit"


def demo_running():
    r = subprocess.run(["pgrep", "-f", "multi_event_demo.py"], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.strip() and int(p) != os.getpid()]


def ffmpeg_cmd(src, live):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if live:
        # 카메라의 메타데이터 트랙 때문에 끊기는 문제를 피하려고 영상 트랙만 받음 (데모·record.py 와 같음)
        cmd += ["-rtsp_transport", "tcp", "-allowed_media_types", "video", "-stimeout", "3000000", "-fflags", "+genpts"]
    else:
        cmd += ["-re"]   # 녹화 영상은 실제 속도로
    cmd += ["-i", src, "-map", "0:v:0", "-vf", f"scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd


class LatestFrame(threading.Thread):
    """실시간 카메라: ffmpeg 출력을 계속 읽어 가장 최근 프레임만 들고 있음 (늦게 처리해도 화면이 밀리지 않게)"""

    def __init__(self, stream):
        super().__init__(daemon=True)
        self.stream = stream
        self.frame = None
        self.count = 0
        self.ended = False

    def run(self):
        size = W * H * 3
        while True:
            buf = self.stream.read(size)
            if not buf or len(buf) < size:
                self.ended = True
                return
            self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.count += 1


def letterbox(img, size=640):
    """데모 YoLoDeepX.letter_box 와 같음: 비율 유지 축소, 가운데, 회색 114"""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nw, nh = int(w * s), int(h * s)
    canvas = np.full((size, size, 3), 114, np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh))
    return canvas, s, dw, dh


def detect(engine, frame, show_thr, iou_thr=0.45):
    """[(x1, y1, x2, y2, 점수, 클래스, [클래스별 점수])] - 점수 >= show_thr, 클래스별 NMS (데모 postprocess 와 같음)"""
    inp, s, dw, dh = letterbox(frame)
    tensor = np.ascontiguousarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    out = engine.run([tensor])
    pred = np.array(out[0])
    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(0, 2, 1)
    if pred.ndim == 3:
        pred = pred[0]
    cls_scores = pred[:, 4:]
    scores = cls_scores.max(axis=1)
    cls = cls_scores.argmax(axis=1)
    keep = scores >= show_thr
    pred, scores, cls, cls_scores = pred[keep], scores[keep], cls[keep], cls_scores[keep]
    if len(pred) == 0:
        return []
    xywh = pred[:, :4].astype(np.float32).copy()
    xywh[:, 0] -= xywh[:, 2] / 2
    xywh[:, 1] -= xywh[:, 3] / 2
    shifted = xywh.copy()
    shifted[:, 0] += cls * 7680   # 클래스별 NMS
    shifted[:, 1] += cls * 7680
    idx = np.array(cv2.dnn.NMSBoxes(shifted.tolist(), scores.astype(float).tolist(), float(show_thr), iou_thr)).reshape(-1)
    fh, fw = frame.shape[:2]
    dets = []
    for i in idx:
        x, y, bw, bh = xywh[i]
        x1, y1 = max(0.0, (x - dw) / s), max(0.0, (y - dh) / s)
        x2, y2 = min(float(fw), (x + bw - dw) / s), min(float(fh), (y + bh - dh) / s)
        dets.append((x1, y1, x2, y2, float(scores[i]), int(cls[i]), [float(v) for v in cls_scores[i]]))
    return dets


def text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def name_of(c, nc):
    names = NAMES_BY_NC.get(nc, [])
    return names[c] if c < len(names) else str(c)


def label_of(c, sc, cs):
    names = NAMES_BY_NC.get(len(cs), [])
    name = name_of(c, len(cs))
    hat = cs[names.index("hat")] if "hat" in names else 0.0
    return f"{name} {sc:.2f}" if name == "hat" else f"{name} {sc:.2f} (hat {hat:.2f})"


def draw(frame, dets, conf, model_name, fps, paused):
    view = cv2.resize(frame, (VIEW_W, VIEW_H))
    sx, sy = VIEW_W / frame.shape[1], VIEW_H / frame.shape[0]
    counts = {"helmet": 0, "head": 0, "hat": 0}
    for x1, y1, x2, y2, sc, c, cs in sorted(dets, key=lambda d: d[4]):
        strong = sc >= conf
        name = name_of(c, len(cs))
        col = COLOR_OF.get(name, (200, 200, 200))
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(view, p1, p2, col, 3 if strong else 1)
        text(view, label_of(c, sc, cs), (p1[0], max(52, p1[1] - 6)), 0.55 if strong else 0.45, col, 2 if strong else 1)
        if strong and name in counts:
            counts[name] += 1
    cv2.rectangle(view, (0, 0), (VIEW_W, 34), (0, 0, 0), -1)
    bar = (f"{model_name}  conf {conf:.2f} (+/-)  FPS {fps:.1f}  |  helmet {counts['helmet']}  head {counts['head']}  hat {counts['hat']}"
           + ("   [PAUSED]" if paused else "") + "   q: quit")
    text(view, bar, (10, 23), 0.6)
    return view, counts


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="새 안전모 모델(헬멧/머리/모자)만 돌려서 화면에 보여 줍니다")
    ap.add_argument("--model", default=str(here / "hat_v1.dxnn"), help="모델 파일 (.dxnn)")
    ap.add_argument("--src", default=URL, help="카메라 RTSP 주소 또는 녹화 영상 파일")
    ap.add_argument("--conf", type=float, default=0.55, help="굵게 그리는 기준 (데모 HELMET2 는 지금 0.8)")
    ap.add_argument("--show", type=float, default=0.25, help="이 점수 이상은 가늘게라도 그림")
    ap.add_argument("--force", action="store_true", help="데모가 켜져 있어도 실행")
    ap.add_argument("--no-view", action="store_true", help="화면 없이 프레임마다 결과만 출력 (시험용)")
    ap.add_argument("--frames", type=int, default=0, help="이 프레임 수만 처리하고 끝 (0 = 계속)")
    args = ap.parse_args()

    if demo_running() and not args.force:
        sys.exit("데모(multi_event_demo.py)가 켜져 있습니다. NPU·카메라를 같이 쓰면 안 되니 데모를 끄고 실행하세요. (무시: --force)")
    model = Path(args.model).expanduser()
    if not model.exists():
        sys.exit(f"모델 파일이 없습니다: {model}")
    try:
        from dx_engine import InferenceEngine, InferenceOption
    except Exception as e:
        sys.exit(f"dx_engine 을 불러올 수 없습니다: {e}")
    engine = InferenceEngine(str(model), InferenceOption())
    print(f"모델 로드: {model.name}", flush=True)

    view_on = not args.no_view and bool(os.environ.get("DISPLAY"))
    if not args.no_view and not view_on:
        print("DISPLAY 가 없어 화면 없이 결과만 출력합니다", flush=True)
    if view_on:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, VIEW_W, VIEW_H)

    live = not os.path.exists(os.path.expanduser(args.src))
    src = args.src if live else os.path.expanduser(args.src)
    conf, paused, done = args.conf, False, 0
    fps, last_t = 0.0, time.time()
    frame, dets = None, []
    proc = subprocess.Popen(ffmpeg_cmd(src, live), stdout=subprocess.PIPE, start_new_session=True)
    reader = LatestFrame(proc.stdout) if live else None
    if reader is not None:
        reader.start()
    last_seen = -1
    print(f"{'카메라' if live else '영상'}: {src}", flush=True)
    try:
        while True:
            if not paused:
                if live:
                    if reader.ended:
                        print("카메라 연결이 끊겼습니다. 다시 연결합니다", flush=True)
                        proc.kill()
                        time.sleep(2)
                        proc = subprocess.Popen(ffmpeg_cmd(src, live), stdout=subprocess.PIPE, start_new_session=True)
                        reader = LatestFrame(proc.stdout)
                        reader.start()
                        last_seen = -1
                        continue
                    if reader.frame is None or reader.count == last_seen:
                        time.sleep(0.005)
                        if view_on and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                        continue
                    last_seen = reader.count
                    frame = reader.frame.copy()
                else:
                    buf = proc.stdout.read(W * H * 3)
                    if not buf or len(buf) < W * H * 3:
                        print("영상 끝", flush=True)
                        break
                    frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
                dets = detect(engine, frame, args.show)
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(1e-3, now - last_t) if fps else 1.0 / max(1e-3, now - last_t)
                last_t = now
                done += 1
                if not view_on:
                    strong = [label_of(c, sc, cs) for _, _, _, _, sc, c, cs in dets if sc >= conf]
                    weak = [label_of(c, sc, cs) for _, _, _, _, sc, c, cs in dets if sc < conf]
                    print(f"frame {done}: {strong} | weak {weak}", flush=True)
                if args.frames and done >= args.frames:
                    break
            if view_on and frame is not None:
                view, _ = draw(frame, dets, conf, model.name, fps, paused)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("+"), ord("=")):
                    conf = min(0.95, conf + 0.05)
                elif key in (ord("-"), ord("_")):
                    conf = max(0.05, conf - 0.05)
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("s"):
                    path = f"/tmp/hat_test_{time.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(path, view)
                    print(f"저장: {path}", flush=True)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            elif paused:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        if view_on:
            cv2.destroyAllWindows()
    print(f"끝 ({done} 프레임)", flush=True)


if __name__ == "__main__":
    main()
