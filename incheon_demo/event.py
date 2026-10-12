#!/usr/bin/env python3
"""두 이벤트 데모: 안전모/조끼 미착용 OR, 손에 든 방망이. 각각 연속 3초.

실행: python3 event.py    종료: Q/ESC    전체화면: F    소리: M
검사: python3 event.py --headless --seconds 30 --mute
multi_event.py의 DeepX 전처리/PPU 디코딩, 사람-머리 매칭과 빨간 안전모
오검출 억제를 참고했습니다. 기존의 여러 구간 누적 대신 연속 3초를 사용합니다.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RTSP_URL = "rtsp://192.168.50.100:554/media/live/1/1"
TRIGGER_SECONDS = 3.0
MAX_EVIDENCE_GAP = 0.8
PERSON_ID, HELMET_ID, BARE_HEAD_ID, VEST_ID, BAT_ID = 2, 0, 1, 0, 34
MODELS = {"person": ("hanjin_cctv_v2.dxnn", "ppu", .50, {PERSON_ID}),
          "helmet": ("helmet_v2.dxnn", "ppu", .55, {HELMET_ID, BARE_HEAD_ID}),
          "vest": ("best_vest1.dxnn", "ppu", .50, {VEST_ID}),
          "bat": ("bat.dxnn", "yolo", .40, {BAT_ID}),
          "pose": ("yolov8m_pose.dxnn", "pose", .45, {0})}
WINDOW = "Demo - PPE OR / BAT - Q: Quit  F: Fullscreen  M: Sound"
LOG = logging.getLogger("event")


def area(b):
    return max(0., b[2] - b[0]) * max(0., b[3] - b[1])


def intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(area(a) + area(b) - overlap, 1.)


def overlap_smaller(a, b):
    return intersection(a, b) / max(1., min(area(a), area(b)))


def head_matches(head, person):
    # multi_event.py HelmetDetector: 위 40%, 가로 15% 여유, 머리 면적 50% 포함.
    x1, y1, x2, y2 = person
    cx, cy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
    margin = (x2 - x1) * .15
    return (x1 - margin <= cx <= x2 + margin and y1 <= cy <= y1 + (y2 - y1) * .4
            and intersection(head, person) / max(area(head), 1.) >= .5)


class DeepXModel:
    def __init__(self, root, spec):
        from dx_engine import InferenceEngine, InferenceOption
        name, self.kind, self.threshold, self.classes = spec
        self.engine = InferenceEngine(str(root / name), InferenceOption())
        info = self.engine.get_input_tensors_info()[0]
        self.shape = tuple(info["shape"])
        if len(self.shape) != 4 or self.shape[0] != 1 or self.shape[-1] != 3:
            raise ValueError(f"지원하지 않는 모델 입력: {name}: {self.shape}")
        self.height, self.width = self.shape[1:3]
        LOG.info("모델 로드: %s (%s)", name, self.kind)

    def infer(self, frame):
        height, width = frame.shape[:2]
        ratio = min(self.width / width, self.height / height)
        nw, nh = int(width * ratio), int(height * ratio)
        dx, dy = (self.width - nw) // 2, (self.height - nh) // 2
        canvas = np.full((self.height, self.width, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh))
        tensor = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None])
        output = self.engine.run([tensor])[0]
        keypoints = None
        if self.kind == "ppu":
            raw = np.ascontiguousarray(output).view(np.uint8).reshape(-1, 32)
            boxes = np.ascontiguousarray(raw[:, :16]).view(np.float32).reshape(-1, 4)
            scores = np.ascontiguousarray(raw[:, 20:24]).view(np.float32).ravel()
            labels = np.ascontiguousarray(raw[:, 24:28]).view(np.uint32).ravel()
        else:
            pred = np.asarray(output)[0]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T
            boxes = pred[:, :4].copy()
            if self.kind == "pose":
                if pred.shape[1] != 56:
                    raise ValueError(f"포즈 출력은 56열이어야 합니다: {pred.shape}")
                scores, labels = pred[:, 4], np.zeros(len(pred), np.int32)
                keypoints = pred[:, 5:56].reshape(-1, 17, 3).copy()
            else:
                labels = pred[:, 4:].argmax(axis=1)
                scores = pred[:, 4:].max(axis=1)
        mask = ((scores >= self.threshold) & np.isfinite(scores)
                & np.isfinite(boxes).all(axis=1) & np.isin(labels, list(self.classes))
                & (boxes[:, 2] > 0) & (boxes[:, 3] > 0))
        boxes, scores, labels = boxes[mask].copy(), scores[mask], labels[mask]
        if keypoints is not None:
            keypoints = keypoints[mask]
        if len(boxes) == 0:
            return []
        boxes[:, :2] -= boxes[:, 2:] / 2
        shifted = boxes.copy()
        shifted[:, :2] += labels[:, None] * 7680
        keep = np.asarray(cv2.dnn.NMSBoxes(shifted.tolist(), scores.tolist(), self.threshold, .45)).reshape(-1)
        result = []
        for index in keep:
            x, y, w, h = boxes[index]
            box = np.array([(x - dx) / ratio, (y - dy) / ratio,
                            (x + w - dx) / ratio, (y + h - dy) / ratio])
            box[[0, 2]] = box[[0, 2]].clip(0, width)
            box[[1, 3]] = box[[1, 3]].clip(0, height)
            if area(box) < 1:
                continue
            item = {"box": box.tolist(), "score": float(scores[index]), "class": int(labels[index])}
            if keypoints is not None:
                kp = keypoints[index]
                kp[:, 0] = (kp[:, 0] - dx) / ratio
                kp[:, 1] = (kp[:, 1] - dy) / ratio
                if (kp[:, 2] < 0).any() or (kp[:, 2] > 1).any():
                    kp[:, 2] = 1 / (1 + np.exp(-np.clip(kp[:, 2], -50, 50)))
                item["keypoints"] = kp.tolist()
            result.append(item)
        return result

    def close(self):
        self.engine.dispose()


class Tracker:
    """기존 SimpleTracker의 IoU 추적을 일대일 전역 매칭으로 적용."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now):
        self.tracks = {tid: t for tid, t in self.tracks.items() if now - t[1] <= 1.0}
        pairs = sorted(((iou(old[0], d["box"]), tid, i)
                        for tid, old in self.tracks.items() for i, d in enumerate(detections)), reverse=True)
        used_ids, used_dets, assigned = set(), set(), {}
        for score, tid, index in pairs:
            if score >= .2 and tid not in used_ids and index not in used_dets:
                assigned[index] = tid
                used_ids.add(tid)
                used_dets.add(index)
        result = []
        for index, detection in enumerate(detections):
            tid = assigned.get(index)
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            self.tracks[tid] = (detection["box"], now)
            result.append(dict(detection, id=tid))
        return result


@dataclass
class Hold:
    start: float | None = None
    last: float | None = None
    seconds: float = 0.

    def update(self, active, now):
        if not active:
            self.start = self.last = None
            self.seconds = 0.
        else:
            if self.start is None or self.last is None or now - self.last > MAX_EVIDENCE_GAP or now < self.last:
                self.start = now
            self.last = now
            self.seconds = now - self.start
        return self.seconds >= TRIGGER_SECONDS - 1e-8


@dataclass
class PersonState:
    helmet: Hold = field(default_factory=Hold)
    vest: Hold = field(default_factory=Hold)
    bat: Hold = field(default_factory=Hold)
    bat_id: int | None = None
    ppe_latched: bool = False
    bat_latched: bool = False
    red_samples: deque = field(default_factory=lambda: deque(maxlen=5))


def confirmed_people(main, poses, heads):
    # 사람 모델이 놓친 사람은 포즈로 보완. 포즈/머리 없는 의자 오검출은 제외.
    result, used = [], set()
    for person in sorted(main, key=lambda p: p["score"], reverse=True):
        choices = [(iou(person["box"], pose["box"]), i) for i, pose in enumerate(poses) if i not in used]
        score, index = max(choices, default=(0., -1))
        if score >= .2:
            result.append(dict(person, keypoints=poses[index]["keypoints"]))
            used.add(index)
        elif any(head_matches(h["box"], person["box"]) for h in heads):
            result.append(dict(person))
    result.extend(dict(p) for i, p in enumerate(poses) if i not in used and not any(iou(p["box"], r["box"]) > .5 for r in result))
    return result


def torso_box(person, image_shape):
    x1, y1, x2, y2 = person["box"]
    height = y2 - y1
    if height < 80 or x2 - x1 < 25:
        return None
    kp = person.get("keypoints")
    if kp:
        shoulders = [kp[i] for i in (5, 6) if kp[i][2] >= .3]
        hips = [kp[i] for i in (11, 12) if kp[i][2] >= .3]
        points = shoulders + hips
        if shoulders and hips and len(points) >= 3:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            pad = max(5., (max(xs) - min(xs)) * .15)
            box = [min(xs) - pad, min(ys), max(xs) + pad, max(ys)]
            if area(box) >= 100 and box[1] > 1 and box[3] < image_shape[0] - 2:
                return box
        # 얼굴/어깨만 보일 때 사람 박스 중앙을 몸통으로 오인하지 않습니다.
        return None
    # 포즈가 아예 없을 때만, 충분히 긴 온전한 사람 박스로 몸통을 추정합니다.
    box = [x1, y1 + .2 * height, x2, y1 + .65 * height]
    margin = max(8., image_shape[0] * .01)
    if height / max(x2 - x1, 1.) >= 1.4 and y1 > margin and y2 < image_shape[0] - margin and x1 > margin and x2 < image_shape[1] - margin:
        return box
    return None


def vest_match(vest, person):
    torso = person["torso"]
    if torso is not None:
        return overlap_smaller(vest, torso)
    # 잘린 몸통에서는 '미착용'을 추측하지 않지만 실제 조끼 검출은 표시합니다.
    body = person["box"]
    cx, cy = (vest[0] + vest[2]) / 2, (vest[1] + vest[3]) / 2
    if body[0] <= cx <= body[2] and body[1] + .3 * (body[3] - body[1]) <= cy <= body[3]:
        return intersection(vest, body) / max(area(vest), 1.)
    return 0.


def hands_for(person):
    kp = person.get("keypoints")
    if not kp:
        return []
    radius = max(8., min(35., (person["box"][3] - person["box"][1]) * .045))
    hands = []
    for elbow, wrist in ((7, 9), (8, 10)):
        if kp[wrist][2] < .35 or not np.isfinite(kp[wrist]).all():
            continue
        x, y = kp[wrist][:2]
        if kp[elbow][2] >= .3:
            # 손목에서 팔꿈치 반대 방향으로 조금 확장해 손바닥 영역을 근사.
            vx, vy = x - kp[elbow][0], y - kp[elbow][1]
            length = max(1., float(np.hypot(vx, vy)))
            advance = min(radius * .6, length * .18)
            x, y = x + vx / length * advance, y + vy / length * advance
        hands.append({"wrist": wrist, "box": [x - radius, y - radius, x + radius, y + radius]})
    return hands


class EventLogic:
    def __init__(self):
        self.people_tracker, self.bat_tracker = Tracker(), Tracker()
        self.states = {}

    def reset(self):
        self.states.clear()
        self.people_tracker.tracks.clear()
        self.bat_tracker.tracks.clear()

    def update(self, frame, detections, now):
        people = self.people_tracker.update(confirmed_people(detections["person"], detections["pose"], detections["helmet"]), now)
        bats = self.bat_tracker.update(detections["bat"], now)
        self.states = {p["id"]: self.states.get(p["id"], PersonState()) for p in people}
        for p in people:
            p["torso"] = torso_box(p, frame.shape)
            p["hands"] = hands_for(p)
        # 하나의 머리/조끼 검출을 여러 사람의 착용 증거로 공유하지 않습니다.
        head_map, vest_map = {}, {}
        for head in detections["helmet"]:
            choices = [(intersection(head["box"], p["box"]) / max(area(head["box"]), 1.), p["id"])
                       for p in people if head_matches(head["box"], p["box"])]
            if choices:
                tid = max(choices)[1]
                if tid not in head_map or head["score"] > head_map[tid]["score"]:
                    head_map[tid] = head
        for vest in detections["vest"]:
            choices = [(vest_match(vest["box"], p), p["id"]) for p in people]
            score, tid = max(choices, default=(0., -1))
            if score >= .4:
                vest_map[tid] = vest
        contacts = []
        for bat in bats:
            for p in people:
                for hand in p["hands"]:
                    if intersection(hand["box"], bat["box"]) > 0:
                        h, b = hand["box"], bat["box"]
                        cx, cy = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2
                        distance = np.hypot(cx - np.clip(cx, b[0], b[2]), cy - np.clip(cy, b[1], b[3]))
                        contacts.append((float(distance), p["id"], bat["id"], hand))
        held, used_bats = {}, set()
        for _, tid, bid, hand in sorted(contacts, key=lambda x: x[:3]):
            if tid not in held and bid not in used_bats:
                held[tid] = (bid, hand)
                used_bats.add(bid)
        events = []
        for p in people:
            tid, state = p["id"], self.states[p["id"]]
            head = head_map.get(tid)
            no_helmet = head is not None and head["class"] == BARE_HEAD_ID
            helmet_label = "MISSING" if no_helmet else ("OK" if head else "UNKNOWN")
            if no_helmet:
                x1, y1, x2, y2 = map(int, head["box"])
                crop = frame[max(0, y1):max(0, y1 + (y2 - y1) // 2), max(0, x1):max(0, x2)]
                if crop.size:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    state.red_samples.append((float(hsv[:, :, 0].mean()), float(hsv[:, :, 1].mean()), float(crop[:, :, 2].mean())))
                if state.red_samples:
                    hue, saturation, red = np.median(state.red_samples, axis=0)
                    if 10 <= hue <= 40 and saturation >= 60 and red >= 100:
                        no_helmet, helmet_label = False, "OK (color veto)"
            else:
                state.red_samples.clear()
            no_vest = p["torso"] is not None and tid not in vest_map
            vest_label = "MISSING" if no_vest else ("OK" if tid in vest_map else "UNKNOWN")
            helmet_alarm = state.helmet.update(no_helmet, now)
            vest_alarm = state.vest.update(no_vest, now)
            reasons = (["no_helmet"] if helmet_alarm else []) + (["no_vest"] if vest_alarm else [])
            if not no_helmet and not no_vest:
                state.ppe_latched = False
            if reasons and not state.ppe_latched:
                events.append({"type": "safety", "person_id": tid, "reasons": reasons, "box": p["box"],
                               "helmet_seconds": state.helmet.seconds, "vest_seconds": state.vest.seconds})
                state.ppe_latched = True
            bid = held.get(tid, (None, None))[0]
            if state.bat_id != bid:
                state.bat.update(False, now)
                state.bat_latched = False
            state.bat_id = bid
            bat_alarm = state.bat.update(bid is not None, now)
            if bid is None:
                state.bat_latched = False
            if bat_alarm and not state.bat_latched:
                events.append({"type": "weapon", "person_id": tid, "reasons": ["bat_in_hand"],
                               "box": p["box"], "bat_id": bid, "held_seconds": state.bat.seconds,
                               "bat_box": next(b["box"] for b in bats if b["id"] == bid)})
                state.bat_latched = True
            p.update(helmet_status=helmet_label, vest_status=vest_label,
                     helmet_seconds=state.helmet.seconds, vest_seconds=state.vest.seconds,
                     bat_seconds=state.bat.seconds, bat_id=bid, safety_alarm=bool(reasons), weapon_alarm=bat_alarm)
        return {"people": people, "bats": bats, "held_bats": list(used_bats), "events": events,
                "heads": detections["helmet"], "vests": detections["vest"]}


class CameraReader(threading.Thread):
    def __init__(self, url, stop):
        super().__init__(daemon=True)
        self.url, self.stop = url, stop
        self.lock = threading.Lock()
        self.frame, self.timestamp, self.sequence, self.epoch = None, 0., 0, 0
        self.connected = False

    def snapshot(self):
        with self.lock:
            return self.frame, self.timestamp, self.sequence, self.epoch, self.connected

    def run(self):
        while not self.stop.is_set():
            capture = cv2.VideoCapture()
            try:
                opened = capture.open(self.url, cv2.CAP_FFMPEG,
                                      [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                if opened:
                    with self.lock:
                        self.epoch += 1
                    while not self.stop.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            break
                        with self.lock:
                            self.frame, self.timestamp = frame, time.monotonic()
                            self.sequence += 1
                            self.connected = True
            except cv2.error:
                LOG.exception("카메라 수신 오류")
            finally:
                capture.release()
                with self.lock:
                    self.connected = False
            if not self.stop.is_set():
                LOG.warning("카메라 연결이 끊겼습니다. 2초 후 재접속합니다.")
                self.stop.wait(2.)


class AlarmWriter(threading.Thread):
    def __init__(self, output, muted):
        super().__init__(daemon=True)
        self.output, self.muted = output, muted
        self.jobs = queue.Queue(maxsize=16)
        self.stopping = threading.Event()
        self.error = None

    def submit(self, event, image):
        event = dict(event, timestamp=datetime.now().astimezone().isoformat(), trigger_seconds=TRIGGER_SECONDS)
        LOG.warning("EVENT %s", json.dumps(event, ensure_ascii=False))
        try:
            self.jobs.put_nowait((event, image.copy()))
        except queue.Full:
            self.error = "알림 저장 대기열이 가득 찼습니다."
            LOG.error(self.error)

    def run(self):
        with tempfile.TemporaryDirectory(prefix="demo-alarm-") as folder:
            sound = Path(folder) / "alarm.wav"
            rate = 22050
            t = np.arange(int(rate * .18)) / rate
            tone = (np.sin(2 * np.pi * 880 * t) * 10000).astype("<i2")
            samples = np.concatenate([tone, np.zeros(int(rate * .1), dtype="<i2"), tone])
            with wave.open(str(sound), "wb") as stream:
                stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                stream.writeframes(samples.tobytes())
            while not self.stopping.is_set() or not self.jobs.empty():
                try:
                    event, image = self.jobs.get(timeout=.2)
                except queue.Empty:
                    continue
                try:
                    self.output.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = self.output / f"{stamp}_{event['type']}_person{event['person_id']}.jpg"
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError(f"이미지 저장 실패: {image_path}")
                    event["snapshot"] = str(image_path)
                    with (self.output / "events.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                except Exception as error:
                    self.error = str(error)
                    LOG.exception("이벤트 저장 오류")
                if not self.muted:
                    player = shutil.which("paplay") or shutil.which("aplay")
                    if player:
                        try:
                            result = subprocess.run([player, str(sound)], timeout=3, capture_output=True)
                            if result.returncode:
                                LOG.warning("스피커 재생 실패: 화면 경고와 이벤트 기록은 유지됩니다.")
                        except (OSError, subprocess.TimeoutExpired):
                            LOG.warning("스피커 재생을 완료하지 못했습니다.")
                    else:
                        print("\a", end="", flush=True)
                self.jobs.task_done()


def draw(frame, result, fps, totals, muted=False):
    h, w = frame.shape[:2]
    scale = min(1280 / w, 720 / h)
    view = cv2.resize(frame, (int(w * scale), int(h * scale)))
    def box(b, color, text=""):
        x1, y1, x2, y2 = [int(v * scale) for v in b]
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(view, text, (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
    for head in result["heads"]:
        box(head["box"], (0, 210, 250), "helmet" if head["class"] == HELMET_ID else "head")
    for vest in result["vests"]:
        box(vest["box"], (255, 180, 0), "vest")
    for bat in result["bats"]:
        box(bat["box"], (210, 50, 240), f"BAT #{bat['id']}" + (" HELD" if bat["id"] in result["held_bats"] else ""))
    for person in result["people"]:
        alarm = person["safety_alarm"] or person["weapon_alarm"]
        color = (50, 50, 255) if alarm else (100, 230, 100)
        text = f"#{person['id']} H:{person['helmet_status']} V:{person['vest_status']}"
        box(person["box"], color, text)
        x, y = int(person["box"][0] * scale), int(person["box"][1] * scale) + 22
        timer = f"H {person['helmet_seconds']:.1f}/3  V {person['vest_seconds']:.1f}/3  BAT {person['bat_seconds']:.1f}/3"
        cv2.putText(view, timer, (max(x, 0), min(y, view.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        for hand in person["hands"]:
            box(hand["box"], (255, 150, 50))
    header = np.full((88, view.shape[1], 3), (25, 25, 30), np.uint8)
    safety = any(p["safety_alarm"] for p in result["people"])
    weapon = any(p["weapon_alarm"] for p in result["people"])
    lines = [(f"PPE: {'ALARM' if safety else 'MONITORING'} | NO HELMET OR NO VEST >= 3s | Events {totals['safety']}", safety),
             (f"BAT: {'ALARM' if weapon else 'MONITORING'} | HAND + BAT >= 3s | Events {totals['weapon']}", weapon)]
    for row, (text, active) in enumerate(lines):
        cv2.putText(header, text, (15, 27 + 35 * row), cv2.FONT_HERSHEY_SIMPLEX, .62,
                    (65, 65, 255) if active else (210, 220, 225), 2, cv2.LINE_AA)
    footer = np.full((30, view.shape[1], 3), (25, 25, 30), np.uint8)
    cv2.putText(footer, f"NPU {fps:.1f} FPS | People {len(result['people'])} | Q/ESC Quit | F Fullscreen | M Sound {'OFF' if muted else 'ON'}",
                (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .53, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack((header, view, footer))


class InferenceWorker(threading.Thread):
    def __init__(self, args, reader, stop, alarm):
        super().__init__(daemon=True)
        self.args, self.reader, self.stop, self.alarm = args, reader, stop, alarm
        self.lock = threading.Lock()
        self.image, self.published_at, self.error = None, 0., None
        self.count, self.max_people, self.max_bats = 0, 0, 0
        self.totals = {"safety": 0, "weapon": 0}
        self.ready = threading.Event()
        self.last_result = None

    def run(self):
        engines = {}
        pool = ThreadPoolExecutor(max_workers=3)
        try:
            for key, spec in MODELS.items():
                engines[key] = DeepXModel(self.args.model_dir, spec)
            self.ready.set()
            logic, previous, epoch, last_frame = EventLogic(), -1, -1, 0.
            while not self.stop.is_set():
                frame, captured, sequence, current_epoch, connected = self.reader.snapshot()
                now = time.monotonic()
                if not connected or now - captured > MAX_EVIDENCE_GAP:
                    logic.reset()
                    self.stop.wait(.03)
                    continue
                if frame is None or sequence == previous:
                    self.stop.wait(.005)
                    continue
                if current_epoch != epoch or captured - last_frame > MAX_EVIDENCE_GAP:
                    logic.reset()
                epoch, previous, last_frame = current_epoch, sequence, captured
                begin = time.monotonic()
                futures = {key: pool.submit(model.infer, frame) for key, model in engines.items()}
                detections = {key: future.result() for key, future in futures.items()}
                # 재접속 중이거나 오래된 프레임으로 끝난 추론은 판정에서 제외합니다.
                _, newest_at, _, newest_epoch, live = self.reader.snapshot()
                if not live or newest_epoch != epoch or time.monotonic() - captured > MAX_EVIDENCE_GAP:
                    logic.reset()
                    continue
                result = logic.update(frame, detections, captured)
                for event in result["events"]:
                    self.totals[event["type"]] += 1
                fps = 1 / max(time.monotonic() - begin, .001)
                image = draw(frame, result, fps, self.totals, self.alarm.muted)
                for event in result["events"]:
                    self.alarm.submit(event, image)
                with self.lock:
                    self.image, self.published_at, self.last_result = image, captured, result
                    self.count += 1
                    self.max_people = max(self.max_people, len(result["people"]))
                    self.max_bats = max(self.max_bats, len(result["bats"]))
                if self.count == 1 or self.count % 100 == 0:
                    LOG.info("실시간 추론: frame=%d, %dx%d, %.1f FPS, people=%d, bats=%d", self.count, frame.shape[1], frame.shape[0], fps, len(result["people"]), len(result["bats"]))
        except Exception as error:
            self.error = str(error)
            LOG.exception("모델/추론 오류: 이벤트 판정을 중단합니다.")
        finally:
            pool.shutdown(wait=True)
            for model in engines.values():
                model.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=os.environ.get("CAMERA_RTSP_URL", RTSP_URL))
    p.add_argument("--model-dir", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "demo_events")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--seconds", type=float, default=0, help="실시간 추론 시작 후 검사 시간")
    p.add_argument("--frames", type=int, default=0, help="추론 프레임 수 제한")
    p.add_argument("--snapshot", type=Path, help="종료 시 표시 화면 저장")
    args = p.parse_args()
    if args.seconds < 0 or args.frames < 0:
        p.error("seconds/frames는 0 이상이어야 합니다.")
    if args.headless and not args.seconds and not args.frames:
        args.seconds = 30
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    missing = [name for name, *_ in MODELS.values() if not (args.model_dir / name).is_file()]
    if missing:
        LOG.error("모델 파일이 없습니다: %s", ", ".join(missing))
        return 1
    if not args.headless and not os.environ.get("DISPLAY"):
        LOG.error("VNC/XRDP 바탕화면 터미널에서 실행하세요. SSH 검사는 --headless를 사용하세요.")
        return 1
    stop = threading.Event()
    reader, alarm = CameraReader(args.url, stop), AlarmWriter(args.output_dir, args.mute)
    worker = InferenceWorker(args, reader, stop, alarm)
    exit_code, fullscreen, first_inference = 0, False, None
    started = time.monotonic()
    reader.start()
    alarm.start()
    worker.start()
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW, 1100, 720)
        while True:
            now = time.monotonic()
            if worker.error:
                exit_code = 1
                break
            with worker.lock:
                image, captured, count = worker.image, worker.published_at, worker.count
            if count and first_inference is None:
                first_inference = now
            if first_inference is None and now - started > 45:
                LOG.error("45초 안에 모델 추론 영상을 받지 못했습니다.")
                exit_code = 1
                break
            if args.frames and count >= args.frames:
                break
            if args.seconds and first_inference and now - first_inference >= args.seconds:
                break
            if args.headless and first_inference and now - captured > 15:
                LOG.error("실시간 영상/추론이 15초 이상 중단됐습니다.")
                exit_code = 1
                break
            if not args.headless:
                if image is None:
                    image = np.zeros((720, 1100, 3), np.uint8)
                    cv2.putText(image, "Loading 5 NPU models / connecting camera...", (40, 330), cv2.FONT_HERSHEY_SIMPLEX, .8, (230, 230, 230), 2)
                elif now - captured > MAX_EVIDENCE_GAP:
                    image = image.copy()
                    cv2.rectangle(image, (0, 0), (image.shape[1], 88), (30, 30, 30), -1)
                    cv2.putText(image, "NO LIVE EVIDENCE - reconnecting / timers reset", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 170, 255), 2)
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("f"), ord("F")):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if key in (ord("m"), ord("M")):
                    alarm.muted = not alarm.muted
            else:
                stop.wait(.05)
    except KeyboardInterrupt:
        pass
    except Exception:
        LOG.exception("실행 오류")
        exit_code = 1
    finally:
        stop.set()
        worker.join(timeout=15)
        reader.join(timeout=6)
        alarm.stopping.set()
        alarm.join(timeout=8)
        if not args.headless:
            cv2.destroyAllWindows()
        if args.snapshot and worker.image is not None:
            if not cv2.imwrite(str(args.snapshot), worker.image):
                LOG.error("검사 화면 저장 실패: %s", args.snapshot)
                exit_code = 1
        if worker.is_alive() or worker.error or alarm.error:
            exit_code = 1
        LOG.info("SUMMARY %s", json.dumps({"inference_frames": worker.count, "models": list(MODELS), "max_people": worker.max_people, "max_bats": worker.max_bats, "events": worker.totals, "exit_code": exit_code}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
