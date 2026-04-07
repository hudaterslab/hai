import argparse
import copy
import os
import threading
from collections import Counter, defaultdict

import cv2

from multi_event import (
    CAMERA_LIST_FILE,
    CONFIG_CAMERAS_FILE,
    CONFIG_COMMON_FILE,
    DEFAULT_COMMON_CONFIG,
    MODEL_FACE_PATH,
    MODEL_GENERAL_PATH,
    MODEL_HELMET_PATH,
    Camera,
    MotionDetector,
    SimpleTracker,
    YoLoDeepX,
    build_runtime_camera_config,
    deep_merge_dict,
    load_json_file,
    logger,
    setup_logging,
)


class NullRecorder:
    """오프라인 테스트에서는 실제 영상 저장을 하지 않는 recorder 대체 구현."""

    def __init__(self):
        self.recording = False

    def update(self, frame):
        return None

    def trigger(self, event_name):
        return None

    def stop(self):
        return None


class OfflineVideoCamera(Camera):
    """동영상 파일 기반 오프라인 테스트용 Camera 경량 래퍼."""

    def __init__(self, ip, conf, helmet_detector, general_detector, face_detector, cam_id, sensitivity):
        self.ip = ip
        self.conf = conf
        self.reader = None
        self.terminal_id = str(conf.get("terminal_id", "3"))
        self.cctv_id = int(conf.get("cctv_id", 1))
        self.event_config = conf.get("event_config", {})
        self.roi_poly = []
        self.roi_lines = []
        self.roi_poly_norm = conf.get("roi_poly_norm", [])
        self.roi_lines_norm = conf.get("roi_lines_norm", [])
        self.using_normalized_roi = bool(self.roi_poly_norm or self.roi_lines_norm)
        if not self.using_normalized_roi:
            self.roi_poly = conf.get("roi_poly", [])
            self.roi_lines = conf.get("roi_lines", [])
        self.events = conf.get("events", [])
        self.helmet_detector = helmet_detector
        self.general_detector = general_detector
        self.face_detector = face_detector
        self.npu_id = 0
        self.cam_id = cam_id
        self.helmet_tracker = SimpleTracker(is_helmet=True)
        self.general_tracker = SimpleTracker(is_helmet=False)
        self.last_draw = None
        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}
        self.face_blur_cache = {}
        self.roi_frame_shape = None
        self.config_lock = threading.Lock()
        self.motion_det = MotionDetector(sensitivity)
        self.recorder = NullRecorder()
        self.event_counter = Counter()
        self.event_history = []
        self.init_handlers()

    def _persist_event(self, saved_img, event_name, bbox, real_tid, now, tid):
        """오프라인 테스트에서는 저장 대신 메모리에 이벤트만 기록한다."""
        self.event_counter[event_name] += 1
        self.event_history.append(
            {
                "event_name": event_name,
                "track_id": int(real_tid),
                "bbox": tuple(int(v) for v in bbox),
                "frame_events_seen": dict(self.event_counter),
            }
        )
        self.alerted[tid].add(event_name)
        self.last_evt_t[event_name] = now
        logger.info(
            f"[OFFLINE TEST] event={event_name} tid={real_tid} bbox={tuple(int(v) for v in bbox)} count={self.event_counter[event_name]}"
        )

    def handle_frame(self, frame, frame_id):
        """한 프레임에 대해 인퍼런스, 이벤트 판정, 렌더링을 수행한다."""
        self._update_runtime_state(frame)

        helmet_detections = []
        general_detections = []
        # 2026-04-07 by dhkim
        # 본코드와 동일하게 no_helmet/intrusion/conveyor_crossing은 best.dxnn(helmet path) 결과를 사용한다.
        if any(event_name in ["no_helmet", "intrusion", "conveyor_crossing"] for event_name in self.events):
            helmet_detections = self.helmet_detector.infer(frame)
        if any(event_name in ["illegal_parking", "signal_vehicle"] for event_name in self.events):
            general_detections = self.general_detector.infer(frame)

        helmet_tracks, general_tracks, alarms = self.run_logic(
            frame,
            frame_id,
            helmet_detections,
            general_detections,
        )
        rendered = self.draw(frame.copy(), helmet_tracks, general_tracks, alarms, connected=True)
        return rendered, alarms


def resolve_default_config_path(primary_path, fallback_path):
    """실제 설정이 있으면 우선 사용하고, 없으면 sample 설정을 사용한다."""
    if os.path.exists(primary_path):
        return primary_path
    return fallback_path


def load_runtime_config(common_path, cameras_path, camera_key, video_path):
    """샘플 또는 실제 설정 파일에서 오프라인 테스트용 카메라 설정을 만든다."""
    common_raw = load_json_file(common_path, {}, description="공통 설정 파일", expected_type=dict)
    common_conf = deep_merge_dict(DEFAULT_COMMON_CONFIG, common_raw)

    camera_map = load_json_file(cameras_path, {}, description="카메라 설정 파일", expected_type=dict)
    if not camera_map:
        raise ValueError(f"카메라 설정이 비어 있습니다: {cameras_path}")

    if camera_key is None:
        camera_key = next(iter(camera_map.keys()))
    if camera_key not in camera_map:
        raise KeyError(f"camera_key를 찾을 수 없습니다: {camera_key}")

    camera_conf = copy.deepcopy(camera_map[camera_key])
    camera_conf["url"] = video_path
    runtime_conf = build_runtime_camera_config(camera_conf, common_conf)
    return common_conf, camera_key, runtime_conf


def open_writer(output_path, width, height, fps):
    """렌더링 결과 저장용 mp4 writer를 연다."""
    if not output_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"출력 비디오를 열 수 없습니다: {output_path}")
    return writer


def parse_args():
    parser = argparse.ArgumentParser(description="오프라인 동영상 기반 인퍼런스/이벤트 렌더 테스트")
    parser.add_argument("--video", required=True, help="입력 동영상 파일 경로")
    parser.add_argument("--camera-key", help="config에서 사용할 카메라 key. 기본값은 첫 번째 항목")
    parser.add_argument("--common-config", default=resolve_default_config_path(CONFIG_COMMON_FILE, "config/common.sample.json"))
    parser.add_argument("--camera-config", default=resolve_default_config_path(CONFIG_CAMERAS_FILE, "config/cameras.sample.json"))
    parser.add_argument("--helmet-model", default=MODEL_HELMET_PATH)
    parser.add_argument("--general-model", default=MODEL_GENERAL_PATH)
    parser.add_argument("--face-model", default=MODEL_FACE_PATH)
    parser.add_argument("--disable-face", action="store_true", help="얼굴 모델을 로드하지 않음")
    parser.add_argument("--cam-id", type=int, default=1)
    parser.add_argument("--sensitivity", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=1, help="N 프레임마다 1회 처리")
    parser.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지 처리")
    parser.add_argument("--output", help="렌더링 결과 저장 mp4 경로")
    parser.add_argument("--no-display", action="store_true", help="화면 출력 없이 결과만 저장")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"입력 동영상을 찾을 수 없습니다: {args.video}")

    common_conf, camera_key, runtime_conf = load_runtime_config(
        args.common_config,
        args.camera_config,
        args.camera_key,
        args.video,
    )
    setup_logging(common_conf)
    logger.info(f"[OFFLINE TEST] video={args.video}")
    logger.info(f"[OFFLINE TEST] common_config={args.common_config}")
    logger.info(f"[OFFLINE TEST] camera_config={args.camera_config} camera_key={camera_key}")
    logger.info(f"[OFFLINE TEST] enabled_events={runtime_conf.get('events', [])}")

    if not os.path.exists(args.helmet_model):
        raise FileNotFoundError(f"helmet model 없음: {args.helmet_model}")
    if not os.path.exists(args.general_model):
        raise FileNotFoundError(f"general model 없음: {args.general_model}")
    if not args.disable_face and not os.path.exists(args.face_model):
        raise FileNotFoundError(f"face model 없음: {args.face_model}")

    helmet_detector = YoLoDeepX(args.helmet_model)
    general_detector = YoLoDeepX(args.general_model)
    face_detector = None if args.disable_face else YoLoDeepX(args.face_model)

    offline_camera = OfflineVideoCamera(
        ip=f"video:{os.path.basename(args.video)}",
        conf=runtime_conf,
        helmet_detector=helmet_detector,
        general_detector=general_detector,
        face_detector=face_detector,
        cam_id=args.cam_id,
        sensitivity=args.sensitivity,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열 수 없습니다: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = args.output
    if output_path is None:
        stem, _ext = os.path.splitext(args.video)
        output_path = f"{stem}.offline_test.mp4"

    writer = open_writer(output_path, width, height, fps)
    processed = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            if args.frame_step > 1 and frame_index % args.frame_step != 0:
                continue

            rendered, alarms = offline_camera.handle_frame(frame, frame_index)
            processed += 1

            cv2.putText(
                rendered,
                f"frame={frame_index} processed={processed}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if writer is not None:
                writer.write(rendered)

            if not args.no_display:
                cv2.imshow("offline-video-event-test", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        offline_camera.stop()

    logger.info(f"[OFFLINE TEST] processed_frames={processed}")
    logger.info(f"[OFFLINE TEST] output={output_path}")
    logger.info(f"[OFFLINE TEST] event_counts={dict(offline_camera.event_counter)}")


if __name__ == "__main__":
    main()
