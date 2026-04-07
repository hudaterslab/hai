import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2

from multi_event import (
    CONFIG_CAMERAS_FILE,
    MODEL_FACE_PATH,
    MODEL_GENERAL_PATH,
    MODEL_HELMET_PATH,
    build_runtime_camera_config,
    load_json_file,
    logger,
    sanitize_camera_url,
    setup_logging,
)


MODEL_MAP = {
    "helmet": MODEL_HELMET_PATH,
    "general": MODEL_GENERAL_PATH,
    "face": MODEL_FACE_PATH,
}


def parse_args():
    parser = argparse.ArgumentParser(description="캡처/모델로드/추론 단계 분리 테스트")
    parser.add_argument("--source", help="직접 입력할 RTSP/동영상/이미지 경로")
    parser.add_argument("--camera-key", help="config/cameras.json 에서 사용할 카메라 key")
    parser.add_argument("--camera-config", default=CONFIG_CAMERAS_FILE)
    parser.add_argument("--warmup-reads", type=int, default=30, help="캡처 후 버릴 프레임 수")
    parser.add_argument("--read-timeout-sec", type=float, default=10.0)
    parser.add_argument("--infer-timeout-sec", type=float, default=15.0)
    parser.add_argument("--load-timeout-sec", type=float, default=15.0)
    parser.add_argument("--models", default="helmet,general,face", help="쉼표 구분 모델 목록 또는 all")
    parser.add_argument("--worker-load", action="store_true")
    parser.add_argument("--worker-infer", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--frame-path")
    return parser.parse_args()


def log_step(step, **fields):
    joined = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(f"[DX_STAGE_TEST] step={step} {joined}".rstrip())


def resolve_source(args):
    if args.source:
        return sanitize_camera_url(args.source)

    if not args.camera_key:
        raise ValueError("--source 또는 --camera-key 중 하나는 필요합니다.")

    camera_map = load_json_file(args.camera_config, {}, description="카메라 설정 파일", expected_type=dict)
    if args.camera_key not in camera_map:
        raise KeyError(f"camera_key를 찾을 수 없습니다: {args.camera_key}")
    runtime_conf = build_runtime_camera_config(camera_map[args.camera_key], {})
    return runtime_conf["url"]


def capture_frame_from_source(source, warmup_reads, read_timeout_sec):
    if os.path.isfile(source):
        image = cv2.imread(source)
        if image is not None:
            log_step("capture_image", source=source, width=image.shape[1], height=image.shape[0])
            return image

    cap_started_at = time.time()
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"입력을 열 수 없습니다: {source}")
    log_step("capture_opened", source=source, elapsed=round(time.time() - cap_started_at, 3))

    frame = None
    read_count = 0
    failed_reads = 0
    warmup_logged = False
    try:
        while time.time() - cap_started_at < read_timeout_sec:
            ret, current = cap.read()
            read_count += 1
            if not ret or current is None:
                failed_reads += 1
                if failed_reads in (1, 5, 10, 20):
                    log_step("capture_read_retry", reads=read_count, failed_reads=failed_reads, elapsed=round(time.time() - cap_started_at, 3))
                time.sleep(0.05)
                continue
            if failed_reads > 0:
                log_step("capture_read_recovered", reads=read_count, failed_reads=failed_reads)
                failed_reads = 0
            frame = current
            if read_count <= warmup_reads:
                if not warmup_logged:
                    log_step("capture_warmup", warmup_reads=warmup_reads)
                    warmup_logged = True
                continue
            break
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError(f"유효 프레임을 읽지 못했습니다. reads={read_count} failed_reads={failed_reads}")

    log_step(
        "capture_done",
        reads=read_count,
        failed_reads=failed_reads,
        warmup_reads=warmup_reads,
        width=frame.shape[1],
        height=frame.shape[0],
    )
    return frame


def write_temp_frame(frame):
    fd, path = tempfile.mkstemp(prefix="dx_stage_test_", suffix=".jpg")
    os.close(fd)
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"임시 프레임 저장 실패: {path}")
    return path


def normalize_models(raw_models):
    if raw_models.strip().lower() == "all":
        return list(MODEL_MAP.keys())
    models = [name.strip().lower() for name in raw_models.split(",") if name.strip()]
    invalid = [name for name in models if name not in MODEL_MAP]
    if invalid:
        raise ValueError(f"지원하지 않는 모델: {invalid}")
    return models


def _worker_print(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_worker_load(model_path):
    import time as _time
    from multi_event import YoLoDeepX

    started_at = _time.time()
    _worker_print({"step": "worker_start", "model_path": model_path})
    YoLoDeepX(model_path)
    model_loaded_at = _time.time()
    _worker_print({"step": "worker_model_loaded", "elapsed": round(model_loaded_at - started_at, 3)})


def run_worker_infer(model_path, frame_path):
    import time as _time
    from multi_event import YoLoDeepX

    started_at = _time.time()
    _worker_print({"step": "worker_start", "model_path": model_path, "frame_path": frame_path})
    model = YoLoDeepX(model_path)
    model_loaded_at = _time.time()
    _worker_print({"step": "worker_model_loaded", "elapsed": round(model_loaded_at - started_at, 3)})
    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError(f"프레임 로드 실패: {frame_path}")
    _worker_print({"step": "worker_frame_loaded", "width": int(frame.shape[1]), "height": int(frame.shape[0])})
    infer_started_at = _time.time()
    _worker_print({"step": "worker_infer_start", "elapsed_since_start": round(infer_started_at - started_at, 3)})
    detections = model.infer(frame)
    infer_done_at = _time.time()
    _worker_print(
        {
            "step": "worker_done",
            "model_path": model_path,
            "load_elapsed": round(model_loaded_at - started_at, 3),
            "infer_elapsed": round(infer_done_at - infer_started_at, 3),
            "detections": 0 if detections is None else int(len(detections)),
        }
    )


def _collect_worker_logs(model_name, stdout_lines, stderr_lines):
    for line in stdout_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log_step("worker_stdout_raw", model=model_name, line=line)
            continue
        step = payload.pop("step", "worker")
        log_step(step, model=model_name, **payload)
    for line in stderr_lines:
        log_step("worker_stderr", model=model_name, line=line)


def run_subprocess_step(script_path, model_name, cmd, timeout_sec, phase_name):
    log_step(f"{phase_name}_start", model=model_name, timeout_sec=timeout_sec, command=" ".join(cmd))
    started_at = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - started_at, 3)
        log_step(f"{phase_name}_timeout", model=model_name, timeout_sec=timeout_sec, elapsed=elapsed)
        return None

    elapsed = round(time.time() - started_at, 3)
    stdout_lines = [line for line in (result.stdout or "").strip().splitlines() if line.strip()]
    stderr_lines = [line for line in (result.stderr or "").strip().splitlines() if line.strip()]
    _collect_worker_logs(model_name, stdout_lines, stderr_lines)
    if result.returncode != 0:
        stderr_tail = stderr_lines[-1] if stderr_lines else ""
        log_step(f"{phase_name}_failed", model=model_name, elapsed=elapsed, returncode=result.returncode, detail=stderr_tail)
        return None

    log_step(f"{phase_name}_ok", model=model_name, elapsed=elapsed)
    return stdout_lines


def run_model_probe(script_path, model_name, model_path, frame_path, load_timeout_sec, infer_timeout_sec):
    if not os.path.exists(model_path):
        log_step("model_missing", model=model_name, model_path=model_path)
        return

    load_cmd = [
        sys.executable,
        script_path,
        "--worker-load",
        "--model-path",
        model_path,
    ]
    if run_subprocess_step(script_path, model_name, load_cmd, load_timeout_sec, "load_subprocess") is None:
        return

    infer_cmd = [
        sys.executable,
        script_path,
        "--worker-infer",
        "--model-path",
        model_path,
        "--frame-path",
        frame_path,
    ]
    run_subprocess_step(script_path, model_name, infer_cmd, infer_timeout_sec, "infer_subprocess")


def main():
    args = parse_args()
    setup_logging({"logging": {"dir": "./logs", "level": "INFO", "retention_days": 14}})

    if args.worker_load:
        run_worker_load(args.model_path)
        return

    if args.worker_infer:
        run_worker_infer(args.model_path, args.frame_path)
        return

    source = resolve_source(args)
    models = normalize_models(args.models)
    log_step("test_start", source=source, models=models)

    frame = capture_frame_from_source(source, args.warmup_reads, args.read_timeout_sec)
    frame_path = write_temp_frame(frame)
    log_step("frame_saved", frame_path=frame_path)

    try:
        for model_name in models:
            run_model_probe(
                script_path=os.path.abspath(__file__),
                model_name=model_name,
                model_path=MODEL_MAP[model_name],
                frame_path=frame_path,
                load_timeout_sec=args.load_timeout_sec,
                infer_timeout_sec=args.infer_timeout_sec,
            )
    finally:
        try:
            os.remove(frame_path)
        except OSError:
            pass

    log_step("test_done")


if __name__ == "__main__":
    main()
