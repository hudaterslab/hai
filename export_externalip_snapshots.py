import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2


DEFAULT_CAMERA_CSV = "cameras.csv"
DEFAULT_REMOTE_USER = None
DEFAULT_REMOTE_HOST = None
DEFAULT_REMOTE_PORT = None
DEFAULT_REMOTE_DIR = "/volume1/Hudaters/dhkim"
DEFAULT_LOCAL_OUTPUT_DIR = "snapshot_exports"
EXTERNAL_IP_SERVICES = (
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://icanhazip.com",
)


def clean_rtsp_url(url):
    return re.sub(r"\s+", "", url or "")


def extract_camera_name(rtsp_url):
    try:
        rest = clean_rtsp_url(rtsp_url).split("@")[-1].split("//")[-1]
        host_port = rest.split("/")[0]
        host, _, port = host_port.partition(":")
        host_tail = host.split(".")[-1]
        return f"{host_tail}_{port}" if port else host_tail
    except Exception:
        return "unknown_camera"


def load_camera_urls(csv_path):
    urls = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                url = (
                    row.get("url")
                    or row.get("rtsp")
                    or row.get("rtsp_url")
                    or row.get("camera_url")
                )
                url = clean_rtsp_url(url)
                if url:
                    urls.append(url)
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                url = clean_rtsp_url(row[0])
                if url and not url.startswith("#"):
                    urls.append(url)
    return urls


def capture_snapshot(rtsp_url):
    cap = cv2.VideoCapture(clean_rtsp_url(rtsp_url), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def ensure_remote_dir(user, host, port, remote_dir):
    cmd = [
        "ssh",
        *ssh_common_args(user, host, port),
        f"mkdir -p {sh_quote(remote_dir)}",
    ]
    subprocess.run(cmd, check=True, text=True)


def sh_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def upload_snapshot(local_path, user, host, port, remote_dir):
    remote_path = f"{remote_dir.rstrip('/')}/{os.path.basename(local_path)}"
    cmd = [
        "ssh",
        *ssh_common_args(user, host, port),
        f"cat > {sh_quote(remote_path)}",
    ]
    with open(local_path, "rb") as src:
        subprocess.run(cmd, stdin=src, check=True)


SSH_CONTROL_PATH = None


def ssh_common_args(user, host, port):
    args = [
        "-p",
        str(port),
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=10m",
        "-o",
        f"ControlPath={SSH_CONTROL_PATH}",
        f"{user}@{host}",
    ]
    return args


def open_ssh_master(user, host, port):
    cmd = [
        "ssh",
        "-MNf",
        *ssh_common_args(user, host, port),
    ]
    subprocess.run(cmd, check=True)


def close_ssh_master(user, host, port):
    cmd = [
        "ssh",
        "-O",
        "exit",
        *ssh_common_args(user, host, port),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sanitize_device_tag(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("_") or "unknown_device"


def detect_external_ip():
    for service_url in EXTERNAL_IP_SERVICES:
        try:
            with urllib.request.urlopen(service_url, timeout=3) as resp:
                ip_text = resp.read().decode("utf-8", errors="ignore").strip()
                if re.fullmatch(r"[0-9a-fA-F:.]+", ip_text):
                    return ip_text
        except Exception:
            continue
    return None


def detect_device_tag():
    env_value = os.environ.get("EDGE_DEVICE_TAG") or os.environ.get("EXTERNAL_IP")
    if env_value:
        return sanitize_device_tag(env_value)

    external_ip = detect_external_ip()
    if external_ip:
        return sanitize_device_tag(external_ip)

    return sanitize_device_tag(os.uname().nodename)


def main():
    parser = argparse.ArgumentParser(
        description="Capture snapshots from camera CSV and upload them to a remote server."
    )
    parser.add_argument("--csv", default=DEFAULT_CAMERA_CSV, help="Path to camera CSV file")
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument(
        "--local-output-dir",
        default=DEFAULT_LOCAL_OUTPUT_DIR,
        help="Local directory to keep captured snapshots for later inspection.",
    )
    parser.add_argument(
        "--device-tag",
        default=None,
        help="Remote folder name for this edge device. Defaults to external IP, then hostname fallback.",
    )
    args = parser.parse_args()
    remote_user = (args.remote_user or input("Remote SSH user: ").strip())
    if not remote_user:
        print("Remote SSH user is required.", file=sys.stderr)
        return 1
    remote_host = (args.remote_host or input("Remote SSH host: ").strip())
    if not remote_host:
        print("Remote SSH host is required.", file=sys.stderr)
        return 1
    remote_port = args.remote_port
    if remote_port is None:
        port_text = input("Remote SSH port: ").strip()
        if not port_text:
            print("Remote SSH port is required.", file=sys.stderr)
            return 1
        try:
            remote_port = int(port_text)
        except ValueError:
            print(f"Invalid remote SSH port: {port_text!r}", file=sys.stderr)
            return 1

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    try:
        camera_urls = load_camera_urls(args.csv)
    except Exception as exc:
        print(f"Failed to load camera CSV: {exc}", file=sys.stderr)
        return 1

    if not camera_urls:
        print("No camera URLs found in CSV.", file=sys.stderr)
        return 1

    device_tag = sanitize_device_tag(args.device_tag) if args.device_tag else detect_device_tag()
    remote_dir = str(Path(args.remote_dir) / device_tag)
    local_output_dir = Path(args.local_output_dir) / device_tag / timestamp_safe_now()
    local_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device tag: {device_tag}")
    print(f"Remote directory: {remote_dir}")
    print(f"Local output directory: {local_output_dir}")

    timestamp = local_output_dir.name
    uploaded = 0
    failed = 0
    global SSH_CONTROL_PATH
    SSH_CONTROL_PATH = os.path.join(tempfile.gettempdir(), f"snapshot_ssh_{os.getpid()}")

    try:
        open_ssh_master(
            remote_user,
            remote_host,
            remote_port,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to establish SSH session: {exc}", file=sys.stderr)
        return 1

    try:
        try:
            ensure_remote_dir(
                remote_user,
                remote_host,
                remote_port,
                remote_dir,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Failed to create remote directory: {exc}", file=sys.stderr)
            return 1

        for index, rtsp_url in enumerate(camera_urls, start=1):
            camera_name = extract_camera_name(rtsp_url)
            filename = f"{index:03d}_{camera_name}_{timestamp}.jpg"
            local_path = str(local_output_dir / filename)

            frame = capture_snapshot(rtsp_url)
            if frame is None:
                failed += 1
                print(f"[FAIL] Snapshot capture failed: {rtsp_url}", file=sys.stderr)
                continue

            if not cv2.imwrite(local_path, frame):
                failed += 1
                print(f"[FAIL] Snapshot save failed: {local_path}", file=sys.stderr)
                continue

            try:
                upload_snapshot(
                    local_path,
                    remote_user,
                    remote_host,
                    remote_port,
                    remote_dir,
                )
                uploaded += 1
                print(f"[OK] Uploaded {filename}")
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(f"[FAIL] Upload failed for {filename}: {exc}", file=sys.stderr)
    finally:
        close_ssh_master(
            remote_user,
            remote_host,
            remote_port,
        )
    print(f"Done. uploaded={uploaded}, failed={failed}")
    return 0 if uploaded > 0 and failed == 0 else 1


def timestamp_safe_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
