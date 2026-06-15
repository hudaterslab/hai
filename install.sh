#!/bin/bash

# 1. 사용자 및 경로 설정
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

# [수정] 스크립트가 실제 위치한 절대 경로를 동적으로 가져옵니다. (어느 위치에서 실행해도 무방함)
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CAM_CONFIG_FILE="$PROJECT_DIR/cameras.json"
CAM_CSV_FILE="$PROJECT_DIR/cameras.csv"
SYS_CONFIG_FILE="$PROJECT_DIR/system_config.json"
TARGET_SCRIPT="$PROJECT_DIR/multi_event.py"
MODEL_DOWNLOAD_SCRIPT="$PROJECT_DIR/downdxnn.sh"
DX_DIR="$USER_HOME/dx-runtime"

sudo apt install ssh -y

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 하이브리드 설치"
echo "====================================================="

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

# 2. system_config.json 부재 시 자동 생성
if [ ! -f "$SYS_CONFIG_FILE" ]; then
    echo "⚙️ [설정 생성] 'system_config.json' 파일을 생성합니다."
    read -p ">> 이 장비의 고유 Terminal ID를 입력하세요 (예: 10001): " USER_TERM_ID
    if [ -z "$USER_TERM_ID" ]; then USER_TERM_ID="99999"; fi

    sudo -u "$ACTUAL_USER" bash -c "cat > $SYS_CONFIG_FILE" << EOL
{
    "terminal_id": "$USER_TERM_ID",
    "INFERENCE_MODE": "auto",
    "logging": {"dir": "./logs", "level": "INFO"},
    "event_config": {
        "intrusion": {"enabled": false, "cooldown_sec": 600},
        "illegal_parking": {"enabled": false, "cooldown_sec": 600, "trigger_sec": 5.0, "move_threshold_ratio": 0.1, "blur_plate": true},
        "no_helmet": {"enabled": false, "cooldown_sec": 600, "blur_face": true, "blur_plate": true, "trigger_sec": 3.0},
        "conveyor_crossing": {"enabled": false, "cooldown_sec": 600, "snapshot_mode": "crossing_moment", "distance_ratio": 0.5, "min_crossing_angle": 20.0, "candidate_ttl_sec": 5.0, "low_body_fallback_sec": 2.0, "blur_face": true, "blur_plate": true},
        "signal_vehicle": {
            "enabled": false, "cooldown_sec": 600, "motion_threshold_ratio": 0.10, "blur_plate": true,
            "line_truck_confirm_frames": 10,
            "line_truck_confirm_ratio": 0.7,
            "line_truck_car_veto_frames": 5,
            "line_truck_min_conf": 0.7,
            "line_truck_car_veto_iou": 0.10,
            "line_truck_car_veto_distance_ratio": 0.60
        }
    },
    "models": {
        "UNIFIED": "signalman.dxnn",
        "MAIN": "hanjin_cctv.dxnn",
        "FACE": "yolov8m-face.dxnn",
        "HELMET": "helmet_3cls_v8.dxnn",
        "SIGNALMAN": "signalman.dxnn",
        "PLATE": "license_plate_detector.dxnn"
    },
    "model_confidences": {"MAIN": 0.6, "FACE": 0.35, "HELMET": 0.55, "PERSON": 0.35, "SIGNALMAN": 0.5, "PLATE": 0.1},
    "model_output_formats": {"UNIFIED": "auto", "MAIN": "auto", "FACE": "auto", "HELMET": "auto", "SIGNALMAN": "auto", "PLATE": "auto"},
    "model_engine_pool_sizes": {"UNIFIED": 1, "MAIN": 1, "FACE": 1, "HELMET": 1, "SIGNALMAN": 1, "PLATE": 1},
    "BATCH_SIZE": 9, "REC_FPS": 3, "REC_PRE_SEC": 10, "REC_POST_SEC": 10,
    "INTERACTIVE_INPUT_GUARD_SEC": 0.35,
    "VISUAL_ALARM_DURATION": 5.0
}
EOL
    echo "✅ 기본 설정 생성이 완료되었습니다."
fi

DEFAULT_SYS_CONFIG_FILE="$(mktemp)"
cat > "$DEFAULT_SYS_CONFIG_FILE" << EOL
{
    "terminal_id": "99999",
    "INFERENCE_MODE": "auto",
    "logging": {"dir": "./logs", "level": "INFO"},
    "event_config": {
        "intrusion": {"enabled": false, "cooldown_sec": 600},
        "illegal_parking": {"enabled": false, "cooldown_sec": 600, "trigger_sec": 5.0, "move_threshold_ratio": 0.1, "blur_plate": true},
        "no_helmet": {"enabled": false, "cooldown_sec": 600, "blur_face": true, "blur_plate": true, "trigger_sec": 3.0},
        "conveyor_crossing": {"enabled": false, "cooldown_sec": 600, "snapshot_mode": "crossing_moment", "distance_ratio": 0.5, "min_crossing_angle": 20.0, "candidate_ttl_sec": 5.0, "low_body_fallback_sec": 2.0, "blur_face": true, "blur_plate": true},
        "signal_vehicle": {
            "enabled": false, "cooldown_sec": 600, "motion_threshold_ratio": 0.10, "blur_plate": true,
            "line_truck_confirm_frames": 10,
            "line_truck_confirm_ratio": 0.7,
            "line_truck_car_veto_frames": 5,
            "line_truck_min_conf": 0.7,
            "line_truck_car_veto_iou": 0.10,
            "line_truck_car_veto_distance_ratio": 0.60
        }
    },
    "models": {
        "UNIFIED": "signalman.dxnn",
        "MAIN": "hanjin_cctv.dxnn",
        "FACE": "yolov8m-face.dxnn",
        "HELMET": "helmet_3cls_v8.dxnn",
        "SIGNALMAN": "signalman.dxnn",
        "PLATE": "license_plate_detector.dxnn"
    },
    "model_confidences": {"MAIN": 0.6, "FACE": 0.35, "HELMET": 0.55, "PERSON": 0.35, "SIGNALMAN": 0.5, "PLATE": 0.1},
    "model_output_formats": {"UNIFIED": "auto", "MAIN": "auto", "FACE": "auto", "HELMET": "auto", "SIGNALMAN": "auto", "PLATE": "auto"},
    "model_engine_pool_sizes": {"UNIFIED": 1, "MAIN": 1, "FACE": 1, "HELMET": 1, "SIGNALMAN": 1, "PLATE": 1},
    "BATCH_SIZE": 9, "REC_FPS": 3, "REC_PRE_SEC": 10, "REC_POST_SEC": 10,
    "INTERACTIVE_INPUT_GUARD_SEC": 0.35,
    "VISUAL_ALARM_DURATION": 5.0
}
EOL

if [ -f "$SYS_CONFIG_FILE" ]; then
    echo "-> system_config.json을 기본 템플릿과 비교합니다. terminal_id는 보존합니다."
    python3 - "$SYS_CONFIG_FILE" "$DEFAULT_SYS_CONFIG_FILE" << 'PY'
import copy
import json
import shutil
import sys
import time

current_path, default_path = sys.argv[1], sys.argv[2]

with open(default_path, "r", encoding="utf-8") as f:
    default_cfg = json.load(f)

already_backed_up = False
try:
    with open(current_path, "r", encoding="utf-8") as f:
        current_cfg = json.load(f)
except Exception as exc:
    backup_path = f"{current_path}.invalid.{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(current_path, backup_path)
    already_backed_up = True
    current_cfg = {}
    print(f"[system_config] invalid JSON backed up: {backup_path} ({exc})")

candidate_cfg = copy.deepcopy(default_cfg)
candidate_cfg["terminal_id"] = current_cfg.get("terminal_id", default_cfg.get("terminal_id", "99999"))

def without_terminal_id(value):
    copied = copy.deepcopy(value)
    if isinstance(copied, dict):
        copied.pop("terminal_id", None)
    return copied

if without_terminal_id(current_cfg) == without_terminal_id(candidate_cfg):
    print("[system_config] unchanged")
    sys.exit(0)

if not already_backed_up:
    backup_path = f"{current_path}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(current_path, backup_path)
else:
    backup_path = "(invalid backup above)"

with open(current_path, "w", encoding="utf-8") as f:
    json.dump(candidate_cfg, f, ensure_ascii=False, indent=4)
    f.write("\n")

print(f"[system_config] updated from template; terminal_id preserved; backup: {backup_path}")
PY
    sudo chown "$ACTUAL_USER:$(id -gn "$ACTUAL_USER")" "$SYS_CONFIG_FILE" 2>/dev/null || true
fi

rm -f "$DEFAULT_SYS_CONFIG_FILE"

echo "-----------------------------------------------------"
echo " 3. Python 환경 및 필수 패키지 설치"
echo "-----------------------------------------------------"
# python 명령어를 python3로 연결 (이미 존재해도 오류 안 나도록 -f 옵션 추가)
echo "-> 'python' 심볼릭 링크를 생성합니다..."
sudo ln -sf /usr/bin/python3 /usr/bin/python

# 필수 파이썬 패키지 설치 (최신 우분투/데비안의 PEP-668 제약 우회를 위해 break-system-packages 옵션 추가 대비)
echo "-> 필수 파이썬 패키지를 설치합니다 (pytz, psutil, requests, opencv-python)..."
python -m pip install pytz psutil requests opencv-python || \
python -m pip install pytz psutil requests opencv-python --break-system-packages

echo "-----------------------------------------------------"
echo " 4. DX runtime precheck"
echo "-----------------------------------------------------"

DX_ENGINE_READY=false
DXRT_CLI_READY=false
DXRT_SERVICE_ACTIVE=false

if python3 -c 'import dx_engine' > /dev/null 2>&1; then
    DX_ENGINE_READY=true
    echo "-> dx_engine import: OK"
else
    echo "-> dx_engine import: missing"
fi

if command -v dxrt-cli > /dev/null 2>&1; then
    DXRT_CLI_READY=true
    echo "-> dxrt-cli: OK ($(command -v dxrt-cli))"
else
    echo "-> dxrt-cli: missing"
fi

if systemctl is-active --quiet dxrt.service 2>/dev/null; then
    DXRT_SERVICE_ACTIVE=true
    echo "-> dxrt.service: active"
else
    echo "-> dxrt.service: inactive or not installed"
fi

echo "-----------------------------------------------------"
echo " 5. DeepX 드라이버 및 런타임 설치/복구"
echo "-----------------------------------------------------"
if [ "$DX_ENGINE_READY" != "true" ]; then
    echo "-> 시스템에서 'dx_engine'을 찾을 수 없어 공식 패키지를 다운로드합니다..."
    DOWNLOAD_DIR="$USER_HOME/Downloads"
    sudo -u "$ACTUAL_USER" mkdir -p "$DOWNLOAD_DIR"

    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/refs/heads/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb
    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt/raw/refs/heads/main/release/3.3.2/libdxrt_3.3.2_all.deb

    sudo apt install -y "$DOWNLOAD_DIR/dxrt-driver-dkms_2.4.0-2_all.deb"
    sudo apt install -y "$DOWNLOAD_DIR/libdxrt_3.3.2_all.deb"

    # 설치 직후 서비스 강제 재시작 (장치 인식 유도)
    sudo systemctl restart dxrt.service 2>/dev/null || true
else
    echo "✅ [확인] dx_engine (런타임)이 이미 글로벌 환경에 설치되어 있습니다."
fi

echo "-----------------------------------------------------"
echo " 6. dx-runtime 설치 보강 및 펌웨어 플래싱"
echo "-----------------------------------------------------"
if [ ! -d "$DX_DIR" ]; then
    echo "-> 펌웨어 및 라이브러리 파일을 가져오기 위해 저장소를 클론합니다 (서브모듈 포함)..."
    # [수정] 서브모듈(fw.bin 등)을 함께 다운로드하기 위해 --recurse-submodules 옵션 추가
    sudo -u "$ACTUAL_USER" git clone --recurse-submodules https://github.com/DEEPX-AI/dx-runtime.git "$DX_DIR"
else
    echo "-> 기존 dx-runtime 저장소를 사용합니다: $DX_DIR"
    sudo -u "$ACTUAL_USER" git -C "$DX_DIR" submodule update --init --recursive || true
fi

if [ "$DX_ENGINE_READY" != "true" ] || [ "$DXRT_CLI_READY" != "true" ]; then
    echo "-> dx-runtime 내장 설치 스크립트(install.sh)를 실행하여 누락 항목을 보강합니다..."
    (cd "$DX_DIR" && sudo bash install.sh)
else
    echo "-> dx_engine과 dxrt-cli가 이미 확인되어 dx-runtime install.sh 실행을 건너뜁니다."
fi

if command -v dxrt-cli &> /dev/null; then
    echo "-> M.2 / PCIe 기반 펌웨어(FW) 업데이트를 시도합니다..."
    dxrt-cli -u "$DX_DIR/dx_fw/m1/latest/mdot2/fw.bin" || echo "⚠️ [안내] 펌웨어 업데이트 건너뜀 (이미 최신이거나 재부팅 필요)"
    dxrt-cli -u "$DX_DIR/dx_fw/m1m/latest/mdot2/fw.bin" > /dev/null 2>&1 || true
else
    echo "⚠️ 'dxrt-cli' 명령어를 찾을 수 없어 펌웨어 업데이트를 건너뜁니다."
fi

echo "-----------------------------------------------------"
echo " 7. dx_engine 최종 검증"
echo "-----------------------------------------------------"
if ! python3 -c 'import dx_engine' > /dev/null 2>&1; then
    echo "❌ [에러] 드라이버 설치 후에도 모듈을 로드할 수 없습니다."
    echo "   >> NPU 장치가 커널에 등록되지 않았습니다."
    echo "   >> 지금 즉시 'sudo reboot' 명령어로 PC를 재부팅한 뒤, 이 스크립트를 다시 실행해 주세요!"
    exit 1
else
    echo "✅ [검증 성공] 로컬(Global) 파이썬 환경에서 'dx_engine'이 완벽하게 로드됩니다."
fi

echo "-----------------------------------------------------"
echo " 8. 모델 파일 다운로드 및 확인"
echo "-----------------------------------------------------"
MODEL_FILES_READY=true
if [ -f "$MODEL_DOWNLOAD_SCRIPT" ]; then
    echo "-> downdxnn.sh를 실행하여 모델 파일을 확인/다운로드합니다..."
    (cd "$PROJECT_DIR" && bash "$MODEL_DOWNLOAD_SCRIPT")
else
    echo "⚠️ downdxnn.sh 파일이 없어 모델 다운로드를 건너뜁니다: $MODEL_DOWNLOAD_SCRIPT"
fi

if ! python3 - "$SYS_CONFIG_FILE" "$PROJECT_DIR" << 'PY'
import json
import os
import sys

config_path, project_dir = sys.argv[1], sys.argv[2]
try:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception as exc:
    print(f"[model-check] system_config.json read failed: {exc}")
    sys.exit(1)

missing = []
for key, value in sorted((cfg.get("models") or {}).items()):
    if not value:
        continue
    path = value if os.path.isabs(value) else os.path.join(project_dir, value)
    if not os.path.exists(path):
        missing.append(f"{key}={value}")

if missing:
    print("[model-check] missing model files: " + ", ".join(missing))
    sys.exit(1)

print("[model-check] all configured model files exist")
PY
then
    MODEL_FILES_READY=false
fi

echo "-----------------------------------------------------"
echo " 9. Systemd 백그라운드 서비스 등록"
echo "-----------------------------------------------------"
sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
Requires=dxrt.service
After=network.target dxrt.service graphical.target
ConditionPathExists=$SYS_CONFIG_FILE
ConditionPathExists=$CAM_CSV_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStartPre=/bin/sleep 20
ExecStart=/bin/bash -c 'yes n | python3 -u multi_event.py'
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl stop $SERVICE_NAME

if [ ! -f "$CAM_CSV_FILE" ]; then
    echo "⚠️ 'cameras.csv' (카메라 설정 파일)이 없어 서비스가 대기 모드로 진입합니다."
elif [ "$MODEL_FILES_READY" != "true" ]; then
    echo "⚠️ 모델 파일이 부족하여 서비스 시작을 보류합니다. downdxnn.sh 또는 system_config.json의 models 경로를 확인하세요."
else
    sudo systemctl start $SERVICE_NAME
    echo "▶️ CCTV AI 백그라운드 서비스가 구동되었습니다."
fi
sudo apt install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
echo "====================================================="
echo " 🎉 설치 완료! 쾌적하고 안정적인 Global 환경으로 설정되었습니다."
echo " ※ 만약 AI 인퍼런스가 정상 작동하지 않는다면, PC를 1회 껐다 켜주세요 (Cold Boot 권장)."
echo "====================================================="
