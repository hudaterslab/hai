#!/bin/bash

# 1. 사용자 및 경로 설정
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
PROJECT_DIR=$(pwd)

# 2. 서비스 및 주요 경로 설정
SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

# [수정] 카메라 설정과 시스템 설정 파일 분리
CAM_CONFIG_FILE="$PROJECT_DIR/cameras.json"
SYS_CONFIG_FILE="$PROJECT_DIR/system_config.json"
TARGET_SCRIPT="$PROJECT_DIR/multi_event.py"

# 가상환경 경로 정의
DX_DIR="$USER_HOME/dx-runtime"
VENV_ACTIVATE="$DX_DIR/venv-dx-runtime/bin/activate"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치"
echo "====================================================="

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

# 3. [핵심 추가] system_config.json 부재 시 터미널 ID 입력받아 자동 생성
if [ ! -f "$SYS_CONFIG_FILE" ]; then
    echo "⚙️ [설정 생성] 'system_config.json' 파일이 존재하지 않습니다."
    
    # sudo로 실행 중이므로 입력 인터페이스 유지
    read -p ">> 이 장비의 고유 Terminal ID를 입력하세요 (예: 10001): " USER_TERM_ID
    
    # 입력값이 비어있을 경우 기본값 세팅
    if [ -z "$USER_TERM_ID" ]; then
        USER_TERM_ID="99999"
    fi
    
    echo "-> Terminal ID [$USER_TERM_ID] 기반으로 기본 시스템 설정을 생성합니다..."
    
    # 파이썬 default_config를 표준 JSON 포맷으로 변환하여 작성
    sudo -u "$ACTUAL_USER" bash -c "cat > $SYS_CONFIG_FILE" << EOL
{
    "terminal_id": "$USER_TERM_ID",
    "logging": {
        "dir": "./logs",
        "level": "INFO"
    },
    "event_config": {
        "intrusion": {
            "enabled": false,
            "cooldown_sec": 600
        },
        "illegal_parking": {
            "enabled": false,
            "cooldown_sec": 600,
            "trigger_sec": 5.0,
            "move_threshold_ratio": 0.1
        },
        "no_helmet": {
            "enabled": false,
            "cooldown_sec": 600,
            "blur_face": true,
            "trigger_sec": 3.0
        },
        "conveyor_crossing": {
            "enabled": false,
            "cooldown_sec": 600,
            "snapshot_mode": "crossing_moment",
            "distance_ratio": 0.5,
            "min_crossing_angle": 20.0,
            "candidate_ttl_sec": 5.0
        },
        "signal_vehicle": {
            "enabled": false,
            "cooldown_sec": 600,
            "motion_threshold_ratio": 0.10
        }
    },
    "models": {
        "MAIN": "hanjin_cctv.dxnn",
        "FACE": "yolov8m-face.dxnn",
        "HELMET": "helmet_3cls_v8.dxnn"
    },
    "model_confidences": {
        "MAIN": 0.6,
        "FACE": 0.35,
        "HELMET": 0.55
    },
    "BATCH_SIZE": 9,
    "REC_FPS": 3,
    "REC_PRE_SEC": 10,
    "REC_POST_SEC": 10,
    "VISUAL_ALARM_DURATION": 5.0
}
EOL
    echo "✅ 기본 'system_config.json' 생성이 완료되었습니다."
else
    echo "ℹ️ 기존 'system_config.json' 파일이 확인되어 생성 단계를 건너뜁니다."
fi

echo "-----------------------------------------------------"
# 4. 환경 검증 및 자동 복구
echo "-> dx_engine 라이브러리가 정상 작동하는 최적의 환경을 탐색합니다..."
DETECTED_MODE=""

if [ -f "$VENV_ACTIVATE" ] && sudo -u "$ACTUAL_USER" -i bash -c "source $VENV_ACTIVATE && python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "✅ [검증 성공] 가상환경(VENV)에서 'dx_engine'이 정상적으로 로드됩니다."
    DETECTED_MODE="VENV"
elif sudo -u "$ACTUAL_USER" -i bash -c "python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "⚠️ [폴백 안내] 로컬(Global) 환경에서 'dx_engine' 구동이 확인되었습니다."
    DETECTED_MODE="GLOBAL"
else
    echo "====================================================="
    echo "⚠️ [자동 복구] 시스템에서 'dx_engine'을 찾을 수 없습니다."
    echo "   -> DeepX Runtime(DXRT) 및 NPU 드라이버 자동 설치를 시작합니다..."
    
    DOWNLOAD_DIR="$USER_HOME/Downloads"
    sudo -u "$ACTUAL_USER" mkdir -p "$DOWNLOAD_DIR"

    echo "   [1/2] 패키지 다운로드 중..."
    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/refs/heads/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb
    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt/raw/refs/heads/main/release/3.3.2/libdxrt_3.3.2_all.deb

    echo "   [2/2] 패키지 설치 중 (시간이 소요될 수 있습니다)..."
    sudo apt install -y "$DOWNLOAD_DIR/dxrt-driver-dkms_2.4.0-2_all.deb"
    sudo apt install -y "$DOWNLOAD_DIR/libdxrt_3.3.2_all.deb"
    
    if sudo -u "$ACTUAL_USER" -i bash -c "python3 -c 'import dx_engine'" > /dev/null 2>&1; then
        echo "✅ [복구 성공] 런타임 설치 후 로컬(Global) 환경에서 'dx_engine' 구동이 확인되었습니다."
        DETECTED_MODE="GLOBAL"
    else
        echo "❌ [설치 중단] 드라이버 설치 후에도 모듈 로드 실패. 재부팅이 필요합니다."
        exit 1
    fi
fi

# 5. Systemd 서비스 파일 생성
echo "-> Systemd 서비스 파일을 생성 중입니다..."
sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
Requires=dxrt.service
After=network.target dxrt.service graphical.target
# system_config.json 이 존재해야 서비스가 실행되도록 조건 변경
ConditionPathExists=$SYS_CONFIG_FILE
ConditionPathExists=$CAM_CSV_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStartPre=/bin/sleep 20
EOL

if [ "$DETECTED_MODE" = "VENV" ]; then
    sudo bash -c "echo \"ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'\" >> $SERVICE_PATH"
    
    BASHRC_PATH="$USER_HOME/.bashrc"
    AUTO_ACTIVATE_STR="source $VENV_ACTIVATE"
    if ! grep -Fxq "$AUTO_ACTIVATE_STR" "$BASHRC_PATH"; then
        echo -e "\n# DeepX Runtime 가상환경 자동 활성화\n$AUTO_ACTIVATE_STR" >> "$BASHRC_PATH"
        echo "ℹ️ ~/.bashrc에 가상환경 자동 활성화 코드를 등록했습니다."
    fi
else
    sudo bash -c "echo \"ExecStart=/bin/bash -c 'yes n | python3 -u multi_event.py'\" >> $SERVICE_PATH"
fi

sudo bash -c "cat >> $SERVICE_PATH" << EOL
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOL

# 6. 서비스 활성화 및 구동
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl stop $SERVICE_NAME

CAM_CSV_FILE="$PROJECT_DIR/cameras.csv"
if [ ! -f "$CAM_CSV_FILE" ]; then
    echo "⚠️ 경고: 'cameras.csv' 파일이 없습니다. RTSP URL 목록이 없으면 서비스가 즉시 종료됩니다."
    echo "   -> $CAM_CSV_FILE 파일을 생성한 뒤 sudo systemctl start $SERVICE_NAME 를 실행하세요."
fi

if [ -f "$CAM_CONFIG_FILE" ]; then
    sudo systemctl start $SERVICE_NAME
    echo "▶️ CCTV AI 백그라운드 서비스가 구동되었습니다."
else
    echo "⚠️ 'cameras.json' 파일이 없어 서비스가 대기 모드로 진입합니다. (카메라 설정 후 sudo systemctl start $SERVICE_NAME)"
fi

echo "====================================================="
echo " 🎉 환경 커스텀 설치 및 백그라운드 구동이 완료되었습니다!"
echo " - 시스템 설정 파일 : $SYS_CONFIG_FILE"
echo " - 적용된 환경      : $DETECTED_MODE 모드"
echo "====================================================="