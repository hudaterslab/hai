#!/bin/bash

# 1. 사용자 및 경로 설정
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
PROJECT_DIR=$(pwd)

# 2. 서비스 및 경로 설정
SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CONFIG_FILE="$PROJECT_DIR/cameras.json"
TARGET_SCRIPT="$PROJECT_DIR/multi_event.py"
DX_DIR="$USER_HOME/dx-runtime"
VENV_ACTIVATE="$DX_DIR/venv-dx-runtime/bin/activate"
VENV_PYTHON="$DX_DIR/venv-dx-runtime/bin/python"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치 (안전성 강화 버전)"
echo "====================================================="

# 3. 파일 검증 (실행 테스트 대신 파이썬 문법 에러만 빠르게 확인)
echo "-> 필수 파일 및 파이썬 문법 검사를 진행합니다..."

if [ ! -f "$TARGET_SCRIPT" ] || [ ! -d "$DX_DIR" ] || [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ [설치 중단] 필수 파일이나 가상환경 경로를 찾을 수 없습니다."
    exit 1
fi

# NPU에 올리지 않고 파이썬 코드의 치명적 오타/문법 에러만 컴파일 테스트 (안전함)
$VENV_PYTHON -m py_compile $TARGET_SCRIPT
if [ $? -ne 0 ]; then
    echo "❌ [설치 중단] 파이썬 코드(multi_event.py)에 문법 오류가 있습니다."
    exit 1
fi
echo "✅ 코드 검증 완료"

# 4. Systemd 서비스 파일 생성
echo "-> Systemd 서비스 파일을 생성 중입니다..."

sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
# [핵심 수정] dxrt.service(DeepX 런타임)가 켜진 이후에만 우리 서비스가 구동되도록 종속성 추가
Requires=dxrt.service
After=network.target dxrt.service
ConditionPathExists=$CONFIG_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOL

# 5. 데몬 리로드 및 서비스 등록
echo "-> 서비스를 시스템에 등록합니다..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

sudo systemctl stop $SERVICE_NAME
if [ -f "$CONFIG_FILE" ]; then
    sudo systemctl start $SERVICE_NAME
    echo "▶️ 백그라운드 서비스가 시작되었습니다."
else
    echo "⚠️ cameras.json 파일이 없어 대기 상태입니다."
fi

echo "====================================================="
echo " 🎉 완료되었습니다! (상태 확인: sudo systemctl status $SERVICE_NAME)"
echo "====================================================="
