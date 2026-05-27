#!/bin/bash

# 서비스 이름 및 파일 경로 설정
SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

# sudo로 실행하더라도 원래 계정 이름과 홈 디렉토리 경로를 정확히 가져옴
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
PROJECT_DIR=$(pwd)

# 카메라 설정 파일 및 가상환경 경로
CONFIG_FILE="$PROJECT_DIR/cameras.json"
VENV_ACTIVATE="$USER_HOME/dx-runtime/venv-dx-runtime/bin/activate"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치 (가상환경 버전)"
echo "====================================================="
echo "프로젝트 경로: $PROJECT_DIR"
echo "실행 계정: $ACTUAL_USER ($USER_HOME)"
echo "가상환경 경로: $VENV_ACTIVATE"
echo "====================================================="

# 가상환경 파일 존재 여부 1차 검증
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "[경고] 가상환경 활성화 파일을 찾을 수 없습니다: $VENV_ACTIVATE"
    echo "경로가 정확한지 확인 후 다시 실행해주세요."
    exit 1
fi

# 1. Systemd 서비스 파일 생성
echo "-> Systemd 서비스 파일을 생성 중입니다..."

sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection (Venv)
After=network.target
ConditionPathExists=$CONFIG_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR

# 1) source 명령어로 가상환경을 켜고
# 2) 파이프라인으로 input()을 자동 통과시키며 python을 실행합니다.
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && echo -e "n\nn" | python3 multi_event.py'

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOL

# 2. Systemd 데몬 리로드 및 서비스 활성화
echo "-> Systemd 데몬을 리로드하고 서비스를 등록합니다..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 3. 서비스 시작 시도
if [ -f "$CONFIG_FILE" ]; then
    echo "-> cameras.json 파일이 확인되어 서비스를 즉시 시작합니다."
    sudo systemctl start $SERVICE_NAME
else
    echo "-> [알림] cameras.json 파일이 없어 서비스가 대기 상태로 유지됩니다."
fi

echo "====================================================="
echo " 설치가 완료되었습니다!"
echo "====================================================="