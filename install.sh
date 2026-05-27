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
echo " Raspberry Pi Edge AI CCTV 서비스 설치 및 자동화"
echo "====================================================="

# 3. 파일 및 환경 검증
if [ ! -f "$TARGET_SCRIPT" ] || [ ! -d "$DX_DIR" ] || [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ [설치 중단] 필수 파일이나 가상환경 경로를 찾을 수 없습니다."
    exit 1
fi

# 4. [기능 1] 유저가 터미널을 켜자마자 가상환경이 자동 활성화되도록 ~/.bashrc 설정
echo "-> 터미널 시작 시 가상환경 자동 활성화 설정을 확인합니다..."
BASHRC_PATH="$USER_HOME/.bashrc"
AUTO_ACTIVATE_STR="source $VENV_ACTIVATE"

if ! grep -Fxq "$AUTO_ACTIVATE_STR" "$BASHRC_PATH"; then
    echo -e "\n# DeepX Runtime 가상환경 자동 활성화\n$AUTO_ACTIVATE_STR" >> "$BASHRC_PATH"
    echo "✅ ~/.bashrc에 가상환경 자동 활성화 등록 완료!"
else
    echo "ℹ️ 이미 가상환경 자동 활성화가 등록되어 있습니다."
fi

# 5. [기능 2] Systemd 서비스 파일 생성 (온전한 부팅 후 실행되도록 튜닝)
echo "-> Systemd 서비스 파일을 생성 중입니다..."

sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
Requires=dxrt.service
# dxrt.service와 모든 그래픽/멀티미디어 환경(graphical.target)이 완료된 후 실행 요청
After=network.target dxrt.service graphical.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR

# [핵심 수정] 온전한 부팅 후 dxrt 데몬이 완전히 준비될 시간을 벌기 위해 20초 지연 후 파이썬 실행
ExecStartPre=/bin/sleep 20
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'

Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOL

# 6. 데몬 리로드 및 서비스 등록
echo "-> 서비스를 시스템에 등록합니다..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 기존 서비스 중지 및 초기화
sudo systemctl stop $SERVICE_NAME

if [ -f "$CONFIG_FILE" ]; then
    echo "▶️ 서비스를 백그라운드로 실행합니다. (설정된 20초 지연 후 실구동 시작)"
    sudo systemctl start $SERVICE_NAME
else
    echo "⚠️ cameras.json 파일이 없어 서비스가 대기 상태로 설정됩니다."
fi

echo "====================================================="
echo " 🎉 모든 설정 및 설치가 완료되었습니다!"
echo "-----------------------------------------------------"
echo " 1. 지금 터미널 창을 새로 열면 가상환경이 자동으로 켜집니다."
echo " 2. 시스템이 리부팅되면 NPU 데몬 안정화를 위해 20초 뒤"
echo "    CCTV AI 프로세스가 자동으로 완벽하게 구동됩니다."
echo "====================================================="
