#!/bin/bash

# 1. sudo로 실행하더라도 원래 계정 이름과 홈 디렉토리 경로를 정확히 추출
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
PROJECT_DIR=$(pwd)

# 2. 서비스 및 주요 경로 설정
SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CONFIG_FILE="$PROJECT_DIR/cameras.json"
TARGET_SCRIPT="$PROJECT_DIR/multi_event.py"
DX_DIR="$USER_HOME/dx-runtime"
VENV_ACTIVATE="$DX_DIR/venv-dx-runtime/bin/activate"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치 준비"
echo "====================================================="
echo "프로젝트 경로 : $PROJECT_DIR"
echo "실행 계정     : $ACTUAL_USER"
echo "유저 홈 경로  : $USER_HOME"
echo "가상환경 경로 : $VENV_ACTIVATE"
echo "====================================================="

# 3. [추가됨] 필수 파일 및 디렉토리 사전 검증 로직
echo "-> 필수 시스템 파일 및 환경을 검증합니다..."

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    echo "   -> 스크립트 실행 위치: $PROJECT_DIR"
    echo "   -> 해결 방법: 파이썬 코드가 있는 정확한 프로젝트 폴더로 이동한 뒤 다시 실행해주세요."
    exit 1
fi

if [ ! -d "$DX_DIR" ]; then
    echo "❌ [설치 중단] 사용자 홈 디렉토리에 'dx-runtime' 폴더가 존재하지 않습니다."
    echo "   -> 확인 경로: $DX_DIR"
    echo "   -> 해결 방법: DeepX 런타임이 해당 계정에 올바르게 설치되어 있는지 확인해주세요."
    exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ [설치 중단] 가상환경 활성화(activate) 파일을 찾을 수 없습니다."
    echo "   -> 확인 경로: $VENV_ACTIVATE"
    echo "   -> 해결 방법: dx-runtime 폴더 내에 가상환경이 정상적으로 구성되었는지 확인해주세요."
    exit 1
fi

echo "✅ 모든 필수 파일 및 환경 검증 완료."
echo "====================================================="

# 4. Systemd 서비스 파일 생성
echo "-> Systemd 서비스 파일을 생성 중입니다..."

sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection (Venv Sourced)
After=network.target
ConditionPathExists=$CONFIG_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR

# bash 쉘을 열고 유저 홈 디렉토리의 activate를 source 한 뒤 파이썬 실행
# yes n: input() 대기 무한 패스 / -u: 실시간 로그 출력
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOL

# 5. Systemd 데몬 리로드 및 서비스 활성화
echo "-> Systemd 데몬을 리로드하고 서비스를 등록합니다..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 6. 기존 서비스 중지 및 재시작
echo "-> 기존 서비스를 정리하고 새로 시작합니다..."
sudo systemctl stop $SERVICE_NAME

if [ -f "$CONFIG_FILE" ]; then
    sudo systemctl start $SERVICE_NAME
    echo "-> 서비스가 성공적으로 시작되었습니다."
else
    echo "-> ⚠️ [알림] cameras.json 파일이 없어 서비스가 대기 상태로 유지됩니다."
    echo "   -> 추후 설정 완료 후 'sudo systemctl start $SERVICE_NAME'을 입력하세요."
fi

echo "====================================================="
echo " 설치 작업이 완료되었습니다!"
echo " - 상태 확인: sudo systemctl status $SERVICE_NAME"
echo " - 실시간 로그 확인: journalctl -u $SERVICE_NAME -f"
echo "====================================================="
