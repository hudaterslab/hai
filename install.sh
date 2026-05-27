#!/bin/bash

# 서비스 이름 및 파일 경로 설정
SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

# 현재 디렉토리 및 사용자 정보 가져오기
PROJECT_DIR=$(pwd)
CURRENT_USER=$USER
PYTHON_EXEC=$(which python3)

# 카메라 설정 파일 경로
CONFIG_FILE="$PROJECT_DIR/cameras.json"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치를 시작합니다."
echo "====================================================="
echo "프로젝트 경로: $PROJECT_DIR"
echo "실행 계정: $CURRENT_USER"
echo "설정 파일 검사 경로: $CONFIG_FILE"
echo "====================================================="

# 1. Systemd 서비스 파일 생성
echo "-> Systemd 서비스 파일을 생성 중입니다..."

sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection Background Service
After=network.target
# 이 조건 덕분에 cameras.json 파일이 없으면 서비스 자체가 시작되지 않습니다.
ConditionPathExists=$CONFIG_FILE

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
# 파이썬 코드의 input()을 우회하기 위해 'n' 두 개를 연속으로 입력합니다.
# (디버그 모드 N -> 기존 설정 무시 N)
ExecStart=/bin/bash -c 'echo -e "n\nn" | $PYTHON_EXEC multi_event.py'

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

# 3. 서비스 시작 시도 (cameras.json 존재 여부에 따라 다르게 동작)
if [ -f "$CONFIG_FILE" ]; then
    echo "-> cameras.json 파일이 확인되어 서비스를 즉시 시작합니다."
    sudo systemctl start $SERVICE_NAME
else
    echo "-> [알림] cameras.json 파일이 존재하지 않아 서비스가 대기 상태로 유지됩니다."
    echo "-> 나중에 'python3 multi_event.py'를 직접 실행하여 설정을 완료한 후,"
    echo "-> 'sudo systemctl start $SERVICE_NAME' 명령어로 서비스를 수동 시작해주세요."
fi

echo "====================================================="
echo " 설치가 완료되었습니다!"
echo "====================================================="
echo " [유용한 명령어 모음]"
echo " - 상태 확인: sudo systemctl status $SERVICE_NAME"
echo " - 로그 보기: journalctl -u $SERVICE_NAME -f"
echo " - 서비스 시작: sudo systemctl start $SERVICE_NAME"
echo " - 서비스 중지: sudo systemctl stop $SERVICE_NAME"
echo "====================================================="
