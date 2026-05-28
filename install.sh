#!/bin/bash

# 1. 사용자 및 경로 설정
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
PROJECT_DIR=$(pwd)

# 2. 서비스 및 주요 경로 설정
SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CONFIG_FILE="$PROJECT_DIR/cameras.json"
TARGET_SCRIPT="$PROJECT_DIR/multi_event.py"

# 가상환경 예상 경로
DX_DIR="$USER_HOME/dx-runtime"
VENV_ACTIVATE="$DX_DIR/venv-dx-runtime/bin/activate"
VENV_PYTHON="$DX_DIR/venv-dx-runtime/bin/python"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치 (환경 자동 감지)"
echo "====================================================="

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

# 3. [핵심] dx_engine 로드 가능 환경 감지 테스트
echo "-> dx_engine 라이브러리를 로드할 수 있는 Python 환경을 탐색합니다..."

EXEC_START_CMD=""
IS_VENV_USED=false

# 케이스 A: 전역(Global) 환경 테스트
if sudo -u "$ACTUAL_USER" bash -c "python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "✅ [통과] 전역(Global) Python 환경에서 dx_engine 로드가 확인되었습니다."
    # 전역 환경용 실행 명령어 
    EXEC_START_CMD="/bin/bash -c 'yes n | python3 -u multi_event.py'"
    
# 케이스 B: 가상환경(Venv) 테스트
elif [ -f "$VENV_PYTHON" ] && sudo -u "$ACTUAL_USER" bash -c "source $VENV_ACTIVATE && python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "✅ [통과] 가상환경(venv) Python 환경에서 dx_engine 로드가 확인되었습니다."
    IS_VENV_USED=true
    # 가상환경용 실행 명령어
    EXEC_START_CMD="/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'"
    
# 케이스 C: 모두 실패
else
    echo "❌ [설치 중단] 전역 및 가상환경 모두에서 dx_engine을 찾을 수 없습니다."
    echo "   -> DeepX 런타임이 올바르게 설치되었는지 확인해주세요."
    exit 1
fi

# 4. 가상환경을 사용하는 경우에만 ~/.bashrc에 자동 활성화 등록
if [ "$IS_VENV_USED" = true ]; then
    BASHRC_PATH="$USER_HOME/.bashrc"
    AUTO_ACTIVATE_STR="source $VENV_ACTIVATE"
    if ! grep -Fxq "$AUTO_ACTIVATE_STR" "$BASHRC_PATH"; then
        echo -e "\n# DeepX Runtime 가상환경 자동 활성화\n$AUTO_ACTIVATE_STR" >> "$BASHRC_PATH"
        echo "ℹ️ ~/.bashrc에 가상환경 자동 활성화 코드를 추가했습니다."
    fi
fi

# 5. Systemd 서비스 파일 생성
echo "-> Systemd 서비스 파일을 생성 중입니다..."

# $EXEC_START_CMD 변수가 환경에 맞게 자동으로 치환되어 들어갑니다.
sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
Requires=dxrt.service
After=network.target dxrt.service graphical.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStartPre=/bin/sleep 20
ExecStart=$EXEC_START_CMD
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

sudo systemctl stop $SERVICE_NAME

if [ -f "$CONFIG_FILE" ]; then
    echo "▶️ 서비스를 백그라운드로 실행합니다."
    sudo systemctl start $SERVICE_NAME
else
    echo "⚠️ cameras.json 파일이 없어 서비스가 대기 상태로 설정됩니다."
fi

echo "====================================================="
echo " 🎉 모든 설정 및 설치가 완료되었습니다!"
echo " - 적용된 실행 방식: $EXEC_START_CMD"
echo " - 상태 확인: sudo systemctl status $SERVICE_NAME"
echo "====================================================="
