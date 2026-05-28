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

# 가상환경 경로 정의
DX_DIR="$USER_HOME/dx-runtime"
VENV_ACTIVATE="$DX_DIR/venv-dx-runtime/bin/activate"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 설치 (가상환경 우선 탐색)"
echo "====================================================="

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

# 3. [핵심] 환경 검증 (가상환경을 최우선으로 검사)
echo "-> 'import dx_engine'이 가능한 최적의 환경을 탐색합니다..."

DETECTED_MODE=""

# 케이스 1: 가상환경이 존재하고 그 안에서 dx_engine이 정상 임포트되는지 확인
if [ -f "$VENV_ACTIVATE" ] && sudo -u "$ACTUAL_USER" bash -c "source $VENV_ACTIVATE && python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "✅ [감지 성공] 가상환경(Venv) 환경에서 'dx_engine' 구동이 확인되었습니다."
    DETECTED_MODE="VENV"

# 케이스 2: 가상환경이 없거나 실패 시, 전역(Global) 환경 확인
elif sudo -u "$ACTUAL_USER" bash -c "python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "✅ [감지 성공] 전역(Global) Python 환경에서 'dx_engine' 구동이 확인되었습니다."
    DETECTED_MODE="GLOBAL"

# 케이스 3: 둘 다 실패
else
    echo "❌ [설치 중단] 전역 및 가상환경 모두에서 'dx_engine'을 로드할 수 없습니다."
    echo "   -> NPU 런타임 설치 상태를 다시 확인해주세요."
    exit 1
fi

echo "-----------------------------------------------------"
echo "-> 시스템 서비스 파일($SERVICE_PATH) 생성을 시작합니다..."

# 4. Systemd 서비스 파일 베이스 작성 (공통 부분)
sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
Requires=dxrt.service
After=network.target dxrt.service graphical.target
ConditionPathExists=$CONFIG_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStartPre=/bin/sleep 20
EOL

# 5. [중요] 감지된 환경에 맞는 ExecStart 구문을 따옴표 깨짐 없이 직접 주입
if [ "$DETECTED_MODE" = "VENV" ]; then
    # 가상환경 실행 구문 직접 주입
    sudo bash -c "echo \"ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'\" >> $SERVICE_PATH"
    
    # ~/.bashrc 자동 활성화 코드 추가 검사
    BASHRC_PATH="$USER_HOME/.bashrc"
    AUTO_ACTIVATE_STR="source $VENV_ACTIVATE"
    if ! grep -Fxq "$AUTO_ACTIVATE_STR" "$BASHRC_PATH"; then
        echo -e "\n# DeepX Runtime 가상환경 자동 활성화\n$AUTO_ACTIVATE_STR" >> "$BASHRC_PATH"
        echo "ℹ️ 사용 편의를 위해 ~/.bashrc에 가상환경 자동 활성화 코드를 등록했습니다."
    fi
else
    # 전역 실행 구문 직접 주입
    sudo bash -c "echo \"ExecStart=/bin/bash -c 'yes n | python3 -u multi_event.py'\" >> $SERVICE_PATH"
fi

# 6. 서비스 파일 마무리 작성 (공통 후반부)
sudo bash -c "cat >> $SERVICE_PATH" << EOL
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOL

# 7. 데몬 리로드 및 서비스 등록
echo "-> 시스템 설정을 반영하고 서비스를 시작합니다..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 기존 인스턴스 확실히 리셋 후 시작
sudo systemctl stop $SERVICE_NAME

if [ -f "$CONFIG_FILE" ]; then
    sudo systemctl start $SERVICE_NAME
    echo "▶️ CCTV AI 백그라운드 서비스가 성공적으로 구동되었습니다."
else
    echo "⚠️ 'cameras.json' 파일이 없어 서비스가 대기 모드로 진입합니다."
fi

echo "====================================================="
echo " 🎉 설치 및 환경 맞춤형 튜닝이 완료되었습니다!"
echo " - 탐색 및 적용된 모드: $DETECTED_MODE 모드"
echo " - 서비스 상태 확인   : sudo systemctl status $SERVICE_NAME"
echo "====================================================="
