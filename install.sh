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
echo " Raspberry Pi Edge AI CCTV 서비스 설치 (자동 복구 지원)"
echo "====================================================="

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

# 3. 가상환경 및 로컬 환경 검증
echo "-> dx_engine 라이브러리가 정상 작동하는 최적의 환경을 탐색합니다..."

DETECTED_MODE=""

# [우선순위 1] 가상환경 검사
if [ -f "$VENV_ACTIVATE" ] && sudo -u "$ACTUAL_USER" -i bash -c "source $VENV_ACTIVATE && python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "✅ [검증 성공] 가상환경(VENV)에서 'dx_engine'이 정상적으로 로드됩니다."
    DETECTED_MODE="VENV"

# [우선순위 2] 로컬(전역) 환경 검사
elif sudo -u "$ACTUAL_USER" -i bash -c "python3 -c 'import dx_engine'" > /dev/null 2>&1; then
    echo "⚠️ [폴백 안내] 로컬(Global) 환경에서 'dx_engine' 구동이 확인되었습니다."
    DETECTED_MODE="GLOBAL"

# [우선순위 3] 모두 실패 시: DeepX 런타임 및 드라이버 자동 설치 시도
else
    echo "====================================================="
    echo "⚠️ [자동 복구] 시스템에서 'dx_engine'을 찾을 수 없습니다."
    echo "   -> DeepX Runtime(DXRT) 및 NPU 드라이버 자동 설치를 시작합니다..."
    
    DOWNLOAD_DIR="$USER_HOME/Downloads"
    sudo -u "$ACTUAL_USER" mkdir -p "$DOWNLOAD_DIR"

    echo "   [1/2] 패키지 다운로드 중 (wget)..."
    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/refs/heads/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb
    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt/raw/refs/heads/main/release/3.3.2/libdxrt_3.3.2_all.deb

    echo "   [2/2] 패키지 설치 중 (apt install)..."
    sudo apt install -y "$DOWNLOAD_DIR/dxrt-driver-dkms_2.4.0-2_all.deb"
    sudo apt install -y "$DOWNLOAD_DIR/libdxrt_3.3.2_all.deb"
    
    echo "-> 런타임 설치 완료! 환경을 재검증합니다..."
    # 설치 후 로컬 환경 재검사
    if sudo -u "$ACTUAL_USER" -i bash -c "python3 -c 'import dx_engine'" > /dev/null 2>&1; then
        echo "✅ [복구 성공] 런타임 설치 후 로컬(Global) 환경에서 'dx_engine' 구동이 확인되었습니다."
        DETECTED_MODE="GLOBAL"
    else
        echo "❌ [설치 중단] 런타임 자동 설치 후에도 모듈을 로드할 수 없습니다. 기기를 재부팅한 후 다시 시도해주세요."
        exit 1
    fi
fi

echo "-----------------------------------------------------"
echo "-> 시스템 서비스 파일($SERVICE_PATH) 생성을 시작합니다..."

# 4. Systemd 서비스 파일 베이스 작성
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

# 5. 최종 확정된 모드에 맞춰 구문 분기 주입
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

# 6. 서비스 파일 마무리 작성
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

sudo systemctl stop $SERVICE_NAME

if [ -f "$CONFIG_FILE" ]; then
    sudo systemctl start $SERVICE_NAME
    echo "▶️ CCTV AI 백그라운드 서비스가 구동되었습니다."
else
    echo "⚠️ 'cameras.json' 파일이 없어 서비스가 대기 모드로 진입합니다."
fi

echo "====================================================="
echo " 🎉 최종 설치가 완료되었습니다!"
echo " - 적용된 구동 환경: $DETECTED_MODE 모드"
echo "====================================================="