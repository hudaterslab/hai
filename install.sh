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
echo " Raspberry Pi Edge AI CCTV 서비스 설치 및 사전 검증"
echo "====================================================="
echo "프로젝트 경로 : $PROJECT_DIR"
echo "실행 계정     : $ACTUAL_USER"
echo "유저 홈 경로  : $USER_HOME"
echo "가상환경 경로 : $VENV_ACTIVATE"
echo "====================================================="

# 3. 필수 파일 및 디렉토리 존재 여부 1차 검증
echo "-> [1단계] 필수 시스템 파일 존재 여부를 확인합니다..."

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

if [ ! -d "$DX_DIR" ]; then
    echo "❌ [설치 중단] 사용자 홈 디렉토리에 'dx-runtime' 폴더가 존재하지 않습니다."
    exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ [설치 중단] 가상환경 활성화(activate) 파일을 찾을 수 없습니다."
    exit 1
fi

echo "✅ 1단계 검증 통과 (파일 확인 완료)"
echo "-----------------------------------------------------"

# 4. [핵심 추가] 가상환경 모의 실행 및 dx_engine 로드 테스트
echo "-> [2단계] 가상환경을 활성화하여 실제 구동 테스트를 진행합니다..."
echo "          (dx_engine 로드 및 파이썬 환경 이상 여부 확인)"

# 가상환경을 켜고 파이썬을 실행하되, 실제 메인 루프에 진입하기 전 컴파일 및 초기 임포트 에러가 나는지 확인
# (실행 직후 바로 종료되도록 파이프라인 우회 처리 환경 구성)
/bin/bash -c "source $VENV_ACTIVATE && echo -e 'n\nn' | python3 $TARGET_SCRIPT" > /dev/null 2>&1

# 파이썬 실행 결과(Exit Code) 확인
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "====================================================="
    echo "❌ [설치 실패] 가상환경에서 파이썬 실행 중 오류가 발생했습니다!"
    echo "   -> 'dx_engine'을 읽어오지 못했거나 의존성 에러가 있을 수 있습니다."
    echo "   -> 수동 명령어 'source $VENV_ACTIVATE && python3 multi_event.py' 로 에러를 먼저 확인하세요."
    echo "====================================================="
    
    # 만약 기존에 서비스가 등록되어 작동 중이었다면 안전하게 끄고 비활성화
    if [ -f "$SERVICE_PATH" ]; then
        echo "-> 기존에 등록되어 있던 서비스를 중지 및 비활성화(disable) 처리합니다."
        sudo systemctl stop $SERVICE_NAME > /dev/null 2>&1
        sudo systemctl disable $SERVICE_NAME > /dev/null 2>&1
    fi
    
    echo "🛑 제대로 서비스가 등록되지 않았습니다. 환경을 재점검해주세요."
    exit 1
fi

echo "✅ 2단계 검증 통과 (가상환경 정상 작동 확인)"
echo "====================================================="

# 5. Systemd 서비스 파일 생성
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

# 가상환경을 적용하여 백그라운드 구동
ExecStart=/bin/bash -c 'source $VENV_ACTIVATE && yes n | python3 -u multi_event.py'

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOL

# 6. Systemd 데몬 리로드 및 서비스 활성화
echo "-> Systemd 데몬을 리로드하고 서비스를 등록합니다..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# 7. 기존 서비스 중지 및 재시작
echo "-> 서비스를 백그라운드로 구동합니다..."
sudo systemctl stop $SERVICE_NAME

if [ -f "$CONFIG_FILE" ]; then
    sudo systemctl start $SERVICE_NAME
    echo "▶️ 백그라운드 서비스가 성공적으로 시작되었습니다."
else
    echo "⚠️ [알림] cameras.json 파일이 없어 서비스가 대기 상태(Active 만 유지)로 대기합니다."
fi

echo "====================================================="
echo " 🎉 최종 설치 및 검증이 완료되었습니다!"
echo " - 상태 확인: sudo systemctl status $SERVICE_NAME"
echo " - 로그 확인: journalctl -u $SERVICE_NAME -f"
echo "====================================================="