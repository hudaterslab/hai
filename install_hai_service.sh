#!/usr/bin/env bash
set -euo pipefail

# === 사용자 설정 ===
USER_NAME="recomputer"
PROJECT_DIR="/home/$USER_NAME/hai"
SERVICE_NAME="hai-multi-event"

# === 가상환경 경로 자동 탐지 ===
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "가상환경이 활성화되어 있지 않습니다. 먼저 가상환경을 activate 해주세요."
    exit 1
fi
VENV_PATH="$(readlink -f "$VIRTUAL_ENV")"
PYTHON_BIN="$VENV_PATH/bin/python"

# === 실행 스크립트 생성 ===
RUN_SCRIPT="$PROJECT_DIR/run_multi_event.sh"

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_DIR"

source "$VENV_PATH/bin/activate"
exec "$PYTHON_BIN" -u "$PROJECT_DIR/multi_event.py"
EOF

chmod +x "$RUN_SCRIPT"
echo "[INFO] 실행 스크립트 생성 완료: $RUN_SCRIPT"

# === systemd 서비스 생성 ===
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=HAI Multi Event Detection Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$RUN_SCRIPT
Restart=always
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30
# 필요한 경우 권한 그룹 추가
# SupplementaryGroups=video render dialout gpio i2c

[Install]
WantedBy=multi-user.target
EOF

echo "[INFO] systemd 서비스 파일 생성 완료: $SERVICE_FILE"

# === systemd 데몬 리로드 & 서비스 등록 ===
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME.service"
sudo systemctl start "$SERVICE_NAME.service"

echo "[INFO] 서비스 시작 완료: $SERVICE_NAME"
echo "상태 확인: sudo systemctl status $SERVICE_NAME"
echo "실시간 로그 확인: sudo journalctl -u $SERVICE_NAME -f"