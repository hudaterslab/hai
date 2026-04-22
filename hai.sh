#!/bin/bash

# ==========================================
# [HAI] Hudaters AI CCTV Solution - Production
# ==========================================

echo "========================================"
echo "🛡️ [HAI] Hudaters AI CCTV 솔루션 부팅 중..."
echo "========================================"

# 1. 필수 패키지 확인
for pkg in jq sshpass sshfs tailscale; do
    if ! command -v $pkg &> /dev/null; then
        echo "⚠️ [HAI] 오류: $pkg 패키지가 설치되어 있지 않습니다."
        exit 1
    fi
done

# ==========================================
# 단말기 ID 초기화
# ==========================================
mkdir -p "$(pwd)/config"
CONFIG_FILE="$(pwd)/config/system_config.json"

if [ ! -f "$CONFIG_FILE" ] || ! jq -e '.terminal_id' "$CONFIG_FILE" > /dev/null 2>&1; then
    echo "⚠️ 초기 단말기 ID 설정이 필요합니다. (새 장비 감지)"
    read -t 15 -p ">> 할당할 단말기 ID를 입력하세요 (15초 대기, 미입력 시 99999): " INPUT_ID
    echo ""

    if [ -z "$INPUT_ID" ]; then
        TERMINAL_ID="99999"
        echo "⏳ 시간 초과. 테스트 모드(99999)로 부팅합니다."
    else
        TERMINAL_ID="$INPUT_ID"
        echo "✅ 단말기 ID가 '$TERMINAL_ID' (으)로 설정되었습니다."
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "{\"terminal_id\": \"$TERMINAL_ID\"}" > "$CONFIG_FILE"
    else
        jq ".terminal_id = \"$TERMINAL_ID\"" "$CONFIG_FILE" > tmp.$$.json && mv tmp.$$.json "$CONFIG_FILE"
    fi
else
    TERMINAL_ID=$(jq -r '.terminal_id' "$CONFIG_FILE")
    echo "📍 단말기 ID 인식 완료: $TERMINAL_ID"
fi

# ==========================================
# Tailscale VPN 연결
# ==========================================
echo "🌐 Tailscale VPN 연결을 시도합니다..."
TAILSCALE_AUTH_KEY="tskey-auth-kvXAjbeFDF11CNTRL-4q2wsZzWaabmKTBzS6ieab9Baru2nCxT8"

if tailscale status &> /dev/null; then
    echo "✅ Tailscale 연결 정상."
else
    sudo tailscale up --authkey="$TAILSCALE_AUTH_KEY" --accept-routes
    if [ $? -eq 0 ]; then
        echo "✅ Tailscale VPN 연결 성공."
    else
        echo "⚠️ Tailscale 연결 실패."
        exit 1
    fi
fi

# ==========================================
# 💡 [핵심] NAS 사전 마운트 및 확실한 검증
# ==========================================
NAS_ID="ai_cctv"
NAS_PW='"n+qJqz2'
NAS_IP="100.65.15.87"
NAS_PORT="21422"
MOUNT_POINT="$HOME/hai_nas"

mkdir -p "$MOUNT_POINT"

# 비정상 마운트 찌꺼기 제거
if grep -qs "$MOUNT_POINT" /proc/mounts || [ ! -r "$MOUNT_POINT" ]; then
    fusermount -uz "$MOUNT_POINT" 2>/dev/null
    sleep 1
fi

echo "🔄 NAS($NAS_IP) 마운트 중..."

# sshpass 대신 안정적인 파이프 입력(password_stdin) 사용 (비밀번호 꼬임 원천 차단)
echo "$NAS_PW" | sshfs -p $NAS_PORT -o password_stdin,StrictHostKeyChecking=no,reconnect "$NAS_ID@$NAS_IP:/" "$MOUNT_POINT"

# 백그라운드 죽음을 판별하기 위해 2초 대기
sleep 2

# NAS 내부의 고유 폴더(hudaters03)가 실제로 보이는지 교차 검증
if [ -d "$MOUNT_POINT/hudaters03" ]; then
    echo "✅ NAS 실제 마운트 검증 완료! (경로: $MOUNT_POINT)"
else
    echo "⚠️ NAS 마운트가 조용히 실패했습니다(Silent Fail). 비밀번호나 네트워크를 확인하세요."
    fusermount -uz "$MOUNT_POINT" 2>/dev/null
fi

# ==========================================
# 메인 AI 프로세스 실행
# ==========================================
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
mkdir -p "$(pwd)/events"
mkdir -p "$(pwd)/CCTV_EVENT_ALERT/events"

python3 src/main.py

EXIT_CODE=$?

# ==========================================
# 프로그램 종료 시 안전한 마운트 해제
# ==========================================
echo "시스템 자원을 정리합니다..."
fusermount -uz "$MOUNT_POINT" 2>/dev/null

if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️ [HAI] 시스템 비정상 종료 (코드: $EXIT_CODE)"
else
    echo "✅ [HAI] 시스템 안전 종료"
fi

exit $EXIT_CODE