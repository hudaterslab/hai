#!/bin/bash

# ==========================================
# [HAI] Hudaters AI CCTV Solution - Production
# ==========================================

echo "========================================"
echo "🛡️ [HAI] Hudaters AI CCTV 솔루션 부팅 중..."
echo "========================================"

# 1. 필수 패키지 확인 (네트워크 관련 의존성 제거)
for pkg in jq; do
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
# Tailscale VPN 연결 (주석 처리됨)
# ==========================================
# echo "🌐 Tailscale VPN 연결을 시도합니다..."
# TAILSCALE_AUTH_KEY="tskey-auth-kvXAjbeFDF11CNTRL-4q2wsZzWaabmKTBzS6ieab9Baru2nCxT8"
# 
# if tailscale status &> /dev/null; then
#     echo "✅ Tailscale 연결 정상."
# else
#     sudo tailscale up --authkey="$TAILSCALE_AUTH_KEY" --accept-routes
#     if [ $? -eq 0 ]; then
#         echo "✅ Tailscale VPN 연결 성공."
#     else
#         echo "⚠️ Tailscale 연결 실패."
#         exit 1
#     fi
# fi

# ==========================================
# 메인 AI 프로세스 실행
# ==========================================
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
mkdir -p "$(pwd)/events"
mkdir -p "$(pwd)/CCTV_EVENT_ALERT/events"

python3 src/main.py

EXIT_CODE=$?

# ==========================================
# 프로그램 종료 처리
# ==========================================
echo "시스템 자원을 정리합니다..."

if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️ [HAI] 시스템 비정상 종료 (코드: $EXIT_CODE)"
else
    echo "✅ [HAI] 시스템 안전 종료"
fi

exit $EXIT_CODE