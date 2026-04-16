#!/bin/bash

# ==========================================
# [HAI] Hudaters AI CCTV Solution - Production
# ==========================================

echo "========================================"
echo "🛡️ [HAI] Hudaters AI CCTV 솔루션 부팅 중..."
echo "========================================"

# 소스 코드가 src 폴더에 있으므로 파이썬 모듈 경로를 환경 변수에 추가합니다.
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 메인 프로세스 실행 (작업 디렉토리는 Root 유지)
python3 src/main.py

# 프로세스 종료 시 상태 코드 반환
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️ [HAI] 시스템이 비정상 종료되었습니다. (코드: $EXIT_CODE)"
else
    echo "✅ [HAI] 시스템이 안전하게 종료되었습니다."
fi
exit $EXIT_CODE