#!/bin/bash

echo "=========================================="
echo " 1. Tailscale 설치 시작 (설치까지만)"
echo "=========================================="
# 필수 패키지 설치
sudo apt update && sudo apt install -y curl gpg

# GPG 키 다운로드 및 추가 (주소 오타 수정됨)
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$(lsb_release -cs).noarmor.gpg" | sudo tee /usr/share/keyrings/tailscale.gpg > /dev/null

# Tailscale 공식 저장소 추가
echo "deb [signed-by=/usr/share/keyrings/tailscale.gpg] https://pkgs.tailscale.com/stable/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tailscale.list

# 패키지 업데이트 및 Tailscale 설치
sudo apt update && sudo apt install -y tailscale

echo "=========================================="
echo " 2. Ubuntu 원격 데스크톱 (RDP) 설정 시작"
echo "=========================================="
# 인증서 생성 경로 (절대 경로)
CERT_DIR="/home/asus"
CRT_FILE="$CERT_DIR/rdp.crt"
KEY_FILE="$CERT_DIR/rdp.key"

# 혹시 모를 기존 인증서 파일 덮어쓰기 방지를 위해 삭제
rm -f $CRT_FILE $KEY_FILE

# 10년짜리 새 보안 인증서 및 키 생성 (2048비트)
openssl req -x509 -newkey rsa:2048 -nodes -keyout $KEY_FILE -out $CRT_FILE -days 3650 -subj "/C=KR/O=Home/CN=ubuntu-rdp"

# 인증서 및 키 권한 설정 (보안 유지)
chmod 600 $KEY_FILE
chmod 644 $CRT_FILE

# RDP 설정에 인증서 등록 (순서 중요: crt 먼저, key 나중에)
grdctl rdp set-tls-cert $CRT_FILE
grdctl rdp set-tls-key $KEY_FILE

# 계정 및 비밀번호 설정 (asus / 1234 통일)
grdctl rdp set-credentials asus 1234

# 보기 전용 모드 해제 (마우스/키보드 입력 허용)
gsettings set org.gnome.desktop.remote-desktop.rdp view-only false

# RDP 활성화 및 백그라운드 서비스 재시작
grdctl rdp enable
systemctl --user restart gnome-remote-desktop

echo "=========================================="
echo " 모든 세팅이 완료되었습니다! 🚀"
echo "=========================================="
