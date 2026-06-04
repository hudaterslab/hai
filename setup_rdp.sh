#!/bin/bash

echo "=========================================="
echo " 1. Tailscale 설치 시작 (설치까지만)"
echo "=========================================="
sudo apt update && sudo apt install -y curl gpg
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$(lsb_release -cs).noarmor.gpg" | sudo tee /usr/share/keyrings/tailscale.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/tailscale.gpg] https://pkgs.tailscale.com/stable/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale

echo "=========================================="
echo " 2. Ubuntu 원격 데스크톱 (RDP) 기본 설정"
echo "=========================================="
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

echo "=========================================="
echo " 3. RDP 속도 최적화 및 절전/잠금 방지 설정"
echo "=========================================="
# 화면 꺼짐 및 자동 잠금 비활성화
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false

# 자동 절전모드 방지
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'

# 애니메이션 비활성화 (반응속도 대폭 향상)
gsettings set org.gnome.desktop.interface enable-animations false

# 시스템 전체 절전 모드 마스킹(원천 차단 - sudo 권한 필요)
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

echo "=========================================="
echo " 4. RDP 서비스 활성화 및 재시작 적용"
echo "=========================================="
grdctl rdp enable
systemctl --user restart gnome-remote-desktop

echo "=========================================="
echo " 🎉 모든 세팅과 최적화가 한 번에 완료되었습니다!"
echo "=========================================="