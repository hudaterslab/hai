#!/bin/bash

# 현재 스크립트를 실행하는 사용자 이름 자동 획득 (예: hudaters)
ACTUAL_USER=$USER

echo "=========================================="
echo " 1. 기존 충돌 서비스(xrdp, systemd vnc) 초기화"
echo "=========================================="
# xrdp 완전 삭제
sudo systemctl stop xrdp 2>/dev/null || true
sudo apt-get purge -y xrdp xorgxrdp 2>/dev/null || true
sudo apt-get autoremove -y 2>/dev/null || true

# 기존 실패한 systemd VNC 서비스 정지 및 삭제
sudo systemctl stop remote-x11vnc.service 2>/dev/null || true
sudo systemctl disable remote-x11vnc.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/remote-x11vnc.service
sudo systemctl daemon-reload

echo "=========================================="
echo " 2. SDDM(Lubuntu) 자동 로그인 설정"
echo "=========================================="
sudo mkdir -p /etc/sddm.conf.d
echo -e "[Autologin]\nUser=$ACTUAL_USER\nSession=lxqt" | sudo tee /etc/sddm.conf.d/autologin.conf > /dev/null

echo "=========================================="
echo " 3. 필수 패키지 설치 (Tailscale & x11vnc)"
echo "=========================================="
sudo apt update && sudo apt install -y curl gpg x11vnc xauth net-tools

# Tailscale 설치
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$(lsb_release -cs).noarmor.gpg" | sudo tee /usr/share/keyrings/tailscale.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/tailscale.gpg] https://pkgs.tailscale.com/stable/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale

echo "=========================================="
echo " 4. 화면 꺼짐 방지 및 VNC 시작 프로그램(Autostart) 등록"
echo "=========================================="
# 시스템 레벨 수면 모드 차단
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target > /dev/null 2>&1

# Autostart 디렉토리 생성
mkdir -p /home/$ACTUAL_USER/.config/autostart

# 화면 꺼짐 방지(DPMS 비활성화) 등록
cat <<EOF > /home/$ACTUAL_USER/.config/autostart/disable-blanking.desktop
[Desktop Entry]
Type=Application
Name=Disable Screen Blanking
Exec=sh -c "xset -dpms && xset s off && xset s noblank"
Terminal=false
EOF

# VNC 비밀번호 설정
VNC_PASSWD="/home/$ACTUAL_USER/.vnc/passwd"
sudo install -d -m 700 -o "$ACTUAL_USER" -g "$ACTUAL_USER" "/home/$ACTUAL_USER/.vnc"

if [ ! -f "$VNC_PASSWD" ]; then
    echo "VNC 접속을 위한 비밀번호를 설정합니다."
    sudo -u "$ACTUAL_USER" x11vnc -storepasswd "$VNC_PASSWD"
fi
sudo chown "$ACTUAL_USER:$ACTUAL_USER" "$VNC_PASSWD"
sudo chmod 600 "$VNC_PASSWD"

# Tailscale IP 감지 (없으면 0.0.0.0)
VNC_LISTEN_ADDR="$(tailscale ip -4 2>/dev/null | awk 'NF {print; exit}')"
if [ -z "$VNC_LISTEN_ADDR" ]; then
    VNC_LISTEN_ADDR="0.0.0.0"
fi

# VNC 서버 Autostart 등록
cat <<EOF > /home/$ACTUAL_USER/.config/autostart/x11vnc.desktop
[Desktop Entry]
Type=Application
Name=VNC Server (Mirroring)
Exec=x11vnc -display :0 -rfbauth $VNC_PASSWD -rfbport 5900 -listen $VNC_LISTEN_ADDR -forever -shared -repeat -noxdamage
Terminal=false
EOF

# 설정 디렉토리 소유권 일괄 정리
sudo chown -R $ACTUAL_USER:$ACTUAL_USER /home/$ACTUAL_USER/.config

echo "=========================================="
echo " 🎉 최종: 시작 프로그램 기반 VNC 세팅 완료!"
echo " ⚠️ 장비를 재부팅(sudo reboot)한 뒤 접속해주세요."
echo " ⚠️ 접속 주소: $VNC_LISTEN_ADDR:5900"
echo "=========================================="
