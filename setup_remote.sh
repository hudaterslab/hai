#!/bin/bash

# 현재 스크립트를 실행하는 사용자 이름 자동 획득
ACTUAL_USER=$USER

echo "=========================================="
echo " 1. 바탕화면 자동 로그인 (GDM3) 자동 설정"
echo "=========================================="
sudo cp /etc/gdm3/custom.conf /etc/gdm3/custom.conf.bak 2>/dev/null || true
sudo touch /etc/gdm3/custom.conf

if ! grep -q "^\[daemon\]" /etc/gdm3/custom.conf; then
    echo "[daemon]" | sudo tee -a /etc/gdm3/custom.conf > /dev/null
fi

if grep -iq "WaylandEnable" /etc/gdm3/custom.conf; then
    sudo sed -i -E 's/^#?\s*WaylandEnable\s*=.*/WaylandEnable=false/i' /etc/gdm3/custom.conf
else
    sudo sed -i "/\[daemon\]/a WaylandEnable=false" /etc/gdm3/custom.conf
fi

if grep -iq "AutomaticLoginEnable" /etc/gdm3/custom.conf; then
    sudo sed -i -E 's/^#?\s*AutomaticLoginEnable\s*=\s*.*/AutomaticLoginEnable=True/i' /etc/gdm3/custom.conf
    sudo sed -i -E "s/^#?\s*AutomaticLogin\s*=\s*.*/AutomaticLogin=$ACTUAL_USER/i" /etc/gdm3/custom.conf
else
    sudo sed -i "/\[daemon\]/a AutomaticLoginEnable=True\nAutomaticLogin=$ACTUAL_USER" /etc/gdm3/custom.conf
fi
echo "✅ 사용자 [$ACTUAL_USER] 바탕화면 자동 로그인 세팅 완료!"

echo "=========================================="
echo " 2. 필수 패키지 설치 (Tailscale & xrdp)"
echo "=========================================="
sudo apt update && sudo apt install -y curl gpg xrdp x11vnc xauth net-tools

# Tailscale 설치
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$(lsb_release -cs).noarmor.gpg" | sudo tee /usr/share/keyrings/tailscale.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/tailscale.gpg] https://pkgs.tailscale.com/stable/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale

echo "=========================================="
echo " 3. 순정 원격 데스크톱 비활성화 (포트 충돌 방지)"
echo "=========================================="
# xrdp와 3389 포트가 충돌하지 않도록 내장 RDP 서비스 강제 종료
systemctl --user stop gnome-remote-desktop 2>/dev/null || true
systemctl --user disable gnome-remote-desktop 2>/dev/null || true
sudo systemctl stop gnome-remote-desktop 2>/dev/null || true
sudo systemctl disable gnome-remote-desktop 2>/dev/null || true

echo "=========================================="
echo " 4. xrdp (헤드리스 RDP) 권한 설정"
echo "=========================================="
# xrdp가 우분투 화면에 접근할 수 있도록 권한 부여
sudo adduser xrdp ssl-cert

echo "=========================================="
echo " 5. RDP 속도 최적화 및 절전/잠금 방지 설정"
echo "=========================================="
# SSH 접속 상태에서도 바탕화면 설정이 적용되도록 환경 변수 주입
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"

# 유휴 상태 끄기 및 잠금 방지, 애니메이션 종료 (xrdp 속도 향상)
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.interface enable-animations false 2>/dev/null || true

# 스키마가 없는 경량 OS일 경우 에러 메시지 무시 처리
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target > /dev/null 2>&1

echo "=========================================="
echo " 6. xrdp 서비스 활성화 및 재시작"
echo "=========================================="
sudo systemctl enable xrdp
sudo systemctl restart xrdp

echo "=========================================="
echo " 7. x11vnc (VNC) service setup"
echo "=========================================="
VNC_USER="${VNC_USER:-$ACTUAL_USER}"
VNC_PORT="${VNC_PORT:-5900}"
VNC_DISPLAY="${VNC_DISPLAY:-:0}"
VNC_LISTEN="${VNC_LISTEN:-auto}"
VNC_LOGIN_SCREEN_ACCESS="${VNC_LOGIN_SCREEN_ACCESS:-true}"
VNC_HOME="$(getent passwd "$VNC_USER" | cut -d: -f6)"
VNC_GROUP="$(id -gn "$VNC_USER")"
VNC_AUTH="${VNC_AUTH:-$VNC_HOME/.Xauthority}"
VNC_AUTH_CANDIDATES="${VNC_AUTH_CANDIDATES:-/var/run/sddm/* /run/sddm/* /run/user/*/gdm/Xauthority /run/gdm3/auth-for-*/database}"
VNC_PASSWD="$VNC_HOME/.vnc/passwd"
VNC_LOG="$VNC_HOME/.vnc/x11vnc.log"
VNC_SERVICE="/etc/systemd/system/remote-x11vnc.service"

if ! id "$VNC_USER" >/dev/null 2>&1; then
    echo "VNC user not found: $VNC_USER"
    exit 1
fi

if ! command -v x11vnc >/dev/null 2>&1; then
    sudo apt update && sudo apt install -y x11vnc xauth
fi
X11VNC_BIN="$(command -v x11vnc)"

if [ "$VNC_LISTEN" = "auto" ]; then
    VNC_LISTEN_ADDR=""
    if command -v tailscale >/dev/null 2>&1; then
        VNC_LISTEN_ADDR="$(tailscale ip -4 2>/dev/null | awk 'NF {print; exit}')"
    fi
    if [ -z "$VNC_LISTEN_ADDR" ]; then
        VNC_LISTEN_ADDR="0.0.0.0"
        echo "Tailscale IP not detected. VNC will listen on 0.0.0.0"
    fi
else
    VNC_LISTEN_ADDR="$VNC_LISTEN"
fi

if [ ! -e "$VNC_AUTH" ]; then
    VNC_AUTH="guess"
fi

if [ "$VNC_LOGIN_SCREEN_ACCESS" = "true" ]; then
    VNC_SERVICE_USER="root"
    VNC_SERVICE_GROUP="root"
    VNC_SERVICE_DISPLAY="$VNC_DISPLAY"
    VNC_SERVICE_AUTH=""
    VNC_SERVICE_EXEC_START="/bin/sh -c 'AUTH_FILE=\"\"; for f in $VNC_AUTH_CANDIDATES; do if [ -r \"\$f\" ]; then AUTH_FILE=\"\$f\"; break; fi; done; if [ -z \"\$AUTH_FILE\" ]; then echo \"No display-manager Xauthority file found in: $VNC_AUTH_CANDIDATES\" >&2; exit 1; fi; exec $X11VNC_BIN -display $VNC_SERVICE_DISPLAY -auth \"\$AUTH_FILE\" -rfbauth $VNC_PASSWD -rfbport $VNC_PORT -listen $VNC_LISTEN_ADDR -forever -shared -repeat -noxdamage -o $VNC_LOG'"
    echo "VNC login-screen access enabled. x11vnc will use display-manager Xauthority files."
else
    VNC_SERVICE_USER="$VNC_USER"
    VNC_SERVICE_GROUP="$VNC_GROUP"
    VNC_SERVICE_DISPLAY="$VNC_DISPLAY"
    VNC_SERVICE_AUTH="$VNC_AUTH"
    VNC_SERVICE_EXEC_START="$X11VNC_BIN -display $VNC_SERVICE_DISPLAY -auth $VNC_SERVICE_AUTH -rfbauth $VNC_PASSWD -rfbport $VNC_PORT -listen $VNC_LISTEN_ADDR -forever -shared -repeat -noxdamage -o $VNC_LOG"
fi

sudo install -d -m 700 -o "$VNC_USER" -g "$VNC_GROUP" "$VNC_HOME/.vnc"

if [ -n "${VNC_PASSWORD:-}" ]; then
    printf '%s\n%s\ny\n' "$VNC_PASSWORD" "$VNC_PASSWORD" | sudo -u "$VNC_USER" "$X11VNC_BIN" -storepasswd "$VNC_PASSWD" >/dev/null
elif [ ! -f "$VNC_PASSWD" ]; then
    echo "VNC password file not found. Please enter a VNC password."
    sudo -u "$VNC_USER" "$X11VNC_BIN" -storepasswd "$VNC_PASSWD"
else
    echo "Using existing VNC password file: $VNC_PASSWD"
fi
sudo chown "$VNC_USER:$VNC_GROUP" "$VNC_PASSWD"
sudo chmod 600 "$VNC_PASSWD"

sudo tee "$VNC_SERVICE" > /dev/null <<EOF
[Unit]
Description=x11vnc desktop sharing for $VNC_DISPLAY
After=network-online.target graphical.target display-manager.service
Wants=network-online.target

[Service]
Type=simple
User=$VNC_SERVICE_USER
Group=$VNC_SERVICE_GROUP
Environment=DISPLAY=$VNC_SERVICE_DISPLAY
Environment=XAUTHORITY=$VNC_SERVICE_AUTH
ExecStartPre=/bin/sh -c 'for i in \$(seq 1 90); do test -S /tmp/.X11-unix/X0 && exit 0; sleep 2; done; exit 1'
ExecStart=$VNC_SERVICE_EXEC_START
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable remote-x11vnc.service
sudo systemctl restart remote-x11vnc.service

echo "VNC service ready: $VNC_LISTEN_ADDR:$VNC_PORT"

echo "=========================================="
echo " 🎉 자동 셋팅 완료!"
echo "=========================================="
echo " ⚠️ 접속 방법 안내:"
echo " 윈도우 '원격 데스크톱 연결'에서 접속 후, 청록색(xrdp) 로그인 창이 뜨면"
echo " 우분투 계정 아이디와 비밀번호를 직접 입력하여 로그인하세요."
echo "=========================================="
