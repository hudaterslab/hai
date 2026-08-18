#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "이 스크립트는 root 권한(sudo)으로 실행해야 합니다."
  exit 1
fi

echo "=== 인텔 전 기종(igc, e1000e) 네트워크 최적화 스크립트 시작 ==="

# 1. 커널 시스템 네트워크 튜닝 (16GB RAM 환경 최적화)
echo -e "\n[1/4] 리눅스 커널 네트워크 버퍼(65536 및 TCP 16MB) 확장..."
cat << 'EOF' > /etc/sysctl.d/99-network-tuning.conf
# 네트워크 인터페이스 백로그 큐 크기 (기본 1000 -> 65536)
net.core.netdev_max_backlog = 65536
# 소켓 수신/송신 버퍼 최대 크기 (약 16MB)
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
# TCP 수신/송신 버퍼 튜닝 (최소, 기본, 최대)
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
EOF
sysctl --system > /dev/null 2>&1
echo " -> sysctl 커널 튜닝 적용 완료."

# 2. GRUB ASPM (전력 관리) 비활성화
echo -e "\n[2/4] GRUB 설정 확인 및 pcie_aspm=off 적용..."
if ! grep -q "pcie_aspm=off" /etc/default/grub; then
    sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 pcie_aspm=off"/' /etc/default/grub
    update-grub > /dev/null 2>&1
    echo " -> GRUB 업데이트 완료 (재부팅 시 적용)"
else
    echo " -> 이미 GRUB에 pcie_aspm=off 설정이 적용되어 있습니다."
fi

# 3. 인텔 랜카드(igc 및 e1000e) 인터페이스 자동 탐색
echo -e "\n[3/4] 인텔 랜카드(igc, e1000e) 인터페이스 탐색 중..."
# 수정된 부분: igc(I225/I226)와 e1000e(I219) 드라이버를 모두 찾습니다.
INTERFACES=$(ls -l /sys/class/net/*/device/driver 2>/dev/null | grep -E 'igc|e1000e' | awk -F'/sys/class/net/' '{print $2}' | awk -F'/' '{print $1}')

if [ -z "$INTERFACES" ]; then
    echo "인텔 랜카드를 찾을 수 없어 스크립트를 종료합니다."
    exit 1
fi
echo " -> 감지된 인터페이스: $INTERFACES"

# 4. Systemd 서비스 파일 생성
SERVICE_FILE="/etc/systemd/system/intel-nic-fix.service"
echo -e "\n[4/4] 하드웨어 버퍼(4096) 및 오프로딩 패치 서비스 등록..."

echo "[Unit]" > $SERVICE_FILE
echo "Description=Intel NIC Hardware Stability Patch" >> $SERVICE_FILE
echo "After=network-online.target" >> $SERVICE_FILE
echo "Wants=network-online.target" >> $SERVICE_FILE
echo "" >> $SERVICE_FILE
echo "[Service]" >> $SERVICE_FILE
echo "Type=oneshot" >> $SERVICE_FILE
echo "RemainAfterExit=yes" >> $SERVICE_FILE

for IFACE in $INTERFACES; do
    echo "ExecStart=-/sbin/ethtool -G $IFACE rx 4096 tx 4096" >> $SERVICE_FILE
    echo "ExecStart=-/sbin/ethtool -K $IFACE tso off gso off gro off" >> $SERVICE_FILE
    echo "ExecStart=-/sbin/ethtool --set-eee $IFACE eee off" >> $SERVICE_FILE
done

echo "" >> $SERVICE_FILE
echo "[Install]" >> $SERVICE_FILE
echo "WantedBy=multi-user.target" >> $SERVICE_FILE

# 5. 서비스 등록 및 실행
systemctl daemon-reload
systemctl enable intel-nic-fix.service > /dev/null 2>&1
systemctl restart intel-nic-fix.service

echo -e "\n=== NUC 13 및 PL64 맞춤형 패치가 완벽하게 적용되었습니다! ==="
