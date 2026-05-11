import os
import json
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)

# 시스템 설정 경로 (multi_event.py와 동일한 환경)
CONFIG_FILE = "cameras.json"
RAM_DISK_DIR = "/dev/shm/cctv_frames"
if not os.path.exists(RAM_DISK_DIR):
    RAM_DISK_DIR = "./web_frames"

# ==========================================
# [1] HTML + Bootstrap 5 통합 템플릿 (Single-file Deployment)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Edge Vision Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .cam-card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; }
        .cam-img { width: 100%; height: 240px; object-fit: cover; background-color: #2c3e50; }
        .status-dot { height: 12px; width: 12px; background-color: #28a745; border-radius: 50%; display: inline-block; margin-right: 5px; }
        .navbar-brand { font-weight: bold; letter-spacing: 1px; }
    </style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
    <div class="container-fluid">
        <a class="navbar-brand" href="#">🖥️ Edge AI Dashboard</a>
        <button class="btn btn-outline-light btn-sm" onclick="location.reload()">새로고침</button>
    </div>
</nav>

<div class="container">
    <div class="row" id="camera-grid">
        {% for ip, conf in configs.items() %}
        <div class="col-md-6 col-lg-4">
            <div class="card cam-card">
                <img src="/img/{{ ip }}" class="card-img-top cam-img" id="img-{{ ip }}" alt="Camera Feed" onerror="this.src='data:image/svg+xml;charset=UTF-8,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100%25\\' height=\\'100%25\\'%3E%3Crect fill=\\'%23ccc\\' width=\\'100%25\\' height=\\'100%25\\'/%3E%3Ctext fill=\\'%23555\\' x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' font-size=\\'20\\'%3ENo Signal / Waiting%3C/text%3E%3C/svg%3E'">
                <div class="card-body">
                    <h5 class="card-title"><span class="status-dot"></span> CAM: {{ ip }}</h5>
                    <p class="card-text text-muted small text-truncate" title="{{ conf.url }}">{{ conf.url }}</p>
                    <div class="mb-3">
                        <span class="badge bg-secondary">Events</span>
                        {% for ev in conf.events %}
                            <span class="badge bg-primary">{{ ev }}</span>
                        {% else %}
                            <span class="badge bg-light text-dark border">None</span>
                        {% endfor %}
                    </div>
                    <button class="btn btn-sm btn-outline-primary w-100" onclick="openConfigModal('{{ ip }}')">설정 변경</button>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>

<div class="modal fade" id="configModal" tabindex="-1">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">이벤트 설정: <span id="modal-ip-title"></span></h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <form id="configForm">
            <input type="hidden" id="modal-ip">
            <div class="mb-3">
                <label class="form-label fw-bold">활성화할 이벤트 선택</label>
                <div class="form-check">
                  <input class="form-check-input event-cb" type="checkbox" value="intrusion" id="ev1">
                  <label class="form-check-label" for="ev1">침입 (Intrusion)</label>
                </div>
                <div class="form-check">
                  <input class="form-check-input event-cb" type="checkbox" value="illegal_parking" id="ev2">
                  <label class="form-check-label" for="ev2">주정차 (Parking)</label>
                </div>
                <div class="form-check">
                  <input class="form-check-input event-cb" type="checkbox" value="no_helmet" id="ev3">
                  <label class="form-check-label" for="ev3">안전모 미착용 (No Helmet)</label>
                </div>
                <div class="form-check">
                  <input class="form-check-input event-cb" type="checkbox" value="conveyor_crossing" id="ev4">
                  <label class="form-check-label" for="ev4">횡단 (Crossing)</label>
                </div>
                <div class="form-check">
                  <input class="form-check-input event-cb" type="checkbox" value="signal_vehicle" id="ev5">
                  <label class="form-check-label" for="ev5">신호수차량 (Signal Vehicle)</label>
                </div>
            </div>
            <div class="alert alert-warning small">
                * 체크박스를 변경하고 저장하면 단말 메인 시스템이 재시작 없이 즉시 감지하여 이벤트를 갱신합니다.<br>
                * 정밀한 다각형/라인 ROI 설정은 터미널 UI 마법사를 이용해주십시오. (현재 웹 MVP 버전)
            </div>
        </form>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">취소</button>
        <button type="button" class="btn btn-primary" onclick="saveConfig()">저장 및 적용</button>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    // 서버 설정 데이터 (Jinja2 렌더링)
    const serverConfigs = {{ configs | tojson }};
    const configModal = new bootstrap.Modal(document.getElementById('configModal'));

    // 프레임 주기적 갱신 로직 (현재 5초. 1시간으로 원할 시 3600000 으로 변경)
    setInterval(() => {
        const images = document.querySelectorAll('.cam-img');
        images.forEach(img => {
            const ip = img.id.replace('img-', '');
            // 쿼리 파라미터에 현재 시간을 주어 브라우저 캐싱 강제 우회
            img.src = `/img/${ip}?t=${new Date().getTime()}`;
        });
    }, 5000); 

    function openConfigModal(ip) {
        document.getElementById('modal-ip-title').innerText = ip;
        document.getElementById('modal-ip').value = ip;
        
        const conf = serverConfigs[ip];
        const activeEvents = conf.events || [];
        
        document.querySelectorAll('.event-cb').forEach(cb => {
            cb.checked = activeEvents.includes(cb.value);
        });
        
        configModal.show();
    }

    function saveConfig() {
        const ip = document.getElementById('modal-ip').value;
        const selectedEvents = [];
        
        document.querySelectorAll('.event-cb').forEach(cb => {
            if (cb.checked) selectedEvents.push(cb.value);
        });
        
        // 서버로 설정 저장 API 호출
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ip: ip, events: selectedEvents })
        })
        .then(response => response.json())
        .then(data => {
            if(data.success) {
                configModal.hide();
                location.reload(); // 성공 시 화면 갱신하여 뱃지 업데이트
            } else {
                alert('설정 저장에 실패했습니다.');
            }
        });
    }
</script>
</body>
</html>
"""

# ==========================================
# [2] 웹 서버 라우팅 (API 및 View)
# ==========================================
def load_configs():
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@app.route('/')
def index():
    configs = load_configs()
    return render_template_string(HTML_TEMPLATE, configs=configs)

@app.route('/img/<ip>')
def get_image(ip):
    """RAM 디스크에서 최신 카메라 프레임 서빙"""
    img_path = os.path.join(RAM_DISK_DIR, f"{ip}.jpg")
    if os.path.exists(img_path):
        # mimetype 설정으로 브라우저 렌더링 최적화
        return send_file(img_path, mimetype='image/jpeg')
    else:
        # 파일이 없을 경우 404를 반환하면 프론트에서 onerror가 발동하여 회색 배경을 띄웁니다.
        return "No Signal", 404

@app.route('/api/config', methods=['POST'])
def update_config():
    """웹 UI에서 전송된 이벤트 설정을 cameras.json에 업데이트합니다."""
    data = request.json
    ip = data.get('ip')
    new_events = data.get('events', [])
    
    configs = load_configs()
    
    if ip in configs:
        configs[ip]['events'] = new_events
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=4)
            # 메인 시스템(multi_event.py)의 Watchdog이 파일 수정 시간(mtime)을 감지하고 무중단 리로드하게 됩니다.
            return jsonify({"success": True, "message": "Saved successfully"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
            
    return jsonify({"success": False, "message": "Camera IP not found"}), 404

if __name__ == '__main__':
    # 0.0.0.0 바인딩을 통해 외부 PC에서도 단말 IP로 접속 가능
    print(f"🌐 웹 서비스가 시작되었습니다. 브라우저에서 http://[단말IP]:5000 으로 접속하십시오.")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)