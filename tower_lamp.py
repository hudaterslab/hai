import logging
import threading
import time

from qlight_etn import DEFAULT_PORT, LampState, QLightETN


logger = logging.getLogger("VMS_SYSTEM")


def _parse_lamp_state(value, default=LampState.OFF):
    if value is None:
        return default
    if isinstance(value, LampState):
        return value
    if isinstance(value, int):
        return LampState(value)
    return LampState[str(value).upper()]


class TowerLampController:
    def __init__(self, config):
        lamp_conf = config.get("tower_lamp", {}) if isinstance(config, dict) else {}
        self.enabled = bool(lamp_conf.get("enabled", False))
        self.host = str(lamp_conf.get("host", "") or "").strip()
        self.port = int(lamp_conf.get("port", DEFAULT_PORT))
        self.timeout = float(lamp_conf.get("timeout_sec", 2.0))
        self.reset_after_sec = float(lamp_conf.get("reset_after_sec", 5.0))
        self.patterns = lamp_conf.get("patterns", {})
        self.default_pattern_key = str(lamp_conf.get("default_pattern", "default_alarm"))
        self._client = QLightETN(self.host, self.port, timeout=self.timeout) if self.enabled and self.host else None
        self._lock = threading.Lock()
        self._active_until = 0.0
        self._active_pattern = None

    def trigger_event(self, event_name):
        if not self._client:
            return

        pattern = self.patterns.get(event_name) or self.patterns.get(self.default_pattern_key)
        if not isinstance(pattern, dict):
            return

        hold_sec = float(pattern.get("hold_sec", self.reset_after_sec))
        with self._lock:
            try:
                # 이벤트별 램프 패턴을 즉시 적용하고, 지정 시간 뒤 자동으로 all_off 되도록 예약한다.
                self._apply_pattern(pattern)
                self._active_until = max(self._active_until, time.time() + hold_sec)
                self._active_pattern = event_name
            except Exception as e:
                logger.error(f"타워램프 제어 실패({event_name}): {e}")

    def update(self, now=None):
        if not self._client:
            return

        now = time.time() if now is None else now
        with self._lock:
            if self._active_pattern is None or now < self._active_until:
                return
            try:
                self._client.all_off()
            except Exception as e:
                logger.error(f"타워램프 종료 실패: {e}")
            finally:
                self._active_pattern = None
                self._active_until = 0.0

    def shutdown(self):
        if not self._client:
            return

        with self._lock:
            try:
                # 프로그램 종료 시 경광등이 남지 않도록 명시적으로 모두 끈다.
                self._client.all_off()
            except Exception as e:
                logger.error(f"타워램프 종료 실패: {e}")
            finally:
                self._active_pattern = None
                self._active_until = 0.0

    def _apply_pattern(self, pattern):
        self._client.write_status(
            red=_parse_lamp_state(pattern.get("red"), LampState.OFF),
            yellow=_parse_lamp_state(pattern.get("yellow"), LampState.OFF),
            green=_parse_lamp_state(pattern.get("green"), LampState.OFF),
            blue=_parse_lamp_state(pattern.get("blue"), LampState.OFF),
            white=_parse_lamp_state(pattern.get("white"), LampState.OFF),
            sound_channel=int(pattern.get("sound_channel", 0)),
            sound_group=int(pattern.get("sound_group", 0)),
        )
