import cv2
import time
import math
import numpy as np
from collections import defaultdict, deque
from common import (
    ID_H_HELMET, ID_H_NO_HELMET, ID_H_PERSON, ID_G_PERSON, TARGET_VEHICLES, 
    get_check_point, get_center_point, 
    get_distance, ccw, calculate_iou
)

class MotionDetector:
    def __init__(self, sensitivity):
        self.threshold = 100 - ((sensitivity - 1) * 9) 
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame):
        if frame is None:
            return None
        small_frame = cv2.resize(frame, (640, 360))
        fg_mask = self.bg_subtractor.apply(small_frame)
        return cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

class BaseEventDetector:
    event_name = "base"
    menu_name = "BASE"    
    gui_name = "BASE"     
    required_models = ["general"] 
    roi_type = "polygon"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        self.config = config
        if roi_poly and len(roi_poly) >= 3:
            self.roi_poly = np.array(roi_poly, dtype=np.int32)
        else:
            self.roi_poly = np.empty((0, 2), dtype=np.int32)
        self.roi_lines = roi_lines or []
        
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        return []

class IntrusionDetector(BaseEventDetector):
    event_name = "intrusion"
    menu_name = "침입"
    gui_name = "INTRUSION"
    required_models = ["helmet"]
    roi_type = "polygon"

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        if self.roi_poly.size == 0:
            return triggered
            
        h_map = track_maps["helmet"]
        for t in helmet_tracks:
            tid = int(t[4])
            if h_map.get(tid) == ID_H_PERSON:
                if cv2.pointPolygonTest(self.roi_poly, get_check_point(*t[:4]), False) >= 0:
                    triggered.append({
                        'tid': tid, 
                        'bbox': t[:4], 
                        'frame': None, 
                        'fid': fid
                    })
        return triggered

class ParkingDetector(BaseEventDetector):
    event_name = "illegal_parking"
    menu_name = "주정차"
    gui_name = "PARKING"
    required_models = ["general"]
    roi_type = "polygon"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        # 주정차 판정 기준은 이벤트 설정에서 조정할 수 있게 유지한다.
        self.stationary_distance_px = float(config.get("stationary_distance_px", 30))
        self.min_stop_sec = float(config.get("min_stop_sec", 5.0))
        self.states = defaultdict(lambda: {'start': 0, 'pos': None})

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        curr_ids = set()
        now = time.time()
        g_map = track_maps["general"]
        
        if self.roi_poly.size == 0:
            return triggered
            
        for t in general_tracks:
            tid = int(t[4])
            if g_map.get(tid) in TARGET_VEHICLES:
                pt = get_check_point(*t[:4])
                if cv2.pointPolygonTest(self.roi_poly, pt, False) >= 0:
                    curr_ids.add(tid)
                    c = get_center_point(*t[:4])
                    
                    # 차량 중심이 일정 픽셀 이상 움직이면 정차 누적 시간을 다시 시작한다.
                    if self.states[tid]['start'] == 0 or get_distance(c, self.states[tid]['pos']) > self.stationary_distance_px:
                        self.states[tid].update({'start': now, 'pos': c})
                    # 지정한 시간 이상 거의 움직이지 않고 ROI 안에 머물면 주정차로 본다.
                    elif now - self.states[tid]['start'] > self.min_stop_sec:
                        triggered.append({
                            'tid': tid, 
                            'bbox': t[:4], 
                            'frame': None, 
                            'fid': fid
                        })
                        
        for tid in list(self.states.keys()):
            if tid not in curr_ids:
                del self.states[tid]
                
        return triggered

class CrossingDetector(BaseEventDetector):
    event_name = "conveyor_crossing"
    menu_name = "횡단"
    gui_name = "CROSSING"
    required_models = ["general"]
    roi_type = "line"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.lines = []
        for i in range(0, len(self.roi_lines), 2):
            if i + 1 < len(self.roi_lines):
                self.lines.append((self.roi_lines[i], self.roi_lines[i + 1]))
                
        self.prev = {}
        self.candidates = {}
        
        self.snapshot_mode = config.get("snapshot_mode", "crossing_moment")
        self.distance_ratio = config.get("distance_ratio", 0.5)
        self.min_distance_px = config.get("min_distance_px", 15)
        self.candidate_ttl_sec = config.get("candidate_ttl_sec", 5.0)
        self.direction_check = config.get("direction_check", True)
        self.max_crossing_angle = config.get("max_crossing_angle", 45.0)

    def _is_intersect(self, p1, p2, p3, p4):
        c1 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
        c2 = ccw(p3, p4, p1) * ccw(p3, p4, p2)
        return c1 < 0 and c2 < 0

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        curr_ids = set()
        now = time.time()
        g_map = track_maps["general"]
        
        for t in general_tracks:
            tid = int(t[4])
            curr_ids.add(tid)
            
            if g_map.get(tid) != ID_G_PERSON:
                continue
                
            # 💡 [핵심] 컨베이어 벨트 등 물리적 접점 계산을 위해 하단 중앙 좌표로 교체
            curr_pos = get_check_point(*t[:4])
            obj_height = t[3] - t[1]
            
            if tid in self.prev and tid not in self.candidates:
                for p1, p2 in self.lines:
                    if self._is_intersect(p1, p2, self.prev[tid], curr_pos):
                        self.candidates[tid] = {
                            'crossing_pt': curr_pos, 
                            'height': obj_height, 
                            'timestamp': now, 
                            'line': (p1, p2),
                            'entry_side': ccw(p1, p2, self.prev[tid]), 
                            'frame': frame.copy() if frame is not None and self.snapshot_mode == "crossing_moment" else None,
                            'bbox': tuple(t[:4]), 
                            'fid': fid
                        }
                        break
                        
            if tid in self.candidates:
                cand = self.candidates[tid]
                p1, p2 = cand['line']
                curr_side = ccw(p1, p2, curr_pos)
                moved_dist = get_distance(cand['crossing_pt'], curr_pos)
                
                dx = curr_pos[0] - cand['crossing_pt'][0]
                dy = curr_pos[1] - cand['crossing_pt'][1]
                angle_deg = math.degrees(math.atan2(abs(dy), abs(dx))) if (dx != 0 or dy != 0) else 0.0
                
                if self.direction_check:
                    direction_ok = (cand['entry_side'] != 0 and curr_side != 0 and cand['entry_side'] * curr_side < 0)
                else:
                    direction_ok = True
                
                if direction_ok and angle_deg <= self.max_crossing_angle:
                    if moved_dist > max(cand['height'] * self.distance_ratio, self.min_distance_px):
                        triggered.append({
                            'tid': tid, 
                            'bbox': cand['bbox'], 
                            'frame': cand['frame'], 
                            'fid': cand['fid']
                        })
                        del self.candidates[tid]
                elif now - cand['timestamp'] > self.candidate_ttl_sec or angle_deg > self.max_crossing_angle:
                    del self.candidates[tid]
                    
            self.prev[tid] = curr_pos
            
        for tid in list(self.prev.keys()):
            if tid not in curr_ids:
                del self.prev[tid]
                if tid in self.candidates:
                    del self.candidates[tid]
                    
        return triggered

class HelmetDetector(BaseEventDetector):
    event_name = "no_helmet"
    menu_name = "안전모"
    gui_name = "NO-HELMET"
    required_models = ["helmet", "general"]
    roi_type = "none"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        # 헬멧이 최근에 관측된 위치를 잠시 유지해, 조명/가림으로 HELMET가 순간 미탐되거나
        # NO_HELMET가 순간 오탐된 경우에도 이벤트가 바로 발생하지 않도록 보호한다.
        self.recent_helmet_hold_sec = float(config.get("recent_helmet_hold_sec", 1.0))
        self.recent_helmet_evidence = deque()

    def _get_intersection_over_head_area(self, head_box, person_box):
        x1 = max(head_box[0], person_box[0])
        y1 = max(head_box[1], person_box[1])
        x2 = min(head_box[2], person_box[2])
        y2 = min(head_box[3], person_box[3])
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        head_area = (head_box[2] - head_box[0]) * (head_box[3] - head_box[1])
        if head_area == 0:
            return 0
        return inter_area / head_area

    def _has_conflicting_helmet_detection(self, no_helmet_box, helmet_tracks, h_map):
        # NO_HELMET와 거의 같은 위치에 HELMET도 함께 존재하면 모델 판단이 충돌한 상황으로 본다.
        # 이 경우 즉시 이벤트를 발생시키지 않고 보류해 순간적인 이벤트 오탐을 줄인다.
        for ht in helmet_tracks:
            if h_map.get(int(ht[4])) != ID_H_HELMET:
                continue
            if self._get_intersection_over_head_area(no_helmet_box, ht[:4]) > 0.5:
                return True
            if self._get_intersection_over_head_area(ht[:4], no_helmet_box) > 0.5:
                return True
        return False

    def _prune_recent_helmet_evidence(self, now):
        while self.recent_helmet_evidence and (now - self.recent_helmet_evidence[0]['timestamp']) > self.recent_helmet_hold_sec:
            self.recent_helmet_evidence.popleft()

    def _remember_current_helmet_detections(self, helmet_tracks, h_map, now):
        for ht in helmet_tracks:
            if h_map.get(int(ht[4])) != ID_H_HELMET:
                continue
            # 현재 프레임의 HELMET 검출을 잠시 보관해, 다음 몇 프레임 동안 보호 증거로 사용한다.
            self.recent_helmet_evidence.append({
                'bbox': tuple(ht[:4]),
                'timestamp': now
            })

    def _has_recent_helmet_evidence(self, no_helmet_box):
        # 현재 프레임에 HELMET가 없더라도, 같은 위치에 최근 HELMET가 있었다면
        # 순간적인 미탐/오탐 가능성을 우선 의심하고 no_helmet 이벤트를 보류한다.
        for item in self.recent_helmet_evidence:
            helmet_box = item['bbox']
            if self._get_intersection_over_head_area(no_helmet_box, helmet_box) > 0.5:
                return True
            if self._get_intersection_over_head_area(helmet_box, no_helmet_box) > 0.5:
                return True
        return False

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        h_map = track_maps["helmet"]
        g_map = track_maps["general"]
        now = time.time()

        self._prune_recent_helmet_evidence(now)
        self._remember_current_helmet_detections(helmet_tracks, h_map, now)
        
        no_helmets = [t for t in helmet_tracks if h_map.get(int(t[4])) == ID_H_NO_HELMET]
        persons = [t for t in general_tracks if g_map.get(int(t[4])) == ID_G_PERSON]
        
        for nh in no_helmets:
            max_ioa = 0
            for p in persons:
                ioa = self._get_intersection_over_head_area(nh[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa

            # 사람과의 정합은 맞더라도, 같은 위치에 HELMET 검출이 함께 있으면
            # 이벤트 단계에서는 보수적으로 no_helmet 경보를 막는다.
            if self._has_conflicting_helmet_detection(nh[:4], helmet_tracks, h_map):
                continue

            # 현재 프레임 충돌이 없더라도, 같은 위치에 최근 HELMET 증거가 있었다면
            # 조명/가림으로 인한 순간적인 NO_HELMET 오탐 가능성을 먼저 의심한다.
            if self._has_recent_helmet_evidence(nh[:4]):
                continue

            if max_ioa > 0.5:
                triggered.append({
                    'tid': int(nh[4]), 
                    'bbox': nh[:4], 
                    'frame': None, 
                    'fid': fid
                })
        return triggered

class SignalVehicleDetector(BaseEventDetector):
    event_name = "signal_vehicle"
    menu_name = "신호수차량감지"
    gui_name = "NO-SIGNAL"
    required_models = ["general"]
    roi_type = "polygon"

    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        # signal_vehicle 판정 민감도는 이벤트 설정에서 조정할 수 있게 유지한다.
        self.motion_threshold_ratio = float(config.get("motion_threshold_ratio", 0.10))
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30))
        # 순간적인 검출 누락이나 motion 흔들림으로 즉시 오탐이 나지 않도록
        # 차량별 조건 만족 횟수를 짧게 누적해 안정적으로 판정한다.
        self.condition_hits = defaultdict(int)
        self.min_condition_hits = int(config.get("min_condition_hits", 3))
        self.max_condition_hits = int(config.get("max_condition_hits", 5))
        # 차량이 신호수를 잠깐 가리는 경우를 고려해 "현재 프레임에 사람이 안 보임"이 아니라
        # "최근 일정 시간 동안 차량 근처에서 사람을 보지 못함"을 기준으로 부재를 판단한다.
        self.person_hold_sec = float(config.get("person_hold_sec", 1.5))
        self.last_person_seen_ts = {}

    def _get_distance_point_to_rect(self, point, bbox):
        px, py = point
        bx1, by1, bx2, by2 = bbox
        return math.sqrt(max(bx1 - px, 0, px - bx2)**2 + max(by1 - py, 0, py - by2)**2)

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        current_vehicle_ids = set()
        g_map = track_maps["general"]
        now = time.time()
        
        if self.roi_poly.size == 0 or motion_mask is None:
            return triggered

        people_points = [get_center_point(*t[:4]) for t in general_tracks if g_map.get(int(t[4])) == ID_G_PERSON]
        mask_h, mask_w = motion_mask.shape[:2]
        
        for t in general_tracks:
            tid = int(t[4])
            if g_map.get(tid) not in TARGET_VEHICLES:
                continue
                
            current_vehicle_ids.add(tid)
            x1, y1, x2, y2 = t[:4]
            center = get_center_point(x1, y1, x2, y2)
            self.vehicle_history[tid].append(center)
            
            if len(self.vehicle_history[tid]) > 5 and get_distance(self.vehicle_history[tid][0], self.vehicle_history[tid][-1]) >= 40.0:
                # motion_mask는 이미 detector 입력 프레임과 같은 640x360 좌표계다.
                # 여기서 다시 SCREEN 기준으로 축소하면 차량 bbox와 다른 영역을 읽게 된다.
                mx1 = max(0, int(x1))
                my1 = max(0, int(y1))
                mx2 = min(mask_w, int(x2))
                my2 = min(mask_h, int(y2))
                
                if mx2 > mx1 and my2 > my1:
                    car_roi_mask = motion_mask[my1:my2, mx1:mx2]
                    _, motion_only = cv2.threshold(car_roi_mask, 250, 255, cv2.THRESH_BINARY)
                    total_pixels = (mx2 - mx1) * (my2 - my1)
                    motion_ratio = (cv2.countNonZero(motion_only) / total_pixels) if total_pixels > 0 else 0.0
                    in_roi = cv2.pointPolygonTest(self.roi_poly, center, False) >= 0
                    safe_radius = y2 - y1
                    has_nearby_person = any(
                        self._get_distance_point_to_rect(pp, (x1, y1, x2, y2)) < safe_radius
                        for pp in people_points
                    )
                    if has_nearby_person:
                        # 사람 박스가 검출된 순간의 시간을 기록해, 이후 잠깐 가려지는 구간에서도
                        # 즉시 "신호수 없음"으로 뒤집히지 않도록 최근 목격 이력을 유지한다.
                        self.last_person_seen_ts[tid] = now
                    last_seen_ts = self.last_person_seen_ts.get(tid, 0.0)
                    person_absent_long_enough = (now - last_seen_ts) > self.person_hold_sec

                    condition_met = (
                        motion_ratio > self.motion_threshold_ratio and
                        in_roi and
                        person_absent_long_enough
                    )

                    if condition_met:
                        self.condition_hits[tid] = min(self.condition_hits[tid] + 1, self.max_condition_hits)
                    else:
                        # 한 프레임 실패했다고 바로 0으로 초기화하지 않고 완만히 감소시켜
                        # 사람/모션 검출이 잠깐 흔들려도 경보 후보를 조금 더 안정적으로 유지한다.
                        self.condition_hits[tid] = max(self.condition_hits[tid] - 1, 0)

                    if self.condition_hits[tid] >= self.min_condition_hits:
                        triggered.append({
                            'tid': tid, 
                            'bbox': t[:4], 
                            'frame': None, 
                            'fid': fid
                        })

        for tid in list(self.vehicle_history.keys()):
            if tid not in current_vehicle_ids:
                del self.vehicle_history[tid]
                if tid in self.condition_hits:
                    del self.condition_hits[tid]
                if tid in self.last_person_seen_ts:
                    del self.last_person_seen_ts[tid]
                
        return triggered

EVENT_REGISTRY = {
    IntrusionDetector.event_name: IntrusionDetector,
    ParkingDetector.event_name: ParkingDetector,
    CrossingDetector.event_name: CrossingDetector,
    HelmetDetector.event_name: HelmetDetector,
    SignalVehicleDetector.event_name: SignalVehicleDetector
}
