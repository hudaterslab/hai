import cv2
import math
import time
import numpy as np
from collections import defaultdict, deque
from common import (
    ID_H_NO_HELMET, ID_H_PERSON, ID_G_PERSON, TARGET_VEHICLES, 
    SCREEN_WIDTH, SCREEN_HEIGHT, get_check_point, get_center_point, get_foot_point,
    get_distance, ccw, calculate_iou
)

class TrajectoryTracker:
    def __init__(self, max_len=30):
        self.history = defaultdict(lambda: deque(maxlen=max_len))
        self.colors = {}

    def update_and_draw(self, frame, tracks):
        curr_ids = set()
        for t in tracks:
            x1, y1, x2, y2, tid, cls_id, conf = t
            tid = int(tid)
            curr_ids.add(tid)
            center = get_foot_point(x1, y1, x2, y2)
            self.history[tid].append(center)
            
            if tid not in self.colors:
                np.random.seed(tid)
                self.colors[tid] = tuple([int(c) for c in np.random.randint(50, 255, 3)])
                
            pts = list(self.history[tid])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], self.colors[tid], 2)
                
        for tid in list(self.history.keys()):
            if tid not in curr_ids: 
                del self.history[tid]

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
    event_name, menu_name, gui_name = "base", "BASE", "BASE"
    required_models, roi_type = ["general"], "polygon"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        self.config = config
        self.roi_poly = np.array(roi_poly, dtype=np.int32) if roi_poly and len(roi_poly) >= 3 else np.empty((0, 2), dtype=np.int32)
        self.roi_lines = roi_lines or []
        
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid): 
        return []

class IntrusionDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "intrusion", "침입", "INTRUSION"
    
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        if self.roi_poly.size == 0: 
            return triggered
            
        g_map = track_maps["general"]
        for t in general_tracks:
            tid = int(t[4])
            if g_map.get(tid) == ID_G_PERSON and cv2.pointPolygonTest(self.roi_poly, get_foot_point(*t[:4]), False) >= 0:
                triggered.append({'tid': tid, 'bbox': t[:4], 'frame': None, 'fid': fid})
                
        return triggered

class ParkingDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "illegal_parking", "주정차", "PARKING"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.states = defaultdict(lambda: {'start': 0, 'pos': None})
        
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered, curr_ids, now = [], set(), time.time()
        if self.roi_poly.size == 0: 
            return triggered
            
        g_map = track_maps["general"]
        for t in general_tracks:
            tid = int(t[4])
            if g_map.get(tid) in TARGET_VEHICLES and cv2.pointPolygonTest(self.roi_poly, get_check_point(*t[:4]), False) >= 0:
                curr_ids.add(tid)
                c = get_center_point(*t[:4])
                if self.states[tid]['start'] == 0 or get_distance(c, self.states[tid]['pos']) > 30:
                    self.states[tid].update({'start': now, 'pos': c})
                elif now - self.states[tid]['start'] > 5.0:
                    triggered.append({'tid': tid, 'bbox': t[:4], 'frame': None, 'fid': fid})
                    
        for tid in list(self.states.keys()):
            if tid not in curr_ids: 
                del self.states[tid]
                
        return triggered

class CrossingDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "conveyor_crossing", "횡단", "CROSSING"
    roi_type = "line"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.lines = [(self.roi_lines[i], self.roi_lines[i+1]) for i in range(len(self.roi_lines)-1)] if len(self.roi_lines) >= 2 else []
        self.prev, self.candidates = {}, {}
        self.pos_history = defaultdict(lambda: deque(maxlen=4))
        
        self.snapshot_mode = config.get("snapshot_mode", "crossing_moment")
        # distance_ratio는 이제 '선으로부터 넘어간 깊이(Penetration)'의 기준으로 사용됩니다.
        self.distance_ratio = config.get("distance_ratio", 0.3)
        self.min_distance_px, self.candidate_ttl_sec = config.get("min_distance_px", 15), config.get("candidate_ttl_sec", 5.0)
        self.direction_check = config.get("direction_check", True)
        
    def _is_intersect(self, p1, p2, p3, p4): 
        return ccw(p1, p2, p3) * ccw(p1, p2, p4) <= 0 and ccw(p3, p4, p1) * ccw(p3, p4, p2) <= 0
        
    # 💡 [핵심 추가] 점과 직선 사이의 수직 거리를 구하는 수학 공식
    def _dist_to_line(self, pt, p1, p2):
        num = abs((p2[0] - p1[0])*(p1[1] - pt[1]) - (p1[0] - pt[0])*(p2[1] - p1[1]))
        den = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return num / den if den > 0 else 0
        
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered, curr_ids, now, g_map = [], set(), time.time(), track_maps["general"]
        for t in general_tracks:
            tid = int(t[4])
            curr_ids.add(tid)
            if g_map.get(tid) != ID_G_PERSON: 
                continue
                
            x1, y1, x2, y2 = t[:4]
            obj_height = y2 - y1
            obj_width = max(1, x2 - x1)
            
            # 💡 [핵심 원복] 팔을 뻗는 모션에 속지 않도록 기준점을 다시 발끝(y2) 하단 중앙으로 원복
            curr_pos = (int((x1 + x2) / 2), int(y2))
            
            is_frame_out = (x1 <= 15) or (x2 >= SCREEN_WIDTH - 15) or (y1 <= 15) or (y2 >= SCREEN_HEIGHT - 15)
            
            self.pos_history[tid].append(curr_pos)
            
            is_ping_pong = False
            if len(self.pos_history[tid]) >= 3:
                p_older = self.pos_history[tid][-3]
                p_prev = self.pos_history[tid][-2] 
                p_curr = self.pos_history[tid][-1] 
                
                dist_jump = get_distance(p_curr, p_prev)
                dist_return = get_distance(p_curr, p_older)
                
                if dist_jump > obj_width * 0.5 and dist_return < obj_width * 0.3:
                    is_ping_pong = True
            
            if is_ping_pong:
                self.prev[tid] = curr_pos
                if tid in self.candidates:
                    del self.candidates[tid]
                continue

            if tid in self.prev and tid not in self.candidates:
                for p1, p2 in self.lines:
                    if self._is_intersect(p1, p2, self.prev[tid], curr_pos):
                        self.candidates[tid] = {
                            'crossing_pt': curr_pos, 'height': obj_height, 'timestamp': now, 'line': (p1, p2),
                            'entry_side': ccw(p1, p2, self.prev[tid]), 'frame': frame.copy() if frame is not None and self.snapshot_mode == "crossing_moment" else None,
                            'bbox': tuple(t[:4]), 'fid': fid
                        }
                        break
                        
            if tid in self.candidates:
                cand = self.candidates[tid]
                
                # 💡 [핵심 보완] 넘어간 선으로부터 현재 발끝까지의 '침투 깊이(수직 거리)'를 계산
                penetration_depth = self._dist_to_line(curr_pos, cand['line'][0], cand['line'][1])
                
                # 가로 비율(Aspect Ratio) 방어 로직은 삭제되었으므로 택배 상자 병합으로 인한 가로 팽창에도 횡단을 정상 감지합니다.
                
                direction_ok = (cand['entry_side'] != 0 and ccw(cand['line'][0], cand['line'][1], curr_pos) != 0 and cand['entry_side'] * ccw(cand['line'][0], cand['line'][1], curr_pos) < 0) if self.direction_check else True
                
                if direction_ok:
                    # 침투 깊이가 객체 너비의 일정 비율(기본 30%) 또는 최소 30픽셀 이상일 때만 진짜 횡단으로 간주
                    dynamic_threshold = max(30.0, obj_width * self.distance_ratio)
                    
                    if penetration_depth > dynamic_threshold or is_frame_out:
                        triggered.append({'tid': tid, 'bbox': cand['bbox'], 'frame': cand['frame'], 'fid': cand['fid']})
                        del self.candidates[tid]
                elif now - cand['timestamp'] > self.candidate_ttl_sec: 
                    del self.candidates[tid]
                    
            self.prev[tid] = curr_pos
            
        for tid in list(self.prev.keys()):
            if tid not in curr_ids:
                del self.prev[tid]
                if tid in self.candidates: 
                    del self.candidates[tid]
        
        for tid in list(self.pos_history.keys()):
            if tid not in curr_ids:
                del self.pos_history[tid]
                    
        return triggered

class HelmetDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "no_helmet", "안전모", "NO-HELMET"
    required_models, roi_type = ["helmet", "general"], "none"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.states, self.trigger_delay = {}, 2.0
        
    def _get_intersection_over_head_area(self, head_box, person_box):
        inter_area = max(0, min(head_box[2], person_box[2]) - max(head_box[0], person_box[0])) * max(0, min(head_box[3], person_box[3]) - max(head_box[1], person_box[1]))
        head_area = (head_box[2] - head_box[0]) * (head_box[3] - head_box[1])
        if head_area != 0:
            return inter_area / head_area
        return 0
        
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered, now = [], time.time() 
        no_helmets = [t for t in helmet_tracks if track_maps["helmet"].get(int(t[4])) == ID_H_NO_HELMET]
        current_nh_person_ids = set()
        
        for p in [t for t in general_tracks if track_maps["general"].get(int(t[4])) == ID_G_PERSON]:
            p_tid, max_ioa, nh_box_match = int(p[4]), 0, None
            for nh in no_helmets:
                ioa = self._get_intersection_over_head_area(nh[:4], p[:4])
                if ioa > max_ioa: 
                    max_ioa, nh_box_match = ioa, nh[:4]
                    
            if max_ioa > 0.5:
                current_nh_person_ids.add(p_tid)
                if p_tid not in self.states: 
                    self.states[p_tid] = now
                elif now - self.states[p_tid] >= self.trigger_delay:
                    triggered.append({'tid': p_tid, 'bbox': nh_box_match, 'frame': None, 'fid': fid})
                    
        for tid in list(self.states.keys()):
            if tid not in current_nh_person_ids: 
                del self.states[tid]
                
        return triggered

class SignalVehicleDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "signal_vehicle", "신호수차량감지", "NO-SIGNAL"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.motion_threshold_ratio, self.vehicle_history = 0.10, defaultdict(lambda: deque(maxlen=30)) 
        
    def _get_distance_point_to_rect(self, point, bbox): 
        return math.sqrt(max(bbox[0] - point[0], 0, point[0] - bbox[2])**2 + max(bbox[1] - point[1], 0, point[1] - bbox[3])**2)
        
    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered, current_vehicle_ids, g_map = [], set(), track_maps["general"]
        if self.roi_poly.size == 0 or motion_mask is None: 
            return triggered
            
        scale_x, scale_y = 640 / SCREEN_WIDTH, 360 / SCREEN_HEIGHT
        people_points = [get_foot_point(*t[:4]) for t in general_tracks if g_map.get(int(t[4])) == ID_G_PERSON]
        
        for t in general_tracks:
            tid = int(t[4])
            if g_map.get(tid) not in TARGET_VEHICLES: 
                continue
                
            current_vehicle_ids.add(tid)
            x1, y1, x2, y2 = t[:4]
            
            foot_center = get_foot_point(x1, y1, x2, y2)
            vehicle_size = max(x2 - x1, y2 - y1)
            
            if len(self.vehicle_history[tid]) > 0:
                prev_foot = self.vehicle_history[tid][-1]
                dynamic_jump_threshold = max(60.0, vehicle_size * 0.6)
                
                if get_distance(prev_foot, foot_center) > dynamic_jump_threshold:
                    self.vehicle_history[tid].clear()
                    continue
                    
            self.vehicle_history[tid].append(foot_center)
            
            history_list = list(self.vehicle_history[tid])
            if len(history_list) > 5:
                start_x, start_y = sum(p[0] for p in history_list[:3])/3.0, sum(p[1] for p in history_list[:3])/3.0
                end_x, end_y = sum(p[0] for p in history_list[-3:])/3.0, sum(p[1] for p in history_list[-3:])/3.0
                smoothed_dist = get_distance((start_x, start_y), (end_x, end_y))
                
                dynamic_move_threshold = max(40.0, vehicle_size * 0.15)
                
                if smoothed_dist >= dynamic_move_threshold and cv2.pointPolygonTest(self.roi_poly, get_center_point(x1, y1, x2, y2), False) >= 0:
                    mx1, my1, mx2, my2 = max(0, int(x1 * scale_x)), max(0, int(y1 * scale_y)), min(640, int(x2 * scale_x)), min(360, int(y2 * scale_y))
                    
                    if mx2 > mx1 and my2 > my1:
                        car_roi_mask = motion_mask[my1:my2, mx1:mx2]
                        _, motion_only = cv2.threshold(car_roi_mask, 250, 255, cv2.THRESH_BINARY)
                        total_pixels = (mx2 - mx1) * (my2 - my1)
                        
                        if total_pixels > 0 and (cv2.countNonZero(motion_only) / total_pixels) > self.motion_threshold_ratio:
                            if not any(self._get_distance_point_to_rect(pp, (x1, y1, x2, y2)) < vehicle_size * 1.5 for pp in people_points):
                                triggered.append({'tid': tid, 'bbox': t[:4], 'frame': None, 'fid': fid})
                                self.vehicle_history[tid].clear()
                                
        for tid in list(self.vehicle_history.keys()):
            if tid not in current_vehicle_ids: 
                del self.vehicle_history[tid]
                
        return triggered

EVENT_REGISTRY = {
    IntrusionDetector.event_name: IntrusionDetector, 
    ParkingDetector.event_name: ParkingDetector, 
    CrossingDetector.event_name: CrossingDetector, 
    HelmetDetector.event_name: HelmetDetector, 
    SignalVehicleDetector.event_name: SignalVehicleDetector
}