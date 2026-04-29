import cv2
import math
import time
import numpy as np
from collections import defaultdict, deque
from common import (
    ID_H_NO_HELMET, ID_G_PERSON, ID_PERSON_LOW, TARGET_VEHICLES, 
    SCREEN_WIDTH, SCREEN_HEIGHT, get_check_point, get_center_point, get_foot_point,
    get_distance, ccw, calculate_iou, SYS_CFG
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
        self.fps = SYS_CFG.get("REC_FPS", 3)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs): 
        return []

class IntrusionDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "intrusion", "침입", "INTRUSION"
    
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        if self.roi_poly.size == 0: 
            return triggered
            
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) == ID_G_PERSON and cv2.pointPolygonTest(self.roi_poly, get_foot_point(*t[:4]), False) >= 0:
                triggered.append({'tid': tid, 'bbox': t[:4], 'frame': None, 'fid': fid})
                
        return triggered

class ParkingDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "illegal_parking", "주정차", "PARKING"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.states = defaultdict(lambda: {'start_fid': 0, 'pos': None})
        trigger_sec = config.get("trigger_sec", 5.0)
        self.move_threshold = config.get("move_threshold_px", 30)
        self.trigger_fid_diff = int(trigger_sec * self.fps)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered, curr_ids = [], set()
        if self.roi_poly.size == 0: 
            return triggered
            
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) in TARGET_VEHICLES and cv2.pointPolygonTest(self.roi_poly, get_check_point(*t[:4]), False) >= 0:
                curr_ids.add(tid)
                c = get_center_point(*t[:4])
                if self.states[tid]['start_fid'] == 0 or get_distance(c, self.states[tid]['pos']) > self.move_threshold:
                    self.states[tid].update({'start_fid': fid, 'pos': c})
                elif fid - self.states[tid]['start_fid'] >= self.trigger_fid_diff:
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
        
        self.snapshot_mode = config.get("snapshot_mode", "crossing_moment")
        self.distance_ratio = config.get("distance_ratio", 0.5)
        self.direction_check = config.get("direction_check", True)
        self.min_crossing_angle = config.get("min_crossing_angle", 20.0) 
        
        candidate_ttl_sec = config.get("candidate_ttl_sec", 5.0)
        self.ttl_fid_diff = int(candidate_ttl_sec * self.fps)
        
    def _is_intersect(self, p1, p2, p3, p4): 
        return ccw(p1, p2, p3) * ccw(p1, p2, p4) <= 0 and ccw(p3, p4, p1) * ccw(p3, p4, p2) <= 0

    def _get_crossing_angle(self, p1, p2, prev_pos, curr_pos):
        lx, ly = p2[0] - p1[0], p2[1] - p1[1]
        mx, my = curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]
        dot = lx * mx + ly * my
        cross = lx * my - ly * mx
        angle_rad = math.atan2(abs(cross), abs(dot))
        return math.degrees(angle_rad)

    def _get_intersection_over_lowbody_area(self, low_box, person_box):
        """사람 영역 대비 하반신 겹침 비율(IoA) 계산"""
        inter_area = max(0, min(low_box[2], person_box[2]) - max(low_box[0], person_box[0])) * max(0, min(low_box[3], person_box[3]) - max(low_box[1], person_box[1]))
        low_area = (low_box[2] - low_box[0]) * (low_box[3] - low_box[1])
        if low_area != 0:
            return inter_area / low_area
        return 0
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered, curr_ids = [], set()
        
        # 💡 [핵심 보완] 트랙 분리: 견고한 전신(Person) 트랙을 기준으로 하반신(LowBody)을 공간 매칭
        persons = [t for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        low_bodies = [t for t in tracks if track_map.get(int(t[4])) == ID_PERSON_LOW]
        
        for p in persons:
            p_tid = int(p[4])
            curr_ids.add(p_tid)
            
            # 사람 BBOX 내부에 포함된 가장 적합한 하반신 BBOX 탐색 (Hierarchical Re-ID)
            best_low_box = None
            max_ioa = 0
            for lb in low_bodies:
                ioa = self._get_intersection_over_lowbody_area(lb[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa
                    best_low_box = lb[:4]
                    
            # 하반신이 명확히 검출되지 않았으면 해당 프레임의 횡단 좌표 갱신은 스킵 (궤적은 유지됨)
            if max_ioa < 0.4 or best_low_box is None:
                continue
                
            # 💡 횡단 판정의 핵심 좌표는 사람 BBOX가 아닌 '하반신의 좌표'
            lx1, ly1, lx2, ly2 = best_low_box
            obj_height = max(1, ly2 - ly1)
            
            # 하반신의 발목 부근을 기준점으로 설정
            curr_pos = (int((lx1 + lx2) / 2), int(ly2 - obj_height * 0.1))
            is_frame_out = (lx1 <= 15) or (lx2 >= SCREEN_WIDTH - 15) or (ly1 <= 15) or (ly2 >= SCREEN_HEIGHT - 15)
            
            # p_tid(전신 ID)를 키값으로 횡단 알고리즘 전개
            if p_tid in self.prev and p_tid not in self.candidates:
                for p1, p2 in self.lines:
                    if self._is_intersect(p1, p2, self.prev[p_tid], curr_pos):
                        cross_angle = self._get_crossing_angle(p1, p2, self.prev[p_tid], curr_pos)
                        
                        if cross_angle >= self.min_crossing_angle:
                            self.candidates[p_tid] = {
                                'crossing_pt': curr_pos, 'height': obj_height, 'timestamp_fid': fid, 'line': (p1, p2),
                                'entry_side': ccw(p1, p2, self.prev[p_tid]), 'frame': frame.copy() if frame is not None and self.snapshot_mode == "crossing_moment" else None,
                                'bbox': tuple(p[:4]), 'fid': fid  # 💡 이벤트 발생 시 API로 전송할 BBOX는 전신(Person) BBOX
                            }
                        break
                        
            if p_tid in self.candidates:
                cand = self.candidates[p_tid]
                moved_dist = get_distance(cand['crossing_pt'], curr_pos)
                
                direction_ok = (cand['entry_side'] != 0 and ccw(cand['line'][0], cand['line'][1], curr_pos) != 0 and cand['entry_side'] * ccw(cand['line'][0], cand['line'][1], curr_pos) < 0) if self.direction_check else True
                
                if direction_ok:
                    dynamic_threshold = max(20.0, obj_height * self.distance_ratio)
                    if moved_dist > dynamic_threshold or is_frame_out:
                        triggered.append({'tid': p_tid, 'bbox': cand['bbox'], 'frame': cand['frame'], 'fid': cand['fid']})
                        del self.candidates[p_tid]
                elif fid - cand['timestamp_fid'] > self.ttl_fid_diff: 
                    del self.candidates[p_tid]
                    
            self.prev[p_tid] = curr_pos
            
        for tid in list(self.prev.keys()):
            if tid not in curr_ids:
                del self.prev[tid]
                if tid in self.candidates: 
                    del self.candidates[tid]
                    
        return triggered

class HelmetDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "no_helmet", "안전모", "NO-HELMET"
    required_models, roi_type = ["helmet", "general"], "none"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.states = {} 
        trigger_sec = config.get("trigger_sec", 3.0) 
        self.trigger_fid_diff = int(trigger_sec * self.fps)
        self.grace_fid_diff = int(2.0 * self.fps) 
        
    def _get_intersection_over_head_area(self, head_box, person_box):
        inter_area = max(0, min(head_box[2], person_box[2]) - max(head_box[0], person_box[0])) * max(0, min(head_box[3], person_box[3]) - max(head_box[1], person_box[1]))
        head_area = (head_box[2] - head_box[0]) * (head_box[3] - head_box[1])
        if head_area != 0:
            return inter_area / head_area
        return 0
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        helmet_tracks = kwargs.get('helmet_tracks', [])
        
        unhelmeted_heads = [t for t in helmet_tracks if int(t[6]) == 1]
        current_nh_person_ids = set()
        
        for p in tracks:
            p_tid = int(p[4])
            if track_map.get(p_tid) != ID_G_PERSON: 
                continue
                
            max_ioa = 0
            nh_box_match = None
            
            for head in unhelmeted_heads:
                ioa = self._get_intersection_over_head_area(head[:4], p[:4])
                if ioa > max_ioa: 
                    max_ioa = ioa
                    nh_box_match = head[:4]
                    
            if max_ioa > 0.5:
                current_nh_person_ids.add(p_tid)
                
                if p_tid not in self.states: 
                    self.states[p_tid] = {'start_fid': fid, 'last_seen': fid, 'bbox': nh_box_match}
                else:
                    self.states[p_tid]['last_seen'] = fid
                    self.states[p_tid]['bbox'] = nh_box_match
                    
                if fid - self.states[p_tid]['start_fid'] >= self.trigger_fid_diff:
                    triggered.append({'tid': p_tid, 'bbox': self.states[p_tid]['bbox'], 'frame': None, 'fid': fid})
                    
        for tid in list(self.states.keys()):
            if fid - self.states[tid]['last_seen'] > self.grace_fid_diff:
                del self.states[tid]
                
        return triggered

class SignalVehicleDetector(BaseEventDetector):
    event_name, menu_name, gui_name = "signal_vehicle", "신호수차량감지", "NO-SIGNAL"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.motion_threshold_ratio = config.get("motion_threshold_ratio", 0.10)
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30)) 
        
    def _get_distance_point_to_rect(self, point, bbox): 
        return math.sqrt(max(bbox[0] - point[0], 0, point[0] - bbox[2])**2 + max(bbox[1] - point[1], 0, point[1] - bbox[3])**2)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered, current_vehicle_ids = [], set()
        if self.roi_poly.size == 0 or motion_mask is None: 
            return triggered
            
        scale_x, scale_y = 640 / SCREEN_WIDTH, 360 / SCREEN_HEIGHT
        people_points = [get_foot_point(*t[:4]) for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) not in TARGET_VEHICLES: 
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