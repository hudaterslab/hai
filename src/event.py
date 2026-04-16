import cv2
import time
import math
import numpy as np
from collections import defaultdict, deque
from common import (
    ID_H_NO_HELMET, ID_H_PERSON, ID_G_PERSON, TARGET_VEHICLES, 
    SCREEN_WIDTH, SCREEN_HEIGHT, get_foot_point, get_check_point, 
    get_center_point, get_distance, ccw, calculate_iou
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
                    
                    if self.states[tid]['start'] == 0 or get_distance(c, self.states[tid]['pos']) > 30:
                        self.states[tid].update({'start': now, 'pos': c})
                    elif now - self.states[tid]['start'] > 5.0:
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
        
        # 💡 [핵심] 기본 스냅샷 모드를 'crossing_moment'로 변경하여 선을 넘는 정확한 찰나를 캡처
        self.snapshot_mode = config.get("snapshot_mode", "crossing_moment")
        self.distance_ratio = config.get("distance_ratio", 0.5)
        self.min_distance_px = config.get("min_distance_px", 15)
        self.candidate_ttl_sec = config.get("candidate_ttl_sec", 5.0)
        self.direction_check = config.get("direction_check", True)

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
                
            curr_pos = get_foot_point(*t[:4])
            obj_width = t[2] - t[0]
            
            if tid in self.prev and tid not in self.candidates:
                for p1, p2 in self.lines:
                    if self._is_intersect(p1, p2, self.prev[tid], curr_pos):
                        self.candidates[tid] = {
                            'crossing_pt': curr_pos, 
                            'width': obj_width, 
                            'timestamp': now, 
                            'line': (p1, p2),
                            'entry_side': ccw(p1, p2, self.prev[tid]), 
                            # 💡 교차하는 순간의 frame을 무조건 복사하여 보관
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
                
                if self.direction_check:
                    direction_ok = (cand['entry_side'] != 0 and curr_side != 0 and cand['entry_side'] * curr_side < 0)
                else:
                    direction_ok = True
                
                if direction_ok and moved_dist > max(cand['width'] * self.distance_ratio, self.min_distance_px):
                    triggered.append({
                        'tid': tid, 
                        'bbox': cand['bbox'], 
                        'frame': cand['frame'], 
                        'fid': cand['fid']
                    })
                    del self.candidates[tid]
                elif now - cand['timestamp'] > self.candidate_ttl_sec:
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

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        h_map = track_maps["helmet"]
        g_map = track_maps["general"]
        
        no_helmets = [t for t in helmet_tracks if h_map.get(int(t[4])) == ID_H_NO_HELMET]
        persons = [t for t in general_tracks if g_map.get(int(t[4])) == ID_G_PERSON]
        
        for nh in no_helmets:
            max_ioa = 0
            for p in persons:
                ioa = self._get_intersection_over_head_area(nh[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa
                    
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
        self.motion_threshold_ratio = 0.10
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30)) 

    def _get_distance_point_to_rect(self, point, bbox):
        px, py = point
        bx1, by1, bx2, by2 = bbox
        return math.sqrt(max(bx1 - px, 0, px - bx2)**2 + max(by1 - py, 0, py - by2)**2)

    def process(self, helmet_tracks, general_tracks, track_maps, motion_mask, frame, fid):
        triggered = []
        current_vehicle_ids = set()
        g_map = track_maps["general"]
        
        if self.roi_poly.size == 0 or motion_mask is None:
            return triggered
        
        scale_x = 640 / SCREEN_WIDTH
        scale_y = 360 / SCREEN_HEIGHT
        people_points = [get_center_point(*t[:4]) for t in general_tracks if g_map.get(int(t[4])) == ID_G_PERSON]
        
        for t in general_tracks:
            tid = int(t[4])
            if g_map.get(tid) not in TARGET_VEHICLES:
                continue
                
            current_vehicle_ids.add(tid)
            x1, y1, x2, y2 = t[:4]
            center = get_center_point(x1, y1, x2, y2)
            self.vehicle_history[tid].append(center)
            
            if len(self.vehicle_history[tid]) > 5 and get_distance(self.vehicle_history[tid][0], self.vehicle_history[tid][-1]) >= 40.0:
                mx1 = max(0, int(x1 * scale_x))
                my1 = max(0, int(y1 * scale_y))
                mx2 = min(640, int(x2 * scale_x))
                my2 = min(360, int(y2 * scale_y))
                
                if mx2 > mx1 and my2 > my1:
                    car_roi_mask = motion_mask[my1:my2, mx1:mx2]
                    _, motion_only = cv2.threshold(car_roi_mask, 250, 255, cv2.THRESH_BINARY)
                    total_pixels = (mx2 - mx1) * (my2 - my1)
                    
                    if total_pixels > 0 and (cv2.countNonZero(motion_only) / total_pixels) > self.motion_threshold_ratio:
                        if cv2.pointPolygonTest(self.roi_poly, center, False) >= 0:
                            safe_radius = y2 - y1
                            if not any(self._get_distance_point_to_rect(pp, (x1, y1, x2, y2)) < safe_radius for pp in people_points):
                                triggered.append({
                                    'tid': tid, 
                                    'bbox': t[:4], 
                                    'frame': None, 
                                    'fid': fid
                                })
                                
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