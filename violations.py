import cv2
import numpy as np
import math
from collections import defaultdict, deque

class ViolationDetector:
    """Streamlined traffic violation detector for helmet, phone, and triple riding"""
    
    def __init__(self, pose_model=None, helmet_model=None):
        self.pose_model = pose_model
        self.helmet_model = helmet_model
        
        # Counters
        self.violations = defaultdict(int)
        self.active_violations = {}
        self.cooldown_frames = 120
        
        # Tracking
        self.bike_tracker = {}
        self.triple_history = {}
        
        # Thresholds (configurable)
        self.conf = {'person': 0.3, 'motorcycle': 0.4, 'phone': 0.25}
        self.triple_min_frames = 5
        self.triple_avg_threshold = 2.0
        
    def detect_violations(self, detections, frame, frame_num):
        """Main detection pipeline - returns list of violations"""
        all_violations = []
        
        # Separate by type
        people = [d for d in detections if d['class'] == 'person' and d['conf'] > self.conf['person']]
        bikes = [d for d in detections if d['class'] == 'motorcycle' and d['conf'] > self.conf['motorcycle']]
        phones = [d for d in detections if d['class'] == 'cell phone' and d['conf'] > self.conf['phone']]
        
        # Update tracking
        self._update_bike_tracking(bikes, people, frame_num)
        
        # Detect each type
        all_violations.extend(self._detect_helmet(bikes, people, frame, frame_num))
        all_violations.extend(self._detect_triple_riding(frame_num))
        all_violations.extend(self._detect_phone_usage(people, phones, frame_num))
        
        return all_violations
    
    def _update_bike_tracking(self, bikes, people, frame_num):
        """Track motorcycles and their riders"""
        current_bikes = {}
        
        for bike in bikes:
            bike_id = self._get_spatial_id(bike['bbox'])
            riders = self._find_riders(bike['bbox'], people)
            
            current_bikes[bike_id] = {
                'bbox': bike['bbox'],
                'riders': riders,
                'last_seen': frame_num
            }
            
            # Update history for triple detection
            if bike_id not in self.triple_history:
                self.triple_history[bike_id] = deque(maxlen=10)
            self.triple_history[bike_id].append(len(riders))
        
        # Update tracker
        self.bike_tracker.update(current_bikes)
        
        # Cleanup old tracks
        cutoff = frame_num - 45
        self.bike_tracker = {k: v for k, v in self.bike_tracker.items() 
                            if v['last_seen'] > cutoff}
        self.triple_history = {k: v for k, v in self.triple_history.items() 
                              if k in self.bike_tracker}
    
    def _find_riders(self, bike_bbox, people):
        """Find people riding the motorcycle"""
        riders = []
        bx1, by1, bx2, by2 = bike_bbox
        bike_center = ((bx1+bx2)/2, (by1+by2)/2)
        bike_width = bx2 - bx1
        
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            person_center = ((px1+px2)/2, (py1+py2)/2)
            
            # Check overlap and position
            iou = self._iou(bike_bbox, person['bbox'])
            dist = math.hypot(bike_center[0]-person_center[0], 
                            bike_center[1]-person_center[1])
            
            # Person above/on bike and close enough
            if iou > 0.1 or (person_center[1] <= bike_center[1] and 
                            dist < bike_width * 0.8):
                riders.append(person)
        
        return riders
    
    def _detect_helmet(self, bikes, people, frame, frame_num):
        """Detect helmet violations"""
        violations = []
        
        for bike in bikes:
            for person in people:
                if self._is_on_bike(person['bbox'], bike['bbox']):
                    has_helmet = self._check_helmet(frame, person['bbox'])
                    
                    if not has_helmet:
                        vid = f"helmet_{self._get_spatial_id(person['bbox'])}"
                        if self._should_count(vid, frame_num):
                            violations.append({
                                'type': 'helmet_violation',
                                'bbox': person['bbox'],
                                'bike_bbox': bike['bbox']
                            })
                            self.violations['helmet'] += 1
        
        return violations
    
    def _detect_triple_riding(self, frame_num):
        """Detect triple riding violations"""
        violations = []
        
        for bike_id, history in self.triple_history.items():
            if len(history) < self.triple_min_frames:
                continue
            
            avg_riders = sum(history) / len(history)
            max_riders = max(history)
            frames_3plus = sum(1 for c in history if c >= 3)
            pct_3plus = frames_3plus / len(history)
            
            # Detection logic
            if (avg_riders >= self.triple_avg_threshold and 
                max_riders >= 3 and pct_3plus >= 0.5):
                
                bike_data = self.bike_tracker[bike_id]
                vid = f"triple_{bike_id}"
                
                if self._should_count(vid, frame_num):
                    violations.append({
                        'type': 'triple_riding',
                        'bbox': bike_data['bbox'],
                        'rider_count': round(avg_riders)
                    })
                    self.violations['triple'] += 1
        
        return violations
    
    def _detect_phone_usage(self, people, phones, frame_num):
        """Detect phone usage while driving"""
        violations = []
        
        for phone in phones:
            for person in people:
                if self._is_phone_near_head(phone['bbox'], person['bbox']):
                    vid = f"phone_{self._get_spatial_id(phone['bbox'])}"
                    
                    if self._should_count(vid, frame_num):
                        violations.append({
                            'type': 'phone_usage',
                            'bbox': phone['bbox'],
                            'person_bbox': person['bbox']
                        })
                        self.violations['phone'] += 1
        
        return violations
    
    def _check_helmet(self, frame, person_bbox):
        """Check if person is wearing helmet"""
        try:
            # Extract head region
            px1, py1, px2, py2 = map(int, person_bbox)
            person_height = py2 - py1
            head_height = int(person_height * 0.25)
            
            head_region = frame[py1:py1+head_height, px1:px2]
            if head_region.size == 0:
                return False
            
            # Use helmet model if available
            if self.helmet_model:
                results = self.helmet_model(head_region, verbose=False)
                if results and len(results[0].boxes) > 0:
                    return True
            
            # Fallback: color-based detection
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Dark colors (common helmet colors)
            dark_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,90]))
            dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
            
            return dark_ratio > 0.3
            
        except:
            return False
    
    def _is_on_bike(self, person_bbox, bike_bbox):
        """Check if person is on motorcycle"""
        iou = self._iou(person_bbox, bike_bbox)
        if iou > 0.1:
            return True
        
        # Check vertical position
        person_cy = (person_bbox[1] + person_bbox[3]) / 2
        bike_cy = (bike_bbox[1] + bike_bbox[3]) / 2
        
        return person_cy <= bike_cy
    
    def _is_phone_near_head(self, phone_bbox, person_bbox):
        """Check if phone is near person's head"""
        person_height = person_bbox[3] - person_bbox[1]
        head_y = person_bbox[1] + person_height * 0.15
        
        phone_center = ((phone_bbox[0]+phone_bbox[2])/2, (phone_bbox[1]+phone_bbox[3])/2)
        head_center = ((person_bbox[0]+person_bbox[2])/2, head_y)
        
        dist = math.hypot(phone_center[0]-head_center[0], phone_center[1]-head_center[1])
        return dist < person_height * 0.35
    
    def _iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter = (x2-x1) * (y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        
        return inter / (area1 + area2 - inter)
    
    def _get_spatial_id(self, bbox):
        """Generate spatial ID for tracking"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return f"{int(cx//50)}_{int(cy//50)}"
    
    def _should_count(self, vid, frame_num):
        """Check if violation should be counted (cooldown logic)"""
        if vid in self.active_violations:
            if frame_num - self.active_violations[vid] < self.cooldown_frames:
                return False
        
        self.active_violations[vid] = frame_num
        return True
    
    def draw_violations(self, frame, violations):
        """Draw violation bounding boxes and labels"""
        colors = {
            'helmet_violation': (0, 0, 255),
            'triple_riding': (255, 165, 0),
            'phone_usage': (255, 0, 255)
        }
        
        for v in violations:
            x1, y1, x2, y2 = map(int, v['bbox'])
            color = colors.get(v['type'], (0, 255, 0))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = v['type'].replace('_', ' ').title()
            if v['type'] == 'triple_riding':
                label = f"Triple Riding: {v.get('rider_count', 3)}"
            
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def get_stats(self):
        """Get violation statistics"""
        return {
            'helmet_violations': self.violations['helmet'],
            'triple_riding': self.violations['triple'],
            'phone_usage': self.violations['phone'],
            'total': sum(self.violations.values())
        }