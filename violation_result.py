"""
Enhanced Traffic Violation Detection System
Optimized for accuracy, efficiency, and clean architecture
"""

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import time


@dataclass
class Detection:
    """Structured detection object"""
    class_id: int
    confidence: float
    bbox: np.ndarray
    center: Tuple[float, float]
    
    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class Violation:
    """Violation record with unique tracking"""
    type: str
    bbox: np.ndarray
    confidence: float
    frame_number: int
    tracker_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class ViolationTracker:
    """Efficient violation tracking to prevent duplicates"""
    
    def __init__(self, cooldown_frames: int = 90):
        self.cooldown_frames = cooldown_frames
        self.active_violations: Dict[str, int] = {}
        
    def get_spatial_id(self, bbox: np.ndarray, v_type: str) -> str:
        """Generate spatial hash for violation location"""
        cx = int((bbox[0] + bbox[2]) / 2 // 40) * 40
        cy = int((bbox[1] + bbox[3]) / 2 // 40) * 40
        return f"{v_type}_{cx}_{cy}"
    
    def should_record(self, bbox: np.ndarray, v_type: str, current_frame: int) -> bool:
        """Check if violation should be recorded (prevents duplicates)"""
        vid = self.get_spatial_id(bbox, v_type)
        
        if vid in self.active_violations:
            last_frame = self.active_violations[vid]
            if current_frame - last_frame < self.cooldown_frames:
                return False
        
        self.active_violations[vid] = current_frame
        return True
    
    def cleanup(self, current_frame: int):
        """Remove old violation records"""
        threshold = current_frame - (self.cooldown_frames * 2)
        self.active_violations = {
            vid: frame for vid, frame in self.active_violations.items() 
            if frame > threshold
        }


class MotorcycleTracker:
    """Advanced motorcycle and rider tracking"""
    
    def __init__(self, max_age: int = 30):
        self.tracks: Dict[str, Dict] = {}
        self.max_age = max_age
        self.next_id = 0
        
    def get_track_id(self, bbox: np.ndarray, frame_number: int) -> str:
        """Match detection to existing track or create new one"""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Find closest existing track
        best_match = None
        best_distance = float('inf')
        
        for tid, track in self.tracks.items():
            if frame_number - track['last_seen'] > self.max_age:
                continue
                
            dist = np.hypot(center[0] - track['center'][0], 
                           center[1] - track['center'][1])
            
            if dist < best_distance and dist < 80:
                best_distance = dist
                best_match = tid
        
        if best_match:
            return best_match
        
        # Create new track
        new_id = f"moto_{self.next_id}"
        self.next_id += 1
        return new_id
    
    def update(self, track_id: str, bbox: np.ndarray, riders: List[Detection], 
               frame_number: int):
        """Update track with current frame data"""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'rider_history': deque(maxlen=15),
                'center': center,
                'last_seen': frame_number,
                'bbox': bbox
            }
        
        track = self.tracks[track_id]
        track['center'] = center
        track['bbox'] = bbox
        track['last_seen'] = frame_number
        track['rider_history'].append(len(riders))
    
    def get_stable_rider_count(self, track_id: str) -> Optional[int]:
        """Get statistically stable rider count"""
        if track_id not in self.tracks:
            return None
        
        history = self.tracks[track_id]['rider_history']
        if len(history) < 8:
            return None
        
        # Use median for stability
        return int(np.median(list(history)))
    
    def cleanup(self, current_frame: int):
        """Remove stale tracks"""
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if current_frame - track['last_seen'] <= self.max_age
        }


class HelmetDetector:
    """Specialized helmet detection"""
    
    def __init__(self, helmet_model_path: Optional[str] = None):
        self.helmet_model = None
        if helmet_model_path and Path(helmet_model_path).exists():
            try:
                self.helmet_model = YOLO(helmet_model_path)
            except Exception as e:
                print(f"Warning: Could not load helmet model: {e}")
    
    def has_helmet(self, frame: np.ndarray, person_bbox: np.ndarray) -> bool:
        """Determine if person is wearing helmet"""
        head_region = self._extract_head(frame, person_bbox)
        if head_region is None:
            return False
        
        # Use specialized model if available
        if self.helmet_model:
            try:
                results = self.helmet_model(head_region, verbose=False)
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for i in range(len(boxes)):
                            if boxes.conf[i] > 0.6:
                                return True
            except:
                pass
        
        # Fallback to color/shape analysis
        return self._analyze_helmet_features(head_region)
    
    def _extract_head(self, frame: np.ndarray, person_bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract head region from person detection"""
        try:
            x1, y1, x2, y2 = map(int, person_bbox)
            h, w = frame.shape[:2]
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Top 25% of person bbox
            person_height = y2 - y1
            head_height = max(30, int(person_height * 0.25))
            head_region = frame[y1:y1+head_height, x1:x2]
            
            return head_region if head_region.size > 0 else None
        except:
            return None
    
    def _analyze_helmet_features(self, head_region: np.ndarray) -> bool:
        """Analyze color and shape features for helmet presence"""
        try:
            if head_region.size == 0:
                return False
            
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Helmet color masks (dark, bright, light colored helmets)
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), 
                                   np.array([180, 255, 80]))
            bright_mask = cv2.inRange(hsv, np.array([0, 120, 120]), 
                                     np.array([180, 255, 255]))
            light_mask = cv2.inRange(hsv, np.array([0, 0, 190]), 
                                    np.array([180, 60, 255]))
            
            combined_mask = cv2.bitwise_or(dark_mask, 
                                          cv2.bitwise_or(bright_mask, light_mask))
            
            helmet_ratio = np.sum(combined_mask > 0) / combined_mask.size
            
            # Helmet typically covers >40% of head region with characteristic colors
            return helmet_ratio > 0.4
        except:
            return False


class EnhancedTrafficDetector:
    """Main detection system with clean architecture"""
    
    # COCO class IDs
    CLASS_IDS = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7,
        'cell phone': 67
    }
    
    def __init__(self, 
                 detection_model: str = 'yolov8x.pt',
                 helmet_model: Optional[str] = None,
                 conf_threshold: float = 0.4):
        
        # Load models
        self.model = YOLO(detection_model)
        self.helmet_detector = HelmetDetector(helmet_model)
        
        # Load pose model for better person analysis
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')
        except:
            self.pose_model = None
        
        self.conf_threshold = conf_threshold
        
        # Trackers
        self.violation_tracker = ViolationTracker(cooldown_frames=90)
        self.moto_tracker = MotorcycleTracker(max_age=30)
        
        # Statistics
        self.stats = {
            'helmet_violations': 0,
            'triple_riding': 0,
            'phone_usage': 0
        }
        
        self.frame_count = 0
    
    def _extract_detections(self, results) -> List[Detection]:
        """Convert YOLO results to Detection objects"""
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return detections
        
        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            detections.append(Detection(
                class_id=int(boxes.cls[i]),
                confidence=float(boxes.conf[i]),
                bbox=bbox,
                center=center
            ))
        
        return detections
    
    def _is_person_on_motorcycle(self, person: Detection, 
                                 motorcycle: Detection) -> bool:
        """Determine if person is riding motorcycle"""
        # IoU check
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            inter = (x2 - x1) * (y2 - y1)
            union = person.area + motorcycle.area - inter
            return inter / union if union > 0 else 0.0
        
        iou = calculate_iou(person.bbox, motorcycle.bbox)
        if iou > 0.15:
            return True
        
        # Spatial relationship check
        moto_center = motorcycle.center
        person_center = person.center
        
        # Person should be above and close to motorcycle center
        moto_width = motorcycle.bbox[2] - motorcycle.bbox[0]
        horizontal_dist = abs(person_center[0] - moto_center[0])
        
        if horizontal_dist > moto_width * 0.6:
            return False
        
        # Person center should be around or slightly above motorcycle center
        vertical_dist = person_center[1] - moto_center[1]
        moto_height = motorcycle.bbox[3] - motorcycle.bbox[1]
        
        return -moto_height * 0.3 < vertical_dist < moto_height * 0.4
    
    def _find_riders(self, motorcycle: Detection, 
                    people: List[Detection]) -> List[Detection]:
        """Find all people riding a motorcycle"""
        riders = []
        
        for person in people:
            if self._is_person_on_motorcycle(person, motorcycle):
                riders.append(person)
        
        return riders
    
    def _detect_helmet_violations(self, frame: np.ndarray, 
                                  detections: List[Detection]) -> List[Violation]:
        """Detect helmet violations"""
        violations = []
        
        motorcycles = [d for d in detections 
                      if d.class_id == self.CLASS_IDS['motorcycle'] 
                      and d.confidence > self.conf_threshold]
        
        people = [d for d in detections 
                 if d.class_id == self.CLASS_IDS['person'] 
                 and d.confidence > 0.35]
        
        for motorcycle in motorcycles:
            riders = self._find_riders(motorcycle, people)
            
            for rider in riders:
                has_helmet = self.helmet_detector.has_helmet(frame, rider.bbox)
                
                if not has_helmet:
                    if self.violation_tracker.should_record(rider.bbox, 'helmet', 
                                                           self.frame_count):
                        violations.append(Violation(
                            type='helmet_violation',
                            bbox=rider.bbox,
                            confidence=rider.confidence,
                            frame_number=self.frame_count,
                            metadata={'motorcycle_bbox': motorcycle.bbox}
                        ))
                        self.stats['helmet_violations'] += 1
        
        return violations
    
    def _detect_triple_riding(self, detections: List[Detection]) -> List[Violation]:
        """Detect triple riding (3+ people on motorcycle)"""
        violations = []
        
        motorcycles = [d for d in detections 
                      if d.class_id == self.CLASS_IDS['motorcycle'] 
                      and d.confidence > self.conf_threshold]
        
        people = [d for d in detections 
                 if d.class_id == self.CLASS_IDS['person'] 
                 and d.confidence > 0.35]
        
        for motorcycle in motorcycles:
            riders = self._find_riders(motorcycle, people)
            
            # Update tracking
            track_id = self.moto_tracker.get_track_id(motorcycle.bbox, 
                                                      self.frame_count)
            self.moto_tracker.update(track_id, motorcycle.bbox, riders, 
                                    self.frame_count)
            
            # Get stable count
            stable_count = self.moto_tracker.get_stable_rider_count(track_id)
            
            if stable_count is not None and stable_count >= 3:
                if self.violation_tracker.should_record(motorcycle.bbox, 'triple', 
                                                       self.frame_count):
                    violations.append(Violation(
                        type='triple_riding',
                        bbox=motorcycle.bbox,
                        confidence=min(0.95, 0.7 + (stable_count / 10)),
                        frame_number=self.frame_count,
                        tracker_id=track_id,
                        metadata={'rider_count': stable_count}
                    ))
                    self.stats['triple_riding'] += 1
        
        return violations
    
    def _detect_phone_usage(self, frame: np.ndarray, 
                           detections: List[Detection]) -> List[Violation]:
        """Detect phone usage while driving"""
        violations = []
        
        people = [d for d in detections 
                 if d.class_id == self.CLASS_IDS['person'] 
                 and d.confidence > 0.35]
        
        phones = [d for d in detections 
                 if d.class_id == self.CLASS_IDS['cell phone'] 
                 and d.confidence > 0.3]
        
        vehicles = [d for d in detections 
                   if d.class_id in [self.CLASS_IDS['car'], 
                                    self.CLASS_IDS['motorcycle'],
                                    self.CLASS_IDS['bus'],
                                    self.CLASS_IDS['truck']] 
                   and d.confidence > 0.4]
        
        for phone in phones:
            for person in people:
                # Check if phone is near person's head
                person_height = person.bbox[3] - person.bbox[1]
                distance = np.hypot(phone.center[0] - person.center[0],
                                   phone.center[1] - person.center[1])
                
                if distance < person_height * 0.4:
                    # Check if person is in a vehicle
                    in_vehicle = False
                    for vehicle in vehicles:
                        vehicle_width = vehicle.bbox[2] - vehicle.bbox[0]
                        v_distance = np.hypot(person.center[0] - vehicle.center[0],
                                            person.center[1] - vehicle.center[1])
                        if v_distance < vehicle_width * 1.2:
                            in_vehicle = True
                            break
                    
                    if in_vehicle:
                        if self.violation_tracker.should_record(phone.bbox, 'phone', 
                                                               self.frame_count):
                            violations.append(Violation(
                                type='phone_usage',
                                bbox=phone.bbox,
                                confidence=phone.confidence,
                                frame_number=self.frame_count,
                                metadata={'person_bbox': person.bbox}
                            ))
                            self.stats['phone_usage'] += 1
        
        return violations
    
    def _draw_violations(self, frame: np.ndarray, 
                        violations: List[Violation]) -> np.ndarray:
        """Draw violations on frame"""
        colors = {
            'helmet_violation': (0, 0, 255),      # Red
            'triple_riding': (0, 165, 255),       # Orange
            'phone_usage': (255, 0, 255)          # Magenta
        }
        
        labels = {
            'helmet_violation': 'No Helmet',
            'triple_riding': 'Triple Riding',
            'phone_usage': 'Phone Usage'
        }
        
        for violation in violations:
            try:
                x1, y1, x2, y2 = map(int, violation.bbox)
                color = colors.get(violation.type, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = labels.get(violation.type, violation.type)
                if violation.type == 'triple_riding' and 'rider_count' in violation.metadata:
                    label = f"{label}: {violation.metadata['rider_count']} riders"
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                                        0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                            (x1 + label_w + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw motorcycle box for helmet violations
                if violation.type == 'helmet_violation' and 'motorcycle_bbox' in violation.metadata:
                    mx1, my1, mx2, my2 = map(int, violation.metadata['motorcycle_bbox'])
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 255), 1)
                
            except Exception as e:
                continue
        
        return frame
    
    def _draw_statistics(self, frame: np.ndarray, 
                        current_violations: List[Violation]):
        """Draw statistics overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (400, 140), (255, 255, 255), 2)
        
        # Statistics
        stats_text = [
            f"Frame: {self.frame_count}",
            f"Current Violations: {len(current_violations)}",
            f"Total Violations: {sum(self.stats.values())}",
            f"  Helmet: {self.stats['helmet_violations']}",
            f"  Triple: {self.stats['triple_riding']}",
            f"  Phone: {self.stats['phone_usage']}"
        ]
        
        colors = [(255, 255, 255), (0, 255, 0), (255, 255, 0), 
                 (0, 0, 255), (0, 165, 255), (255, 0, 255)]
        
        for i, (text, color) in enumerate(zip(stats_text, colors)):
            y = 35 + i * 20
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     display: bool = True):
        """Process video and detect violations"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\nProcessing: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}\n")
        
        start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Run detection
                results = self.model(frame, verbose=False, conf=self.conf_threshold)
                detections = self._extract_detections(results)
                
                # Detect all violation types
                all_violations = []
                all_violations.extend(self._detect_helmet_violations(frame, detections))
                all_violations.extend(self._detect_triple_riding(detections))
                all_violations.extend(self._detect_phone_usage(frame, detections))
                
                # Draw results
                frame = self._draw_violations(frame, all_violations)
                self._draw_statistics(frame, all_violations)
                
                # Periodic cleanup
                if self.frame_count % 100 == 0:
                    self.violation_tracker.cleanup(self.frame_count)
                    self.moto_tracker.cleanup(self.frame_count)
                
                # Write output
                if writer:
                    writer.write(frame)
                
                # Display
                if display:
                    cv2.imshow('Traffic Violation Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                # Progress update
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = self.frame_count / elapsed
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f} | "
                          f"Violations: {sum(self.stats.values())}", end='\r')
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final results
        elapsed = time.time() - start_time
        print(f"\n\n{'='*60}")
        print("DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Processing Time: {elapsed:.2f}s")
        print(f"Average FPS: {self.frame_count / elapsed:.2f}")
        print(f"\nViolations Detected:")
        print(f"  Helmet Violations: {self.stats['helmet_violations']}")
        print(f"  Triple Riding: {self.stats['triple_riding']}")
        print(f"  Phone Usage: {self.stats['phone_usage']}")
        print(f"  Total: {sum(self.stats.values())}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Traffic Violation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input video path')
    parser.add_argument('--output', '-o', default=None,
                       help='Output video path (optional)')
    parser.add_argument('--detection-model', default='yolov8x.pt',
                       help='YOLO detection model (default: yolov8x.pt)')
    parser.add_argument('--helmet-model', default=None,
                       help='Optional specialized helmet detection model')
    parser.add_argument('--confidence', type=float, default=0.4,
                       help='Confidence threshold (default: 0.4)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display window')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Initialize detector
    detector = EnhancedTrafficDetector(
        detection_model=args.detection_model,
        helmet_model=args.helmet_model,
        conf_threshold=args.confidence
    )
    
    # Process video
    detector.process_video(
        video_path=args.input,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()