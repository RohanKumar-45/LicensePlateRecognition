import cv2
import numpy as np
import torch
from ultralytics import YOLO
from config import *

def check_cuda():
    """Check CUDA availability"""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA Devices: {device_count}")
        print(f"Current Device: {current_device} - {device_name}")
        return True
    else:
        print("CUDA not available, using CPU")
        return False

def load_models(use_cuda=True):
    """Load YOLO models for vehicle and plate detection"""
    print("Loading YOLOv8 models...")
    device = 0 if use_cuda and torch.cuda.is_available() else 'cpu'
    
    try:
        vehicle_model = YOLO(VEHICLE_DETECTION_MODEL_PATH)
        vehicle_model.to(device)
        print(f"Vehicle detection model loaded successfully on {device}")
        
        if os.path.exists(PLATE_DETECTION_MODEL_PATH):
            plate_model = YOLO(PLATE_DETECTION_MODEL_PATH)
            plate_model.to(device)
            print(f"License plate detection model loaded successfully on {device}")
        else:
            print(f"License plate model not found, using vehicle model for both")
            plate_model = vehicle_model
            
        return vehicle_model, plate_model
    except Exception as e:
        print(f"Error loading YOLOv8 models: {str(e)}")
        return None, None

def detect_vehicles(frame, model):
    """Detect vehicles in frame"""
    vehicles = []
    
    try:
        results = model(frame, stream=True, verbose=False, conf=0.40)
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls.item())
                
                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    class_name, color = VEHICLE_CLASSES[cls_id]
                    
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0
                    area = width * height
                    
                    # Vehicle class refinement logic
                    if cls_id == 6:  # Auto
                        if aspect_ratio < 0.7:
                            cls_id = 3
                            class_name, color = VEHICLE_CLASSES[cls_id]
                            confidence *= 0.95
                        elif aspect_ratio > 1.5:
                            cls_id = 2
                            class_name, color = VEHICLE_CLASSES[cls_id]
                            confidence *= 0.95
                    
                    is_two_wheeler = cls_id in TWO_WHEELER_CLASSES
                    
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': cls_id,
                        'class_name': class_name,
                        'color': color,
                        'is_two_wheeler': is_two_wheeler,
                        'aspect_ratio': aspect_ratio,
                        'area': area
                    })
        
        return vehicles
    except Exception as e:
        print(f"Error in vehicle detection: {str(e)}")
        return []

def detect_plates(frame, model, vehicle_boxes=None):
    """Detect license plates in frame"""
    plates = []
    
    try:
        if vehicle_boxes:
            for vehicle in vehicle_boxes:
                x1, y1, x2, y2 = vehicle['bbox']
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                vehicle_roi = frame[y1:y2, x1:x2]
                
                if vehicle_roi.size == 0:
                    continue
                
                # Adjust confidence threshold based on vehicle type
                if vehicle.get('is_two_wheeler', False):
                    conf_threshold = 0.18
                elif vehicle.get('class_name') == 'Auto':
                    conf_threshold = 0.22
                else:
                    conf_threshold = 0.25
                
                results = model(vehicle_roi, conf=conf_threshold, verbose=False)
                
                # Additional detection for two-wheelers (lower half)
                if vehicle.get('is_two_wheeler', False) and vehicle_roi.shape[0] > 20:
                    lower_half_y = y1 + (y2 - y1) // 2
                    lower_half_roi = frame[lower_half_y:y2, x1:x2]
                    if lower_half_roi.size > 0:
                        lower_results = model(lower_half_roi, conf=0.15, verbose=False)
                        
                        for result in lower_results:
                            for box in result.boxes:
                                roi_x1, roi_y1, roi_x2, roi_y2 = map(int, box.xyxy[0].tolist())
                                confidence = float(box.conf.item())
                                
                                orig_x1 = x1 + roi_x1
                                orig_y1 = lower_half_y + roi_y1
                                orig_x2 = x1 + roi_x2
                                orig_y2 = lower_half_y + roi_y2
                                
                                plate_width = orig_x2 - orig_x1
                                plate_height = orig_y2 - orig_y1
                                if plate_height > 0:
                                    plate_aspect = plate_width / plate_height
                                    if 1.7 <= plate_aspect <= 5.5:
                                        plates.append({
                                            'bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                                            'confidence': confidence,
                                            'vehicle_type': vehicle['class_name'],
                                            'vehicle_color': vehicle['color'],
                                            'is_two_wheeler': vehicle.get('is_two_wheeler', False),
                                            'vehicle_id': vehicle.get('track_id', -1),
                                            'plate_aspect_ratio': plate_aspect
                                        })
                
                # Process main ROI detections
                for result in results:
                    for box in result.boxes:
                        roi_x1, roi_y1, roi_x2, roi_y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf.item())
                        
                        orig_x1 = x1 + roi_x1
                        orig_y1 = y1 + roi_y1
                        orig_x2 = x1 + roi_x2
                        orig_y2 = y1 + roi_y2
                        
                        plate_width = orig_x2 - orig_x1
                        plate_height = orig_y2 - orig_y1
                        if plate_height > 0:
                            plate_aspect = plate_width / plate_height
                            if 1.5 <= plate_aspect <= 6.0:
                                plates.append({
                                    'bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                                    'confidence': confidence,
                                    'vehicle_type': vehicle['class_name'],
                                    'vehicle_color': vehicle['color'],
                                    'is_two_wheeler': vehicle.get('is_two_wheeler', False),
                                    'vehicle_id': vehicle.get('track_id', -1),
                                    'plate_aspect_ratio': plate_aspect
                                })
        else:
            # Direct plate detection without vehicle context
            results = model(frame, conf=0.3, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    
                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    if plate_height > 0:
                        plate_aspect = plate_width / plate_height
                        if 1.5 <= plate_aspect <= 6.0:
                            plates.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'vehicle_type': 'Unknown',
                                'vehicle_color': (255, 255, 255),
                                'is_two_wheeler': False,
                                'vehicle_id': -1,
                                'plate_aspect_ratio': plate_aspect
                            })
        
        if len(plates) > 1:
            plates = filter_overlapping_plates(plates)
        
        return plates
        
    except Exception as e:
        print(f"Error in license plate detection: {str(e)}")
        return []

def filter_overlapping_plates(plates, iou_threshold=0.5):
    """Filter overlapping plate detections"""
    filtered_plates = []
    plates_sorted = sorted(plates, key=lambda x: x['confidence'], reverse=True)
    
    while plates_sorted:
        best = plates_sorted.pop(0)
        filtered_plates.append(best)
        
        plates_sorted = [
            plate for plate in plates_sorted
            if compute_iou(best['bbox'], plate['bbox']) < iou_threshold
        ]
    
    return filtered_plates


def detect_objects(frame, model, conf=0.25):
    """Detect generic objects like person and cell phone from a YOLO model.

    Returns a list of dicts with keys: 'bbox' (x1,y1,x2,y2), 'conf', 'class'
    where class is a lowercase string such as 'person', 'cell phone', 'motorcycle'.
    """
    objects = []
    try:
        results = model(frame, conf=conf, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                # We only care about person(0) and cell phone(67) and motorcycle(3)
                if cls_id in (0, 67, 3):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    if cls_id == 0:
                        cls_name = 'person'
                    elif cls_id == 67:
                        cls_name = 'cell phone'
                    elif cls_id == 3:
                        cls_name = 'motorcycle'
                    else:
                        cls_name = str(cls_id)

                    objects.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': confidence,
                        'class': cls_name
                    })
    except Exception as e:
        print(f"Error in detect_objects: {e}")

    return objects

def detect_plate_color(image, bbox):
    """Detect license plate color type"""
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    plate_img = image[y1:y2, x1:x2]
    
    if plate_img.size == 0:
        return "Private"
    
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    
    # Color ranges
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([50, 255, 255])
    
    lower_green = np.array([35, 25, 25])
    upper_green = np.array([95, 255, 255])
    
    lower_red1 = np.array([0, 60, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 60, 50])
    upper_red2 = np.array([180, 255, 255])
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 100, 60])
    
    # Create masks
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                               cv2.inRange(hsv, lower_red2, upper_red2))
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    total_pixels = (y2-y1) * (x2-x1)
    if total_pixels == 0:
        return "Private"
    
    yellow_percentage = cv2.countNonZero(yellow_mask) / total_pixels * 100
    green_percentage = cv2.countNonZero(green_mask) / total_pixels * 100
    red_percentage = cv2.countNonZero(red_mask) / total_pixels * 100
    black_percentage = cv2.countNonZero(black_mask) / total_pixels * 100
    
    # Determine plate type
    # NOTE: There is no special 'Government' classification in this system.
    # Red or black plates will be treated as Private unless explicitly
    # identified by some other reliable marker in future.
    if yellow_percentage > 18:
        return "Commercial"
    elif green_percentage > 18:
        return "Electric Vehicle"
    else:
        return "Private"

def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou