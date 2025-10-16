import cv2
import numpy as np
import pandas as pd
import time
import os
import uuid
from collections import defaultdict
import matplotlib
# Use non-interactive backend to avoid tkinter/TkAgg issues in headless or multi-threaded environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sort.sort import Sort

from config import *
from detection import (check_cuda, load_models, detect_vehicles, 
                       detect_plates, detect_plate_color, compute_iou,
                       detect_objects)
from violations import ViolationDetector
from ocr_utils import (extract_text_with_plate_recognizer, process_plate_text,
                       is_valid_indian_plate, is_duplicate_plate, plate_similarity)

def generate_visualization_data(detections):
    """Generate data for visualization charts"""
    if not detections:
        return None
    
    data = {
        'Vehicle Types': defaultdict(int),
        'Plate Types': defaultdict(int)
    }
    
    for detection in detections:
        data['Vehicle Types'][detection['Vehicle Type']] += 1
        data['Plate Types'][detection['Plate Type']] += 1
    
    return data

def save_visualization(data, output_path):
    """Save visualization charts"""
    if not data:
        return
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    labels = list(data['Vehicle Types'].keys())
    sizes = list(data['Vehicle Types'].values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Vehicle Type Distribution')
    
    plt.subplot(1, 2, 2)
    labels = list(data['Plate Types'].keys())
    sizes = list(data['Plate Types'].values())
    plt.bar(labels, sizes)
    plt.title('License Plate Types')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_violation_visualization(violations, output_path):
    """Save a bar chart comparing violation types"""
    try:
        if not violations:
            return
        counts = defaultdict(int)
        for v in violations:
            vtype = v.get('type') if isinstance(v, dict) else getattr(v, 'type', 'unknown')
            counts[vtype] += 1

        labels = list(counts.keys())
        sizes = [counts[k] for k in labels]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, sizes, color='tab:orange')
        plt.title('Violation Types Comparison')
        plt.ylabel('Count')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Failed to save violation visualization: {e}")


def _save_thumb_from_bbox(img, bbox, prefix="violation"):
    """Save a thumbnail crop for a bounding box to OUTPUT_FOLDER and return filename."""
    try:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = max(0, int(x2)); y2 = max(0, int(y2))
        h, w = img.shape[:2]
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Resize thumbnail with fixed height
        thumb_h = 160
        try:
            thumb_w = int((crop.shape[1] / crop.shape[0]) * thumb_h)
        except Exception:
            thumb_w = 160
        thumb_w = max(64, thumb_w)
        thumb = cv2.resize(crop, (thumb_w, thumb_h))
        thumb_name = f"{prefix}_{uuid.uuid4().hex}.jpg"
        thumb_path = os.path.join(OUTPUT_FOLDER, thumb_name)
        cv2.imwrite(thumb_path, thumb)
        return thumb_name
    except Exception as e:
        print(f"_save_thumb_from_bbox error: {e}")
        return None

def process_image(image_path):
    """Process a single image for license plate detection"""
    print(f"Processing image: {image_path}")
    
    cuda_available = check_cuda()
    vehicle_model, plate_model = load_models(use_cuda=cuda_available)
    
    if vehicle_model is None:
        return {"error": "Failed to load vehicle detection model"}
    
    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": "Failed to read image file"}
    
    output_frame = frame.copy()
    # Run object detection useful for violations (people, phones, motorcycles)
    object_detections = detect_objects(frame, vehicle_model, conf=0.25)
    violation_detector = ViolationDetector(pose_model=None, helmet_model=None)
    try:
        image_violations = violation_detector.detect_violations(object_detections, frame, frame_num=1)
    except Exception as e:
        print(f"ViolationDetector error on image: {e}")
        image_violations = []

    # Draw violations on output image
    if image_violations:
        output_frame = violation_detector.draw_violations(output_frame, image_violations)
    
    # use module-level _save_thumb_from_bbox
    
    # Detect vehicles
    vehicles = detect_vehicles(frame, vehicle_model)
    raw_detections = []
    
    # Draw vehicle bounding boxes
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle['bbox']
        color = vehicle['color']
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_frame, vehicle['class_name'],
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Detect plates
    plates = detect_plates(frame, plate_model, 
                          vehicles if plate_model != vehicle_model else None)
    
    # Process each detected plate
    for plate in plates:
        x1, y1, x2, y2 = plate['bbox']
        vehicle_type = plate['vehicle_type']
        
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        plate_type = detect_plate_color(frame, (x1, y1, x2, y2))
        cv2.putText(output_frame, f"Plate: {plate_type}",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Extract and process text
        plate_text = extract_text_with_plate_recognizer(frame, (x1, y1, x2, y2))
        processed_text = process_plate_text(plate_text)
        
        if is_valid_indian_plate(processed_text):
            if not is_duplicate_plate(processed_text, raw_detections, threshold=2):
                raw_detections.append({
                    "License Plate": processed_text,
                    "Plate Type": plate_type,
                    "Vehicle Type": vehicle_type,
                    "Confidence": plate['confidence'],
                    "text": processed_text,
                    "Box": (x1, y1, x2, y2)
                })
    
    # Save output image
    output_filename = f"output_{uuid.uuid4().hex}.jpg"
    output_image_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_image_path, output_frame)
    
    # Save thumbnails for image violations (they are dicts from ViolationDetector)
    saved_image_violations = []
    for v in image_violations:
        try:
            # v is expected to be a dict with 'bbox','type','confidence' etc.
            if isinstance(v, dict):
                bbox = v.get('bbox') or v.get('Box')
                v_out = dict(v)
            else:
                bbox = getattr(v, 'bbox', None)
                v_out = {
                    'type': getattr(v, 'type', None),
                    'bbox': tuple(map(int, bbox)) if bbox is not None else None,
                    'confidence': float(getattr(v, 'confidence', 0.0)),
                    'frame_number': int(getattr(v, 'frame_number', 0))
                }

            if bbox:
                try:
                    bbox_tuple = tuple(map(int, bbox))
                    thumb = _save_thumb_from_bbox(frame, bbox_tuple, prefix='violation_img')
                    if thumb:
                        v_out['thumb'] = thumb
                except Exception:
                    pass

            saved_image_violations.append(v_out)
        except Exception as e:
            print(f"Error saving image violation thumb: {e}")
    
    # Generate visualization
    visualization_data = generate_visualization_data(raw_detections)
    visualization_filename = f"visualization_{uuid.uuid4().hex}.png"
    visualization_path = os.path.join(OUTPUT_FOLDER, visualization_filename)
    save_visualization(visualization_data, visualization_path)
    
    # Save Excel file
    excel_filename = f"plates_{uuid.uuid4().hex}.xlsx"
    excel_path = os.path.join(OUTPUT_FOLDER, excel_filename)
    
    if raw_detections:
        plate_df = pd.DataFrame(raw_detections)
        plate_df = plate_df.drop(columns=["text", "Box"], errors='ignore')
        # Write plates to Excel and add a sheet for violations if any
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                plate_df.to_excel(writer, sheet_name='Plates', index=False)
                if saved_image_violations:
                    vdf = pd.DataFrame(saved_image_violations)
                    vdf.to_excel(writer, sheet_name='Violations', index=False)
        except Exception:
            plate_df.to_excel(excel_path, index=False)
    else:
        pd.DataFrame(columns=["License Plate", "Plate Type", "Vehicle Type", 
                              "Confidence"]).to_excel(excel_path, index=False)
        # Still write violations sheet if present
        if saved_image_violations:
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                    vdf = pd.DataFrame(saved_image_violations)
                    vdf.to_excel(writer, sheet_name='Violations', index=False)
            except Exception:
                pass

    # Save violation visualization (bar chart)
    violation_vis_filename = None
    if saved_image_violations:
        violation_vis_filename = f"violations_vis_{uuid.uuid4().hex}.png"
        violation_vis_path = os.path.join(OUTPUT_FOLDER, violation_vis_filename)
        save_violation_visualization(saved_image_violations, violation_vis_path)
    
    return {
        "total_plates": len(plates),
        "plates": [plate for plate in raw_detections if "text" not in plate],
        "violations": saved_image_violations,
        "output_image": output_filename,
        "excel_file": excel_filename,
        "visualization": visualization_filename,
        "violation_visualization": violation_vis_filename
    }

def process_video(video_path):
    """Process a video for license plate detection"""
    print(f"Processing video: {video_path}")
    
    cuda_available = check_cuda()
    vehicle_model, plate_model = load_models(use_cuda=cuda_available)
    
    if vehicle_model is None:
        return {"error": "Failed to load vehicle detection model"}
    
    tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, 
                   iou_threshold=SORT_IOU_THRESHOLD)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video {video_path}"}
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate output filenames
    output_filename = f"output_{uuid.uuid4().hex}.mp4"
    excel_filename = f"plates_{uuid.uuid4().hex}.xlsx"
    visualization_filename = f"visualization_{uuid.uuid4().hex}.png"
    
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    excel_path = os.path.join(OUTPUT_FOLDER, excel_filename)
    visualization_path = os.path.join(OUTPUT_FOLDER, visualization_filename)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    raw_detections = []
    vehicle_track_data = {}
    frame_number = 0
    # Initialize lightweight ViolationDetector for video violation detection
    try:
        violation_detector = ViolationDetector(pose_model=None, helmet_model=None)
    except Exception as e:
        print(f"Failed to initialize ViolationDetector for video: {e}")
        violation_detector = None
    all_video_violations = []
    
    print("Starting video processing...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            process_detections = (frame_number % VIDEO_PROCESSING_INTERVAL == 0)
            
            if frame_number % 100 == 0:
                print(f"Processing frame {frame_number}/{total_frames} "
                      f"({(frame_number/total_frames)*100:.1f}%)")
            
            # Detect vehicles
            vehicles = detect_vehicles(frame, vehicle_model)
            
            # Track vehicles
            if vehicles:
                vehicle_dets = np.array([[v['bbox'][0], v['bbox'][1], 
                                        v['bbox'][2], v['bbox'][3], 
                                        v['confidence']] for v in vehicles])
                tracked_objects = tracker.update(vehicle_dets)
            else:
                tracked_objects = tracker.update(np.empty((0, 5)))
            
            # Process tracked vehicles
            current_vehicles = []
            for track in tracked_objects:
                track_id = int(track[4])
                bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
                
                vehicle_type = "Unknown"
                vehicle_color = (255, 255, 255)
                
                # Match with detected vehicles
                for v in vehicles:
                    v_bbox = v['bbox']
                    iou = compute_iou(bbox, v_bbox)
                    if iou > 0.5:
                        vehicle_type = v['class_name']
                        vehicle_color = v['color']
                        break
                
                # Update tracking data
                if track_id not in vehicle_track_data:
                    vehicle_track_data[track_id] = {
                        'type': vehicle_type,
                        'first_seen': frame_number,
                        'last_seen': frame_number,
                        'boxes': [bbox]
                    }
                else:
                    vehicle_track_data[track_id]['last_seen'] = frame_number
                    vehicle_track_data[track_id]['boxes'].append(bbox)
                    if vehicle_type != "Unknown":
                        vehicle_track_data[track_id]['type'] = vehicle_type
                
                current_vehicles.append({
                    'bbox': bbox,
                    'track_id': track_id,
                    'class_name': vehicle_type,
                    'color': vehicle_color
                })
                
                # Draw vehicle box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            vehicle_color, 2)
                cv2.putText(frame, f"ID: {track_id}, {vehicle_type}",
                          (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, vehicle_color, 2)
            
            # Detect plates
            plates = detect_plates(frame, plate_model, 
                                  current_vehicles if plate_model != vehicle_model else None)
            
            # Process plates
            for plate in plates:
                x1, y1, x2, y2 = plate['bbox']
                vehicle_type = plate['vehicle_type']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, "License Plate",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 0, 255), 2)
                
                # Extract text periodically
                if process_detections:
                    plate_text = extract_text_with_plate_recognizer(frame, 
                                                                    (x1, y1, x2, y2))
                    processed_text = process_plate_text(plate_text)
                    
                    if processed_text and is_valid_indian_plate(processed_text):
                        if not is_duplicate_plate(processed_text, raw_detections, 
                                                 threshold=2):
                            plate_type = detect_plate_color(frame, (x1, y1, x2, y2))
                            raw_detections.append({
                                "Frame": frame_number,
                                "License Plate": processed_text,
                                "Plate Type": plate_type,
                                "Vehicle Type": vehicle_type,
                                "Vehicle ID": plate['vehicle_id'],
                                "Confidence": plate['confidence'],
                                "text": processed_text,
                                "Box": (x1, y1, x2, y2)
                            })
            
            out.write(frame)

            # Run violation detection every processing interval using ViolationDetector
            if process_detections and violation_detector is not None:
                try:
                    object_detections = detect_objects(frame, vehicle_model, conf=0.25)
                    frame_violations = violation_detector.detect_violations(object_detections, frame, frame_number)
                    if frame_violations:
                        # convert each violation (dict-like) and save thumbs
                        for vf in frame_violations:
                            try:
                                if isinstance(vf, dict):
                                    bbox = vf.get('bbox') or vf.get('Box')
                                    vdict = dict(vf)
                                else:
                                    bbox = getattr(vf, 'bbox', None)
                                    vdict = {
                                        'type': getattr(vf, 'type', None),
                                        'bbox': tuple(map(int, bbox)) if bbox is not None else None,
                                        'confidence': float(getattr(vf, 'confidence', 0.0)),
                                        'frame_number': frame_number
                                    }

                                if bbox:
                                    try:
                                        btuple = tuple(map(int, bbox))
                                        thumb_name = _save_thumb_from_bbox(frame, btuple, prefix='violation_vid')
                                        if thumb_name:
                                            vdict['thumb'] = thumb_name
                                    except Exception:
                                        pass

                                all_video_violations.append(vdict)
                            except Exception as e:
                                print(f"Error processing frame violation: {e}")

                        # Draw violations on frame
                        frame = violation_detector.draw_violations(frame, frame_violations)
                except Exception as e:
                    print(f"ViolationDetector error during video processing: {e}")
            
            if process_detections and len(plates) > 0:
                time.sleep(0.2)
        
        # Remove duplicates
        unique_plates = []
        plate_texts = []
        
        for detection in raw_detections:
            plate_text = detection["License Plate"]
            
            is_duplicate = False
            for existing in unique_plates:
                if plate_similarity(plate_text, existing["License Plate"]) <= 2:
                    if detection["Confidence"] > existing["Confidence"]:
                        existing.update(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate and plate_text not in plate_texts:
                unique_plates.append(detection)
                plate_texts.append(plate_text)
        
        # Generate visualization
        visualization_data = generate_visualization_data(unique_plates)
        save_visualization(visualization_data, visualization_path)
        
        # Save Excel
        if unique_plates:
            plate_df = pd.DataFrame(unique_plates)
            plate_df = plate_df.drop(columns=["text", "Box"], errors='ignore')
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    plate_df.to_excel(writer, sheet_name='Plates', index=False)
                    if all_video_violations:
                        vdf = pd.DataFrame(all_video_violations)
                        vdf.to_excel(writer, sheet_name='Violations', index=False)
            except Exception:
                plate_df.to_excel(excel_path, index=False)
        else:
            pd.DataFrame(columns=["Frame", "License Plate", "Plate Type", 
                                 "Vehicle Type", "Vehicle ID", 
                                 "Confidence"]).to_excel(excel_path, index=False)
            # Append violations if present
            if all_video_violations:
                try:
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                        vdf = pd.DataFrame(all_video_violations)
                        vdf.to_excel(writer, sheet_name='Violations', index=False)
                except Exception:
                    pass

        # Save violation visualization for video
        violation_vis_filename = None
        if all_video_violations:
            violation_vis_filename = f"violations_vis_{uuid.uuid4().hex}.png"
            violation_vis_path = os.path.join(OUTPUT_FOLDER, violation_vis_filename)
            save_violation_visualization(all_video_violations, violation_vis_path)
        
        cap.release()
        out.release()
        
        return {
            "total_frames": total_frames,
            "processed_frames": frame_number,
            "total_plates": len(unique_plates),
            "plates": [dict([(k, v) for k, v in plate.items() 
                           if k != "text" and k != "Box"]) 
                      for plate in unique_plates[:10]],
            "violations": all_video_violations,
            "output_video": output_filename,
            "excel_file": excel_filename,
            "visualization": visualization_filename,
            "violation_visualization": violation_vis_filename
        }
        
    
    except Exception as e:
        if cap.isOpened():
            cap.release()
        if out.isOpened():
            out.release()
        
        print(f"Error processing video: {str(e)}")
        return {"error": str(e)}