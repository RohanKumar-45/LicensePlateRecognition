import re
import time
import requests
import cv2
import Levenshtein
from io import BytesIO
from PIL import Image
from config import PLATE_RECOGNIZER_API_KEY, VALID_STATE_CODES, STATE_CODE_CORRECTIONS

def extract_text_with_plate_recognizer(image, bounding_box, max_retries=3, retry_delay=1.5):
    """Extract text from license plate using Plate Recognizer API"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    x1, y1, x2, y2 = bounding_box
    cropped_image = pil_image.crop((x1, y1, x2, y2))
    
    img_byte_arr = BytesIO()
    cropped_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    url = "https://api.platerecognizer.com/v1/plate-reader/"
    headers = {"Authorization": f"Token {PLATE_RECOGNIZER_API_KEY}"}
    files = {"upload": img_byte_arr}
    params = {"regions": "in", "camera_id": "license-plate-recognition"}
    
    attempts = 0
    
    while attempts < max_retries:
        try:
            response = requests.post(url, headers=headers, files=files, data=params)
            
            if response.status_code == 429:
                wait_time = retry_delay * (2 ** attempts)
                time.sleep(wait_time)
                attempts += 1
                continue
                
            response.raise_for_status()
            
            result = response.json()
            
            if result and "results" in result and result["results"]:
                plate_text = result["results"][0]["plate"]
                confidence = result["results"][0]["score"]
                print(f"Plate Recognizer extracted: {plate_text} (conf: {confidence:.2f})")
                return plate_text
            else:
                print("No plate text detected by Plate Recognizer")
                return ""
                
        except requests.exceptions.HTTPError:
            if response.status_code == 429:
                continue
            return ""
        except Exception:
            return ""
    
    return ""

def process_plate_text(raw_text):
    """Process and format raw plate text"""
    if not raw_text:
        return ""
    
    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    print(f"Processing: Raw='{raw_text}' -> Cleaned='{cleaned_text}'")
    
    if len(cleaned_text) < 2:
        return ""
        
    potential_state_code = cleaned_text[:2]
    
    if potential_state_code in STATE_CODE_CORRECTIONS:
        potential_state_code = STATE_CODE_CORRECTIONS[potential_state_code]
    
    cleaned_text = potential_state_code + cleaned_text[2:]
    
    if len(cleaned_text) >= 7:
        try:
            state_code = cleaned_text[:2]
            remaining_text = cleaned_text[2:]
            
            match = re.match(r'(\d{1,2})([A-Z]{1,3})(\d{1,4})', remaining_text)
            
            if match:
                district_code = match.group(1)
                series_code = match.group(2)
                vehicle_number = match.group(3).zfill(4)
                
                formatted_text = f"{state_code} {district_code} {series_code} {vehicle_number}"
                return formatted_text.strip()
            else:
                return f"{state_code} {remaining_text}"
        except Exception as e:
            print(f"Error formatting plate text: {str(e)}")
    
    return cleaned_text

def is_valid_indian_plate(text):
    """Validate Indian license plate format"""
    if not text:
        return False
    
    clean_text = re.sub(r'\s+', '', text)
    
    if len(clean_text) < 8:
        return False
    
    state_code = clean_text[:2]
    if state_code not in VALID_STATE_CODES:
        if state_code not in STATE_CODE_CORRECTIONS:
            return False
    
    # Standard pattern check
    std_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')
    if std_pattern.match(clean_text):
        return True
    
    # Flexible validation
    if not re.search(r'^[A-Z]{2}\d{1,2}', clean_text):
        return False
    
    if not re.search(r'\d{4}$', clean_text):
        return False
    
    middle_part = clean_text[4:-4]
    if not re.search(r'[A-Z]{1,3}', middle_part):
        return False
    
    return True

def plate_similarity(text1, text2):
    """Calculate similarity between two plate texts"""
    if not text1 or not text2:
        return 100
        
    clean_text1 = re.sub(r'\s+', '', text1.upper())
    clean_text2 = re.sub(r'\s+', '', text2.upper())
    
    distance = Levenshtein.distance(clean_text1, clean_text2)
    
    # Penalize state code mismatch
    if clean_text1[:2] != clean_text2[:2]:
        distance += 2
    
    # Penalize number mismatch
    num_pattern = r'\d{3,4}$'
    num1 = re.search(num_pattern, clean_text1)
    num2 = re.search(num_pattern, clean_text2)
    
    if num1 and num2:
        if num1.group() != num2.group():
            distance += 3
    
    return distance

def is_duplicate_plate(new_plate, existing_plates, threshold=3):
    """Check if plate is duplicate"""
    for plate in existing_plates:
        if plate_similarity(new_plate, plate["text"]) <= threshold:
            return True
    return False