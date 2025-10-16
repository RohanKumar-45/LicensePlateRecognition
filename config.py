import os
from pathlib import Path

# Optionally load environment variables from a local .env file if python-dotenv
# is installed. This keeps API keys out of source code.
try:
    from dotenv import load_dotenv
    # Load .env in project root
    _env_path = Path(__file__).resolve().parent / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
except Exception:
    # dotenv is optional; fall back to environment variables only
    pass

# Folder paths
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
MODELS_FOLDER = 'models'

# Model paths
PLATE_DETECTION_MODEL_PATH = "./models/best copy.pt"
VEHICLE_DETECTION_MODEL_PATH = "yolov8n.pt"

# File configurations
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Vehicle class definitions
VEHICLE_CLASSES = {
    1: ('Bicycle', (255, 0, 0)),
    2: ('Car', (0, 0, 255)),
    3: ('Motorcycle', (0, 255, 0)),
    5: ('Bus', (0, 255, 255)),
    7: ('Truck', (128, 0, 128)),
    6: ('Auto', (255, 165, 0)),
    4: ('Other', (128, 128, 128))
}

TWO_WHEELER_CLASSES = {1, 3}

# Valid Indian state codes
VALID_STATE_CODES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN',
    'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR',
    'TS', 'UK', 'UP', 'WB'
]

# State code corrections mapping
STATE_CODE_CORRECTIONS = {
    '01': 'DL', '6J': 'GJ', '1N': 'TN', '14P': 'MP',
    'N1H': 'MH', 'KR': 'KA', 'A9': 'AP', 'T8': 'TS',
    '0D': 'OD', 'BR': 'BR', '88': 'BR', 'B8': 'BR',
    'T7': 'TN', 'KI': 'KL', 'ML': 'MH', 'N4': 'MP',
    'HR': 'HR', 'H8': 'HR', 'PY': 'PB', 'P8': 'PB',
    'U9': 'UP', 'UX': 'UK', 'RJ': 'RJ', '8J': 'RJ',
    'NS': 'MH', '1A': 'TA'
}

PLATE_RECOGNIZER_API_KEY = os.environ.get('PLATE_RECOGNIZER_API_KEY', '')

# Processing parameters
VIDEO_PROCESSING_INTERVAL = 40
SORT_MAX_AGE = 20
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# Create necessary folders
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)