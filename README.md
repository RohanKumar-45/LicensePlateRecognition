# ALPR WebApp — License Plate Detection

This project is a Flask web application for detecting vehicles and reading license plates from images and videos. It produces annotated output images/videos and an Excel report of detected plates and vehicle/violation metadata.

Core features
- Vehicle detection using YOLO models
- License plate detection and OCR
- Plate type classification (Private / Commercial / Electric)
- Simple violation detection (helmet, phone use, etc.)
- Exports: annotated media, thumbnails, and an Excel (.xlsx) report

Getting started (developer)
- Install dependencies from `requirements.txt` in a virtual environment.
- Configure environment variables (e.g., `PLATE_RECOGNIZER_API_KEY`) or place a local `.env` file.
- Run the app: `python app.py` for development. For production, run under a WSGI server (e.g., `gunicorn app:app`).

Project layout
- `app.py` — Flask application routes and entry point
- `processors.py` — high-level image/video processing orchestration
- `detection.py` — model loading and detection helper functions
- `ocr_utils.py` — OCR and plate text post-processing utilities
- `violations.py`, `violation_result.py` — violation detection logic and data structures
- `templates/`, `static/` — web UI and output storage
- `models/` — local models (not committed to repository)

Notes
- Do not commit heavy model files or uploaded media to the repository. Keep secrets (API keys) out of source code.
- The application defaults to CPU inference if no CUDA device is available.
