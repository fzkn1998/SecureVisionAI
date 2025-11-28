# SecureVision AI

SecureVision AI is a Flask-based real-time PPE compliance system that detects people, helmets, and safety vests using YOLOv8. It provides live video streaming, re-identification for consistent person IDs, configurable detection thresholds, and a modern dashboard with violation logging and exports.

## Key Features
- Real-time helmet/vest detection with custom YOLOv8 model (`models/exp9/weights/best.pt`) and person detector (`models/yolov8n.pt`).
- Optional ONNX export for faster inference (`scripts/convert_to_onnx.py`).
- Re-identification and frame-confirmation to reduce false positives; per-person violation logging with CSV/JSON outputs.
- Live MJPEG stream endpoint, snapshot endpoint, and stats API.
- Dashboard with camera selection, detection sliders, violations page, and CSV download.
- Utility scripts: add sample violations, clear logs with backups, convert models to ONNX.

## Project Structure
- `app.py` — Flask app, video pipeline, detection, tracking, APIs.
- `templates/` — `login.html`, `index.html`, `violations.html`.
- `static/` — CSS/JS/assets for the UI.
- `models/` — YOLO weights (custom PPE + person), optional ONNX exports.
- `data/` — `violations.csv` log; `videos/` sample clips (`input1.mp4`, `input2.mp4`, `input3.mp4`).
- `scripts/` — maintenance and conversion helpers.

## Quick Start
1) Python 3.10+ (GPU optional but recommended).  
2) Create and activate a virtual env:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3) Ensure weights exist:
   - PPE model: `models/exp9/weights/best.pt`
   - Person model: `models/yolov8n.pt`
   - (Optional) ONNX: run `python scripts/convert_to_onnx.py` to export next to the `.pt` files.
4) Run the app:
   ```bash
   python app.py
   ```
   Open `http://localhost:5000`.

## Using the UI
- Login (front-end only): username `admin`, password `admin123`.
- Select a camera: `input1`, `input2`, `input3` map to sample videos; any RTSP/HTTP URL or local file path also works.
- Adjust thresholds (person/helmet/vest), confirmation frames, recording toggles.
- View/export violations from the Violations Report page.

## API Endpoints
- `GET /video_feed?source=<id>&person_conf=0.35&helmet_conf=0.30&vest_conf=0.30&confirmation_frames=5&w=800&skip=2` — MJPEG stream.
- `GET /stats` — latest counts/fps.
- `GET /snapshot` — latest JPEG.
- `GET /violations/view` — JSON list.
- `GET /violations/download` — CSV export.
- `GET /violations/stats` — aggregates and recent items.
- `POST /violations/reset` — clears logs (backs up existing CSV/JSON).

## Maintenance Scripts
- `python scripts/add_test_violations.py` — append sample rows for all cameras.
- `python scripts/clear_violations.py` — back up and reset `violations.csv`/`violations.json`.
- `python scripts/convert_to_onnx.py` — export both models to ONNX.

## Notes
- `USE_ONNX` is currently set to `False` in `app.py`; switch to `True` after exporting and installing `onnxruntime`/`onnxruntime-gpu`.
- GPU inference benefits from CUDA/TensorRT if available.
- Data is logged to `data/violations.csv`; back it up before cleanup.

## GitHub Upload
If you want to push this project:
```bash
git init
git add .
git commit -m "Initial commit: SecureVision AI"
git branch -M main
git remote add origin https://github.com/<your-username>/SecureVisionAI.git
git push -u origin main
```
(Replace the remote with your repo URL.)
