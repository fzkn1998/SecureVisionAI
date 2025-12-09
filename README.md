# 🛡️ SecureVision AI

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-blue)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.0-00B3FF)](https://ultralytics.com/yolov8)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-000000)](https://flask.palletsprojects.com/)

</div>

## 🚀 Overview
SecureVision AI is an advanced computer vision solution for real-time Personal Protective Equipment (PPE) compliance monitoring. Built with Flask and powered by YOLOv8, it provides automated detection of safety gear violations in real-time video streams. The system is designed for construction sites, factories, and other industrial environments where safety compliance is critical.

## ✨ Key Features

- 🔍 **Real-time Detection**: Instant identification of PPE compliance using YOLOv8
- 🎯 **High Accuracy**: Custom-trained models for precise helmet and vest detection
- 📊 **Comprehensive Dashboard**: Intuitive web interface for monitoring and management
- 📈 **Violation Logging**: Detailed records of all safety violations with timestamps
- 🚀 **Optimized Performance**: Support for both CPU and GPU acceleration
- 🔄 **Flexible Integration**: Works with RTSP streams, IP cameras, and video files
- 📱 **Responsive Design**: Accessible from any device with a modern web browser

## 🏗️ Project Structure

```
SecureVisionAI/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── models/               # Model weights and configurations
│   ├── exp9/weights/     # Custom YOLOv8 model
│   └── yolov8n.pt        # Base YOLO model
├── static/               # Static files (CSS, JS, assets)
│   ├── css/              # Stylesheets
│   └── js/               # JavaScript files
├── templates/            # HTML templates
│   ├── index.html        # Main dashboard
│   ├── login.html        # Login page
│   └── violations.html   # Violations log
├── data/                 # Data storage
│   ├── violations.csv    # Violation logs
│   └── videos/           # Sample video files
└── scripts/              # Utility scripts
    ├── add_test_violations.py
    ├── clear_violations.py
    └── convert_to_onnx.py
```

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

## 🖥️ User Guide

### Dashboard Overview
- **Live Feed**: View real-time video with detection overlays
- **Camera Selection**: Switch between different video sources
- **Settings Panel**: Adjust detection sensitivity and thresholds
- **Violation Logs**: Review and export safety violations

### Key Features
- **Real-time Monitoring**: Watch live detections with visual indicators
- **Customizable Alerts**: Set up notifications for violations
- **Export Data**: Download violation reports in CSV or JSON format
- **Multi-camera Support**: Monitor multiple locations simultaneously

### Configuration
Edit `config.py` to customize:
- Detection thresholds
- Camera settings
- Storage locations
- Notification preferences

## 🔌 API Reference

### Video Feed
- `GET /video_feed` - Stream live video with detections
  - Parameters:
    - `source`: Camera ID or stream URL
    - `person_conf`: Confidence threshold for person detection (0-1)
    - `helmet_conf`: Confidence threshold for helmet detection (0-1)
    - `vest_conf`: Confidence threshold for vest detection (0-1)
    - `confirmation_frames`: Number of frames to confirm detection
    - `w`: Frame width
    - `skip`: Frame skip rate

### Statistics
- `GET /stats` - Get real-time detection statistics
  - Returns: JSON with FPS, object counts, and system status

### Snapshots
- `GET /snapshot` - Capture current frame
  - Returns: JPEG image

### Violation Management
- `GET /violations/view` - List all violations (JSON)
- `GET /violations/download` - Download violations as CSV
- `GET /violations/stats` - Get violation statistics
- `POST /violations/reset` - Clear violation logs (with backup)

## ⚙️ Advanced Configuration

### ONNX Optimization
For improved performance, you can export models to ONNX format:

```bash
python scripts/convert_to_onnx.py
```
Then enable ONNX in `app.py`:
```python
USE_ONNX = True  # Set to True after ONNX export
```

### GPU Acceleration
To enable GPU acceleration:
1. Install CUDA and cuDNN
2. Install PyTorch with CUDA support
3. Set `device=0` in `app.py` for GPU usage

## 🛠️ Maintenance

### Utility Scripts
- `add_test_violations.py`: Generate sample violation data
- `clear_violations.py`: Reset violation logs with backup
- `convert_to_onnx.py`: Convert models to ONNX format

### Data Management
- Violation logs are stored in `data/violations.csv`
- Regular backups are recommended
- Use the web interface or API to export data

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://ultralytics.com/yolov8) for the object detection framework
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision operations

