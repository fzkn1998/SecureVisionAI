import os
import threading
import time
import json
import csv
from typing import Generator
from datetime import datetime

from flask import Flask, Response, request, send_from_directory
import torch
import cv2
import numpy as np

# Allow ultralytics to auto-install missing requirements (needed for legacy weights such as "models.yolo")
os.environ["YOLO_AUTOINSTALL"] = "1"

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static', template_folder=TEMPLATES_DIR)

# Path to custom YOLO model for helmet/jacket detection
MODEL_PATH = os.path.join(MODELS_DIR, 'exp9', 'weights', 'best.pt')
PERSON_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8n.pt')  # Built-in YOLO model for person detection

# ONNX model paths (preferred for faster inference)
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, 'exp9', 'weights', 'best.onnx')
ONNX_PERSON_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8n.onnx')
VIOLATIONS_CSV = os.path.join(DATA_DIR, 'violations.csv')
VIOLATIONS_JSON = os.path.join(DATA_DIR, 'violations.json')

# Use ONNX by default if available
USE_ONNX = False  # Force PyTorch path


class VideoCamera:
    def __init__(self, source=0, width=800, skip=1):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.width = width
        self.skip = max(1, int(skip))
        self.frame_lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.is_file = isinstance(source, str) and os.path.exists(source)
        self.read_thread = threading.Thread(target=self.update, daemon=True)
        self.read_thread.start()

    def update(self):
        idx = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # if video file, loop back to start
                if self.is_file:
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:
                        pass
                    time.sleep(0.01)
                    continue
                time.sleep(0.05)
                continue
            if self.width and frame is not None:
                h, w = frame.shape[:2]
                scale = self.width / float(w)
                if scale < 1.5:  # avoid upscaling too much
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            with self.frame_lock:
                # only keep every nth frame to reduce CPU
                if idx % self.skip == 0:
                    self.frame = frame
            idx += 1
            # Small delay for video files to control playback speed
            if self.is_file:
                time.sleep(0.02)  # ~50 fps max from camera reader

    def read(self):
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


_model = None
_person_model = None
_onnx_model = None
_onnx_person_model = None
_use_yolov5 = False
_state_lock = threading.Lock()
_latest_count = 0
_latest_fps = 0.0
_latest_jpg: bytes | None = None
_latest_no_helmet = 0
_latest_no_vest = 0

# Re-identification system
_person_tracker = {}  # {person_id: (box, conf, feature_vector, last_seen_frame)}
_next_person_id = 1
_current_frame_number = 0
_person_features = {}  # {person_id: [feature_vectors]}

# Frame-based tracking for violations
_person_frame_history = {}  # {person_id: [(has_helmet, has_vest), ...]}
_violation_log = []  # Store violation events
CONFIRMATION_FRAMES = 5  # Number of frames to confirm status
MAX_DISAPPEARED_FRAMES = 30  # Max frames before removing ID


def get_model(model_name: str = MODEL_PATH):
    global _model
    global _use_yolov5
    if _model is None:
        if YOLO is not None:
            load_errors = []
            try:
                _model = YOLO(model_name)
                _ = _model.names  # sanity access
                _use_yolov5 = False
                return _model
            except Exception as e:
                load_errors.append(f"{model_name}: {e}")
                _model = None
            
            # Fallback to bundled PPE model if present
            fallback_model = os.path.join(MODELS_DIR, 'ppe_best.pt')
            if _model is None and os.path.exists(fallback_model):
                try:
                    _model = YOLO(fallback_model)
                    _ = _model.names
                    _use_yolov5 = False
                    print(f"Loaded fallback model: {fallback_model}")
                    return _model
                except Exception as e2:
                    load_errors.append(f"{fallback_model}: {e2}")
                    _model = None

            # Surface detailed errors to help diagnose missing dependencies like models.yolo
            joined_errors = " | ".join(load_errors) if load_errors else "no errors captured"
        # Disable YOLOv5 torch.hub fallback to avoid missing models.common dependency
        raise RuntimeError(f'Failed to load model via YOLOv8. Ensure ultralytics is installed and model path is correct. Attempts: {joined_errors}')
    return _model

def get_person_model():
    global _person_model
    if _person_model is None:
        if YOLO is not None:
            try:
                _person_model = YOLO(PERSON_MODEL_PATH)
                return _person_model
            except Exception:
                pass
        raise RuntimeError('Failed to load person detection model via YOLOv8. Ensure ultralytics is installed and person model path is correct.')
    return _person_model

def get_onnx_model(model_path: str):
    """Load ONNX model for faster inference with TensorRT support"""
    global _onnx_model
    if _onnx_model is None and os.path.exists(model_path):
        try:
            # Configure ONNX Runtime for optimal performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Try providers in order of preference: TensorRT > CUDA > CPU
            # TensorRT provides 2-5x faster inference than CUDA
            providers = []
            available_providers = ort.get_available_providers()
            
            if 'TensorrtExecutionProvider' in available_providers:
                providers.append(('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,  # Enable FP16 for faster inference
                    'trt_max_workspace_size': 2147483648,  # 2GB
                    'trt_engine_cache_enable': True,  # Cache TensorRT engines
                    'trt_engine_cache_path': './trt_cache'  # Cache directory
                }))
                print("✓ TensorRT execution provider available")
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            
            providers.append('CPUExecutionProvider')
            
            _onnx_model = ort.InferenceSession(model_path, sess_options, providers=providers)
            print(f"✓ Loaded ONNX model from {model_path}")
            print(f"  Active providers: {_onnx_model.get_providers()}")
        except Exception as e:
            print(f"✗ Failed to load ONNX model: {e}")
            _onnx_model = None
    return _onnx_model

def get_onnx_person_model(model_path: str):
    """Load ONNX person detection model with TensorRT support"""
    global _onnx_person_model
    if _onnx_person_model is None and os.path.exists(model_path):
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Try providers in order: TensorRT > CUDA > CPU
            providers = []
            available_providers = ort.get_available_providers()
            
            if 'TensorrtExecutionProvider' in available_providers:
                providers.append(('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_max_workspace_size': 2147483648,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache'
                }))
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            
            providers.append('CPUExecutionProvider')
            
            _onnx_person_model = ort.InferenceSession(model_path, sess_options, providers=providers)
            print(f"✓ Loaded ONNX person model from {model_path}")
            print(f"  Active providers: {_onnx_person_model.get_providers()}")
        except Exception as e:
            print(f"✗ Failed to load ONNX person model: {e}")
            _onnx_person_model = None
    return _onnx_person_model

def preprocess_for_onnx(frame, img_size=640):
    """Preprocess frame for ONNX inference"""
    # Resize and pad image while meeting stride-multiple constraints
    img = cv2.resize(frame, (img_size, img_size))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_onnx_output(output, conf_threshold=0.25, iou_threshold=0.45, img_size=640, orig_shape=None):
    """Post-process ONNX model output to get bounding boxes"""
    predictions = output[0]
    
    # predictions shape: (1, 25200, 85) for YOLOv8
    # Format: [x, y, w, h, confidence, class_scores...]
    boxes = []
    
    for pred in predictions[0]:  # Iterate over detections
        # Extract box coordinates and confidence
        if len(pred) > 4:
            x, y, w, h = pred[:4]
            obj_conf = pred[4] if len(pred) > 4 else 1.0
            class_scores = pred[5:] if len(pred) > 5 else pred[4:]
            
            if len(class_scores) > 0:
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                conf = obj_conf * class_conf
            else:
                conf = obj_conf
                class_id = 0
            
            if conf >= conf_threshold:
                # Convert from center format to corner format
                x1 = int((x - w / 2) * orig_shape[1] / img_size) if orig_shape else int(x - w / 2)
                y1 = int((y - h / 2) * orig_shape[0] / img_size) if orig_shape else int(y - h / 2)
                x2 = int((x + w / 2) * orig_shape[1] / img_size) if orig_shape else int(x + w / 2)
                y2 = int((y + h / 2) * orig_shape[0] / img_size) if orig_shape else int(y + h / 2)
                
                boxes.append({
                    'box': [x1, y1, x2, y2],
                    'conf': float(conf),
                    'class_id': int(class_id)
                })
    
    return boxes

def resolve_source(alias: str):
    base = VIDEOS_DIR
    mapping = {
        'input1': os.path.join(base, 'input1.mp4'),
        'input2': os.path.join(base, 'input2.mp4'),
        'input3': os.path.join(base, 'input3.mp4')
    }
    if alias in mapping and os.path.exists(mapping[alias]):
        return mapping[alias]
    if isinstance(alias, str) and (alias.startswith('rtsp://') or alias.startswith('http') or os.path.exists(alias)):
        return alias
    return 0


def _allowed_label_set() -> set[str]:
    try:
        names = get_model().names
    except Exception:
        names = {}
    if isinstance(names, dict):
        labels = {str(v).lower() for v in names.values()}
    else:
        labels = {str(v).lower() for v in names}
    # Include person, helmet, and jacket detection
    return labels

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def get_box_center(box):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def box_distance(box1, box2):
    """Calculate distance between two box centers"""
    c1 = get_box_center(box1)
    c2 = get_box_center(box2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

def extract_appearance_features(frame, box):
    """Extract simple appearance features from person bounding box"""
    try:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return None
        
        # Resize to standard size for feature extraction
        person_img = cv2.resize(person_img, (64, 128))
        
        # Extract color histogram features
        hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
        
        # Histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Concatenate features
        features = np.concatenate([h_hist, s_hist, v_hist])
        
        return features
    except Exception as e:
        return None

def calculate_feature_similarity(feat1, feat2):
    """Calculate similarity between two feature vectors"""
    if feat1 is None or feat2 is None:
        return 0.0
    
    # Cosine similarity
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def track_persons_with_reid(frame, current_persons, max_distance=150, similarity_threshold=0.6):
    """Advanced person tracker with re-identification"""
    global _person_tracker, _next_person_id, _person_frame_history, _current_frame_number, _person_features
    
    _current_frame_number += 1
    tracked_persons = []
    used_ids = set()
    
    # Match current detections with tracked persons using both position and appearance
    for person_box, conf in current_persons:
        # Extract appearance features
        features = extract_appearance_features(frame, person_box)
        
        best_match_id = None
        best_score = 0.0
        
        for person_id, (prev_box, _, prev_features, last_seen) in list(_person_tracker.items()):
            if person_id in used_ids:
                continue
            
            # Calculate position distance (normalized)
            dist = box_distance(person_box, prev_box)
            position_score = max(0, 1 - (dist / max_distance))
            
            # Calculate appearance similarity
            appearance_score = 0.0
            if features is not None and prev_features is not None:
                appearance_score = calculate_feature_similarity(features, prev_features)
            
            # Combined score (weighted)
            combined_score = 0.4 * position_score + 0.6 * appearance_score
            
            # Check if person was seen recently
            frames_disappeared = _current_frame_number - last_seen
            if frames_disappeared > MAX_DISAPPEARED_FRAMES:
                continue
            
            if combined_score > best_score and combined_score > similarity_threshold:
                best_score = combined_score
                best_match_id = person_id
        
        if best_match_id is not None:
            person_id = best_match_id
            used_ids.add(person_id)
            
            # Update feature history (keep last 5 feature vectors)
            if person_id not in _person_features:
                _person_features[person_id] = []
            if features is not None:
                _person_features[person_id].append(features)
                if len(_person_features[person_id]) > 5:
                    _person_features[person_id] = _person_features[person_id][-5:]
                
                # Average features for more robust matching
                avg_features = np.mean(_person_features[person_id], axis=0)
            else:
                avg_features = _person_tracker[person_id][2]
        else:
            # New person
            person_id = _next_person_id
            _next_person_id += 1
            _person_frame_history[person_id] = []  # Initialize frame history
            _person_features[person_id] = [features] if features is not None else []
            avg_features = features
        
        _person_tracker[person_id] = (person_box, conf, avg_features, _current_frame_number)
        tracked_persons.append((person_id, person_box, conf))
    
    # Clean up old tracked persons
    ids_to_remove = []
    for person_id, (_, _, _, last_seen) in list(_person_tracker.items()):
        if person_id not in used_ids:
            frames_disappeared = _current_frame_number - last_seen
            if frames_disappeared > MAX_DISAPPEARED_FRAMES:
                ids_to_remove.append(person_id)
    
    for pid in ids_to_remove:
        del _person_tracker[pid]
        if pid in _person_frame_history:
            del _person_frame_history[pid]
        if pid in _person_features:
            del _person_features[pid]
    
    return tracked_persons

# Track which violations have been logged to avoid duplicates
_logged_violations = {}  # {person_id: {'no_helmet': timestamp, 'no_vest': timestamp}}
_csv_file_lock = threading.Lock()

# Initialize CSV file with headers
def initialize_csv_file():
    """Create violations CSV file with headers if it doesn't exist"""
    if not os.path.exists(VIOLATIONS_CSV):
        try:
            with open(VIOLATIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Date', 'Time', 'Person_ID', 'Violation_Type', 'Camera_Area', 'Frame_Count'])
            print(f"✓ Created violations CSV file: {VIOLATIONS_CSV}")
        except Exception as e:
            print(f"✗ Error creating CSV file: {e}")

# Initialize CSV on startup
initialize_csv_file()

def save_violation(person_id, violation_type, timestamp, camera_area='Unknown'):
    """Save violation data to CSV and JSON log (only once per person per violation type per 30 minutes)"""
    global _violation_log, _logged_violations
    
    # Check if already logged recently (within last 30 minutes = 1800 seconds)
    if person_id not in _logged_violations:
        _logged_violations[person_id] = {}
    
    current_time = time.time()
    
    # Check each violation type
    for v_type in violation_type.split(', '):
        last_logged = _logged_violations[person_id].get(v_type, 0)
        
        # Only log if not logged in last 30 minutes (1800 seconds)
        if current_time - last_logged > 1800:
            # Create violation record
            dt_now = datetime.now()
            violation = {
                'person_id': person_id,
                'violation_type': v_type,
                'timestamp': timestamp,
                'camera_area': camera_area,
                'frame_count': len(_person_frame_history.get(person_id, []))
            }
            _violation_log.append(violation)
            
            # Mark as logged with current timestamp (30 minute cooldown)
            _logged_violations[person_id][v_type] = current_time
            
            # Keep only last 100 violations in memory
            if len(_violation_log) > 100:
                _violation_log = _violation_log[-100:]
            
            # Save to CSV file (thread-safe)
            try:
                with _csv_file_lock:
                    with open(VIOLATIONS_CSV, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            dt_now.strftime('%Y-%m-%d %H:%M:%S'),  # Full timestamp
                            dt_now.strftime('%Y-%m-%d'),           # Date
                            dt_now.strftime('%H:%M:%S'),           # Time
                            person_id,
                            v_type,
                            camera_area,
                            violation['frame_count']
                        ])
                print(f"✓ Violation logged: Person {person_id} - {v_type} at {camera_area}")
            except Exception as e:
                print(f"Error saving to CSV: {e}")
            
            # Also save to JSON file (backward compatibility)
            try:
                with open(VIOLATIONS_JSON, 'a') as f:
                    violation['timestamp'] = dt_now.strftime('%Y-%m-%d %H:%M:%S')
                    json.dump(violation, f)
                    f.write('\n')
            except Exception as e:
                print(f"Error saving to JSON: {e}")

def get_person_status(person_id, has_helmet, has_vest, confirmation_frames=5):
    """Check person status based on consecutive frame history with stricter requirements"""
    global _person_frame_history
    
    # Add current frame status
    if person_id not in _person_frame_history:
        _person_frame_history[person_id] = []
    
    _person_frame_history[person_id].append((has_helmet, has_vest))
    
    # Use MINIMUM 15 frames for better accuracy (override low values)
    min_required_frames = max(15, confirmation_frames)
    
    # Keep longer history for better stability
    if len(_person_frame_history[person_id]) > min_required_frames:
        _person_frame_history[person_id] = _person_frame_history[person_id][-min_required_frames:]
    
    # Check if we have enough frames
    history = _person_frame_history[person_id]
    if len(history) < min_required_frames:
        return 'checking', has_helmet, has_vest  # Still checking - white box
    
    # Count frames with helmet and vest in the last min_required_frames
    helmet_count = sum(1 for h, v in history if h)
    vest_count = sum(1 for h, v in history if v)
    
    # Need 90% of frames to confirm status (stricter than before)
    helmet_threshold = int(min_required_frames * 0.90)
    vest_threshold = int(min_required_frames * 0.90)
    
    confirmed_helmet = helmet_count >= helmet_threshold
    confirmed_vest = vest_count >= vest_threshold
    
    # Status updates continuously based on sliding window
    if confirmed_helmet and confirmed_vest:
        return 'safe', True, True  # Green box
    else:
        return 'violation', confirmed_helmet, confirmed_vest  # Red box


def draw_detections(frame: np.ndarray, person_results, gear_results, conf_thresh: float = 0.25, helmet_conf: float = 0.3, vest_conf: float = 0.3, confirmation_frames: int = 5, use_onnx: bool = False, camera_area: str = 'Unknown') -> tuple[np.ndarray, int, int, int]:
    person_count = 0
    no_helmet_count = 0
    no_vest_count = 0
    
    # Collect all detections
    persons = []
    helmets = []
    vests = []

    def _class_flags(cls_name: str):
        """Normalize class names from different PPE models (helmet/hardhat, vest/jacket)."""
        name = cls_name.lower()
        is_negative = 'no' in name
        is_helmet = ('helmet' in name) or ('hardhat' in name)
        is_vest = ('vest' in name) or ('jacket' in name)
        helmet_positive = is_helmet and not is_negative
        vest_positive = is_vest and not is_negative
        helmet_negative = is_helmet and is_negative
        vest_negative = is_vest and is_negative
        return helmet_positive, vest_positive, helmet_negative, vest_negative

    # Process person detections - ONNX or YOLO
    if use_onnx and person_results is not None:
        # ONNX format: list of detections
        for det in person_results:
            if det['class_id'] == 0 and det['conf'] >= conf_thresh:  # 0 is person class
                x1, y1, x2, y2 = det['box']
                persons.append(((x1, y1, x2, y2), det['conf']))
    elif person_results:
        if hasattr(person_results, 'xyxy'):
            # YOLOv5 format
            try:
                preds = person_results.xyxy[0].cpu().numpy()
                for x1, y1, x2, y2, conf, cls_id in preds:
                    if int(cls_id) == 0 and conf >= conf_thresh:  # 0 is person class in COCO
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        persons.append(((x1, y1, x2, y2), float(conf)))
            except Exception as e:
                print(f"Person detection error (YOLOv5): {e}")
        else:
            # YOLOv8 format
            try:
                res = person_results[0]
                if hasattr(res, 'boxes') and res.boxes is not None:
                    boxes = res.boxes
                    for xyxy, cls_id, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                        if int(cls_id) == 0 and conf >= conf_thresh:  # person class in COCO
                            x1, y1, x2, y2 = map(int, xyxy)
                            persons.append(((x1, y1, x2, y2), float(conf)))
            except Exception as e:
                print(f"Person detection error (YOLOv8): {e}")
    
    # Process helmet/vest detections - ONNX or YOLO
    if use_onnx and gear_results is not None:
        # ONNX format: list of detections
        gear_model = get_model()
        gear_names = gear_model.names if hasattr(gear_model, 'names') else {0: 'helmet', 1: 'vest', 2: 'jacket'}
        
        for det in gear_results:
            cls_id = det['class_id']
            conf = det['conf']
            
            try:
                cls_name = str(gear_names[cls_id]).lower() if not isinstance(gear_names, dict) else str(gear_names.get(cls_id, str(cls_id))).lower()
            except Exception:
                cls_name = str(cls_id)
            h_pos, v_pos, h_neg, v_neg = _class_flags(cls_name)
            
            if h_pos and conf >= helmet_conf:
                x1, y1, x2, y2 = det['box']
                helmets.append(((x1, y1, x2, y2), conf))
            elif v_pos and conf >= vest_conf:
                x1, y1, x2, y2 = det['box']
                vests.append(((x1, y1, x2, y2), conf))
    elif gear_results:
        # YOLO format
        gear_model = get_model()
        gear_names = gear_model.names
        if hasattr(gear_results, 'xyxy'):
            # YOLOv5 format
            try:
                preds = gear_results.xyxy[0].cpu().numpy()
                for x1, y1, x2, y2, conf, cls_id in preds:
                    try:
                        cls_name = str(gear_names[int(cls_id)]).lower() if not isinstance(gear_names, dict) else str(gear_names.get(int(cls_id), str(cls_id))).lower()
                    except Exception:
                        cls_name = str(int(cls_id))
                    h_pos, v_pos, h_neg, v_neg = _class_flags(cls_name)
                    
                    # Apply specific threshold for helmet or vest
                    if h_pos and conf >= helmet_conf:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        box = (x1, y1, x2, y2)
                        helmets.append((box, float(conf)))
                    elif v_pos and conf >= vest_conf:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        box = (x1, y1, x2, y2)
                        vests.append((box, float(conf)))
            except Exception:
                pass
        else:
            # YOLOv8 format
            try:
                res = gear_results[0]
                if hasattr(res, 'boxes') and res.boxes is not None:
                    boxes = res.boxes
                    for xyxy, cls_id, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                        try:
                            cls_name = str(gear_names[int(cls_id)]).lower() if not isinstance(gear_names, dict) else str(gear_names.get(int(cls_id), str(cls_id))).lower()
                        except Exception:
                            cls_name = str(int(cls_id))
                        h_pos, v_pos, h_neg, v_neg = _class_flags(cls_name)
                        
                        # Apply specific threshold for helmet or vest
                        if h_pos and conf >= helmet_conf:
                            x1, y1, x2, y2 = map(int, xyxy)
                            box = (x1, y1, x2, y2)
                            helmets.append((box, float(conf)))
                        elif v_pos and conf >= vest_conf:
                            x1, y1, x2, y2 = map(int, xyxy)
                            box = (x1, y1, x2, y2)
                            vests.append((box, float(conf)))
            except Exception:
                pass
    
    # Track persons with re-identification
    tracked_persons = track_persons_with_reid(frame, persons)
    
    # Check each tracked person for violations
    IOU_THRESHOLD = 0.05  # Very low threshold for better overlap detection
    
    def check_overlap(person_box, gear_box):
        """Check if gear box is within or overlaps with person box"""
        p_x1, p_y1, p_x2, p_y2 = person_box
        g_x1, g_y1, g_x2, g_y2 = gear_box
        
        # Check if gear center is within person box
        gear_center_x = (g_x1 + g_x2) // 2
        gear_center_y = (g_y1 + g_y2) // 2
        
        if p_x1 <= gear_center_x <= p_x2 and p_y1 <= gear_center_y <= p_y2:
            return True
        
        # Check IoU as fallback
        iou = calculate_iou(person_box, gear_box)
        return iou > IOU_THRESHOLD
    
    for person_id, person_box, person_conf in tracked_persons:
        person_count += 1
        x1, y1, x2, y2 = person_box
        
        # Check helmet overlap - use center-based method
        has_helmet = False
        max_helmet_iou = 0.0
        for helmet_box, _ in helmets:
            if check_overlap(person_box, helmet_box):
                has_helmet = True
                iou = calculate_iou(person_box, helmet_box)
                if iou > max_helmet_iou:
                    max_helmet_iou = iou
                break
        
        # Check vest overlap - use center-based method
        has_vest = False
        max_vest_iou = 0.0
        for vest_box, _ in vests:
            if check_overlap(person_box, vest_box):
                has_vest = True
                iou = calculate_iou(person_box, vest_box)
                if iou > max_vest_iou:
                    max_vest_iou = iou
                break
        
        # Debug output for troubleshooting
        # print(f"Person ID:{person_id} - Helmet: {has_helmet} (IoU: {max_helmet_iou:.3f}), Vest: {has_vest} (IoU: {max_vest_iou:.3f})")
        
        # Get confirmed status based on dynamic confirmation frames (sliding window)
        status, confirmed_helmet, confirmed_vest = get_person_status(person_id, has_helmet, has_vest, confirmation_frames)
        
        # Log violation if confirmed (only once per minute per violation type)
        if status == 'violation':
            violation_types = []
            if not confirmed_helmet:
                violation_types.append('no_helmet')
            if not confirmed_vest:
                violation_types.append('no_vest')
            if violation_types:
                save_violation(person_id, ', '.join(violation_types), time.time(), camera_area)
        
        # Count violations based on confirmed status
        if status == 'violation':
            if not confirmed_helmet:
                no_helmet_count += 1
                print(f"Helmet violation detected for person {person_id}")
            if not confirmed_vest:
                no_vest_count += 1
                print(f"Vest violation detected for person {person_id}")
        
        # Determine color and create status label with tick/cross
        helmet_status = "✓" if confirmed_helmet else "✗"
        vest_status = "✓" if confirmed_vest else "✗"
        
        if status == 'checking':
            # WHITE for checking status (initial detection)
            color = (255, 255, 255)
            status_text = f"CHECKING {len(_person_frame_history[person_id])}/{confirmation_frames}"
        elif status == 'safe':
            # GREEN for full compliance (5 consecutive frames with helmet + vest)
            color = (0, 255, 0)
            status_text = "SAFE"
        else:
            # RED for confirmed violations
            color = (0, 0, 255)
            status_text = "VIOLATION"
        
        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background and ID
        label = f"ID:{person_id} {status_text}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 50), (x1 + max(tw, 120) + 10, y1), color, -1)
        # Use black text for white background, white text for others
        text_color = (0, 0, 0) if status == 'checking' else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 5, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # Draw helmet status with tick/cross (use black for white background)
        # Set colors based on status
        if status == 'checking':
            # Use black text for checking status
            text_color = (0, 0, 0)
            box_color = (255, 255, 255)  # White box while checking
        else:
            # If any protection is missing, use red box
            if not confirmed_helmet or not confirmed_vest:
                box_color = (0, 0, 255)  # Red box for any violation
            else:
                box_color = (0, 255, 0)  # Green box only if both present
            color = box_color
            text_color = (255, 255, 255)  # White text for colored boxes
        
        # Display status as simple text with tick/cross
        #helmet_text = f"Helmet {'✓' if confirmed_helmet else '✗'}"
        #vest_text = f"Vest {'✓' if confirmed_vest else '✗'}"
        
        # Draw the status text
        #cv2.putText(frame, helmet_text, (x1 + 5, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
        #cv2.putText(frame, vest_text, (x1 + 5, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
    
    # Draw helmets and vests with visible boxes (for debugging)
    for helmet_box, conf in helmets:
        x1, y1, x2, y2 = helmet_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"Helmet {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 0), -1)
    
    for vest_box, conf in vests:
        x1, y1, x2, y2 = vest_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"Vest {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 165, 255), -1)

    # Overlay count disabled - stats shown in sidebar instead
    # cv2.rectangle(frame, (10, 10), (280, 90), (15, 23, 42), -1)
    # cv2.putText(frame, f"People: {person_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(frame, f"No Helmet: {no_helmet_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(frame, f"No Vest: {no_vest_count}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, cv2.LINE_AA)
    return frame, person_count, no_helmet_count, no_vest_count


@app.route('/')
def index():
    return send_from_directory(TEMPLATES_DIR, 'login.html')

@app.route('/index.html')
def dashboard():
    return send_from_directory(TEMPLATES_DIR, 'index.html')

@app.route('/login.html')
def login():
    return send_from_directory(TEMPLATES_DIR, 'login.html')

@app.route('/violations.html')
def violations_page():
    return send_from_directory(TEMPLATES_DIR, 'violations.html')


@app.route('/video_feed')
def video_feed():
    # Query params
    source = request.args.get('source', default='0')
    person_conf = float(request.args.get('person_conf', default='0.35'))
    helmet_conf = float(request.args.get('helmet_conf', default='0.30'))
    vest_conf = float(request.args.get('vest_conf', default='0.30'))
    confirmation_frames = int(request.args.get('confirmation_frames', default='5'))
    width = int(request.args.get('w', default='800'))
    skip = int(request.args.get('skip', default='2'))

    # Resolve alias to file path, URL, or default webcam
    source_val = resolve_source(source)
    
    # Get camera area name for CSV logging (matching frontend camera names)
    camera_area_mapping = {
        'input1': 'Outside Plant Area',
        'input2': 'Inner Corridor',
        'input3': 'Inner Plant Area',
        '0': 'Default Camera'
    }
    camera_area = camera_area_mapping.get(source, f'Camera - {source}')

    camera = VideoCamera(source=source_val, width=width, skip=skip)
    
    # Load models - PyTorch only (ONNX disabled)
    gear_model = get_model()
    person_model = get_person_model()

    def generate() -> Generator[bytes, None, None]:
        global _latest_count, _latest_fps, _latest_jpg, _latest_no_helmet, _latest_no_vest
        last_time = time.time()
        target_fps = 25  # Target FPS for video playback
        frame_delay = 1.0 / target_fps
        
        try:
            while True:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Run inference on both models (PyTorch path only)
                person_results = None
                gear_results = None
                try:
                    if hasattr(person_model, 'predict'):
                        person_results = person_model.predict(source=frame, verbose=False, imgsz=640, conf=person_conf)
                    else:
                        person_results = person_model(frame, size=640)
                except Exception:
                    person_results = None
                
                # Helmet/Vest detection using custom model (use minimum of helmet and vest thresholds)
                gear_conf = min(helmet_conf, vest_conf)
                try:
                    if _use_yolov5:
                        gear_results = gear_model(frame, size=640)
                    else:
                        gear_results = gear_model.predict(source=frame, verbose=False, imgsz=640, conf=gear_conf)
                except Exception:
                    gear_results = None
                
                frame, person_count, no_helmet, no_vest = draw_detections(frame, person_results, gear_results, conf_thresh=person_conf, helmet_conf=helmet_conf, vest_conf=vest_conf, confirmation_frames=confirmation_frames, use_onnx=False, camera_area=camera_area)

                now = time.time()
                dt = now - last_time
                fps = 1.0 / dt if dt > 0 else 0.0

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue
                jpg_bytes = buffer.tobytes()
                with _state_lock:
                    _latest_count = person_count
                    _latest_no_helmet = no_helmet
                    _latest_no_vest = no_vest
                    _latest_fps = fps
                    _latest_jpg = jpg_bytes
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
                
                # Control playback speed to match target FPS
                elapsed = time.time() - last_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
                last_time = time.time()
        finally:
            camera.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    with _state_lock:
        return {
            'count': int(_latest_count),
            'no_helmet': int(_latest_no_helmet),
            'no_vest': int(_latest_no_vest),
            'fps': float(_latest_fps)
        }


@app.route('/snapshot')
def snapshot():
    with _state_lock:
        if _latest_jpg is None:
            return Response(status=204)
        return Response(_latest_jpg, mimetype='image/jpeg')


@app.route('/violations/download')
def download_violations():
    """Download violations CSV file"""
    try:
        return send_from_directory(DATA_DIR, 'violations.csv', as_attachment=True, download_name=f'violations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except FileNotFoundError:
        return Response('No violations recorded yet', status=404)


@app.route('/violations/view')
def view_violations():
    """Get violations data as JSON"""
    try:
        violations = []
        with open(VIOLATIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                violations.append(row)
        return {'violations': violations, 'total': len(violations)}
    except FileNotFoundError:
        return {'violations': [], 'total': 0}


@app.route('/violations/stats')
def violations_stats():
    """Get violation statistics"""
    try:
        stats = {
            'total': 0,
            'no_helmet': 0,
            'no_vest': 0,
            'by_camera': {},
            'by_date': {},
            'recent': []
        }
        
        with open(VIOLATIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            violations = list(reader)
            
            stats['total'] = len(violations)
            
            for v in violations:
                # Count by type
                if 'no_helmet' in v.get('Violation_Type', ''):
                    stats['no_helmet'] += 1
                if 'no_vest' in v.get('Violation_Type', ''):
                    stats['no_vest'] += 1
                
                # Count by camera
                camera = v.get('Camera_Area', 'Unknown')
                stats['by_camera'][camera] = stats['by_camera'].get(camera, 0) + 1
                
                # Count by date
                date = v.get('Date', 'Unknown')
                stats['by_date'][date] = stats['by_date'].get(date, 0) + 1
            
            # Get recent 10 violations
            stats['recent'] = violations[-10:][::-1] if len(violations) > 0 else []
        
        return stats
    except FileNotFoundError:
        return {
            'total': 0,
            'no_helmet': 0,
            'no_vest': 0,
            'by_camera': {},
            'by_date': {},
            'recent': []
        }


@app.route('/violations/reset', methods=['POST'])
def reset_violations():
    """Reset/clear all violations data"""
    try:
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup CSV if exists
        if os.path.exists(VIOLATIONS_CSV):
            backup_csv = os.path.join(DATA_DIR, f'violations_backup_{backup_timestamp}.csv')
            try:
                import shutil
                shutil.copy2(VIOLATIONS_CSV, backup_csv)
                print(f"✓ Backed up violations.csv to {backup_csv}")
            except Exception as e:
                print(f"Backup warning: {e}")
        
        # Backup JSON if exists
        if os.path.exists(VIOLATIONS_JSON):
            backup_json = os.path.join(DATA_DIR, f'violations_backup_{backup_timestamp}.json')
            try:
                import shutil
                shutil.copy2(VIOLATIONS_JSON, backup_json)
                print(f"✓ Backed up violations.json to {backup_json}")
            except Exception as e:
                print(f"Backup warning: {e}")
        
        # Clear CSV and recreate with headers
        with _csv_file_lock:
            with open(VIOLATIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Date', 'Time', 'Person_ID', 'Violation_Type', 'Camera_Area', 'Frame_Count'])
        
        # Clear JSON
        with open(VIOLATIONS_JSON, 'w') as f:
            pass
        
        # Clear in-memory log
        global _violation_log, _logged_violations
        _violation_log = []
        _logged_violations = {}
        
        print("✓ Violations data reset successfully")
        
        return {
            'success': True,
            'message': 'All violations cleared successfully',
            'backup_csv': backup_csv if os.path.exists(VIOLATIONS_CSV) else None,
            'backup_json': backup_json if os.path.exists(VIOLATIONS_JSON) else None
        }
        
    except Exception as e:
        print(f"✗ Error resetting violations: {e}")
        return {'success': False, 'message': str(e)}, 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
