"""
Convert YOLO models to ONNX format for faster inference.
Run this script once to convert both models.
"""
from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
CUSTOM_MODEL = MODELS_DIR / 'exp9' / 'weights' / 'best.pt'
PERSON_MODEL = MODELS_DIR / 'yolov8n.pt'


def convert_models():
    """Export both models to ONNX in-place next to the .pt files."""
    print("Converting custom helmet/vest detection model to ONNX...")
    if CUSTOM_MODEL.exists():
        try:
            model = YOLO(str(CUSTOM_MODEL))
            model.export(format='onnx', imgsz=640, simplify=True)
            print(f"✓ Successfully converted {CUSTOM_MODEL} to ONNX")
            print(f"  Output: {CUSTOM_MODEL.with_suffix('.onnx')}")
        except Exception as e:
            print(f"✗ Error converting custom model: {e}")
    else:
        print(f"✗ Custom model not found at {CUSTOM_MODEL}")
    
    print("\nConverting person detection model to ONNX...")
    try:
        model = YOLO(str(PERSON_MODEL))
        model.export(format='onnx', imgsz=640, simplify=True)
        print(f"✓ Successfully converted {PERSON_MODEL} to ONNX")
        print(f"  Output: {PERSON_MODEL.with_suffix('.onnx')}")
    except Exception as e:
        print(f"✗ Error converting person model: {e}")
    
    print("\n" + "="*60)
    print("ONNX Conversion Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install ONNX Runtime: pip install onnxruntime")
    print("2. For GPU support: pip install onnxruntime-gpu")
    print("3. Run the updated app.py")


if __name__ == '__main__':
    convert_models()
