"""
Add test violations for multiple cameras to verify filter
This script adds sample violations to test the camera filter
"""
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CSV_PATH = DATA_DIR / 'violations.csv'


def add_test_violations():
    """Add test violations from all 3 cameras"""
    DATA_DIR.mkdir(exist_ok=True)
    
    print("Adding test violations to verify camera filter...")
    print("=" * 60)
    
    # Test violations for each camera
    test_violations = [
        # Inner Corridor violations
        ("Inner Corridor", "Person 200", "no_helmet"),
        ("Inner Corridor", "Person 201", "no_vest"),
        ("Inner Corridor", "Person 202", "no_helmet"),
        
        # Inner Plant Area violations
        ("Inner Plant Area", "Person 300", "no_vest"),
        ("Inner Plant Area", "Person 301", "no_helmet"),
        ("Inner Plant Area", "Person 302", "no_vest"),
    ]
    
    # Add to CSV
    try:
        with CSV_PATH.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for camera, person_id, violation_type in test_violations:
                dt_now = datetime.now()
                writer.writerow([
                    dt_now.strftime('%Y-%m-%d %H:%M:%S'),  # Timestamp
                    dt_now.strftime('%Y-%m-%d'),           # Date
                    dt_now.strftime('%H:%M:%S'),           # Time
                    person_id,                              # Person ID
                    violation_type,                         # Violation Type
                    camera,                                 # Camera Area
                    5                                       # Frame Count
                ])
                print(f"✓ Added: {person_id} - {violation_type} at {camera}")
        
        print("\n" + "=" * 60)
        print("✓ Test violations added successfully!")
        print("\nNow check violations dashboard:")
        print("  http://localhost:5000/violations.html")
        print("\nFilter should now show all 3 cameras:")
        print("  - Outside Plant Area")
        print("  - Inner Corridor")
        print("  - Inner Plant Area")
        
    except Exception as e:
        print(f"✗ Error adding test violations: {e}")

if __name__ == '__main__':
    add_test_violations()
