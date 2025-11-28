"""
Clear old violations data
Run this script to clear violations.csv and violations.json files
"""
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CSV_PATH = DATA_DIR / 'violations.csv'
JSON_PATH = DATA_DIR / 'violations.json'


def clear_violations():
    """Clear all violations data and create fresh files"""
    DATA_DIR.mkdir(exist_ok=True)
    
    # Backup old files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("Clearing Violations Data")
    print("="*60)
    
    # Backup CSV if exists
    if CSV_PATH.exists():
        backup_csv = DATA_DIR / f'violations_backup_{timestamp}.csv'
        try:
            CSV_PATH.rename(backup_csv)
            print(f"✓ Backed up violations.csv to {backup_csv}")
        except Exception as e:
            print(f"✗ Error backing up CSV: {e}")
    
    # Backup JSON if exists
    if JSON_PATH.exists():
        backup_json = DATA_DIR / f'violations_backup_{timestamp}.json'
        try:
            JSON_PATH.rename(backup_json)
            print(f"✓ Backed up violations.json to {backup_json}")
        except Exception as e:
            print(f"✗ Error backing up JSON: {e}")
    
    # Create new CSV with headers
    try:
        with CSV_PATH.open('w', newline='', encoding='utf-8') as f:
            f.write('Timestamp,Date,Time,Person_ID,Violation_Type,Camera_Area,Frame_Count\n')
        print("✓ Created fresh violations.csv")
    except Exception as e:
        print(f"✗ Error creating CSV: {e}")
    
    print("\n" + "="*60)
    print("Violations data cleared successfully!")
    print("="*60)
    print("\nOld data backed up with timestamp.")
    print("New violations will now show correct camera names.")
    print("\nRestart the app: python app.py")


if __name__ == '__main__':
    response = input("Clear all violations data? Old data will be backed up. (yes/no): ")
    if response.lower() in ['yes', 'y']:
        clear_violations()
    else:
        print("Operation cancelled.")
