"""
Download script for Bangkok Traffy dataset.

This script downloads the raw data from Google Drive or other cloud storage.
Update the file_id with your actual Google Drive file ID.
"""

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("gdown not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Get file ID from shareable link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
GOOGLE_DRIVE_FILE_ID = "19QkF8i1my99gjbyHe7de_qZNwgrca6R5"
OUTPUT_PATH = DATA_DIR / "bangkok_traffy.csv"

def download_data():
    """Download the raw Bangkok Traffy dataset."""
    
    if OUTPUT_PATH.exists():
        print(f"Data already exists at {OUTPUT_PATH}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    if GOOGLE_DRIVE_FILE_ID == "YOUR_FILE_ID_HERE":
        print("WARNING: Please update the GOOGLE_DRIVE_FILE_ID in this script.")
        print("\nTo get the file ID:")
        print("1. Upload bangkok_traffy.csv to Google Drive")
        print("2. Right-click → Share → Copy link")
        print("3. Extract FILE_ID from: https://drive.google.com/file/d/FILE_ID/view")
        print("4. Update GOOGLE_DRIVE_FILE_ID in scripts/download_data.py")
        print("\nFor now, please manually place bangkok_traffy.csv in the data/ folder.")
        return
    
    print(f"Downloading data from Google Drive...")
    print(f"File ID: {GOOGLE_DRIVE_FILE_ID}")
    
    try:
        gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(OUTPUT_PATH),
            quiet=False,
            fuzzy=True
        )
        print(f"Successfully downloaded to {OUTPUT_PATH}")
        
        # Verify file size
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Manually download and place in data/ folder")
        sys.exit(1)

if __name__ == "__main__":
    download_data()
