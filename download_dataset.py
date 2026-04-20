"""
INRIA Person Dataset Downloader
Downloads and extracts the INRIA Person dataset for the HOG project.

Usage:
    python3 download_dataset.py

If automatic download fails, instructions for manual download are provided.
"""

import os
import sys
import tarfile
import zipfile
import shutil
import urllib.request
from pathlib import Path


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Known mirrors / download sources
DOWNLOAD_URLS = [
    # Official INRIA source
    "http://pascal.inrialpes.fr/data/human/INRIAPerson.tar",
    # HTTPS variant
    "https://pascal.inrialpes.fr/data/human/INRIAPerson.tar",
]


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress indicator."""
    print(f"  Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        with open(dest_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\r  Progress: {downloaded/1e6:.1f}MB / {total_size/1e6:.1f}MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r  Downloaded: {downloaded/1e6:.1f}MB...", end="")
        
        print()
        
        # Check if we got a real file (not an error page)
        if os.path.getsize(dest_path) < 10000:
            os.remove(dest_path)
            print(f"  File too small, likely an error page. Skipping.")
            return False
        
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar or zip archive."""
    print(f"  Extracting to {extract_to}...")
    
    if archive_path.endswith('.tar') or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        mode = 'r:gz' if archive_path.endswith('.gz') or archive_path.endswith('.tgz') else 'r'
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(path=extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(path=extract_to)
    else:
        print(f"  Unknown archive format: {archive_path}")
        return False
    
    return True


def verify_dataset(data_dir):
    """Verify the dataset structure is correct."""
    required = [
        os.path.join(data_dir, "Train", "pos"),
        os.path.join(data_dir, "Train", "neg"),
        os.path.join(data_dir, "Test", "pos"),
        os.path.join(data_dir, "Test", "neg"),
    ]
    
    # Check in subdirectories too
    for root, dirs, files in os.walk(data_dir):
        for d in ['INRIAPerson', '96X160H96']:
            candidate = os.path.join(root, d)
            if os.path.isdir(candidate):
                sub_required = [os.path.join(candidate, r.replace(data_dir + "/", "")) for r in required]
                if all(os.path.isdir(r) for r in sub_required):
                    return True
    
    return all(os.path.isdir(r) for r in required)


def print_manual_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
Automatic download failed. Please download the dataset manually:

Option 1 (Kaggle - Recommended):
  1. Go to: https://www.kaggle.com/datasets/jcoral/inriaperson
  2. Click 'Download' (you may need to sign in)
  3. Extract the downloaded ZIP file
  4. Place the contents so the structure looks like:
     data/
     ├── Train/
     │   ├── pos/    (positive training images)
     │   └── neg/    (negative training images)
     └── Test/
         ├── pos/    (positive test images)
         └── neg/    (negative test images)

Option 2 (Direct download):
  1. Go to: http://pascal.inrialpes.fr/data/human/
  2. Download INRIAPerson.tar
  3. Extract to the 'data/' folder in this project

After downloading, run this script again to verify:
  python3 download_dataset.py --verify
""")


def main():
    """Main download flow."""
    print("=" * 60)
    print("  INRIA Person Dataset Downloader")
    print("=" * 60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if dataset already exists
    if verify_dataset(DATA_DIR):
        print("\n✓ Dataset already exists and is valid!")
        return True
    
    if "--verify" in sys.argv:
        print("\n✗ Dataset not found or incomplete.")
        print_manual_instructions()
        return False
    
    # Try automatic download
    print("\nAttempting automatic download...")
    archive_path = os.path.join(DATA_DIR, "INRIAPerson.tar")
    
    for url in DOWNLOAD_URLS:
        if download_file(url, archive_path):
            if extract_archive(archive_path, DATA_DIR):
                os.remove(archive_path)  # Clean up
                if verify_dataset(DATA_DIR):
                    print("\n✓ Dataset downloaded and verified successfully!")
                    return True
                else:
                    print("\n⚠ Downloaded but structure looks different. Checking...")
    
    # If we get here, automatic download failed
    print_manual_instructions()
    return False


if __name__ == "__main__":
    main()
