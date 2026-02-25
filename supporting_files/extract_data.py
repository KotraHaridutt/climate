#!/usr/bin/env python3
import zipfile
import os

zip_path = '../telangana_weather_data.nc'
extract_dir = '.'

if os.path.exists(zip_path):
    print(f"Found ZIP file: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print("✓ Successfully extracted files:")
            for name in zip_ref.namelist():
                print(f"  - {name}")
    except Exception as e:
        print(f"✗ Error extracting: {e}")
else:
    print(f"✗ File not found: {zip_path}")
