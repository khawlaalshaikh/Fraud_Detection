import zipfile
import os

zip_path = "data/raw/archive.zip"
extract_path = "data/external"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Data extracted to:", extract_path)