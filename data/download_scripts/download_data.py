import os
import urllib.request

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES = [
    "processed.cleveland.data",
    "heart-disease.names"
]

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

for file in FILES:
    url = BASE_URL + file
    output_path = os.path.join(RAW_DIR, file)
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(url, output_path)

print("Download completed.")
