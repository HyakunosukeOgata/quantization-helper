# scripts/download_test_image.py

import os
import urllib.request

os.makedirs("tests", exist_ok=True)

url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
save_path = "tests/sample_cat.jpg"

print(f"Downloading test image to: {save_path} ...")
urllib.request.urlretrieve(url, save_path)
print("✅ Download complete.")
