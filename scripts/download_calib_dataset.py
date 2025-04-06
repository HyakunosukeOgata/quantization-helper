# scripts/download_calib_dataset.py

import os
import zipfile
import urllib.request


def download_imagenet_mini(dest_dir="data/imagenet-mini"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "imagenet-mini.zip")

    if os.path.exists(os.path.join(dest_dir, "val")):
        print(f"✔️ Calibration dataset already exists at: {dest_dir}/val")
        return

    print("📦 Downloading ImageNet Mini calibration dataset...")
    url = "https://huggingface.co/datasets/ML4CV/imagenet-mini-calib/resolve/main/imagenet-mini.zip"
    urllib.request.urlretrieve(url, zip_path)

    print("🧩 Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    os.remove(zip_path)
    print(f"✅ Done. Calibration dataset ready at: {dest_dir}/val")


if __name__ == "__main__":
    download_imagenet_mini()
