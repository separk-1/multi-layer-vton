#!/usr/bin/env python3
# resizing.py

import os
from PIL import Image

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
BASE_DIR = "./datasets/my_vest_data/test"
SUBFOLDERS = ["image", "image-densepose", "agnostic-mask", "cloth"]
TARGET_SIZE = (512, 768)  # (width, height)
# -------------------------------------------------------------------


def resize_image_file(input_path: str, size: tuple, resample):
    """
    Open an image, convert to RGB (drop alpha), resize in-place.
    """
    # force RGB → drop any alpha channel
    img = Image.open(input_path).convert("RGB")
    img_resized = img.resize(size, resample=resample)
    img_resized.save(input_path)


def resize_folder(folder: str, size: tuple):
    """
    Walk through all files in a folder, resize each in-place.
    Use NEAREST for masks, BILINEAR for others.
    """
    folder_path = os.path.join(BASE_DIR, folder)
    for fname in os.listdir(folder_path):
        input_path = os.path.join(folder_path, fname)
        if not os.path.isfile(input_path):
            continue

        resample = Image.NEAREST if folder == "agnostic-mask" else Image.BILINEAR
        print(f"Resizing {folder}/{fname} → {size}, drop alpha → RGB")
        resize_image_file(input_path, size, resample)


if __name__ == "__main__":
    for sub in SUBFOLDERS:
        resize_folder(sub, TARGET_SIZE)
    print("✅ All done! Images are now RGB and uniformly sized.")
