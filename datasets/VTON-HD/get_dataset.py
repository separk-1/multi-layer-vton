import os
import shutil

# -----------------------------------------------------------
# This script copies selected image-cloth pairs from the local
# VTON-HD dataset (downloaded in ~/Downloads/archive/test/)
# into the current project's dataset folder.
# 
# It reads image-cloth pairs from "test_pairs.txt" and copies
# the corresponding files to ./test/image and ./test/cloth.
#
# Make sure to prepare your own "test_pairs.txt" to control
# which samples you want to use.
# -----------------------------------------------------------

# ====== Step 1: Path Settings ======
source_image_dir = os.path.expanduser("~/Downloads/archive/test/image")  # Source folder for person images
source_cloth_dir = os.path.expanduser("~/Downloads/archive/test/cloth")  # Source folder for cloth images
target_image_dir = "./test/image"  # Target folder for person images (local project)
target_cloth_dir = "./test/cloth"  # Target folder for cloth images (local project)
pair_file = "test_pairs.txt"        # File containing pairs: [person_image cloth_image]

# ====== Step 2: Create Target Directories If Not Exist ======
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_cloth_dir, exist_ok=True)

# ====== Step 3: Read Image-Cloth Pairs ======
with open(pair_file, "r") as f:
    lines = f.readlines()

# ====== Step 4: Copy Files According to Pairs ======
for line in lines:
    image_name, cloth_name = line.strip().split()  # Each line has: person_image cloth_image

    # Build full source and destination paths
    src_image_path = os.path.join(source_image_dir, image_name)
    src_cloth_path = os.path.join(source_cloth_dir, cloth_name)
    dst_image_path = os.path.join(target_image_dir, image_name)
    dst_cloth_path = os.path.join(target_cloth_dir, cloth_name)

    # Copy person image
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
        print(f"Copied {image_name} to {target_image_dir}")
    else:
        print(f"[WARNING] Image file {src_image_path} not found!")

    # Copy cloth image
    if os.path.exists(src_cloth_path):
        shutil.copy(src_cloth_path, dst_cloth_path)
        print(f"Copied {cloth_name} to {target_cloth_dir}")
    else:
        print(f"[WARNING] Cloth file {src_cloth_path} not found!")
