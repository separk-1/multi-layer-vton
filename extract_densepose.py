import sys
import os
import argparse
import torch
import numpy as np
import cv2

# ====== Step 1: Parse arguments ======
parser = argparse.ArgumentParser(description="DensePose generation script.")
parser.add_argument('--config', type=str, required=True, help='Path to DensePose config file (.yaml)')
parser.add_argument('--model', type=str, required=True, help='Path to DensePose model checkpoint (.pkl)')
parser.add_argument('--input', type=str, required=True, help='Input image path')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
parser.add_argument('--width', type=int, default=576, help='Target width for output images')
parser.add_argument('--height', type=int, default=768, help='Target height for output images')
args = parser.parse_args()

# ====== Step 2: Setup paths ======
sys.path.append(os.path.abspath("./detectron2/projects/DensePose"))
from apply_net import main

results_pkl_path = os.path.join(args.output_dir, "results.pkl")
I_npy_path = os.path.join(args.output_dir, "I.npy")

pose_filename = os.path.basename(args.input)  # <-- Extract input filename like 'joon.jpg'
pose_image_path = os.path.join(args.output_dir, pose_filename)

os.makedirs(args.output_dir, exist_ok=True)


# ====== Step 3: Run apply_net to generate results.pkl ======
sys.argv = [
    "apply_net.py",
    "dump",
    args.config,
    args.model,
    args.input,
    "--output", results_pkl_path,
    "-v"
]
main()

# ====== Step 4: Load results.pkl and generate I.npy ======
data = torch.load(results_pkl_path)

canvas = np.zeros((args.height, args.width), dtype=np.uint8)

for idx, item in enumerate(data):
    boxes = item['pred_boxes_XYXY']
    densepose_outputs = item['pred_densepose']

    for densepose_output, box in zip(densepose_outputs, boxes):
        labels = densepose_output.labels.cpu().numpy()

        x_min, y_min, x_max, y_max = map(int, box.tolist())

        # Clip coordinates to valid image range
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, args.width)
        y_max = min(y_max, args.height)

        if x_max <= x_min or y_max <= y_min:
            continue  # Skip invalid boxes

        resized_labels = cv2.resize(labels, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
        canvas[y_min:y_max, x_min:x_max] = resized_labels

np.save(I_npy_path, canvas)

# ====== Step 5: Load I.npy and generate pose image ======
I = np.load(I_npy_path)

DENSEPOSE_COLORS = [
    (0, 0, 0), (255, 0, 85), (194, 80, 20), (224, 98, 4), (221, 110, 8),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (198, 166, 6),
    (184, 173, 22), (0, 170, 255), (0, 85, 255), (25, 235, 251), (65, 235, 251),
    (116, 191, 145), (104, 191, 168), (96, 189, 192), (87, 187, 216),
    (73, 191, 227), (60, 198, 240), (46, 207, 252), (38, 220, 250),
    (25, 235, 251), (14, 251, 248)
]

h, w = I.shape
segmentation_color = np.zeros((h, w, 3), dtype=np.uint8)

for part_id in range(1, 25):
    segmentation_color[I == part_id] = DENSEPOSE_COLORS[part_id]

cv2.imwrite(pose_image_path, segmentation_color)

print(f"Done! DensePose and pose image saved at {args.output_dir}")
