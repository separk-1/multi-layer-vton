import sys
import os
import argparse
import torch
import numpy as np
import cv2

# ====== Step 1: Parse arguments ======
parser = argparse.ArgumentParser(description="DensePose generation script for all images in a folder.")
parser.add_argument('--config', type=str, required=True, help='Path to DensePose config file (.yaml)')
parser.add_argument('--model', type=str, required=True, help='Path to DensePose model checkpoint (.pkl)')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
parser.add_argument('--width', type=int, default=576, help='Target width for output images')
parser.add_argument('--height', type=int, default=768, help='Target height for output images')
args = parser.parse_args()

# ====== Step 2: Setup paths ======
sys.path.append(os.path.abspath("./detectron2/projects/DensePose"))
from apply_net import main as apply_net_main

os.makedirs(args.output_dir, exist_ok=True)

# ====== Step 3: Process each image ======
for filename in os.listdir(args.input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(args.input_dir, filename)
        output_prefix = os.path.splitext(filename)[0]

        results_pkl_path = os.path.join(args.output_dir, f"{output_prefix}_results.pkl")
        I_npy_path = os.path.join(args.output_dir, f"{output_prefix}.npy")
        pose_image_path = os.path.join(args.output_dir, f"{output_prefix}.jpg")

        # Run apply_net to generate results.pkl
        sys.argv = [
            "apply_net.py",
            "dump",
            args.config,
            args.model,
            input_path,
            "--output", results_pkl_path,
            "-v"
        ]
        apply_net_main()

        # Load results.pkl and generate I.npy
        data = torch.load(results_pkl_path)
        canvas = np.zeros((args.height, args.width), dtype=np.uint8)

        for idx, item in enumerate(data):
            boxes = item['pred_boxes_XYXY']
            densepose_outputs = item['pred_densepose']

            for densepose_output, box in zip(densepose_outputs, boxes):
                labels = densepose_output.labels.cpu().numpy()

                x_min, y_min, x_max, y_max = map(int, box.tolist())
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, args.width)
                y_max = min(y_max, args.height)

                if x_max <= x_min or y_max <= y_min:
                    continue

                resized_labels = cv2.resize(labels, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                canvas[y_min:y_max, x_min:x_max] = resized_labels

        np.save(I_npy_path, canvas)

        # Load I.npy and generate pose image
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

        print(f"Processed {filename}: outputs saved to {args.output_dir}")

print("Done! All DensePose maps and pose images are generated.")
