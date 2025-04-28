import os
import argparse
import torch
import numpy as np
from PIL import Image
from preprocess.humanparsing.run_parsing import Parsing
import cv2

# ====== Utility functions ======
def remove_zipper_line_straight(mask, ratio):
    h, w = mask.shape
    centers = []
    zipper_widths = []
    for y in range(h):
        x_indices = np.where(mask[y] == 255)[0]
        if len(x_indices) == 0:
            continue
        left = np.min(x_indices)
        right = np.max(x_indices)
        center = (left + right) // 2
        zipper_width = int((right - left) * ratio / 2)
        centers.append(center)
        zipper_widths.append(zipper_width)
    if centers and zipper_widths:
        avg_center = int(np.mean(centers))
        avg_zipper_width = int(np.mean(zipper_widths))
        mask[:, avg_center - avg_zipper_width:avg_center + avg_zipper_width] = 0
    return mask

def remove_zipper_line_natural(mask, ratio):
    h, w = mask.shape
    new_mask = mask.copy()
    for y in range(h):
        x_indices = np.where(mask[y] == 255)[0]
        if len(x_indices) == 0:
            continue
        left = np.min(x_indices)
        right = np.max(x_indices)
        center = (left + right) // 2
        zipper_width = int((right - left) * ratio / 2)
        x_start = max(center - zipper_width, 0)
        x_end = min(center + zipper_width, w)
        new_mask[y, x_start:x_end] = 0
    return new_mask

def extract_upper_cloth(input_image_path, output_mask_path_jpg, output_mask_path_png, densepose_path, naturality, ratio, vest, vest_padding, gpu_id=0, remove_zipper=True):
    parser = Parsing(gpu_id=gpu_id)
    input_image = Image.open(input_image_path).convert("RGB")
    parsed_image, _ = parser(input_image)
    parsed_np = np.array(parsed_image)
    upper_cloth_mask = (parsed_np == 4).astype(np.uint8) * 255

    if remove_zipper:
        if naturality == 0:
            upper_cloth_mask = remove_zipper_line_straight(upper_cloth_mask, ratio)
        elif naturality == 1:
            upper_cloth_mask = remove_zipper_line_natural(upper_cloth_mask, ratio)

    if vest == 1 and densepose_path is not None:
        I = np.load(densepose_path)
        if I.shape != upper_cloth_mask.shape:
            raise ValueError(f"[ERROR] Shape mismatch: DensePose {I.shape} vs upper_cloth_mask {upper_cloth_mask.shape}")
        arms_mask = np.isin(I, [15, 16, 17, 18, 19, 20, 21, 22]).astype(np.uint8)
        if vest_padding != 0:
            kernel = np.ones((vest_padding, vest_padding), np.uint8)
            arms_mask = cv2.dilate(arms_mask, kernel, iterations=1)
        arms_mask = arms_mask.astype(bool)
        upper_cloth_mask[arms_mask] = 0

    upper_cloth_image = Image.fromarray(upper_cloth_mask)
    upper_cloth_image.save(output_mask_path_jpg)
    upper_cloth_image.save(output_mask_path_png)

# ====== Main script ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upper cloth mask extraction script for a folder of images.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing person images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save masks')
    parser.add_argument('--densepose_dir', type=str, default=None, help='Directory containing DensePose npy files (optional)')
    parser.add_argument('--naturality', type=int, default=1, help='Naturality of zipper line removal (1: natural, 0: straight)')
    parser.add_argument('--ratio', type=float, default=0.1, help='Ratio for zipper line removal')
    parser.add_argument('--vest', type=int, default=1, help='Vest mode (1: vest, 0: non-vest)')
    parser.add_argument('--vest_padding', type=int, default=10, help='Padding size for arms removal')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(args.input_dir, filename)
            output_jpg = os.path.join(args.output_dir, filename)
            output_png = os.path.join(args.output_dir, os.path.splitext(filename)[0] + "_mask.png")

            densepose_path = None
            if args.densepose_dir:
                densepose_file = os.path.splitext(filename)[0] + ".npy"
                densepose_path = os.path.join(args.densepose_dir, densepose_file)
                if not os.path.exists(densepose_path):
                    densepose_path = None  # Ignore if not found

            extract_upper_cloth(
                input_image_path=input_path,
                output_mask_path_jpg=output_jpg,
                output_mask_path_png=output_png,
                densepose_path=densepose_path,
                naturality=args.naturality,
                ratio=args.ratio,
                vest=args.vest,
                vest_padding=args.vest_padding
            )

            print(f"Processed {filename}")

    print(f"Done! Upper cloth masks saved to {args.output_dir}")
