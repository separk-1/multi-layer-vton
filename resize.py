import argparse
from PIL import Image
import os

def resize_and_crop_center(input_path, output_path, target_width, target_height):
    # Open the image
    img = Image.open(input_path).convert("RGB")
    original_width, original_height = img.size

    # Calculate the scaling factor to maintain aspect ratio
    scale = max(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.BILINEAR)

    # Calculate cropping coordinates (center crop)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    img_cropped = img_resized.crop((left, top, right, bottom))

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the result
    img_cropped.save(output_path)
    print(f"Saved {output_path}")

def process_directory(input_dir, output_dir, target_width, target_height):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_and_crop_center(input_path, output_path, target_width, target_height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and center-crop all images in a folder to the target size.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory")
    parser.add_argument("--width", type=int, required=True, help="Target width of the output images")
    parser.add_argument("--height", type=int, required=True, help="Target height of the output images")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.width, args.height)
