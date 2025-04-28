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
    print(f"Saved center-cropped resized image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and center-crop an image to the target size.")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to save the resized and cropped image")
    parser.add_argument("--width", type=int, required=True, help="Target width of the output image")
    parser.add_argument("--height", type=int, required=True, help="Target height of the output image")

    args = parser.parse_args()

    resize_and_crop_center(args.input, args.output, args.width, args.height)
