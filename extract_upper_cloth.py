import os
import torch
import numpy as np
from PIL import Image
from preprocess.humanparsing.run_parsing import Parsing

def remove_zipper_line(mask, ratio=0.2):
    """
    Function to remove the zipper area in the middle of the upper clothing mask.
    :param mask: Upper clothing mask (255=upper clothing, 0=background)
    :param ratio: The percentage of the image width to remove from the center (default is 20%)
    :return: Mask with the zipper area removed
    """
    h, w = mask.shape
    center = w // 2
    zipper_width = int(w * ratio / 2)
    mask[:, center - zipper_width:center + zipper_width] = 0
    return mask

def extract_upper_cloth(input_image_path, output_mask_path, gpu_id=0, remove_zipper=True):
     # Initialize the parsing class (Load the ONNX model)
    parser = Parsing(gpu_id=gpu_id)

    # Load the input image (in PIL format)
    input_image = Image.open(input_image_path).convert("RGB")

    # Perform parsing
    parsed_image, _ = parser(input_image) # parsed_image is a PIL image (palette applied to indexed image)

    # Convert to numpy array
    parsed_np = np.array(parsed_image)

    # Extract only the upper clothing area (class ID 4)
    upper_cloth_mask = (parsed_np == 4).astype(np.uint8) * 255

    # Remove zipper area
    if remove_zipper:
        upper_cloth_mask = remove_zipper_line(upper_cloth_mask)

    # Save the mask
    upper_cloth_image = Image.fromarray(upper_cloth_mask)
    upper_cloth_image.save(output_mask_path)
    print(f"[INFO] Upper cloth mask saved to: {output_mask_path}")

# Example execution
if __name__ == "__main__":
    input_image_path = "datasets/my_vest_data/test/image/suit_man.png"
    output_mask_path = "datasets/my_vest_data/test/agnostic-mask/suit_man_mask.png"
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    extract_upper_cloth(input_image_path, output_mask_path)