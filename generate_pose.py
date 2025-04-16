from PIL import Image
import os

def create_rgb_dummy_densepose(image_path, output_path):
    # 원본 사이즈 그대로 가져오기
    with Image.open(image_path) as im:
        w, h = im.size
    
    # RGB 3채널 회색(dummy) 이미지 생성
    dummy = Image.new("RGB", (w, h), (128, 128, 128))
    dummy.save(output_path)
    print(f"✅ Saved RGB dummy densepose to {output_path}")

# suit_man.png
create_rgb_dummy_densepose(
    "datasets/my_vest_data/test/image/suit_man.png",
    "datasets/my_vest_data/test/image-densepose/suit_man.png"
)

# construction.jpg
create_rgb_dummy_densepose(
    "datasets/my_vest_data/test/image/construction.jpg",
    "datasets/my_vest_data/test/image-densepose/construction.jpg"
)
