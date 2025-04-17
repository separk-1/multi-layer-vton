from PIL import Image
import os

def create_dummy_densepose_png(image_path, png_output_path):
    # 출력 디렉토리 자동 생성
    os.makedirs(os.path.dirname(png_output_path), exist_ok=True)

    # 이미지 열어서 사이즈 확인
    with Image.open(image_path) as im:
        w, h = im.size

    # PNG용 회색(dummy) 이미지 저장
    dummy_image = Image.new("RGB", (w, h), (128, 128, 128))
    dummy_image.save(png_output_path)
    print(f"✅ Saved RGB dummy densepose to {png_output_path}")

if __name__ == "__main__":
    items = [
        ("./datasets/my_vest_data/test/image/suit_man.png",
         "./datasets/my_vest_data/test/image-densepose/suit_man.png"),
        ("./datasets/my_vest_data/test/image/model.jpg",
         "./datasets/my_vest_data/test/image-densepose/model.jpg"),
    ]

    for img_path, png_path in items:
        create_dummy_densepose_png(img_path, png_path)
