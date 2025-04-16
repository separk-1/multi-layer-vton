import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

def generate_upper_body_mask(image_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = pose.process(image_rgb)
    if not result.pose_landmarks:
        print(f"No person detected in {image_path}")
        return

    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    keypoints = result.pose_landmarks.landmark
    y_coords = [
        keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        keypoints[mp_pose.PoseLandmark.LEFT_HIP].y,
        keypoints[mp_pose.PoseLandmark.RIGHT_HIP].y
    ]
    x_coords = [
        keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
        keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
        keypoints[mp_pose.PoseLandmark.LEFT_HIP].x,
        keypoints[mp_pose.PoseLandmark.RIGHT_HIP].x
    ]

    y_min = int(min(y_coords) * h)
    y_max = int(max(y_coords) * h)
    x_min = int(min(x_coords) * w)
    x_max = int(max(x_coords) * w)

    # 상체 영역을 흰색(255)으로
    mask[y_min:y_max, x_min:x_max] = 255

    Image.fromarray(mask).save(output_path)
    print(f"Saved mask to {output_path}")


generate_upper_body_mask(
    "datasets/my_vest_data/test/image/suit_man.png",
    "datasets/my_vest_data/test/agnostic-mask/suit_man.png"
)

generate_upper_body_mask(
    "datasets/my_vest_data/test/image/construction.jpg",
    "datasets/my_vest_data/test/agnostic-mask/construction.jpg"
)
