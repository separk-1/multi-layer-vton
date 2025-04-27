import torch
import numpy as np
import os
import cv2

# 1. 결과 pkl 불러오기
data = torch.load('pose_generator/detectron2/projects/DensePose/results.pkl')

# 2. 디렉토리 생성
os.makedirs("pose_generator/detectron2/projects/DensePose/densepose_outputs", exist_ok=True)

# 3. 타겟 해상도 (원본 이미지 크기)
target_h, target_w = 768, 576

# 4. 하나씩 I 뽑아서 저장
for idx, item in enumerate(data):
    densepose_outputs = item['pred_densepose']  # ★ 여기

    for i, densepose_output in enumerate(densepose_outputs):
        labels = densepose_output.labels.cpu().numpy()   # (H, W)

        # (추가) 원본 사이즈로 리사이즈
        labels_resized = cv2.resize(labels, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 파일 저장
        np.save(f"pose_generator/detectron2/projects/DensePose/densepose_outputs/I.npy", labels_resized)

print("Done! Saved resized I npy files!")