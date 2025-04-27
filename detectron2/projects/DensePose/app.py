import sys
import os
import torch
import numpy as np
import cv2

#1. apply_net.py -> generate results.pkl
sys.path.append(os.path.abspath("pose_generator/detectron2/projects/DensePose"))   
from apply_net import main
if __name__ == "__main__":
    sys.argv = [
        "apply_net.py",
        "dump",
        "pose_generator/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml", 
        "pose_generator/detectron2/projects/DensePose/ckpt/model_final_162be9.pkl", #download: https://huggingface.co/yisol/IDM-VTON/tree/main/densepose
        "datasets/my_vest_data/test/image/joon.jpg", #input image
        "--output", "datasets/my_vest_data/test/image-densepose/results.pkl", #output directory and file name
        "-v"
    ]

    main()

#2. Load results.pkl, generate I.npy
data = torch.load('datasets/my_vest_data/test/image-densepose/results.pkl')

target_h, target_w = 768, 576

for idx, item in enumerate(data):
    densepose_outputs = item['pred_densepose']

    for i, densepose_output in enumerate(densepose_outputs):
        labels = densepose_output.labels.cpu().numpy() 

        labels_resized = cv2.resize(labels, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        np.save(f"datasets/my_vest_data/test/image-densepose/I.npy", labels_resized) #I.npy directory

#3. Load I.npy, generate "pose" image
I = np.load('datasets/my_vest_data/test/image-densepose/I.npy')  #I.npy directory

##!!!! the direction(right, left) is from your view!!!!
DENSEPOSE_COLORS = [
    (0, 0, 0),
    (255, 0, 85),
    (194, 80, 20), #upper body
    (224, 98, 4), #left hand 
    (221, 110, 8), #right hand
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (198, 166, 6), #left leg(upper)
    (184, 173, 22),#right leg(upper)
    (0, 170, 255), 
    (0, 85, 255), 
    (25, 235, 251),  #left leg(lower)
    (65, 235, 251), #right leg(lower)
    (116, 191, 145), #right arm(upper, inside) 
    (104, 191, 168), #left arm(upper, inside)
    (96, 189, 192), #right arm(upper, outside)
    (87, 187, 216), #left arm(upper, outside)
    (73, 191, 227), #right arm(lower, inside)
    (60, 198, 240), #left arm(lower, inside)
    (46, 207, 252),#right arm(lower, outside)
    (38, 220, 250), #left arm(lower, outside)
    (25, 235, 251), #left head 
    (14, 251, 248) #right head
]

h, w = I.shape
segmentation_color = np.zeros((h, w, 3), dtype=np.uint8)

for part_id in range(1, 25):  # 1~24
    segmentation_color[I == part_id] = DENSEPOSE_COLORS[part_id]

cv2.imwrite('datasets/my_vest_data/test/image-densepose/joon.jpg', segmentation_color) # "pose" image location

print("Done! DensePose saved!")