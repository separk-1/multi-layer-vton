import torch
import cv2
import numpy as np

from densepose.config import add_densepose_config
from detectron2.config.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import numpy as np
import cv2
import os

def save_iuv_fullsize(image_rgb, outputs, save_dir):
    """
    DensePose output을 원본 이미지 크기에 맞춰 I, U, V map 저장
    """
    densepose_outputs, pred_boxes = outputs  # tuple 언팩
    height, width, _ = image_rgb.shape

    I_full = np.zeros((height, width), dtype=np.uint8)
    U_full = np.zeros((height, width), dtype=np.float32)
    V_full = np.zeros((height, width), dtype=np.float32)

    for idx in range(len(densepose_outputs)):
        dp_output = densepose_outputs[idx]

        bbox = pred_boxes[idx]
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1

        # 🔥 batch dimension 제거
        coarse_segm = dp_output.coarse_segm.squeeze(0)  # (2,112,112)
        fine_segm = dp_output.fine_segm.squeeze(0)      # (25,112,112)
        U_all = dp_output.u.squeeze(0).cpu().numpy()    # (25,112,112)
        V_all = dp_output.v.squeeze(0).cpu().numpy()    # (25,112,112)

        # 1. 사람/배경 segmentation
        S = coarse_segm.argmax(dim=0).cpu().numpy()  # (112,112)

        # 2. 부위 segmentation
        I_all = fine_segm.argmax(dim=0).cpu().numpy()  # (112,112)

        # 3. 사람인 부분만 살리기
        I = np.where(S == 1, I_all, 0).astype(np.uint8)

        # 4. 🔥 U, V에서 부위별 픽셀 선택
        H, W = I.shape
        U = np.zeros((H, W), dtype=np.float32)
        V = np.zeros((H, W), dtype=np.float32)

        for part_id in range(1, 25):  # 1~24번 부위만 (0은 background)
            mask = (I == part_id)
            U[mask] = U_all[part_id][mask]
            V[mask] = V_all[part_id][mask]

        # 5. 🔥 bbox 사이즈에 맞게 resize
        I_resized = cv2.resize(I, (w, h), interpolation=cv2.INTER_NEAREST)
        U_resized = cv2.resize(U, (w, h), interpolation=cv2.INTER_LINEAR)
        V_resized = cv2.resize(V, (w, h), interpolation=cv2.INTER_LINEAR)

        # 6. 🔥 Full size map에 삽입
        I_full[y1:y2, x1:x2] = I_resized
        U_full[y1:y2, x1:x2] = U_resized
        V_full[y1:y2, x1:x2] = V_resized

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "I.npy"), I_full)
    np.save(os.path.join(save_dir, "U.npy"), U_full)
    np.save(os.path.join(save_dir, "V.npy"), V_full)

def create_segmentation_image(I_full, save_path):
    """
    I_full (DensePose의 부위 인덱스 map)을 원본 크기로 컬러 시각화해서 저장.

    Args:
        I_full: (H, W) np.uint8, 각 픽셀에 0~24 부위 번호
        save_path: 저장할 경로
    """

    # 1. (H, W, 3) 색깔 입히기
    height, width = I_full.shape
    color_map = np.array([
        [0, 0, 0],        # 0: background → black
        [255, 0, 0],      # 1: body part 1 → blue
        [0, 255, 0],      # 2: body part 2 → green
        [0, 0, 255],      # 3: body part 3 → red
        [255, 255, 0],    # 4: ...
        [255, 0, 255], #5
        [0, 255, 255],#6
        [128, 0, 0],#7
        [0, 128, 0],#8
        [0, 0, 128],#9
        [128, 128, 0],#10
        [128, 0, 128],#11
        [0, 128, 128],#12
        [64, 0, 0],#13
        [0, 64, 0],#14
        [0, 0, 64],#15
        [64, 64, 0],#16
        [64, 0, 64],#17
        [0, 64, 64],#18
        [192, 0, 0],#19
        [0, 192, 0],#20
        [0, 0, 192],#21
        [192, 192, 0],#22
        [192, 0, 192],#23
        [0, 192, 192],#24
    ], dtype=np.uint8)

    seg_img = np.zeros((height, width, 3), dtype=np.uint8)

    for part_id in range(25):
        seg_img[I_full == part_id] = color_map[part_id]

    # 2. 🔥 원본 해상도로 부드럽게 upsample
    seg_img_upscaled = cv2.resize(seg_img, (width, height), interpolation=cv2.INTER_LINEAR)
    seg_img_upscaled = cv2.GaussianBlur(seg_img_upscaled, (3, 3), sigmaX=0.5)

    # 2. 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, seg_img_upscaled)
    print(f"✅ Segmentation image saved at: {save_path}")
# ---------------------

# 1. Config 설정
cfg = get_cfg()
add_densepose_config(cfg)

cfg.merge_from_file("configs/densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "ckpt/densepose/model_final_162be9.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DENSEPOSE_VISUALIZE = "I"  # (🔥 중요)
cfg.freeze()

# 2. Predictor 생성
predictor = DefaultPredictor(cfg)

# 3. 이미지 불러오기
image = cv2.imread("datasets/my_vest_data/test/image/suit.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. 모델에 넣기
outputs = predictor(image_rgb)

# 5. Instances 꺼내기
instances = outputs["instances"].to("cpu")

# 6. DensePose output 있는지 체크
if instances.has("pred_densepose"):

    densepose_outputs = instances.pred_densepose  # List of DensePoseOutput
    pred_boxes = instances.pred_boxes.tensor.numpy()  # (N, 4) shape

    # 7. Visualizer 만들기
    from densepose.vis.densepose_outputs_iuv import DensePoseOutputsVisualizer
    visualizer = DensePoseOutputsVisualizer(cfg, to_visualize="I")

    # 8. visualize
    vis_output = visualizer.visualize(
        image_rgb,
        (densepose_outputs, pred_boxes)  # tuple 넘김
    )

    vis_output_bgr = cv2.cvtColor(vis_output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("densepose_result.jpg", vis_output_bgr)

    # 🔥 여기서 save
    save_dir = "datasets/my_vest_data/test/image-densepose"  # 경로 수정 (슬래시)
    save_iuv_fullsize(image_rgb, (densepose_outputs, pred_boxes), save_dir)
    
    I_full = np.load("datasets/my_vest_data/test/image-densepose/I.npy")  # (H, W)

    save_path = "datasets/my_vest_data/test/image-densepose/segmentation_clean.jpg"

    create_segmentation_image(I_full, save_path)
    print("No DensePose prediction found.")

'''
# 5. DensePose 결과 가져오기
densepose_output = outputs["instances"].get("pred_densepose")

# 6. (Optional) DensePose 결과 시각화
if densepose_output is not None:
    from densepose.vis.densepose_outputs_vertex import DensePoseOutputsVertexVisualizer

    visualizer = DensePoseOutputsVertexVisualizer(cfg)
    vis_output = visualizer.visualize(image_rgb, outputs["instances"])

    vis_output_bgr = cv2.cvtColor(vis_output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("densepose_result.jpg", vis_output_bgr)
    '''